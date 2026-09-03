"""Session monitoring service — thin coordinator and poll loop.

Orchestrates the session-monitoring subsystem:
  1. Reads hook events via event_reader and dispatches them.
  2. Reconciles session_map changes via SessionLifecycle.
  3. Reads transcript updates via TranscriptReader.
  4. Emits NewMessage / NewWindowEvent to registered callbacks.

All heavy logic lives in the extracted modules:
  - event_reader.py   — reads events.jsonl incrementally
  - idle_tracker.py   — per-session idle timers
  - session_lifecycle.py — session-map diff, claude_task_state authority
  - transcript_reader.py — transcript I/O and parsing

Key classes: SessionMonitor, NewMessage, NewWindowEvent, SessionInfo.
Re-exported from transcript_reader for backward-compatible imports.
"""

import asyncio
import contextlib
import structlog
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from telegram.error import TelegramError

from .config import config
from .delivery_contract import (
    DeliveryReceipt,
    activate_delivery_receipt,
    deactivate_delivery_receipt,
    new_delivery_receipt,
    settled_prefix,
    settled_run_offset,
)
from .event_reader import read_new_events
from .idle_tracker import IdleTracker
from .monitor_state import BacklogSkipIntent, MonitorState, TrackedSession
from .providers import get_provider_for_window, registry  # noqa: F401 (used by test patches)
from .session_map import (
    acknowledge_replay_from_start,
    parse_session_map,
    read_session_map_raw,
    session_map_prefix,
)
from .session_lifecycle import session_lifecycle
from .multiplexer import multiplexer as tmux_manager
from .multiplexer.reconciliation import list_windows_for_reconciliation
from .multiplexer.window_liveness import note_live_windows
from .multiplexer.topic_mapping import is_agent_topic_window
from .monitor_events import NewMessage, NewWindowEvent, SessionInfo
from .transcript_reader import TranscriptReader
from .utils import task_done_callback

import json

# Re-export for backward-compatible imports from other modules
__all__ = [
    "NewMessage",
    "NewWindowEvent",
    "SessionInfo",
    "SessionMonitor",
    "get_active_monitor",
    "set_active_monitor",
]

_CallbackError = Exception
_LoopError = (OSError, RuntimeError, json.JSONDecodeError, ValueError, TelegramError)

_BACKOFF_MIN = 2.0
_BACKOFF_MAX = 30.0
_SKIP_RETRY_BASE_SECONDS = 2.0
_SKIP_RETRY_MAX_SECONDS = 60.0
_MSG_PREVIEW_LENGTH = 80

logger = structlog.get_logger()


class SessionMonitor:
    """Monitors Claude Code sessions for new assistant messages.

    Thin coordinator: delegates I/O to TranscriptReader, event reading to
    event_reader, session-map diffing to SessionLifecycle, and idle tracking
    to IdleTracker.
    """

    def __init__(
        self,
        projects_path: Path | None = None,
        poll_interval: float | None = None,
        state_file: Path | None = None,
    ):
        self.projects_path = (
            projects_path if projects_path is not None else config.claude_projects_path
        )
        self.poll_interval = (
            poll_interval if poll_interval is not None else config.monitor_poll_interval
        )

        self.state = MonitorState(state_file=state_file or config.monitor_state_file)
        self.state.load()

        self._running = False
        self._task: asyncio.Task | None = None
        self._message_callback: Callable[[NewMessage], Awaitable[None]] | None = None
        self._new_window_callback: (
            Callable[[NewWindowEvent], Awaitable[None]] | None
        ) = None
        # Lazy: providers.base imports HookEvent and gets imported back
        # through tmux_manager → providers; keep at call site.
        # Lazy: HookEvent pulled by hook dispatch path; defer until that path runs
        from .providers.base import HookEvent

        self._hook_event_callback: Callable[[HookEvent], Awaitable[None]] | None = None

        self._idle_tracker = IdleTracker()
        self._transcript_reader = TranscriptReader(
            self.state,
            self._idle_tracker,
            on_session_retired=self._discard_session_delivery_state,
        )
        # Receipts are grouped by transcript session so one failed send only
        # freezes its own watermark.
        self._delivery_receipts: dict[str, list[DeliveryReceipt]] = {}
        # Backlog skips cross the monitor/queue boundary through injected
        # adapters, preserving this module's handler independence.
        self._skip_purge_callback: (
            Callable[[BacklogSkipIntent], Awaitable[int | None]] | None
        ) = None
        self._skip_validate_callback: Callable[[BacklogSkipIntent], bool] | None = None
        self._skip_notice_callback: (
            Callable[[BacklogSkipIntent], Awaitable[None]] | None
        ) = None
        self._skip_notice_receipts: dict[str, DeliveryReceipt] = {}
        self._skip_retry_attempts: dict[str, int] = {}
        self._skip_retry_at: dict[str, float] = {}

    # Delegation properties for backward-compatible test access
    @property
    def _last_session_map(self) -> dict:
        return session_lifecycle.last_session_map

    @_last_session_map.setter
    def _last_session_map(self, value: dict) -> None:
        session_lifecycle.initialize(value)

    @property
    def _last_activity(self) -> dict:
        return self._idle_tracker._last_activity

    @property
    def _file_mtimes(self) -> dict:
        return self._transcript_reader._file_mtimes

    @property
    def _pending_tools(self) -> dict:
        return self._transcript_reader._pending_tools

    def get_last_activity(self, session_id: str) -> float | None:
        """Get monotonic timestamp of last transcript activity for a session."""
        return self._idle_tracker.get_last_activity(session_id)

    def _discard_session_delivery_state(self, session_id: str) -> None:
        """Drop receipts tied to a session identity that no longer exists."""
        self._delivery_receipts.pop(session_id, None)
        self._skip_notice_receipts.pop(session_id, None)
        self._clear_skip_retry(session_id)

    def set_message_callback(
        self, callback: Callable[[NewMessage], Awaitable[None]]
    ) -> None:
        self._message_callback = callback

    def set_new_window_callback(
        self, callback: Callable[[NewWindowEvent], Awaitable[None]]
    ) -> None:
        self._new_window_callback = callback

    def set_hook_event_callback(self, callback: Callable[..., Awaitable[None]]) -> None:
        self._hook_event_callback = callback

    def set_skip_callbacks(
        self,
        *,
        purge: Callable[[BacklogSkipIntent], Awaitable[int | None]],
        notice: Callable[[BacklogSkipIntent], Awaitable[None]],
        validate: Callable[[BacklogSkipIntent], bool],
    ) -> None:
        """Install the queue adapters used by confirmed backlog skips."""
        self._skip_purge_callback = purge
        self._skip_validate_callback = validate
        self._skip_notice_callback = notice

    async def request_backlog_skip(
        self, user_id: int, window_id: str, thread_id: int | None, chat_id: int
    ) -> BacklogSkipIntent | None:
        """Freeze one source at EOF, persist its barrier, then retire its queue work."""
        if (
            self._skip_purge_callback is None
            or self._skip_notice_callback is None
            or self._skip_validate_callback is None
        ):
            raise RuntimeError("backlog skip callbacks are not wired")
        session_id = session_lifecycle.resolve_session_id(window_id)
        if not session_id:
            return None
        session = self.state.get_session(session_id)
        if session is None or session_id in self.state.pending_skips:
            return None
        try:
            snapshot_offset = Path(session.file_path).stat().st_size
        except OSError:
            logger.warning(
                "Cannot snapshot transcript for backlog skip: %s", session.file_path
            )
            return None
        if snapshot_offset < session.last_byte_offset:
            logger.warning(
                "Refusing backlog skip with regressed transcript EOF: %s", session_id
            )
            return None
        intent = BacklogSkipIntent(
            session_id=session_id,
            window_id=window_id,
            user_id=user_id,
            thread_id=thread_id,
            chat_id=chat_id,
            snapshot_offset=snapshot_offset,
            range_start=session.last_byte_offset,
        )
        # The durable barrier is written before destructive queue retirement.
        self.state.begin_skip(intent)
        if not self.state.save_if_dirty():
            self.state.cancel_skip(session_id)
            return None
        prepared = await self._prepare_pending_skip(intent)
        if prepared is not True:
            return None
        if not await self._enqueue_pending_skip_notice(intent):
            return None
        return intent

    def _skip_is_current(self, intent: BacklogSkipIntent) -> bool:
        callback = self._skip_validate_callback
        if callback is None:
            return False
        try:
            return callback(intent)
        except Exception:
            logger.exception(
                "Failed to validate backlog skip for %s", intent.session_id
            )
            return False

    def _skip_retry_due(self, session_id: str) -> bool:
        return time.monotonic() >= self._skip_retry_at.get(session_id, 0.0)

    def _schedule_skip_retry(self, session_id: str) -> None:
        attempt = self._skip_retry_attempts.get(session_id, 0) + 1
        delay = min(
            _SKIP_RETRY_MAX_SECONDS,
            _SKIP_RETRY_BASE_SECONDS * (2 ** min(attempt - 1, 5)),
        )
        self._skip_retry_attempts[session_id] = attempt
        self._skip_retry_at[session_id] = time.monotonic() + delay
        logger.warning(
            "Backlog skip step failed; retrying later",
            session_id=session_id,
            retry=attempt,
            retry_in_seconds=delay,
        )

    def _clear_skip_retry(self, session_id: str) -> None:
        self._skip_retry_attempts.pop(session_id, None)
        self._skip_retry_at.pop(session_id, None)

    async def _prepare_pending_skip(self, intent: BacklogSkipIntent) -> bool | None:
        """Retire frozen source work before a skip notice may be sent."""
        if not self._skip_is_current(intent):
            logger.warning("Cancelling stale backlog skip for %s", intent.session_id)
            self.state.cancel_skip(intent.session_id)
            self.state.save_if_dirty()
            self._clear_skip_retry(intent.session_id)
            return None
        if intent.purge_complete:
            return True
        callback = self._skip_purge_callback
        if callback is None or not self._skip_retry_due(intent.session_id):
            return False
        try:
            skipped = await callback(intent)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to purge backlog for %s", intent.session_id)
            self._schedule_skip_retry(intent.session_id)
            return False
        if skipped is None:
            logger.warning("Cancelling stale backlog skip for %s", intent.session_id)
            self.state.cancel_skip(intent.session_id)
            self.state.save_if_dirty()
            self._clear_skip_retry(intent.session_id)
            return None
        self.state.update_skip_count(intent.session_id, skipped)
        if not self.state.save_if_dirty():
            intent.purge_complete = False
            self._schedule_skip_retry(intent.session_id)
            return False
        self._clear_skip_retry(intent.session_id)
        return True

    async def _enqueue_pending_skip_notice(self, intent: BacklogSkipIntent) -> bool:
        """Create one receipt-tracked visible notice for a persisted barrier."""
        if not self._skip_is_current(intent):
            self.state.cancel_skip(intent.session_id)
            self.state.save_if_dirty()
            self._clear_skip_retry(intent.session_id)
            return False
        if intent.session_id in self._skip_notice_receipts or not self._skip_retry_due(
            intent.session_id
        ):
            return False
        callback = self._skip_notice_callback
        if callback is None:
            return False
        receipt = new_delivery_receipt(checkpoint=intent.snapshot_offset)
        token = activate_delivery_receipt(receipt)
        try:
            await callback(intent)
        except asyncio.CancelledError:
            receipt.fail()
            raise
        except Exception:
            receipt.fail()
            logger.exception(
                "Failed to enqueue backlog skip notice for %s", intent.session_id
            )
            self._schedule_skip_retry(intent.session_id)
        finally:
            deactivate_delivery_receipt(token)
            receipt.close()
        if receipt.failed:
            return False
        self._clear_skip_retry(intent.session_id)
        self._skip_notice_receipts[intent.session_id] = receipt
        return True

    async def _resume_pending_skip_notices(self) -> None:
        """Resume persisted skip barriers before reading any skipped bytes."""
        for intent in tuple(self.state.pending_skips.values()):
            if not self._skip_retry_due(intent.session_id):
                continue
            prepared = await self._prepare_pending_skip(intent)
            if prepared:
                await self._enqueue_pending_skip_notice(intent)

    def _commit_pending_skips(self) -> None:
        """Advance only barriers whose visible notices reached Telegram."""
        for session_id, receipt in tuple(self._skip_notice_receipts.items()):
            if receipt.failed:
                # A failed notice has no acknowledgement boundary. Remove the
                # failed receipt so the persisted barrier can retry in-process;
                # the source remains paused until a notice is delivered.
                self._skip_notice_receipts.pop(session_id, None)
                self._schedule_skip_retry(session_id)
                continue
            if not receipt.commit_ready:
                continue
            intent = self.state.pending_skips.get(session_id)
            if intent is None or not self._skip_is_current(intent):
                # A topic may rebind after the notice was queued or delivered.
                # Never advance the old source watermark across that boundary.
                self.state.cancel_skip(session_id)
                self.state.save_if_dirty()
                self._delivery_receipts.pop(session_id, None)
                self._skip_notice_receipts.pop(session_id, None)
                self._clear_skip_retry(session_id)
                continue
            if self.state.complete_skip(session_id):
                self.state.save_if_dirty()
                self._delivery_receipts.pop(session_id, None)
            else:
                # The tracked session was removed or re-keyed. Do not retain a
                # barrier that can no longer be committed or replay safely.
                self.state.cancel_skip(session_id)
                self.state.save_if_dirty()
                self._delivery_receipts.pop(session_id, None)
            self._skip_notice_receipts.pop(session_id, None)
            self._clear_skip_retry(session_id)

    def record_hook_activity(self, window_id: str) -> None:
        """Record hook-based activity for a window (resets idle timers)."""
        session_id = session_lifecycle.resolve_session_id(window_id)
        if session_id:
            self._idle_tracker.record_activity(session_id)

    def commit_delivered_watermarks(self) -> None:
        """Persist receipts acknowledged by the delivery boundary.

        Called after a normal monitor cycle and after the bounded shutdown
        drain. It intentionally has no queue implementation knowledge.
        """
        self._commit_watermark_prefixes()

    def _commit_watermark_prefixes(self) -> None:
        """Commit each session's longest settled receipt run (#205).

        Waiting for every receipt of a session to close (the previous
        policy) never commits under sustained output: in-flight tasks hold
        back the whole settled run, so a restart replays it in full into
        the outbound queue. The run policy lives in
        delivery_contract.settled_prefix / settled_run_offset (including
        the shared-batch-checkpoint tie rule: persistence lags delivery by
        at most one in-flight batch); this coordinator groups receipts by
        session, keeps the pending-tools and pending-skip fences, and persists
        one batched commit per cycle. The settled run
        is consumed even when the tie rule defers its commit: replay then
        re-delivers those messages, which at-least-once permits.
        """
        delivered_offsets: dict[str, int] = {}
        consumed: dict[str, int] = {}
        self._commit_pending_skips()
        for session_id, receipts in self._delivery_receipts.items():
            if (
                not receipts
                or session_id in self.state.pending_skips
                or session_id in self._pending_tools
            ):
                continue
            prefix = settled_prefix(receipts)
            if not prefix:
                continue
            fence = receipts[len(prefix)] if len(prefix) < len(receipts) else None
            offset = settled_run_offset(prefix, fence)
            if offset is not None:
                delivered_offsets[session_id] = offset
            consumed[session_id] = len(prefix)
        if delivered_offsets and self.state.commit_parsed_offsets(
            set(delivered_offsets), delivered_offsets=delivered_offsets
        ):
            self.state.save_if_dirty()
        for session_id, count in consumed.items():
            remainder = self._delivery_receipts[session_id][count:]
            if remainder:
                self._delivery_receipts[session_id] = remainder
            else:
                self._delivery_receipts.pop(session_id, None)

    def _reserve_replay_start(
        self, window_id: str, session_id: str, path: Path
    ) -> bool:
        """Persist offset zero before consuming a session-map replay marker."""
        tracked = self.state.get_session(session_id)
        if tracked is None or tracked.file_path != str(path):
            self.state.update_session(
                TrackedSession(
                    session_id=session_id,
                    file_path=str(path),
                    last_byte_offset=0,
                )
            )
        if not self.state.save_if_dirty():
            return False
        if not acknowledge_replay_from_start(window_id, session_id):
            logger.warning(
                "Could not consume replay-from-start marker for session %s",
                session_id,
            )
        return True

    async def check_for_updates(self, current_map: dict) -> list[NewMessage]:
        """Check all sessions for new assistant messages.

        Routes sessions to _process_session_file (allowing test spying) and
        delegates the actual I/O to TranscriptReader. Uses _get_active_cwds()
        for fallback session discovery so tests can stub tmux calls.
        """
        new_messages: list[NewMessage] = []
        sid_to_wid = {v["session_id"]: wid for wid, v in current_map.items()}

        direct_sessions: list[tuple[str, Path]] = []
        fallback_session_ids: set[str] = set()

        for window_id, details in current_map.items():
            session_id = details["session_id"]
            transcript_path = details.get("transcript_path", "")
            if transcript_path:
                path = Path(transcript_path)
                replay_from_start = details.get("replay_from_start") is True
                if replay_from_start and not self._reserve_replay_start(
                    window_id, session_id, path
                ):
                    continue
                if path.exists():
                    direct_sessions.append((session_id, path))
                    continue
                if replay_from_start:
                    # The explicit hook path is authoritative. Scanning can
                    # find a different generation and seed it at EOF before
                    # this file is created.
                    continue
            fallback_session_ids.add(session_id)

        for session_id, file_path in direct_sessions:
            if session_id in self.state.pending_skips:
                continue
            try:
                await self._process_session_file(
                    session_id,
                    file_path,
                    new_messages,
                    window_id=sid_to_wid.get(session_id, ""),
                )
            except Exception:
                logger.exception("Error processing session %s", session_id)

        if fallback_session_ids:
            active_cwds = await self._get_active_cwds()
            sessions = self._scan_projects_sync(active_cwds) if active_cwds else []
            for session_info in sessions:
                if (
                    session_info.session_id not in fallback_session_ids
                    or session_info.session_id in self.state.pending_skips
                ):
                    continue
                try:
                    await self._process_session_file(
                        session_info.session_id,
                        session_info.file_path,
                        new_messages,
                        window_id=sid_to_wid.get(session_info.session_id, ""),
                    )
                except Exception:
                    logger.exception(
                        "Error processing session %s", session_info.session_id
                    )

        self.state.save_if_dirty()
        return new_messages

    async def _process_session_file(
        self, session_id: str, file_path: Path, new_messages: list, window_id: str = ""
    ) -> None:
        """Process a single session file (delegates to TranscriptReader)."""
        await self._transcript_reader._process_session_file(
            session_id, file_path, new_messages, window_id=window_id
        )

    def _scan_projects_sync(self, active_cwds: set) -> list:
        """Scan projects synchronously (delegates to TranscriptReader)."""
        return self._transcript_reader._scan_projects_sync(
            self.projects_path, active_cwds
        )

    async def _get_active_cwds(self) -> set[str]:
        """Get normalized cwds of all active tmux windows (delegates to TranscriptReader)."""
        return await self._transcript_reader._get_active_cwds()

    async def _read_new_lines(
        self, session: Any, file_path: Path, window_id: str = ""
    ) -> list:
        """Read new lines from session file (delegates to TranscriptReader)."""
        return await self._transcript_reader._read_new_lines(
            session, file_path, window_id
        )

    async def _read_hook_events(self) -> None:
        """Read new lines from events.jsonl and dispatch via callback."""
        if not self._hook_event_callback:
            return

        offset_before = self.state.events_offset
        events, new_offset = await read_new_events(
            config.events_file, self.state.events_offset
        )
        self.state.events_offset = new_offset
        if new_offset != offset_before:
            self.state._dirty = True

        for event in events:
            try:
                await self._hook_event_callback(event)
            except _CallbackError:
                logger.exception("Hook event callback error for %s", event.event_type)

    async def _load_current_session_map(
        self, raw: dict | None = None
    ) -> dict[str, dict[str, str]]:
        """Load a validated session_map mapping.

        Callers that reconcile or prune first read the raw map and explicitly
        preserve a failed read. This compatibility helper keeps its historical
        mapping return type for callers that only need a parsed snapshot.
        """
        if raw is None:
            raw = await read_session_map_raw()
        if not isinstance(raw, dict):
            return {}
        prefix = session_map_prefix()
        return parse_session_map(raw, prefix)

    async def _cleanup_all_stale_sessions(self) -> None:
        """Clean up all tracked sessions not in current session_map (startup)."""
        raw = await read_session_map_raw()
        if raw is None:
            logger.warning("Startup cleanup skipped: session_map is unreadable")
            return
        current_map = await self._load_current_session_map(raw)
        active_session_ids = {v["session_id"] for v in current_map.values()}

        stale_sessions = [
            sid for sid in self.state.tracked_sessions if sid not in active_session_ids
        ]
        if stale_sessions:
            logger.info(
                "[Startup cleanup] Removing %d stale sessions", len(stale_sessions)
            )
            for session_id in stale_sessions:
                self._transcript_reader.clear_session(session_id)
                self._idle_tracker.clear_session(session_id)
            self.state.save_if_dirty()

    async def _detect_and_cleanup_changes(
        self, raw: dict | None = None
    ) -> dict[str, dict[str, str]]:
        """Reconcile session_map; clean up replaced/removed sessions; fire new-window events."""
        if raw is None:
            raw = await read_session_map_raw()
        if raw is None:
            logger.warning("Session-map reconciliation skipped: map is unreadable")
            return session_lifecycle.last_session_map
        current_map = await self._load_current_session_map(raw)
        result = session_lifecycle.reconcile(current_map, self._idle_tracker)

        for session_id in result.sessions_to_remove:
            self._transcript_reader.clear_session(session_id)
        if result.sessions_to_remove:
            self.state.save_if_dirty()

        adoption_windows = dict(result.new_windows)
        # Lazy: thread_router is wired into session_manager which imports
        # session_monitor; hoisting forms a startup cycle.
        # Lazy: proxies wired by SessionManager constructor
        from .thread_router import thread_router

        for window_id, details in result.changed_windows.items():
            if not thread_router.has_window(window_id):
                adoption_windows[window_id] = details

        if adoption_windows:
            # Lazy: session.py imports session_monitor at top; hoisting
            # session_manager forms a hard cycle on bootstrap.
            from .session import session_manager as _sm

            for window_id, details in adoption_windows.items():
                provider_name = details.get("provider_name", "")
                if provider_name:
                    _sm.set_window_provider(window_id, provider_name)

                if thread_router.has_window(window_id):
                    # A key that is new to the map is not a window that is new
                    # to ccgram. Identity folding runs first (``_monitor_loop``),
                    # so a re-keyed or late-published identity already carries
                    # the topic it was bound under; announcing it here would
                    # ask for a second topic for the same agent. Both other
                    # discovery paths skip bound windows for the same reason.
                    continue

                if self._new_window_callback:
                    event = NewWindowEvent(
                        window_id=window_id,
                        session_id=details["session_id"],
                        window_name=details.get("window_name", ""),
                        cwd=details.get("cwd", ""),
                    )
                    try:
                        await self._new_window_callback(event)
                    except _CallbackError:
                        logger.exception(
                            "New window callback error (session_map path) for %s",
                            window_id,
                        )

        return result.current_map

    async def _emit_unbound_window_events(
        self, all_windows: list, known_window_ids: set[str]
    ) -> None:
        """Fire a NewWindowEvent for each live window not in session_map / bound.

        Surfaces windows the hook never registered (no session_map entry) so
        they can become topics. On backends that expose agent status natively
        (herdr), only agent panes qualify — a bare shell pane is not a topic;
        tmux surfaces every window, preserving today's behavior. The gate is the
        ``native_agent_status`` capability, not a backend name.
        """
        if not self._new_window_callback:
            return
        # Lazy: thread_router is wired into session_manager which imports
        # session_monitor; hoisting forms a startup cycle.
        from .thread_router import thread_router

        caps = tmux_manager.capabilities
        bound_window_ids = {wid for _, _, wid in thread_router.iter_thread_bindings()}
        for window in all_windows:
            if window.window_id in known_window_ids:
                continue
            if window.window_id in bound_window_ids:
                continue
            if not is_agent_topic_window(window, caps):
                continue
            event = NewWindowEvent(
                window_id=window.window_id,
                session_id="",
                window_name=window.window_name,
                cwd=window.cwd,
            )
            try:
                await self._new_window_callback(event)
            except _CallbackError:
                logger.exception(
                    "New window callback error (unbound window path) for %s",
                    window.window_id,
                )

    async def _emit_known_unbound_window_events(
        self,
        current_map: dict,
        live_window_ids: set[str],
    ) -> None:
        """Fire a NewWindowEvent for each session_map window that is not bound.

        Steady-state self-heal: a tab that was in session_map at startup (known,
        so never a delta) but not yet bound to a Telegram topic retries on every
        poll until it succeeds. ``handle_new_window`` is idempotent — it skips
        windows that are already bound — so this generates no spam for bound tabs.

        ``live_window_ids`` is the set from ``list_windows``. Because ``list_windows``
        already filters ``__*__`` workspace/tab labels, any such tab is absent from
        ``live_window_ids`` and is silently skipped here as well.
        """
        if not self._new_window_callback:
            return
        # Lazy: thread_router is wired into session_manager which imports
        # session_monitor; hoisting forms a startup cycle.
        from .thread_router import thread_router

        bound_window_ids = {wid for _, _, wid in thread_router.iter_thread_bindings()}
        for window_id, details in current_map.items():
            if window_id not in live_window_ids:
                continue  # dead / __*__-filtered — skip
            if window_id in bound_window_ids:
                continue  # already has a topic
            event = NewWindowEvent(
                window_id=window_id,
                session_id=details.get("session_id", ""),
                window_name=details.get("window_name", ""),
                cwd=details.get("cwd", ""),
            )
            try:
                await self._new_window_callback(event)
            except _CallbackError:
                logger.exception(
                    "New window callback error (known-unbound path) for %s",
                    window_id,
                )

    def _register_delivery_receipts(
        self, messages: list[NewMessage]
    ) -> list[tuple[NewMessage, DeliveryReceipt]]:
        """Register a non-ready receipt for every parsed message synchronously."""
        pending: list[tuple[NewMessage, DeliveryReceipt]] = []
        if self._message_callback is None:
            return pending
        for msg in messages:
            session = self.state.get_session(msg.session_id)
            checkpoint = session.parsed_offset if session is not None else None
            receipt = new_delivery_receipt(checkpoint=checkpoint)
            self._delivery_receipts.setdefault(msg.session_id, []).append(receipt)
            pending.append((msg, receipt))
        return pending

    async def _dispatch_message_with_receipt(
        self, msg: NewMessage, receipt: DeliveryReceipt | None = None
    ) -> None:
        """Run one transcript callback under a delivery-boundary receipt."""
        if self._message_callback is None:
            return
        if receipt is None:
            session = self.state.get_session(msg.session_id)
            checkpoint = session.parsed_offset if session is not None else None
            receipt = new_delivery_receipt(checkpoint=checkpoint)
            self._delivery_receipts.setdefault(msg.session_id, []).append(receipt)
        token = activate_delivery_receipt(receipt)
        try:
            await self._message_callback(msg)
        except asyncio.CancelledError:
            receipt.fail()
            raise
        except _CallbackError:
            receipt.fail()
            logger.exception("Message callback error for session=%s", msg.session_id)
        finally:
            deactivate_delivery_receipt(token)
            receipt.close()

    async def _monitor_loop(self) -> None:
        """Background poll loop."""
        logger.info("Session monitor started, polling every %ss", self.poll_interval)

        # Lazy: session_map imports session_monitor types via shared
        # state cycle; keep at call site.
        # Lazy: proxies wired by SessionManager constructor
        from .session_map import session_map_sync

        await self._cleanup_all_stale_sessions()
        initial_raw = await read_session_map_raw()
        initial_map = await self._load_current_session_map(initial_raw)
        session_lifecycle.initialize(initial_map)

        error_streak = 0
        while self._running:
            try:
                # The same long-lived task handles every cycle. Do not attach a
                # prior message's session_id to reconciliation and hook logs.
                structlog.contextvars.clear_contextvars()
                raw_session_map = await read_session_map_raw()

                # A fresh listing owns identity convergence. It must precede
                # session-map loading because loading rejects raw legacy keys;
                # after a successful fold, re-read the hook file under its
                # normal parser so the canonical key is what lifecycle sees.
                all_windows = await list_windows_for_reconciliation(tmux_manager)
                if all_windows is None:
                    logger.warning(
                        "Multiplexer listing unavailable; skipping window reconciliation"
                    )
                else:
                    # Before anything keys off these ids, let the backend
                    # reconcile only aliases it explicitly attests as safe.
                    # Herdr publishes no raw locator aliases, so a missing or
                    # changed session target remains unresolved until an
                    # operator explicitly rebinds it.
                    # Lazy: importing session_manager at module scope forms a
                    # hard cycle on bootstrap (same reason as below).
                    from .session import session_manager as _sm

                    # Lazy: thread routing imports monitor-facing helpers.
                    from .thread_router import thread_router

                    _sm.reconcile_window_aliases(all_windows)
                    note_live_windows(all_windows, thread_router.all_bound_window_ids())
                    raw_session_map = await read_session_map_raw()

                # Dispatch only after identity convergence and the session-map
                # re-read: hook routing is exact-bound, so consuming a canonical
                # event before moving a legacy topic binding would drop it.
                await self._read_hook_events()

                await session_map_sync.load_session_map(raw_session_map)
                current_map = await self._detect_and_cleanup_changes(raw_session_map)

                monitored_map = current_map
                if all_windows is not None:
                    live_window_ids = {w.window_id for w in all_windows}
                    session_map_sync.prune_session_map(live_window_ids)
                    known_window_ids = set(current_map.keys())
                    await self._emit_unbound_window_events(
                        all_windows, known_window_ids
                    )
                    await self._emit_known_unbound_window_events(
                        current_map, live_window_ids
                    )
                    monitored_map = {
                        window_id: details
                        for window_id, details in current_map.items()
                        if window_id in live_window_ids
                    }

                # A persisted barrier must be noticed before its source is read
                # again; this preserves the exact EOF snapshot across restarts.
                await self._resume_pending_skip_notices()
                self._commit_pending_skips()
                new_messages = await self.check_for_updates(monitored_map)
                # Register every parsed message before the next await. A
                # shutdown cancellation between parse and dispatch must leave
                # a non-ready receipt so its offset remains replayable.
                pending_dispatches = self._register_delivery_receipts(new_messages)

                for msg, receipt in pending_dispatches:
                    structlog.contextvars.clear_contextvars()
                    structlog.contextvars.bind_contextvars(session_id=msg.session_id)
                    status = "complete" if msg.is_complete else "streaming"
                    preview = msg.text[:_MSG_PREVIEW_LENGTH] + (
                        "..." if len(msg.text) > _MSG_PREVIEW_LENGTH else ""
                    )
                    logger.debug("[%s] session=%s: %s", status, msg.session_id, preview)
                    await self._dispatch_message_with_receipt(msg, receipt)

                self.commit_delivered_watermarks()

            except _LoopError:
                logger.exception("Monitor loop error")
                backoff_delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**error_streak))
                error_streak += 1
                await asyncio.sleep(backoff_delay)
                continue
            except Exception:
                logger.exception("Unexpected error in monitor loop")
                backoff_delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**error_streak))
                error_streak += 1
                await asyncio.sleep(backoff_delay)
                continue

            error_streak = 0
            await asyncio.sleep(self.poll_interval)

        logger.info("Session monitor stopped")

    def start(self) -> None:
        if self._running:
            logger.debug("Monitor already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        self._task.add_done_callback(task_done_callback)

    def stop(self) -> None:
        """Request producer cancellation; use ``stop_and_wait`` before drain."""
        self._running = False
        if self._task:
            self._task.cancel()
        self.state.save()
        # Distinct from the loop's "Session monitor stopped" (logged when the
        # poll loop actually exits) — this marks the stop request + state save.
        logger.info("Session monitor stop requested; state saved")

    async def stop_and_wait(self) -> None:
        """Cancel the monitor producer and wait until it cannot enqueue again."""
        self.stop()
        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if self._task is task:
                self._task = None


_active_monitor: SessionMonitor | None = None


def set_active_monitor(monitor: SessionMonitor) -> None:
    """Set the active SessionMonitor instance (called by bot.py post_init)."""
    global _active_monitor  # noqa: PLW0603
    _active_monitor = monitor


def clear_active_monitor() -> None:
    """Clear the active SessionMonitor singleton (shutdown / test reset)."""
    global _active_monitor  # noqa: PLW0603
    _active_monitor = None


def get_active_monitor() -> SessionMonitor | None:
    """Return the active SessionMonitor instance."""
    return _active_monitor
