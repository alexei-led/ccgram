"""Transcript reading and processing for agent session files.

Handles the full lifecycle of reading agent transcripts:
  - Scanning Claude projects for active session files
  - Incremental byte-offset reads for JSONL providers
  - Whole-file reads for JSON providers (e.g. Gemini)
  - Parsing transcript entries into NewMessage objects
  - mtime caching to skip unchanged files
  - Pending tool-use state carried across poll cycles

Key class: TranscriptReader.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import aiofiles
import structlog

from .monitor_events import NewMessage, SessionInfo
from .monitor_state import MonitorState, TrackedSession
from .providers import (
    detect_provider_from_transcript_path,
    get_provider_for_window,
    registry,
)
from .utils import log_throttle_reset, log_throttled, read_cwd_from_jsonl

if TYPE_CHECKING:
    from .idle_tracker import IdleTracker

logger = structlog.get_logger()

_PathResolveError = (OSError, ValueError)


class _StartupBoundary(NamedTuple):
    size: int
    device: int
    inode: int


class _StableRead(NamedTuple):
    entries: list[dict]
    stat: Any
    reset_generation: bool


class _GenerationCheck(NamedTuple):
    changed: bool
    consumed_intact: bool


_MARKER_BYTES = 128


def _prefix_digest(file_path: Path, size: int) -> bytes:
    digest = hashlib.sha256()
    with file_path.open("rb") as transcript:
        remaining = size
        while remaining:
            chunk = transcript.read(min(64 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.digest()


def _tail_marker(file_path: Path, offset: int) -> bytes:
    """Read a small marker immediately before a consumed byte offset."""
    if offset <= 0:
        return b""
    start = max(0, offset - _MARKER_BYTES)
    with file_path.open("rb") as transcript:
        transcript.seek(start)
        return transcript.read(offset - start)


def _resolve_provider_for_file(window_id: str, file_path: Path) -> Any:
    """Prefer transcript-path provider hints when a hookful state goes stale."""
    provider_name: str | None = None
    try:
        # Lazy: window_state_ports.identity_state imports the kernel which
        # may not yet be wired during early transcript-discovery paths.
        # RuntimeError comes from the unwired _WindowStoreProxy;
        # ImportError guards against an unfinished port package on disk.
        from .window_state_ports import identity_state

        provider_name = identity_state.get_provider_name(window_id)
    except ImportError, RuntimeError:
        pass
    provider = get_provider_for_window(window_id, provider_name=provider_name)
    inferred = detect_provider_from_transcript_path(str(file_path))
    current = provider.capabilities.name
    if (
        inferred
        and inferred != current
        and provider.capabilities.supports_hook
        and registry.is_valid(inferred)
    ):
        # Throttled debug, not warning: this read-path observation repeats every
        # poll until session_map corrects the in-memory state. The read itself
        # self-heals (we return the inferred provider below), and when the hook
        # is functional session_map._sync_window_from_session_map emits the
        # authoritative WARNING on the state mutation. Caveat: if the hook is
        # broken so session_map never updates, that correction (and its WARNING)
        # never fires and this stays debug-only — accepted, since the read still
        # works and a per-poll WARNING here would just flood.
        log_throttled(
            logger,
            f"provider-mismatch:{window_id}",
            "Provider mismatch for window %s: state=%s transcript=%s; using %s",
            window_id,
            current,
            str(file_path),
            inferred,
        )
        return registry.get(inferred)
    return provider


class TranscriptReader:
    """Reads and processes agent transcript files for new messages.

    Owns: mtime cache, pending_tools per session, MonitorState updates.
    Delegates activity recording to IdleTracker (via session_id).
    """

    def __init__(
        self,
        state: MonitorState,
        idle_tracker: IdleTracker,
        on_session_retired: Callable[[str], None] | None = None,
    ) -> None:
        self._state = state
        self._idle_tracker = idle_tracker
        self._on_session_retired = on_session_retired
        self._pending_tools: dict[str, dict[str, Any]] = {}
        self._file_mtimes: dict[str, float] = {}
        self._file_ctimes: dict[str, int] = self._snapshot_file_ctimes()
        self._file_sizes = self._snapshot_file_sizes()
        self._file_prefixes = self._snapshot_file_prefixes()
        self._file_generations = self._snapshot_file_generations()
        self._file_markers = self._snapshot_file_markers()
        self._startup_file_boundaries = self._snapshot_startup_boundaries()

    def _snapshot_file_ctimes(self) -> dict[str, int]:
        ctimes: dict[str, int] = {}
        for session_id, tracked in self._state.tracked_sessions.items():
            try:
                ctimes[session_id] = Path(tracked.file_path).stat().st_ctime_ns
            except OSError:
                continue
        return ctimes

    def _snapshot_file_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for session_id, tracked in self._state.tracked_sessions.items():
            try:
                sizes[session_id] = Path(tracked.file_path).stat().st_size
            except OSError:
                continue
        return sizes

    def _snapshot_file_prefixes(self) -> dict[str, tuple[int, bytes]]:
        prefixes: dict[str, tuple[int, bytes]] = {}
        for session_id, tracked in self._state.tracked_sessions.items():
            try:
                size = Path(tracked.file_path).stat().st_size
                prefixes[session_id] = (
                    size,
                    _prefix_digest(Path(tracked.file_path), size),
                )
            except OSError:
                continue
        return prefixes

    def _snapshot_file_generations(self) -> dict[str, tuple[int, int]]:
        generations: dict[str, tuple[int, int]] = {}
        for session_id, tracked in self._state.tracked_sessions.items():
            try:
                st = Path(tracked.file_path).stat()
            except OSError:
                continue
            generations[session_id] = (st.st_dev, st.st_ino)
        return generations

    def _snapshot_file_markers(self) -> dict[str, tuple[int, bytes]]:
        markers: dict[str, tuple[int, bytes]] = {}
        for session_id, tracked in self._state.tracked_sessions.items():
            try:
                marker = _tail_marker(Path(tracked.file_path), tracked.last_byte_offset)
            except OSError:
                continue
            markers[session_id] = (tracked.last_byte_offset, marker)
        return markers

    def _snapshot_startup_boundaries(self) -> dict[str, _StartupBoundary]:
        """Capture each tracked transcript generation and its pre-start EOF."""
        boundaries: dict[str, _StartupBoundary] = {}
        for session_id, tracked in self._state.tracked_sessions.items():
            try:
                st = Path(tracked.file_path).stat()
            except OSError:
                continue
            boundaries[session_id] = _StartupBoundary(
                size=st.st_size,
                device=st.st_dev,
                inode=st.st_ino,
            )
        return boundaries

    def _prepare_startup_boundary(
        self,
        session_id: str,
        tracked: TrackedSession,
        st: Any,
        *,
        consumed_intact: bool,
    ) -> _StartupBoundary | None:
        """Reset post-start replacements and return the activity boundary."""
        boundary = self._startup_file_boundaries.get(session_id)
        if boundary is None:
            return None
        generation_changed = (st.st_dev, st.st_ino) != (
            boundary.device,
            boundary.inode,
        )
        if (generation_changed or st.st_size < boundary.size) and not consumed_intact:
            tracked.parsed_offset = st.st_size
            boundary = _StartupBoundary(
                size=0,
                device=st.st_dev,
                inode=st.st_ino,
            )
            self._startup_file_boundaries[session_id] = boundary
        return boundary

    def _prefix_covers_consumed(self, session_id: str, consumed: int, st: Any) -> bool:
        """True when the cached prefix digest spans every byte already delivered."""
        saved = self._file_prefixes.get(session_id)
        return (
            consumed > 0
            and saved is not None
            and saved[0] >= consumed
            and st.st_size >= consumed
        )

    async def _consumed_prefix_intact(
        self, session_id: str, consumed: int, file_path: Path, st: Any
    ) -> bool:
        """True when every byte already delivered is still byte-identical on disk.

        A replacement signal that fires while this holds is a false positive:
        nothing we sent was replaced, so rewinding would replay the whole
        transcript instead of the new content.
        """
        if not self._prefix_covers_consumed(session_id, consumed, st):
            return False
        size, digest = self._file_prefixes[session_id]
        try:
            current = await asyncio.to_thread(_prefix_digest, file_path, size)
        except OSError:
            return False
        return current == digest

    async def _marker_changed(
        self, session_id: str, tracked: TrackedSession, file_path: Path
    ) -> bool:
        pos = self._parse_pos(tracked)
        saved = self._file_markers.get(session_id)
        if saved is None or saved[0] != pos:
            return False
        try:
            current = await asyncio.to_thread(_tail_marker, file_path, pos)
        except OSError:
            return False
        return current != saved[1]

    async def _prepare_observed_generation(
        self,
        session_id: str,
        tracked: TrackedSession,
        file_path: Path,
        st: Any,
        *,
        check_marker: bool,
    ) -> _GenerationCheck:
        generation = (st.st_dev, st.st_ino)
        previous = self._file_generations.get(session_id)
        previous_ctime = self._file_ctimes.get(session_id)
        changed = previous is not None and previous != generation
        previous_size = self._file_sizes.get(session_id)
        previous_prefix = self._file_prefixes.get(session_id)
        prefix_changed = False
        prefix_verified = False
        if previous_prefix is not None:
            try:
                prefix_changed = (
                    await asyncio.to_thread(
                        _prefix_digest, file_path, previous_prefix[0]
                    )
                    != previous_prefix[1]
                )
            except OSError:
                prefix_changed = False
            else:
                prefix_verified = True
        changed = (
            changed
            or prefix_changed
            or (
                previous_ctime is not None
                and previous_ctime != st.st_ctime_ns
                and previous_size is not None
                and st.st_size <= previous_size
            )
        )
        pos = self._parse_pos(tracked)
        changed = changed or st.st_size < pos
        if not changed and check_marker:
            changed = await self._marker_changed(session_id, tracked, file_path)
        consumed_intact = False
        if changed:
            consumed_intact = (
                check_marker
                and prefix_verified
                and not prefix_changed
                and self._prefix_covers_consumed(session_id, pos, st)
            )
            if consumed_intact:
                logger.debug(
                    "Ignoring transcript replacement signal, delivered bytes intact: %s",
                    session_id,
                )
            else:
                # Replaced transcript: jump the parse position to EOF so the
                # existing bytes are treated as already-notified history
                # (incident 2026-08-17: never replay), and let the persisted
                # watermark follow via commit_parsed_offsets().
                tracked.parsed_offset = st.st_size
                self._startup_file_boundaries.pop(session_id, None)
        return _GenerationCheck(changed and not consumed_intact, consumed_intact)

    async def _read_session_entries(
        self,
        session_id: str,
        tracked: TrackedSession,
        file_path: Path,
        window_id: str,
        *,
        check_marker: bool,
    ) -> _StableRead | None:
        """Read a stable file generation, retrying once after a write race."""
        reset_generation = False
        for _attempt in range(2):
            try:
                before = file_path.stat()
                start_offset = self._parse_pos(tracked)
                entries = await self._read_new_lines(tracked, file_path, window_id)
                after = file_path.stat()
            except OSError:
                return None
            same_generation = (before.st_dev, before.st_ino) == (
                after.st_dev,
                after.st_ino,
            )
            rewritten_in_place = before.st_ctime_ns != after.st_ctime_ns
            marker_changed = False
            saved = self._file_markers.get(session_id) if check_marker else None
            if saved is not None and saved[0] == start_offset:
                try:
                    marker = await asyncio.to_thread(
                        _tail_marker, file_path, start_offset
                    )
                except OSError:
                    return None
                marker_changed = marker != saved[1]
            if same_generation and not rewritten_in_place and not marker_changed:
                return _StableRead(entries, after, reset_generation)
            consumed_survived = check_marker and await self._consumed_prefix_intact(
                session_id, start_offset, file_path, after
            )
            if consumed_survived:
                # Concurrent append or metadata churn, not a replacement: the
                # bytes already delivered survived, so keep this read.
                return _StableRead(entries, after, reset_generation)
            tracked.parsed_offset = after.st_size
            self._startup_file_boundaries.pop(session_id, None)
            reset_generation = True
        return None

    async def _commit_stable_read(
        self,
        session_id: str,
        tracked: TrackedSession,
        file_path: Path,
        stable_read: _StableRead,
    ) -> bool:
        """Commit caches only after a stable read and marker capture."""
        stable_stat = stable_read.stat
        try:
            marker = await asyncio.to_thread(
                _tail_marker, file_path, self._parse_pos(tracked)
            )
        except OSError:
            return False
        self._file_mtimes[session_id] = stable_stat.st_mtime
        self._file_generations[session_id] = (
            stable_stat.st_dev,
            stable_stat.st_ino,
        )
        self._file_ctimes[session_id] = stable_stat.st_ctime_ns
        self._file_sizes[session_id] = stable_stat.st_size
        self._file_prefixes[session_id] = (
            stable_stat.st_size,
            await asyncio.to_thread(_prefix_digest, file_path, stable_stat.st_size),
        )
        self._file_markers[session_id] = (self._parse_pos(tracked), marker)
        return True

    def clear_session(self, session_id: str) -> None:
        """Remove all per-session state for a cleaned-up session."""
        # A skip barrier belongs to the tracked session lifetime. Once that
        # session disappears, cancel it so it cannot block a future replay.
        self._state.cancel_skip(session_id)
        if self._on_session_retired is not None:
            self._on_session_retired(session_id)
        self._state.remove_session(session_id)
        self._file_mtimes.pop(session_id, None)
        self._pending_tools.pop(session_id, None)
        self._file_generations.pop(session_id, None)
        self._file_ctimes.pop(session_id, None)
        self._file_sizes.pop(session_id, None)
        self._file_prefixes.pop(session_id, None)
        self._file_markers.pop(session_id, None)
        self._startup_file_boundaries.pop(session_id, None)
        log_throttle_reset(f"partial-jsonl:{session_id}")

    def _adopt_tracking_for_file(
        self, session_id: str, file_path: Path
    ) -> TrackedSession | None:
        """Move offset state when the same transcript appears under a refreshed id."""
        try:
            target = file_path.resolve()
        except _PathResolveError:
            target = file_path

        for old_session_id, old_session in list(self._state.tracked_sessions.items()):
            if old_session_id == session_id:
                continue
            try:
                existing = Path(old_session.file_path).resolve()
            except _PathResolveError:
                existing = Path(old_session.file_path)
            if existing != target:
                continue

            tracked = TrackedSession(
                session_id=session_id,
                file_path=str(file_path),
                last_byte_offset=old_session.last_byte_offset,
            )
            # Session IDs can be refreshed for the same transcript. A pending
            # skip is tied to the old identity and must not follow it silently.
            self._state.cancel_skip(old_session_id)
            if self._on_session_retired is not None:
                self._on_session_retired(old_session_id)
            self._state.remove_session(old_session_id)
            self._state.update_session(tracked)
            if old_session_id in self._file_mtimes:
                self._file_mtimes[session_id] = self._file_mtimes.pop(old_session_id)
            if old_session_id in self._file_generations:
                self._file_generations[session_id] = self._file_generations.pop(
                    old_session_id
                )
            if old_session_id in self._file_ctimes:
                self._file_ctimes[session_id] = self._file_ctimes.pop(old_session_id)
            if old_session_id in self._file_sizes:
                self._file_sizes[session_id] = self._file_sizes.pop(old_session_id)
            if old_session_id in self._file_prefixes:
                self._file_prefixes[session_id] = self._file_prefixes.pop(
                    old_session_id
                )
            if old_session_id in self._pending_tools:
                self._pending_tools[session_id] = self._pending_tools.pop(
                    old_session_id
                )
            if old_session_id in self._file_markers:
                self._file_markers[session_id] = self._file_markers.pop(old_session_id)
            if old_session_id in self._startup_file_boundaries:
                self._startup_file_boundaries[session_id] = (
                    self._startup_file_boundaries.pop(old_session_id)
                )
            log_throttle_reset(f"partial-jsonl:{old_session_id}")
            logger.debug(
                "Adopted transcript offset for refreshed session: %s -> %s (%s)",
                old_session_id,
                session_id,
                str(file_path),
            )
            return tracked
        return None

    @staticmethod
    def _parse_pos(session: TrackedSession) -> int:
        """Current parse position (parsed_offset once set, watermark before)."""
        return (
            session.parsed_offset
            if session.parsed_offset >= 0
            else session.last_byte_offset
        )

    async def _process_session_file(
        self,
        session_id: str,
        file_path: Path,
        new_messages: list[NewMessage],
        window_id: str = "",
        current_map: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Process a single session file for new messages."""
        tracked = self._state.get_session(session_id)
        provider = _resolve_provider_for_file(window_id, file_path)

        if tracked is None:
            tracked = self._adopt_tracking_for_file(session_id, file_path)

        if tracked is None:
            try:
                st = file_path.stat()
                file_size, current_mtime = st.st_size, st.st_mtime
            except OSError:
                file_size = 0
                current_mtime = 0.0
                st = None
                generation = None
            else:
                generation = (st.st_dev, st.st_ino)

            if provider.capabilities.supports_incremental_read:
                initial_offset = file_size
            else:
                _, initial_offset = await asyncio.to_thread(
                    provider.read_transcript_file, str(file_path), 0
                )

            tracked = TrackedSession(
                session_id=session_id,
                file_path=str(file_path),
                last_byte_offset=initial_offset,
            )
            self._state.update_session(tracked)
            self._file_mtimes[session_id] = current_mtime
            if generation is not None and st is not None:
                self._file_generations[session_id] = generation
                self._file_ctimes[session_id] = st.st_ctime_ns
                self._file_sizes[session_id] = st.st_size
                self._file_prefixes[session_id] = (
                    st.st_size,
                    await asyncio.to_thread(_prefix_digest, file_path, st.st_size),
                )
            try:
                marker = await asyncio.to_thread(
                    _tail_marker, file_path, tracked.last_byte_offset
                )
            except OSError:
                pass
            else:
                self._file_markers[session_id] = (
                    tracked.last_byte_offset,
                    marker,
                )
            if provider.capabilities.supports_task_tracking and window_id:
                await provider.seed_task_state(window_id, session_id, str(file_path))
            logger.debug("Started tracking session: %s", session_id)
            return

        try:
            st = file_path.stat()
            current_mtime, current_size = st.st_mtime, st.st_size
        except OSError:
            return

        generation_changed, consumed_intact = await self._prepare_observed_generation(
            session_id,
            tracked,
            file_path,
            st,
            check_marker=provider.capabilities.supports_incremental_read,
        )
        last_mtime = self._file_mtimes.get(session_id, 0.0)
        if provider.capabilities.supports_incremental_read:
            if (
                not generation_changed
                and current_mtime <= last_mtime
                and current_size <= self._parse_pos(tracked)
            ):
                return
        elif not generation_changed and current_mtime <= last_mtime:
            return

        startup_boundary = (
            None
            if generation_changed or not provider.capabilities.supports_incremental_read
            else self._prepare_startup_boundary(
                session_id,
                tracked,
                st,
                consumed_intact=consumed_intact,
            )
        )
        stable_read = await self._read_session_entries(
            session_id,
            tracked,
            file_path,
            window_id,
            check_marker=provider.capabilities.supports_incremental_read,
        )
        if stable_read is None:
            return
        new_entries, _, reset_during_read = stable_read
        if not await self._commit_stable_read(
            session_id, tracked, file_path, stable_read
        ):
            return
        if reset_during_read:
            startup_boundary = None

        # Deliver pre-start unread history, but count activity only when a
        # complete entry advances beyond the startup file-size boundary.
        has_live_entries = startup_boundary is None or (
            self._parse_pos(tracked) > startup_boundary.size
        )
        if startup_boundary is not None and (
            self._parse_pos(tracked) >= startup_boundary.size
        ):
            self._startup_file_boundaries.pop(session_id, None)
        if new_entries and has_live_entries:
            self._idle_tracker.record_activity(session_id)
        self._append_provider_messages(
            session_id,
            new_entries,
            provider,
            current_map,
            window_id,
            new_messages,
        )
        self._state.update_session(tracked)

    def _append_provider_messages(
        self,
        session_id: str,
        new_entries: list[dict],
        provider: Any,
        current_map: dict[str, dict[str, Any]] | None,
        window_id: str,
        new_messages: list[NewMessage],
    ) -> None:
        if provider.capabilities.supports_task_tracking and window_id:
            provider.apply_task_entries(window_id, session_id, new_entries)
        session_cwd = next(
            (
                details.get("cwd")
                for details in (current_map or {}).values()
                if details.get("session_id") == session_id
            ),
            None,
        )
        agent_messages, remaining = provider.parse_transcript_entries(
            new_entries,
            pending_tools=self._pending_tools.get(session_id, {}),
            cwd=session_cwd,
        )
        if remaining:
            self._pending_tools[session_id] = remaining
        else:
            self._pending_tools.pop(session_id, None)
        new_messages.extend(
            NewMessage(
                session_id=session_id,
                text=entry.text,
                is_complete=entry.is_complete,
                content_type=entry.content_type,
                phase=entry.phase,
                tool_use_id=entry.tool_use_id,
                role=entry.role,
                tool_name=entry.tool_name,
            )
            for entry in agent_messages
            if entry.text
        )

    async def _read_new_lines(
        self,
        session: TrackedSession,
        file_path: Path,
        window_id: str = "",
        provider: Any = None,
    ) -> list[dict]:
        """Read new lines from a session file using byte offset.

        ``provider`` may be passed by callers that already resolved it
        (avoids a second resolution per file per poll cycle).
        """
        if provider is None:
            provider = _resolve_provider_for_file(window_id, file_path)

        if not provider.capabilities.supports_incremental_read:
            return await self._read_whole_file(session, file_path, provider)

        new_entries: list[dict] = []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                await f.seek(0, 2)
                file_size = await f.tell()

                pos = self._parse_pos(session)
                if pos > file_size:
                    # A shrunken file is almost always a REPLACED session
                    # transcript (session refresh/rotation), not a genuine
                    # truncation. Resuming from 0 would replay the entire
                    # history as fresh notifications, flooding Telegram and
                    # starving every other topic (incident 2026-08-17).
                    # Resume from EOF instead: treat existing bytes as
                    # already-notified history.
                    logger.info(
                        "File shrunk/replaced for session %s "
                        "(offset %d > size %d): resuming from EOF, "
                        "no replay.",
                        session.session_id,
                        pos,
                        file_size,
                    )
                    pos = file_size

                await f.seek(pos)

                if pos > 0:
                    first_byte = await f.read(1)
                    if first_byte and first_byte != "{":
                        logger.warning(
                            "Corrupted offset for session %s (byte %d is %r, not '{'). "
                            "Advancing to next line.",
                            session.session_id,
                            pos,
                            first_byte,
                        )
                        await f.readline()
                        pos = await f.tell()
                    else:
                        await f.seek(pos)

                safe_offset = pos
                async for line in f:
                    data = provider.parse_transcript_line(line)
                    if data:
                        new_entries.append(data)
                        safe_offset = await f.tell()
                    elif line.strip():
                        log_throttled(
                            logger,
                            f"partial-jsonl:{session.session_id}",
                            "Partial JSONL line in session %s, will retry next cycle",
                            session.session_id,
                        )
                        break
                    else:
                        safe_offset = await f.tell()

                # Parsed position advances in memory only; the persisted
                # last_byte_offset (delivered watermark) is committed by the
                # monitor once the queues have drained (TASK-5).
                session.parsed_offset = safe_offset

        except OSError:
            logger.exception("Error reading session file %s", file_path)
            raise
        return new_entries

    async def _read_whole_file(
        self,
        session: TrackedSession,
        file_path: Path,
        provider: Any,
    ) -> list[dict]:
        """Read a whole-file transcript (e.g. Gemini JSON) via the provider."""
        try:
            new_entries, new_offset = await asyncio.to_thread(
                provider.read_transcript_file,
                str(file_path),
                self._parse_pos(session),
            )
            session.parsed_offset = new_offset
            return new_entries
        except OSError:
            logger.exception("Error reading transcript file %s", file_path)
            raise

    async def _get_active_cwds(self) -> set[str]:
        """Get normalised cwds of every live window.

        Reads the complete listing, not the adoption-filtered one: this set
        decides which transcripts are discoverable at all, so a live window a
        backend merely will not auto-adopt (out of the agterm workspace scope,
        under a herdr internal workspace, tmux-hidden) would have its history
        silently invisible to /restore and the history pager.
        """
        # Lazy: importing the reconciliation seam at module load forms a cycle
        # through providers, which import transcript_reader for their formats.
        from .multiplexer.reconciliation import list_windows_for_reconciliation

        cwds: set[str] = set()
        windows = await list_windows_for_reconciliation()
        for w in windows or []:
            try:
                cwds.add(str(Path(w.cwd).resolve()))
            except _PathResolveError:
                cwds.add(w.cwd)
        return cwds

    def _scan_projects_sync(
        self, projects_path: Path, active_cwds: set[str]
    ) -> list[SessionInfo]:
        """Scan filesystem for session files matching active cwds (sync)."""
        sessions: list[SessionInfo] = []

        if not projects_path.exists():
            return sessions

        for project_dir in projects_path.iterdir():
            if not project_dir.is_dir():
                continue

            index_file = project_dir / "sessions-index.json"
            original_path = ""
            indexed_ids: set[str] = set()

            if index_file.exists():
                try:
                    index_data = json.loads(index_file.read_text())
                    entries = index_data.get("entries", [])
                    original_path = index_data.get("originalPath", "")

                    for entry in entries:
                        session_id = entry.get("sessionId", "")
                        full_path = entry.get("fullPath", "")
                        project_path = entry.get("projectPath", original_path)

                        if not session_id or not full_path:
                            continue

                        try:
                            norm_pp = str(Path(project_path).resolve())
                        except _PathResolveError:
                            norm_pp = project_path
                        if norm_pp not in active_cwds:
                            continue

                        indexed_ids.add(session_id)
                        file_path = Path(full_path)
                        if file_path.exists():
                            sessions.append(
                                SessionInfo(
                                    session_id=session_id,
                                    file_path=file_path,
                                )
                            )

                except (json.JSONDecodeError, OSError) as e:
                    # Degraded discovery: index unreadable, falling back to a
                    # glob scan below. Worth surfacing — not a per-poll hot path.
                    logger.warning("Error reading index %s: %s", index_file, e)

            try:
                for jsonl_file in project_dir.glob("*.jsonl"):
                    session_id = jsonl_file.stem
                    if session_id in indexed_ids:
                        continue

                    file_project_path = original_path
                    if not file_project_path:
                        file_project_path = read_cwd_from_jsonl(jsonl_file)
                    if not file_project_path:
                        continue

                    try:
                        norm_fp = str(Path(file_project_path).resolve())
                    except _PathResolveError:
                        norm_fp = file_project_path

                    if norm_fp not in active_cwds:
                        continue

                    sessions.append(
                        SessionInfo(
                            session_id=session_id,
                            file_path=jsonl_file,
                        )
                    )
            except OSError as e:
                logger.warning("Error scanning jsonl files in %s: %s", project_dir, e)

        return sessions
