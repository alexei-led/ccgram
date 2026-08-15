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
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

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

    def __init__(self, state: MonitorState, idle_tracker: IdleTracker) -> None:
        self._state = state
        self._idle_tracker = idle_tracker
        self._pending_tools: dict[str, dict[str, Any]] = {}
        self._file_mtimes: dict[str, float] = {}
        self._file_generations = self._snapshot_file_generations()
        self._startup_file_boundaries = self._snapshot_startup_boundaries()

    def _snapshot_file_generations(self) -> dict[str, tuple[int, int]]:
        generations: dict[str, tuple[int, int]] = {}
        for session_id, tracked in self._state.tracked_sessions.items():
            try:
                st = Path(tracked.file_path).stat()
            except OSError:
                continue
            generations[session_id] = (st.st_dev, st.st_ino)
        return generations

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
        self, session_id: str, tracked: TrackedSession, st: Any
    ) -> _StartupBoundary | None:
        """Reset post-start replacements and return the activity boundary."""
        boundary = self._startup_file_boundaries.get(session_id)
        if boundary is None:
            return None
        generation_changed = (st.st_dev, st.st_ino) != (
            boundary.device,
            boundary.inode,
        )
        if generation_changed or st.st_size < boundary.size:
            tracked.last_byte_offset = 0
            boundary = _StartupBoundary(
                size=0,
                device=st.st_dev,
                inode=st.st_ino,
            )
            self._startup_file_boundaries[session_id] = boundary
        return boundary

    async def _read_session_entries(
        self, tracked: TrackedSession, file_path: Path, window_id: str
    ) -> list[dict] | None:
        """Read once, preserving startup state when the file is unavailable."""
        try:
            return await self._read_new_lines(tracked, file_path, window_id)
        except OSError:
            return None

    def clear_session(self, session_id: str) -> None:
        """Remove all per-session state for a cleaned-up session."""
        self._state.remove_session(session_id)
        self._file_mtimes.pop(session_id, None)
        self._pending_tools.pop(session_id, None)
        self._file_generations.pop(session_id, None)
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
            self._state.remove_session(old_session_id)
            self._state.update_session(tracked)
            if old_session_id in self._file_mtimes:
                self._file_mtimes[session_id] = self._file_mtimes.pop(old_session_id)
            if old_session_id in self._file_generations:
                self._file_generations[session_id] = self._file_generations.pop(
                    old_session_id
                )
            if old_session_id in self._pending_tools:
                self._pending_tools[session_id] = self._pending_tools.pop(
                    old_session_id
                )
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
            if generation is not None:
                self._file_generations[session_id] = generation
            if provider.capabilities.supports_task_tracking and window_id:
                await provider.seed_task_state(window_id, session_id, str(file_path))
            logger.debug("Started tracking session: %s", session_id)
            return

        try:
            st = file_path.stat()
            current_mtime, current_size = st.st_mtime, st.st_size
        except OSError:
            return

        generation = (st.st_dev, st.st_ino)
        previous_generation = self._file_generations.get(session_id)
        generation_changed = (
            previous_generation is not None and previous_generation != generation
        )
        if generation_changed:
            tracked.last_byte_offset = 0
            self._startup_file_boundaries.pop(session_id, None)
        self._file_generations[session_id] = generation

        last_mtime = self._file_mtimes.get(session_id, 0.0)
        if provider.capabilities.supports_incremental_read:
            if current_mtime <= last_mtime and current_size <= tracked.last_byte_offset:
                return
        else:
            if current_mtime <= last_mtime:
                return

        startup_boundary = (
            None
            if generation_changed or not provider.capabilities.supports_incremental_read
            else self._prepare_startup_boundary(session_id, tracked, st)
        )
        new_entries = await self._read_session_entries(tracked, file_path, window_id)
        if new_entries is None:
            return
        self._file_mtimes[session_id] = current_mtime

        # Deliver pre-start unread history, but count activity only when a
        # complete entry advances beyond the startup file-size boundary.
        has_live_entries = startup_boundary is None or (
            tracked.last_byte_offset > startup_boundary.size
        )
        if startup_boundary is not None and (
            tracked.last_byte_offset >= startup_boundary.size
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
        self, session: TrackedSession, file_path: Path, window_id: str = ""
    ) -> list[dict]:
        """Read new lines from a session file using byte offset."""
        provider = _resolve_provider_for_file(window_id, file_path)

        if not provider.capabilities.supports_incremental_read:
            return await self._read_whole_file(session, file_path, provider)

        new_entries: list[dict] = []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                await f.seek(0, 2)
                file_size = await f.tell()

                if session.last_byte_offset > file_size:
                    logger.info(
                        "File truncated for session %s "
                        "(offset %d > size %d). Resetting.",
                        session.session_id,
                        session.last_byte_offset,
                        file_size,
                    )
                    session.last_byte_offset = 0

                await f.seek(session.last_byte_offset)

                if session.last_byte_offset > 0:
                    first_byte = await f.read(1)
                    if first_byte and first_byte != "{":
                        logger.warning(
                            "Corrupted offset for session %s (byte %d is %r, not '{'). "
                            "Advancing to next line.",
                            session.session_id,
                            session.last_byte_offset,
                            first_byte,
                        )
                        await f.readline()
                        session.last_byte_offset = await f.tell()
                    else:
                        await f.seek(session.last_byte_offset)

                safe_offset = session.last_byte_offset
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

                session.last_byte_offset = safe_offset

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
                session.last_byte_offset,
            )
            session.last_byte_offset = new_offset
            return new_entries
        except OSError:
            logger.exception("Error reading transcript file %s", file_path)
            raise

    async def _get_active_cwds(self) -> set[str]:
        """Get normalized cwds of all active tmux windows."""
        # Lazy: tmux_manager imports providers which transitively imports
        # transcript_reader through provider format modules.
        # Lazy: tmux_manager pulls providers eagerly; defer until pane lookup runs
        from .multiplexer import multiplexer as tmux_manager

        cwds: set[str] = set()
        windows = await tmux_manager.list_windows()
        for w in windows:
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
