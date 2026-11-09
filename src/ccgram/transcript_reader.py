"""Transcript reading and processing for agent session files.

Handles the full lifecycle of reading agent transcripts:
  - Scanning Claude projects for active session files
  - Incremental byte-offset reads for JSONL providers
  - Whole-document reads for providers without append-only JSONL
  - Parsing transcript entries into NewMessage objects
  - mtime caching to skip unchanged files
  - Pending tool-use state carried across poll cycles

Key class: TranscriptReader.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import aiofiles
import structlog

from .config import config
from .monitor_events import NewMessage, SessionInfo
from .expandable_quote import format_expandable_quote
from .window_state_ports import identity_state
from .monitor_state import MonitorState, TrackedSession
from .providers import (
    detect_provider_from_transcript_path,
    get_provider_for_window,
    registry,
)
from .utils import log_throttle_reset, log_throttled, read_cwd_from_jsonl
from .session_protocol.messages import (
    SourceEntry,
    TranscriptBatch,
    TranscriptRecord,
    TranscriptParserState,
    parse_records,
    read_entries,
)
from .tool_format import format_tool_line

if TYPE_CHECKING:
    from .idle_tracker import IdleTracker

logger = structlog.get_logger()

_PathResolveError = (OSError, ValueError)
_TOOL_RESULT_QUOTE_THRESHOLD = 3


class _StartupBoundary(NamedTuple):
    size: int
    device: int
    inode: int


class _StableRead(NamedTuple):
    entries: list[SourceEntry]
    stat: Any
    reset_generation: bool
    start_cursor: int  # raw parsed_offset before this read, in its file generation


class _GenerationCheck(NamedTuple):
    changed: bool
    consumed_intact: bool


@dataclass(slots=True)
class _ToolDeliveryFence:
    call_checkpoint: int
    release_checkpoint: int | None = None


_MARKER_BYTES = 128


def _transcript_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("output", "result", "text", "error"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested.strip()
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text", "")) for item in value if isinstance(item, dict)
        )
    return str(value)


def _render_tool_result(
    provider_name: str,
    tool_name: str | None,
    summary: str | None,
    output: Any,
    failed: bool,
    diff: dict[str, Any] | None,
) -> str:
    text = _transcript_output_text(output)
    if failed:
        error = text.splitlines()[0] if text else "Error"
        prefix = format_tool_line(tool_name or "", summary or "")
        quote = f"\n{format_expandable_quote(text)}" if "\n" in text else ""
        return f"{prefix}\n  ⎿  ⚠️ {error}{quote}"
    if provider_name == "claude" and tool_name == "Edit" and diff:
        diff_text = diff.get("text")
        if isinstance(diff_text, str) and diff_text:
            prefix = format_tool_line(tool_name, summary or "")
            return (
                f"{prefix}\n  ⎿  +{diff.get('added', 0)} −{diff.get('removed', 0)}\n"
                f"{format_expandable_quote(diff_text)}"
            )
    if not text:
        return "Done"
    lines = text.count("\n") + 1
    quote = format_expandable_quote(text)
    if provider_name == "claude":
        prefix = format_tool_line(tool_name or "", summary or "")
        return f"{prefix}\n  ⎿  {lines} lines\n{quote}"
    raw_name = tool_name or ""
    if provider_name == "codex" and raw_name in {"exec_command", "shell"}:
        return f"  ⎿  {lines} lines\n{quote}"
    if provider_name == "pi" and raw_name == "bash":
        unit = "line" if lines == 1 else "lines"
        return f"  ⎿  {lines} {unit}\n{quote}"
    if lines > _TOOL_RESULT_QUOTE_THRESHOLD:
        return f"  ⎿  {lines} lines\n{quote}"
    return text


def render_protocol_records(
    session_id: str,
    provider_name: str,
    records: tuple[tuple[SourceEntry, TranscriptRecord], ...],
    state: TranscriptParserState,
) -> list[NewMessage]:
    """Render canonical protocol records into Telegram monitor events."""
    rendered: list[NewMessage] = []
    tool_aliases = {
        "apply_patch": "Edit",
        "request_user_input": "AskUserQuestion",
    }
    for _, record in records:
        role = "assistant" if record.role == "agent" else record.role

        def render_parts(parts: tuple[dict[str, Any], ...]) -> None:
            for raw_part in parts:
                part_type = raw_part.get("type")
                text = raw_part.get("text")
                content_type = "text"
                tool_id = raw_part.get("tool_call_id")
                tool_name = raw_part.get("name")
                plan_text: str | None = None
                if part_type == "thinking":
                    content_type = "thinking"
                    if isinstance(text, str) and text:
                        text = format_expandable_quote(text)
                elif part_type == "tool":
                    content_type = "tool_use"
                    if isinstance(tool_name, str):
                        display_name = tool_aliases.get(tool_name, tool_name)
                        title = raw_part.get("title")
                        text = format_tool_line(
                            display_name, title if isinstance(title, str) else ""
                        )
                        tool_input = raw_part.get("input")
                        plan = (
                            tool_input.get("plan")
                            if tool_name == "ExitPlanMode"
                            and isinstance(tool_input, dict)
                            else None
                        )
                        if isinstance(plan, str) and plan.strip():
                            plan_text = format_expandable_quote(plan.strip())
                        tool_name = display_name
                elif part_type == "notice":
                    text = raw_part.get("text")
                    if raw_part.get("source") == "local_command":
                        command = raw_part.get("command")
                        if isinstance(command, str) and command:
                            text = (
                                f"❯ `{command}`\n```\n{text}\n```"
                                if isinstance(text, str) and "\n" in text
                                else f"❯ `{command}`\n`{text}`"
                            )
                        elif isinstance(text, str) and "\n" in text:
                            text = f"```\n{text}\n```"
                        else:
                            text = f"`{text}`"
                        content_type = "local_command"
                if not isinstance(text, str) or not text:
                    continue
                if plan_text:
                    rendered.append(
                        NewMessage(
                            session_id=session_id,
                            text=plan_text,
                            is_complete=record.complete,
                            content_type="text",
                            phase=record.phase,
                            role=role,
                            provider_name=provider_name,
                        )
                    )
                rendered.append(
                    NewMessage(
                        session_id=session_id,
                        text=text,
                        is_complete=record.complete,
                        content_type=content_type,
                        phase=record.phase,
                        tool_use_id=tool_id if isinstance(tool_id, str) else None,
                        role=role,
                        tool_name=tool_name if isinstance(tool_name, str) else None,
                        provider_name=provider_name,
                    )
                )

        def render_results() -> None:
            for tool_id, output, failed in record.results:
                raw_name = state.tool_names.get(tool_id)
                output_text = _render_tool_result(
                    provider_name,
                    raw_name,
                    state.tool_summaries.get(tool_id),
                    output,
                    failed,
                    state.tool_diffs.get(tool_id),
                )
                rendered.append(
                    NewMessage(
                        session_id=session_id,
                        text=output_text,
                        is_complete=record.complete,
                        content_type="tool_result",
                        phase=record.phase,
                        tool_use_id=tool_id or None,
                        role="assistant",
                        tool_name=tool_aliases.get(raw_name or "", raw_name),
                        provider_name=provider_name,
                    )
                )

        if provider_name == "gemini":
            render_parts(
                tuple(part for part in record.parts if part.get("type") == "tool")
            )
            render_results()
            render_parts(
                tuple(part for part in record.parts if part.get("type") != "tool")
            )
        else:
            render_parts(record.parts)
            render_results()
    return rendered


def render_transcript_batch(batch: TranscriptBatch) -> tuple[NewMessage, ...]:
    state = TranscriptParserState(
        tool_names=dict(batch.tool_names),
        tool_summaries=dict(batch.tool_summaries),
        tool_diffs=dict(batch.tool_diffs),
    )
    return tuple(
        render_protocol_records(
            batch.session_id, batch.provider_name, batch.records, state
        )
    )


def _hash_range(digest: Any, file_path: Path, start: int, end: int) -> int:
    """Feed bytes [start, end) to *digest*; return how many were actually there."""
    hashed = 0
    with file_path.open("rb") as transcript:
        transcript.seek(start)
        while hashed < end - start:
            chunk = transcript.read(min(64 * 1024, end - start - hashed))
            if not chunk:
                break
            digest.update(chunk)
            hashed += len(chunk)
    return hashed


def _prefix_digest(file_path: Path, size: int) -> bytes:
    """Digest of bytes [0, size): only for signal events, never per poll."""
    digest = hashlib.sha256()
    _hash_range(digest, file_path, 0, size)
    return digest.digest()


def _advance_prefix(
    saved: tuple[int, Any] | None, file_path: Path, size: int
) -> tuple[int, Any]:
    """Running digest of bytes [0, size), hashing only bytes it has not seen.

    Append-only growth costs O(appended). A missing baseline, or one that
    covers more than *size* (truncation), is hashed from the start once.
    The saved digest is copied, so a failed read leaves the baseline intact.
    A short read (file truncated meanwhile) records only the bytes hashed, so
    the baseline never claims coverage it does not have.
    """
    if saved is None or saved[0] > size:
        digest, start = hashlib.sha256(), 0
    else:
        digest, start = saved[1].copy(), saved[0]
    return (start + _hash_range(digest, file_path, start, size), digest)


_REBUILD_CHUNK_BYTES = 4 * 1024 * 1024


def _replay_committed_prefix(
    file_path: Path, provider_name: str, floor: int, generation: tuple[int, int]
) -> TranscriptParserState | None:
    """Rebuild parser state from the committed prefix [0, floor) in bounded chunks.

    Bytes past *floor* are read live by the reader and may grow concurrently,
    so they are never read here. Only a replacement (device/inode change) or a
    file that shrank during the read invalidates the rebuild; growth does not.
    A file shorter than *floor* is replayed to its end, as a watermark can
    outlive a replaced transcript until the monitor commits the clamp.
    Memory stays at one chunk instead of the whole transcript.
    """

    def same_file(st: Any, min_size: int) -> bool:
        return (st.st_dev, st.st_ino) == generation and st.st_size >= min_size

    parser_state = TranscriptParserState()

    def feed(raw: bytes, base: int) -> None:
        nonlocal parser_state
        entries = [
            SourceEntry(base + entry.offset, entry.data)
            for entry in read_entries(raw, provider_name)
            if base + entry.offset < floor
        ]
        _, parser_state = parse_records(
            entries, provider_name, parser_state, deduplicate_updates=True
        )

    with file_path.open("rb") as transcript:
        opened = os.fstat(transcript.fileno())
        if not same_file(opened, 0):
            return None
        base = 0
        # Pieces of the unfinished last line; joined once so a line spanning
        # many chunks is not re-copied per chunk.
        pending: list[bytes] = []
        remaining = floor
        while remaining > 0:
            block = transcript.read(min(_REBUILD_CHUNK_BYTES, remaining))
            if not block:
                break
            remaining -= len(block)
            cut = block.rfind(b"\n") + 1
            if not cut:
                pending.append(block)
                continue
            complete = b"".join(pending) + block[:cut]
            feed(complete, base)
            base += len(complete)
            pending = [block[cut:]]
        if pending and (tail := b"".join(pending)):
            # A parse position inside a line still owns that whole line.
            feed(tail + transcript.readline(), base)
    if not same_file(file_path.stat(), opened.st_size):
        return None
    return parser_state


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
    # The identity store is not wired during early transcript discovery.
    with suppress(RuntimeError):
        provider_name = identity_state.get_provider_name(window_id)
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
        self._tool_delivery_fences: dict[str, _ToolDeliveryFence] = {}
        self._protocol_parser_states: dict[str, tuple[str, TranscriptParserState]] = {}
        self._parser_rebuild_pending: set[str] = set()
        self._pending_batches: list[TranscriptBatch] = []
        self._file_mtimes: dict[str, float] = {}
        self._file_ctimes: dict[str, int] = self._snapshot_file_ctimes()
        self._file_sizes = self._snapshot_file_sizes()
        # Empty at startup on purpose: the first committed read of each session
        # hashes its baseline in a worker thread. Hashing every tracked
        # transcript here blocked the event loop for seconds. A transcript
        # replaced before that first read has no baseline and is treated as
        # replaced (skip to EOF, never replay).
        self._file_prefixes: dict[str, tuple[int, Any]] = {}
        self._file_generations = self._snapshot_file_generations()
        self._file_markers = self._snapshot_file_markers()
        self._startup_file_boundaries = self._snapshot_startup_boundaries()
        # Sessions whose startup replay was already bounded in this process.
        self._startup_replay_checked: set[str] = set()
        # Sessions that existed before this process started. A session first
        # seen now is a fresh launch, not a backlog from before the restart.
        self._startup_sessions: set[str] = set(self._state.tracked_sessions)
        # Deliberate replay-from-start requests, which must never be clamped.
        self._replay_allowed: set[str] = set()
        # Sessions this process clamped, with the offset it clamped to. A
        # later rollback must not undo the clamp, and must not skip output
        # written after it either.
        self._startup_clamped: dict[str, int] = {}

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

    async def _saved_prefix_matches(
        self, session_id: str, file_path: Path
    ) -> bool | None:
        """Digest the baseline range once; None when there is no verdict."""
        saved = self._file_prefixes.get(session_id)
        if saved is None:
            return None
        try:
            current = await asyncio.to_thread(_prefix_digest, file_path, saved[0])
        except OSError:
            return None
        return current == saved[1].digest()

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
        return await self._saved_prefix_matches(session_id, file_path) is True

    async def _marker_changed(
        self, session_id: str, file_path: Path, st: Any
    ) -> bool | None:
        """Compare the bytes before the last committed offset; None if unverifiable.

        Checked at the committed offset, not the live parse position, so a
        rollback that moved the position still leaves rewrites detectable.
        """
        saved = self._file_markers.get(session_id)
        if saved is None:
            return None
        if st.st_size < saved[0]:
            return True
        try:
            current = await asyncio.to_thread(_tail_marker, file_path, saved[0])
        except OSError:
            return None
        return current != saved[1]

    def _stat_unchanged(self, session_id: str, st: Any) -> bool:
        """True when nothing observable moved since the last committed read."""
        return (
            self._file_generations.get(session_id) == (st.st_dev, st.st_ino)
            and self._file_sizes.get(session_id) == st.st_size
            and self._file_ctimes.get(session_id) == st.st_ctime_ns
        )

    async def _prepare_observed_generation(
        self,
        session_id: str,
        tracked: TrackedSession,
        file_path: Path,
        st: Any,
        *,
        check_marker: bool,
    ) -> _GenerationCheck:
        """Detect a replaced transcript without hashing on the steady-state path.

        Same inode: replaced or rewritten only if the file is shorter than the
        parse position or the bytes just before it changed (one small pread,
        skipped while the stat is unchanged). ctime alone proves nothing, since
        every append moves it. A new inode costs one digest of the consumed
        prefix, on that event only.
        """
        pos = self._parse_pos(tracked)
        previous = self._file_generations.get(session_id)
        generation_changed = previous is not None and previous != (
            st.st_dev,
            st.st_ino,
        )
        changed = generation_changed or st.st_size < pos
        consumed_intact = False
        if changed:
            consumed_intact = (
                generation_changed
                and check_marker
                and await self._consumed_prefix_intact(session_id, pos, file_path, st)
            )
        elif not self._stat_unchanged(session_id, st):
            # Whole-document providers have no append-only contract, so their
            # signal is the saved prefix; they re-parse the file on change anyway.
            if check_marker:
                verdict = await self._marker_changed(session_id, file_path, st)
            else:
                matches = await self._saved_prefix_matches(session_id, file_path)
                verdict = None if matches is None else not matches
            changed = verdict is True
            if verdict is False:
                # Verified metadata-only change (chmod, touch): remember it so
                # an idle file is not re-inspected every poll. An unverifiable
                # check (no baseline, I/O error) must not be memoised.
                self._file_ctimes[session_id] = st.st_ctime_ns
        if changed and not consumed_intact:
            # Replaced transcript: jump the parse position to EOF so the
            # existing bytes are treated as already-notified history
            # (incident 2026-08-17: never replay), and let the persisted
            # watermark follow via commit_parsed_offsets().
            tracked.parsed_offset = st.st_size
            self._startup_file_boundaries.pop(session_id, None)
            # A replaced transcript has no pre-restart backlog left: a clamp
            # that survived would bound parser rebuild past the new file and
            # swallow live answers.
            self._startup_clamped.pop(session_id, None)
            # The old baseline describes bytes that are gone; the next commit
            # re-baselines from the new file.
            self._file_prefixes.pop(session_id, None)
        elif consumed_intact:
            logger.debug(
                "Ignoring transcript replacement signal, delivered bytes intact: %s",
                session_id,
            )
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
        start_cursor = tracked.parsed_offset  # rebound per attempt below
        # Every None exit restores the cursor to where the failed attempt
        # started: the bytes it read were never emitted, and the next poll
        # must read them again instead of resuming after them.
        for _attempt in range(2):
            start_cursor = tracked.parsed_offset
            start_offset = self._parse_pos(tracked)
            try:
                before = file_path.stat()
                entries = await self._read_new_lines(tracked, file_path, window_id)
                after = file_path.stat()
            except OSError:
                self._rollback_read(session_id, tracked, start_cursor)
                return None
            same_generation = (before.st_dev, before.st_ino) == (
                after.st_dev,
                after.st_ino,
            )
            marker_changed = False
            saved = self._file_markers.get(session_id) if check_marker else None
            if saved is not None and saved[0] == start_offset:
                try:
                    marker = await asyncio.to_thread(
                        _tail_marker, file_path, start_offset
                    )
                except OSError:
                    self._log_read_abandoned(session_id)
                    self._rollback_read(session_id, tracked, start_cursor)
                    return None
                marker_changed = marker != saved[1]
            if same_generation and not marker_changed:
                return _StableRead(entries, after, reset_generation, start_cursor)
            if (
                not same_generation
                and check_marker
                and await self._consumed_prefix_intact(
                    session_id, start_offset, file_path, after
                )
            ):
                # The inode changed but the bytes already delivered survived,
                # so keep this read.
                return _StableRead(entries, after, reset_generation, start_cursor)
            tracked.parsed_offset = after.st_size
            self._startup_file_boundaries.pop(session_id, None)
            self._startup_clamped.pop(session_id, None)
            self._file_prefixes.pop(session_id, None)
            reset_generation = True
        # Two write races in a row. The cursor goes back to the last attempt's
        # start; if the file was replaced, the stale generation cache makes the
        # next poll jump to EOF (the no-replay policy), otherwise it re-reads.
        self._rollback_read(session_id, tracked, start_cursor)
        return None

    async def _commit_stable_read(
        self,
        session_id: str,
        tracked: TrackedSession,
        file_path: Path,
        stable_read: _StableRead,
        *,
        incremental: bool,
    ) -> bool:
        """Commit caches only after a stable read and marker capture.

        On an I/O failure nothing is cached and the cursor returns to where the
        read began, so the next poll reads the same range again.
        """
        stable_stat = stable_read.stat
        try:
            marker = await asyncio.to_thread(
                _tail_marker, file_path, self._parse_pos(tracked)
            )
            prefix = await asyncio.to_thread(
                _advance_prefix,
                # A rewritten document has no append-only baseline to extend.
                self._file_prefixes.get(session_id) if incremental else None,
                file_path,
                stable_stat.st_size,
            )
        except OSError:
            self._log_read_abandoned(session_id)
            self._rollback_read(session_id, tracked, stable_read.start_cursor)
            return False
        self._file_prefixes[session_id] = prefix
        self._file_mtimes[session_id] = stable_stat.st_mtime
        self._file_generations[session_id] = (
            stable_stat.st_dev,
            stable_stat.st_ino,
        )
        self._file_ctimes[session_id] = stable_stat.st_ctime_ns
        self._file_sizes[session_id] = stable_stat.st_size
        self._file_markers[session_id] = (self._parse_pos(tracked), marker)
        return True

    async def _rebuild_committed_parser_context(
        self,
        session_id: str,
        file_path: Path,
        provider_name: str,
        expected_stat: Any,
        *,
        supports_incremental_read: bool,
    ) -> bool:
        """Recreate parser context from a stable, committed JSONL prefix."""
        if not supports_incremental_read:
            # Live built-in providers with transcript records use byte-addressed
            # JSONL. Whole-document history snapshots do not have a byte range
            # for each entry, so they cannot be reconstructed from this watermark.
            self.reset_parser_state(session_id)
            return True
        tracked = self._state.get_session(session_id)
        if tracked is None:
            return False
        # Rebuild to the position the reader will resume from, which is the
        # clamp point when a startup clamp applied. Rebuilding only to the
        # delivered watermark would miss the skipped turn and replay stale
        # parser context (a Codex task_complete "Done" that the skipped
        # user turn had cleared), discarding the next real answer.
        floor = self._clamped_floor(session_id, tracked.last_byte_offset)
        try:
            parser_state = await asyncio.to_thread(
                _replay_committed_prefix,
                file_path,
                provider_name,
                floor,
                (expected_stat.st_dev, expected_stat.st_ino),
            )
        except OSError:
            return False
        if parser_state is None:
            return False
        self._protocol_parser_states[session_id] = (provider_name, parser_state)
        self._pending_tools[session_id] = dict(parser_state.pending_tools)
        self.retire_skipped_tool_state(session_id)
        self._parser_rebuild_pending.discard(session_id)
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
        self.reset_parser_state(session_id)
        self._pending_batches = [
            batch for batch in self._pending_batches if batch.session_id != session_id
        ]
        self._file_generations.pop(session_id, None)
        self._file_ctimes.pop(session_id, None)
        self._file_sizes.pop(session_id, None)
        self._file_prefixes.pop(session_id, None)
        self._file_markers.pop(session_id, None)
        self._startup_file_boundaries.pop(session_id, None)
        # The startup replay bound's per-session state goes too: a session id
        # reused later must not inherit the old session's replay exemption,
        # clamp offset, or pre-existing-at-startup status (#367).
        self._replay_allowed.discard(session_id)
        self._startup_clamped.pop(session_id, None)
        self._startup_sessions.discard(session_id)
        self._startup_replay_checked.discard(session_id)
        log_throttle_reset(f"partial-jsonl:{session_id}")

    def retire_skipped_tool_state(self, session_id: str) -> None:
        """Retire only tool calls whose source range a completed skip consumed."""
        self._pending_tools.pop(session_id, None)
        self._tool_delivery_fences.pop(session_id, None)

        parser_state = self._protocol_parser_states.get(session_id)
        if parser_state is not None:
            parser_state[1].pending_tools.clear()

    def reset_parser_state(self, session_id: str) -> None:
        """Drop parser context before replaying a source from its watermark."""
        self.retire_skipped_tool_state(session_id)
        self._protocol_parser_states.pop(session_id, None)
        self._parser_rebuild_pending.discard(session_id)

    def _clamped_floor(self, session_id: str, watermark: int) -> int:
        """Lower bound for a resumed parse position: the clamp, when applied."""
        return max(self._startup_clamped.get(session_id, watermark), watermark)

    def replay_from_watermark(self, session_id: str) -> None:
        """Discard uncommitted parse context and resume from durable progress.

        A canceled or expired backlog skip restores the delivered watermark.
        For a clamped session that watermark still precedes the range this
        process decided to skip, so resuming from it alone would replay the
        backlog the clamp exists to prevent.
        """
        tracked = self._state.get_session(session_id)
        if tracked is None:
            return
        tracked.parsed_offset = self._clamped_floor(
            session_id, tracked.last_byte_offset
        )
        self.reset_parser_state(session_id)
        self._parser_rebuild_pending.add(session_id)
        self._file_mtimes.pop(session_id, None)
        self._pending_batches = [
            batch for batch in self._pending_batches if batch.session_id != session_id
        ]

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
            self._file_mtimes.pop(old_session_id, None)
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
            self.reset_parser_state(old_session_id)
            if old_session_id in self._file_markers:
                self._file_markers[session_id] = self._file_markers.pop(old_session_id)
            if old_session_id in self._startup_file_boundaries:
                self._startup_file_boundaries[session_id] = (
                    self._startup_file_boundaries.pop(old_session_id)
                )
            # A refresh can arrive before the old id was ever polled, so the
            # clamp has not run yet: carry the membership instead of the clamp,
            # and clear the checked flag so the bound is applied to this id.
            if old_session_id in self._startup_sessions:
                self._startup_sessions.add(session_id)
                self._startup_replay_checked.discard(session_id)
            # An explicit replay request belongs to the transcript, not the id.
            if old_session_id in self._replay_allowed:
                self._replay_allowed.add(session_id)
            # The clamp is keyed by session id, so a refreshed id would read as
            # a brand-new session and replay the backlog the old id clamped.
            if old_session_id in self._startup_clamped:
                clamped = self._startup_clamped.pop(old_session_id)
                self._startup_clamped[session_id] = clamped
                self._startup_sessions.add(session_id)
                self._startup_replay_checked.add(session_id)
                # The replacement cursor starts at -1, which falls back to the
                # delivered watermark and would replay the clamped range. Point
                # it at the clamp instead: the clamp state moved with the id.
                tracked.parsed_offset = max(clamped, tracked.last_byte_offset)
            # The old id is retired: leaving its membership behind would leak.
            self._replay_allowed.discard(old_session_id)
            self._startup_sessions.discard(old_session_id)
            self._startup_replay_checked.discard(old_session_id)
            self._parser_rebuild_pending.add(session_id)
            log_throttle_reset(f"partial-jsonl:{old_session_id}")
            logger.debug(
                "Adopted transcript offset for refreshed session: %s -> %s (%s)",
                old_session_id,
                session_id,
                str(file_path),
            )
            return tracked
        return None

    def allow_replay(self, session_id: str) -> None:
        """Exempt *session_id* from the startup replay bound.

        Called when a session map explicitly asks for a replay from the start:
        that request is authoritative, so the clamp must not silently skip it.
        """
        self._replay_allowed.add(session_id)

    def _clamp_startup_replay(
        self, session_id: str, tracked: TrackedSession, file_size: int
    ) -> None:
        """Bound what one restart may replay from an undelivered transcript.

        The delivered watermark only advances when the queue fully drains, so
        a wedged or flood-limited queue leaves it far behind. Replaying that
        whole range on the next start re-sends thousands of messages, which
        trips Telegram flood control and freezes delivery again — a loop.
        Once per process per session, a range larger than
        CCGRAM_STARTUP_REPLAY_MAX_KB is treated as history: the parse position
        jumps to EOF, exactly like a replaced transcript. The bytes stay in
        the transcript file; only their Telegram delivery is skipped.
        """
        if session_id in self._startup_replay_checked:
            return
        self._startup_replay_checked.add(session_id)
        if (
            session_id not in self._startup_sessions
            or session_id in self._replay_allowed
        ):
            # Freshly created in this process, or an explicit replay request:
            # there is no pre-restart backlog to bound.
            return
        limit = config.startup_replay_max_bytes
        if limit <= 0:
            return
        pending = file_size - self._parse_pos(tracked)
        if pending <= limit:
            return
        tracked.parsed_offset = file_size
        self._startup_clamped[session_id] = file_size
        self._startup_file_boundaries.pop(session_id, None)
        # Rebuild the parser context from the clamped prefix. Without this a
        # provider whose delivery mirrors state (Codex task_complete) can emit
        # a skipped final answer again, because the rebuild guard only fires
        # for a nonzero watermark.
        self._parser_rebuild_pending.add(session_id)
        logger.warning(
            "Startup replay bounded: skipped %.1f KiB of undelivered transcript "
            "for session %s (limit %d KiB, CCGRAM_STARTUP_REPLAY_MAX_KB)",
            pending / 1024,
            session_id,
            limit // 1024,
        )

    @staticmethod
    def _parse_pos(session: TrackedSession) -> int:
        """Current parse position (parsed_offset once set, watermark before)."""
        return (
            session.parsed_offset
            if session.parsed_offset >= 0
            else session.last_byte_offset
        )

    async def _start_tracking_session(
        self,
        session_id: str,
        file_path: Path,
        provider: Any,
        window_id: str,
        provider_name: str,
    ) -> None:
        try:
            st = file_path.stat()
            file_size, current_mtime = st.st_size, st.st_mtime
        except OSError:
            file_size, current_mtime, st = 0, 0.0, None
        initial_offset = file_size
        tracked = TrackedSession(
            session_id=session_id,
            file_path=str(file_path),
            last_byte_offset=initial_offset,
        )
        self._file_mtimes[session_id] = current_mtime
        if st is not None:
            self._file_generations[session_id] = (st.st_dev, st.st_ino)
            self._file_ctimes[session_id] = st.st_ctime_ns
            self._file_sizes[session_id] = st.st_size
            with suppress(OSError):
                self._file_prefixes[session_id] = await asyncio.to_thread(
                    _advance_prefix, None, file_path, st.st_size
                )
        try:
            marker = await asyncio.to_thread(
                _tail_marker, file_path, tracked.last_byte_offset
            )
        except OSError:
            pass
        else:
            self._file_markers[session_id] = (tracked.last_byte_offset, marker)
        if not identity_state.accepts_provider_observation(window_id, provider_name):
            return
        self._state.update_session(tracked)
        if provider.capabilities.supports_task_tracking and window_id:
            await provider.seed_task_state(window_id, session_id, str(file_path))
        logger.debug("Started tracking session: %s", session_id)

    def _admit_read(
        self,
        window_id: str,
        provider_name: str,
        tracked: TrackedSession,
        previous_offset: int,
    ) -> bool:
        """Roll back speculative progress if selection changed during file I/O."""
        if identity_state.accepts_provider_observation(window_id, provider_name):
            return True
        self._rollback_read(tracked.session_id, tracked, previous_offset)
        return False

    @staticmethod
    def _log_read_abandoned(session_id: str) -> None:
        log_throttled(
            logger,
            f"read-abandoned:{session_id}",
            "Transcript I/O failed for session %s; the read will be retried next poll",
            session_id,
            level="warning",
        )

    def _rollback_read(
        self, session_id: str, tracked: TrackedSession, offset: int
    ) -> None:
        """Undo the cursor move of a read whose entries were not emitted.

        The exit for every read abandoned after it moved the cursor. The mtime gate is
        cleared so a whole-document provider re-reads on the next poll even
        though the file did not change again.
        """
        tracked.parsed_offset = offset
        self._file_mtimes.pop(session_id, None)

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
        provider_name = (
            detect_provider_from_transcript_path(str(file_path))
            or provider.capabilities.name
        )
        if not identity_state.accepts_provider_observation(window_id, provider_name):
            return

        if tracked is None:
            tracked = self._adopt_tracking_for_file(session_id, file_path)

        if tracked is None:
            await self._start_tracking_session(
                session_id, file_path, provider, window_id, provider_name
            )
            return

        try:
            st = file_path.stat()
            current_mtime, current_size = st.st_mtime, st.st_size
        except OSError:
            return

        self._clamp_startup_replay(session_id, tracked, current_size)
        probe_offset = tracked.parsed_offset
        probe_boundary = self._startup_file_boundaries.get(session_id)
        generation_changed, consumed_intact = await self._prepare_observed_generation(
            session_id,
            tracked,
            file_path,
            st,
            check_marker=provider.capabilities.supports_incremental_read,
        )
        if not identity_state.accepts_provider_observation(window_id, provider_name):
            tracked.parsed_offset = probe_offset
            if probe_boundary is not None:
                self._startup_file_boundaries[session_id] = probe_boundary
            return
        if (
            not generation_changed
            and (tracked.last_byte_offset > 0 or session_id in self._startup_clamped)
            and session_id not in self._protocol_parser_states
        ):
            # Parser-only context is volatile. Recreate it from the durable
            # prefix before parsing a post-restart suffix or Codex mirror.
            self._parser_rebuild_pending.add(session_id)
        rebuild_parser = session_id in self._parser_rebuild_pending
        last_mtime = self._file_mtimes.get(session_id, 0.0)
        if provider.capabilities.supports_incremental_read:
            # A pending rebuild alone is no reason to read: parser context is
            # only needed once there are entries to parse (see below).
            if (
                not generation_changed
                and current_mtime <= last_mtime
                and current_size <= self._parse_pos(tracked)
            ):
                return
        elif (
            not generation_changed
            and not rebuild_parser
            and current_mtime <= last_mtime
        ):
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
        if stable_read is None or not self._admit_read(
            window_id, provider_name, tracked, stable_read.start_cursor
        ):
            return
        new_entries, _, reset_during_read, _ = stable_read
        committed = await self._commit_stable_read(
            session_id,
            tracked,
            file_path,
            stable_read,
            incremental=provider.capabilities.supports_incremental_read,
        )
        if (
            not self._admit_read(
                window_id, provider_name, tracked, stable_read.start_cursor
            )
            or not committed
        ):
            return
        if reset_during_read:
            startup_boundary = None
        if generation_changed or reset_during_read:
            if self._on_session_retired is not None:
                self._on_session_retired(session_id)
            self.reset_parser_state(session_id)
        elif rebuild_parser and new_entries:
            # Deferred until this poll has entries to parse. The rebuilt
            # context only affects how later entries render: its pending tools
            # are retired right after the rebuild. Its floor cannot move while
            # the flag waits: the watermark advances only through this
            # session's batches, and every batch without entries comes from a
            # replacement or replay, which reset or re-arm the flag in that
            # same poll. Rebuilding every idle restored session on the first
            # poll stalled that whole cycle (~4 s per 100 MB).
            rebuilt = await self._rebuild_committed_parser_context(
                session_id,
                file_path,
                provider_name,
                stable_read.stat,
                supports_incremental_read=provider.capabilities.supports_incremental_read,
            )
            if not rebuilt:
                # Rolling back to the delivered watermark is right for a live
                # session, but a clamped one would then replay the backlog this
                # process decided to skip. Roll back to the clamp point, not to
                # EOF: records written after it are live output and must be
                # read on the next poll.
                self._rollback_read(
                    session_id,
                    tracked,
                    max(self._startup_clamped[session_id], tracked.last_byte_offset)
                    if session_id in self._startup_clamped
                    else tracked.last_byte_offset,
                )
                new_entries = []
            elif not self._admit_read(
                window_id, provider_name, tracked, stable_read.start_cursor
            ):
                self.reset_parser_state(session_id)
                self._parser_rebuild_pending.add(session_id)
                if probe_boundary is not None:
                    self._startup_file_boundaries[session_id] = probe_boundary
                new_entries = []

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
            byte_checkpoint=self._parse_pos(tracked),
            emit_checkpoint_if_empty=generation_changed or reset_during_read,
        )
        self._state.update_session(tracked)

    def _append_provider_messages(
        self,
        session_id: str,
        new_entries: list[SourceEntry],
        provider: Any,
        _current_map: dict[str, dict[str, Any]] | None,
        window_id: str,
        new_messages: list[NewMessage],
        *,
        byte_checkpoint: int = 0,
        emit_checkpoint_if_empty: bool = False,
    ) -> None:
        if window_id and not identity_state.accepts_provider_observation(
            window_id, provider.capabilities.name
        ):
            return
        if not new_entries:
            if emit_checkpoint_if_empty:
                self._pending_batches.append(
                    TranscriptBatch(
                        session_id=session_id,
                        provider_name=provider.capabilities.name,
                        records=(),
                        byte_checkpoint=byte_checkpoint,
                    )
                )
            return
        if provider.capabilities.supports_task_tracking and window_id:
            provider.apply_task_entries(
                window_id, session_id, [entry.data for entry in new_entries]
            )
        provider_name = provider.capabilities.name
        previous = self._protocol_parser_states.get(session_id)
        if previous is None or previous[0] != provider_name:
            state = TranscriptParserState()
        else:
            state = previous[1]
        records, state = parse_records(
            new_entries,
            provider_name,
            state,
            deduplicate_updates=True,
        )
        self._protocol_parser_states[session_id] = (provider_name, state)
        if state.pending_tools:
            self._pending_tools[session_id] = dict(state.pending_tools)
        else:
            self._pending_tools.pop(session_id, None)
        fence = self._tool_delivery_fences.get(session_id)
        if state.pending_tools:
            if fence is None:
                self._tool_delivery_fences[session_id] = _ToolDeliveryFence(
                    byte_checkpoint
                )
            else:
                fence.release_checkpoint = None
        elif fence is not None and fence.release_checkpoint is None and new_entries:
            fence.release_checkpoint = byte_checkpoint
        batch = TranscriptBatch(
            session_id=session_id,
            provider_name=provider_name,
            records=records,
            byte_checkpoint=byte_checkpoint,
            tool_names=tuple(sorted(state.tool_names.items())),
            tool_summaries=tuple(sorted(state.tool_summaries.items())),
            tool_diffs=tuple(sorted(state.tool_diffs.items())),
        )
        if records or new_entries:
            self._pending_batches.append(batch)
        new_messages.extend(render_transcript_batch(batch))

    def drain_batches(self) -> tuple[TranscriptBatch, ...]:
        """Return normalized transcript batches accumulated since the last poll."""
        batches = tuple(self._pending_batches)
        self._pending_batches.clear()
        return batches

    async def _read_new_lines(
        self,
        session: TrackedSession,
        file_path: Path,
        window_id: str = "",
        provider: Any = None,
    ) -> list[SourceEntry]:
        """Read new lines from a session file using byte offset.

        ``provider`` may be passed by callers that already resolved it
        (avoids a second resolution per file per poll cycle).
        """
        if provider is None:
            provider = _resolve_provider_for_file(window_id, file_path)

        if provider.capabilities.name == "shell":
            session.parsed_offset = file_path.stat().st_size
            return []
        if not provider.capabilities.supports_incremental_read:
            return await self._read_whole_file(session, file_path, provider)

        new_entries: list[SourceEntry] = []
        try:
            async with aiofiles.open(file_path, "rb") as f:
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
                    if first_byte and first_byte != b"{":
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
                remaining = await f.read()
                relative_offset = 0
                for line in remaining.splitlines(keepends=True):
                    if not line.endswith(b"\n"):
                        break
                    try:
                        data = json.loads(line)
                    except ValueError, UnicodeDecodeError:
                        data = None
                    if isinstance(data, dict):
                        new_entries.append(SourceEntry(pos + relative_offset, data))
                        safe_offset = pos + relative_offset + len(line)
                    elif line.strip():
                        log_throttled(
                            logger,
                            f"partial-jsonl:{session.session_id}",
                            "Partial JSONL line in session %s, will retry next cycle",
                            session.session_id,
                        )
                        break
                    else:
                        safe_offset = pos + relative_offset + len(line)
                    relative_offset += len(line)

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
    ) -> list[SourceEntry]:
        """Read a whole-document transcript through the canonical source reader."""
        try:
            entries = await asyncio.to_thread(
                read_entries, file_path, provider.capabilities.name
            )
            session.parsed_offset = file_path.stat().st_size
            return entries
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
            if not w.cwd:
                continue
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
