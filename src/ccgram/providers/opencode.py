"""OpenCode CLI provider behind the AgentProvider protocol.

OpenCode (``opencode``) is a terminal AI coding agent that persists sessions
in a SQLite database (``~/.local/share/opencode/opencode.db`` on Linux,
``~/Library/Application Support/opencode/opencode.db`` on macOS).  Unlike the
JSONL-based providers (Claude, Codex, Gemini, Pi) its "transcript" is an
event-sourced table::

    event(aggregate_id = session_id, seq, type, data)

with a per-session monotonic ``seq`` (unique index on ``aggregate_id, seq``).

ccgram reads new events by ``seq`` cursor and mirrors the resulting normalized
entries into a per-session JSONL file under ``~/.ccgram/opencode/`` so the
existing file-based transcript machinery (byte offsets, mtime caching,
``/history`` reads) works unchanged.  The mirror file's mtime is bumped on
every successful sync so the whole-file poll path stays active.

Known event types (OpenCode 1.18.x):

  - ``session.created.1`` / ``session.updated.1``
  - ``message.updated.1``       ``data.info`` = message ``{id, role, time, agent, model}``
  - ``message.part.updated.1``  ``data.part``  = part ``{id, messageID, type, ...}``
  - ``message.removed.1``

Part types observed: ``text``, ``reasoning``, ``tool``, ``step-start``,
``step-finish``.  Parts are written in place while streaming, so each part is
emitted at most once, when it settles (text with ``time.end`` or no ``time``;
tool with a terminal ``state.status``).

Hooks: none in v1 (OpenCode has no config-file hook mechanism — hooks live in
its plugin system).  Status detection relies on the multiplexer's native agent
status (herdr) plus this provider's conservative permission-prompt heuristic.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import structlog

from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import (
    AgentMessage,
    DiscoveredCommand,
    MessageRole,
    ProviderCapabilities,
    RESUME_ID_RE,
    ResumableSession,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.tool_format import compact_arg, format_tool_line

logger = structlog.get_logger()

# ── OpenCode TUI built-in slash commands (bare names, no leading slash) ────
_OPENCODE_BUILTINS: dict[str, str] = {
    "compact": "Compact the current session",
    "connect": "Add a provider and API key",
    "export": "Export the session",
    "help": "Show help",
    "init": "Guided AGENTS.md setup",
    "models": "Pick a model",
    "new": "Start a new session",
    "redo": "Redo the last undone turn",
    "sessions": "Switch saved sessions",
    "share": "Share a session link",
    "undo": "Revert the last turn and files",
}

# Part state.status values that mean "still in flight".
_IN_FLIGHT_STATUSES = frozenset({"running", "pending", "in-progress", "streaming"})

# Terminal tool statuses that count as a finished (possibly failed) call.
_TERMINAL_TOOL_STATUSES = frozenset({"completed", "error", "cancelled", "failed"})

# Mirrored-part bitmask markers.
_EMITTED_USE = 1
_EMITTED_RESULT = 2

# Cap on mirrored sessions kept in the in-memory part-id cache.
_MAX_CACHED_SESSIONS = 512

_EVENT_BATCH_SIZE = 500

# Cap on normalized entries returned per read. After downtime a session can
# have thousands of pending parts; relaying them all in one poll floods
# Telegram and stalls the monitor's sequential delivery loop. Capping
# staggers catch-up over several polls instead.
_MAX_ENTRIES_PER_READ = 1000

# mtime bump: strictly forward so the whole-file poll gate (current_mtime >
# last_mtime) always advances, even on coarse-mtime filesystems.
_MTIME_BUMP_SECONDS = 1.0

_ARCHIVED_FILTER = "(time_archived IS NULL OR time_archived IN (0, ''))"


def resolve_opencode_db_path() -> Path:
    """Resolve the OpenCode SQLite database path with deterministic precedence.

    1. ``CCGRAM_OPENCODE_DB`` environment override
    2. Platform default candidates (first existing one wins)
    3. Linux default as final fallback
    """
    override = os.environ.get("CCGRAM_OPENCODE_DB", "").strip()
    if override:
        return Path(override).expanduser()
    home = Path.home()
    candidates = (
        home / ".local" / "share" / "opencode" / "opencode.db",
        home / "Library" / "Application Support" / "opencode" / "opencode.db",
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return candidates[0]


def resolve_opencode_mirror_root() -> Path:
    """Return the directory holding per-session JSONL mirror files."""
    override = os.environ.get("CCGRAM_OPENCODE_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ccgram" / "opencode"


def _mirror_path_for(session_id: str) -> Path:
    return resolve_opencode_mirror_root() / f"{session_id}.jsonl"


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a read-only SQLite connection (never locks OpenCode's writer)."""
    uri = f"file:{quote(str(db_path))}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _query_role_for_message(conn: sqlite3.Connection, message_id: str) -> str | None:
    row = conn.execute(
        "SELECT json_extract(data, '$.role') FROM message WHERE id = ?", (message_id,)
    ).fetchone()
    if not row or not row[0]:
        return None
    return str(row[0])


def _handle_message_event(raw: str, role_cache: dict[str, str]) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError, TypeError:
        return
    info = payload.get("info")
    if not isinstance(info, dict):
        return
    message_id = info.get("id")
    role = info.get("role")
    if message_id and isinstance(role, str):
        role_cache[message_id] = role


def _handle_part_event(
    conn: sqlite3.Connection,
    raw: str,
    session_id: str,  # noqa: ARG001 — part payload carries sessionID
    role_cache: dict[str, str],
    entries: list[dict[str, Any]],
) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError, TypeError:
        return
    part = payload.get("part")
    if not isinstance(part, dict):
        return
    message_id = part.get("messageID")
    if not isinstance(message_id, str):
        return
    role = _resolve_part_role(conn, message_id, role_cache)
    if role is None:
        return
    entries.append(
        {
            "type": "opencode_part",
            "message_id": message_id,
            "role": role,
            "part": part,
        }
    )


def _resolve_part_role(
    conn: sqlite3.Connection, message_id: str, role_cache: dict[str, str]
) -> str | None:
    role = role_cache.get(message_id)
    if role is None:
        role = _query_role_for_message(conn, message_id)
        if role:
            role_cache[message_id] = role
    if role not in ("user", "assistant"):
        return None
    return role


def _sync_opencode_events(
    session_id: str,
    since_seq: int,
    db_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch OpenCode events for *session_id* after *since_seq*.

    Returns ``(normalized_entries, last_seq)``.  On any database error the
    call fails closed: ``([], since_seq)`` so polling does not spin.
    """
    try:
        conn = _connect_readonly(db_path)
    except (sqlite3.Error, OSError) as exc:
        logger.debug("opencode: cannot open db %s: %s", db_path, exc)
        return [], since_seq

    entries: list[dict[str, Any]] = []
    role_cache: dict[str, str] = {}
    last_seq = since_seq
    try:
        cursor = since_seq
        while True:
            rows = conn.execute(
                "SELECT seq, type, data FROM event "
                "WHERE aggregate_id = ? AND seq > ? ORDER BY seq LIMIT ?",
                (session_id, cursor, _EVENT_BATCH_SIZE),
            ).fetchall()
            if not rows:
                break
            for seq, etype, raw in rows:
                last_seq = seq
                if etype == "message.updated.1":
                    _handle_message_event(raw, role_cache)
                elif etype == "message.part.updated.1":
                    _handle_part_event(conn, raw, session_id, role_cache, entries)
                # session.* / message.removed.* events carry no message content.
                if len(entries) >= _MAX_ENTRIES_PER_READ:
                    break
            if len(entries) >= _MAX_ENTRIES_PER_READ or len(rows) < _EVENT_BATCH_SIZE:
                break
            cursor = last_seq
    except (sqlite3.Error, OSError) as exc:
        logger.warning("opencode: event sync failed for %s: %s", session_id, exc)
        return [], since_seq
    finally:
        conn.close()
    return entries, last_seq


def _scan_mirror_part_ids(mirror: Path) -> dict[str, int]:
    """Re-read emitted-part markers from an existing mirror file (restart-safe)."""
    seen: dict[str, int] = {}
    if not mirror.exists():
        return seen
    try:
        with mirror.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                part = (
                    entry.get("part") if entry.get("type") == "opencode_part" else None
                )
                if isinstance(part, dict) and part.get("id"):
                    seen[str(part["id"])] = _EMITTED_USE | _EMITTED_RESULT
    except OSError:
        pass
    return seen


def _part_is_settled(part: dict[str, Any]) -> bool:
    ptype = part.get("type")
    if ptype == "text":
        ptime = part.get("time")
        # Streaming assistant text has time.start but no time.end yet.
        return not (isinstance(ptime, dict) and "end" not in ptime)
    if ptype == "tool":
        state = part.get("state")
        if isinstance(state, dict):
            status = str(state.get("status", "")).lower()
            return status not in _IN_FLIGHT_STATUSES
        return True
    return False


def _desired_markers(part: dict[str, Any]) -> int:
    """Bitmask of emissions this part should produce once settled."""
    if part.get("type") == "tool":
        desired = _EMITTED_USE
        if _part_is_settled(part):
            desired |= _EMITTED_RESULT
        return desired
    if _part_is_settled(part):
        return _EMITTED_USE
    return 0

    return 0


def _entry_line(seen: dict[str, int], entry: dict[str, Any]) -> str | None:
    """Return the JSON line to mirror for *entry*, or None when already emitted."""
    part = entry.get("part")
    if not isinstance(part, dict):
        return None
    part_id = part.get("id")
    if not isinstance(part_id, str):
        return None
    wanted = _desired_markers(part) & ~seen.get(part_id, 0)
    if not wanted:
        return None
    seen[part_id] = seen.get(part_id, 0) | wanted
    return json.dumps(entry, ensure_ascii=False)


def _extract_tool_result_summary(part: dict[str, Any], cap: int = 400) -> str:
    """Short human-readable summary of a settled tool part."""
    state = part.get("state")
    if not isinstance(state, dict):
        return ""
    output = state.get("output")
    if isinstance(output, str):
        return compact_arg(output, cap=cap)
    if isinstance(output, dict):
        try:
            return compact_arg(json.dumps(output, ensure_ascii=False), cap=cap)
        except TypeError, ValueError:
            return ""
    return ""


def _part_timestamp(part: dict[str, Any]) -> str | None:
    ptime = part.get("time")
    if isinstance(ptime, dict) and ptime.get("end"):
        return str(ptime["end"])
    return None


def _read_mirror_meta(mirror: Path) -> dict[str, str]:
    if not mirror.exists():
        return {}
    try:
        with mirror.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "session_meta":
                    return {
                        "session_id": str(entry.get("session_id") or ""),
                        "cwd": str(entry.get("cwd") or ""),
                    }
    except OSError:
        pass
    return {}


def _session_snapshot_row(
    session_id: str,
) -> tuple[str, str, float] | None:
    """Return ``(title, model, cost)`` for a session id, or None."""
    try:
        conn = _connect_readonly(resolve_opencode_db_path())
    except sqlite3.Error, OSError:
        return None
    try:
        row = conn.execute(
            "SELECT title, model, cost FROM session WHERE id = ?", (session_id,)
        ).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        conn.close()
    if not row:
        return None
    return str(row[0] or ""), str(row[1] or ""), float(row[2] or 0.0)


class OpenCodeProvider(JsonlProvider):
    """Provider for OpenCode CLI (``opencode``) with SQLite-backed transcripts."""

    _CAPS = ProviderCapabilities(
        name="opencode",
        launch_command="opencode",
        supports_hook=False,
        supports_resume=True,
        supports_resume_picker=True,
        supports_continue=True,
        supports_structured_transcript=True,
        supports_incremental_read=False,  # DB-backed whole-file JSON path
        uses_pane_title=False,
        uses_pyte_status_parsing=False,
        builtin_commands=tuple(sorted(_OPENCODE_BUILTINS.keys())),
        supports_user_command_discovery=False,
        supports_status_snapshot=True,
        has_yolo_confirmation=False,
        tui_picker_commands=frozenset(),
    )
    _BUILTINS = _OPENCODE_BUILTINS

    def __init__(self) -> None:
        self._emitted_parts: dict[str, dict[str, int]] = {}

    # ── Launch / resume ────────────────────────────────────────────────────

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        if resume_id:
            if not RESUME_ID_RE.match(resume_id):
                raise ValueError(f"Invalid resume_id: {resume_id!r}")
            return f"--session {resume_id}"
        if use_continue:
            return "--continue"
        return ""

    def discover_resumable_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[ResumableSession]:
        query = (
            "SELECT id, title, directory, time_updated FROM session "
            f"WHERE {_ARCHIVED_FILTER}"
        )
        params: list[Any] = []
        if cwd:
            query += " AND directory = ?"
            params.append(cwd)
        query += " ORDER BY time_updated DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        try:
            conn = _connect_readonly(resolve_opencode_db_path())
        except (sqlite3.Error, OSError) as exc:
            logger.debug("opencode: cannot open db for resume list: %s", exc)
            return []
        try:
            rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            logger.warning("opencode: resume list query failed: %s", exc)
            return []
        finally:
            conn.close()
        sessions: list[ResumableSession] = []
        for session_id, title, directory, time_updated in rows:
            summary = (title or session_id[:12]).strip() or session_id[:12]
            sessions.append(
                ResumableSession(
                    session_id=session_id,
                    summary=summary,
                    cwd=directory or "",
                    provider_name="opencode",
                    mtime=(time_updated or 0) / 1000.0,
                )
            )
        return sessions

    # ── Transcript discovery + reading ─────────────────────────────────────

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,  # noqa: ARG002 — DB lookup, no file age gate
    ) -> SessionStartEvent | None:
        resolved_cwd = self._resolve_cwd(cwd)
        if resolved_cwd is None:
            return None
        try:
            conn = _connect_readonly(resolve_opencode_db_path())
        except (sqlite3.Error, OSError) as exc:
            logger.debug("opencode: cannot open db for discovery: %s", exc)
            return None
        try:
            row = conn.execute(
                "SELECT id FROM session WHERE directory = ? "
                f"AND {_ARCHIVED_FILTER} "
                "ORDER BY time_updated DESC LIMIT 1",
                (resolved_cwd,),
            ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("opencode: discovery query failed: %s", exc)
            return None
        finally:
            conn.close()
        if not row:
            return None
        session_id = str(row[0])
        mirror = _mirror_path_for(session_id)
        try:
            resolve_opencode_mirror_root().mkdir(parents=True, exist_ok=True)
            if not mirror.exists():
                mirror.write_text(
                    json.dumps(
                        {
                            "type": "session_meta",
                            "session_id": session_id,
                            "cwd": resolved_cwd,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except OSError as exc:
            logger.warning("opencode: cannot create mirror %s: %s", mirror, exc)
            return None
        return SessionStartEvent(
            session_id=session_id,
            cwd=resolved_cwd,
            transcript_path=str(mirror),
            window_key=window_key,
        )

    def read_transcript_file(
        self, file_path: str, last_offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Sync new OpenCode events into the mirror and return them.

        *last_offset* is the per-session event ``seq`` cursor (not a byte
        offset); the mirror file is appended to and its mtime bumped so the
        whole-file poll path keeps running.
        """
        mirror = Path(file_path)
        session_id = mirror.stem
        entries, last_seq = _sync_opencode_events(
            session_id, last_offset, resolve_opencode_db_path()
        )
        if last_seq <= last_offset and not entries:
            # DB unreachable or no change — keep poll gate closed.
            return [], last_offset
        self._append_entries(mirror, session_id, entries)
        self._bump_mtime(mirror)
        return entries, last_seq

    def _append_entries(
        self, mirror: Path, session_id: str, entries: list[dict[str, Any]]
    ) -> None:
        if not entries:
            return
        seen = self._emitted_parts.get(session_id)
        if seen is None:
            if len(self._emitted_parts) >= _MAX_CACHED_SESSIONS:
                self._emitted_parts.clear()
            seen = _scan_mirror_part_ids(mirror)
            self._emitted_parts[session_id] = seen
        to_write: list[str] = []
        for entry in entries:
            line = _entry_line(seen, entry)
            if line is not None:
                to_write.append(line)
        if not to_write:
            return
        try:
            with mirror.open("a", encoding="utf-8") as fh:
                for line in to_write:
                    fh.write(line + "\n")
        except OSError as exc:
            logger.warning("opencode: mirror append failed %s: %s", mirror, exc)

    @staticmethod
    def _bump_mtime(path: Path) -> None:
        try:
            now = time.time() + _MTIME_BUMP_SECONDS
            os.utime(path, (now, now))
        except OSError:
            pass

    @staticmethod
    def _resolve_cwd(cwd: str | None) -> str | None:
        if not cwd:
            return None
        try:
            return str(Path(cwd).expanduser().resolve())
        except OSError:
            return None

    # ── Entry parsing ──────────────────────────────────────────────────────

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
        cwd: str | None = None,  # noqa: ARG002
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        messages: list[AgentMessage] = []
        pending = dict(pending_tools)
        for entry in entries:
            if entry.get("type") != "opencode_part":
                continue
            role = entry.get("role")
            if role not in ("user", "assistant"):
                continue
            part = entry.get("part")
            if not isinstance(part, dict):
                continue
            message = self._message_from_part(part, role, pending)
            if message is not None:
                messages.append(message)
        return messages, pending

    def _message_from_part(
        self,
        part: dict[str, Any],
        role: MessageRole,
        pending: dict[str, Any],
    ) -> AgentMessage | None:
        ptype = part.get("type")
        if ptype == "text":
            text = str(part.get("text") or "").strip()
            if not text:
                return None
            return AgentMessage(
                text=text,
                role=role,
                content_type="text",
                timestamp=_part_timestamp(part),
            )
        if ptype == "reasoning":
            return None  # hidden in v1 to keep transcripts quiet
        if ptype == "tool":
            return self._tool_message(part, pending)
        return None  # step-start / step-finish / unknown

    def _tool_message(
        self, part: dict[str, Any], pending: dict[str, Any]
    ) -> AgentMessage | None:
        tool = str(part.get("tool") or "tool")
        call_id = str(part.get("callID") or part.get("id") or "")
        state = part.get("state")
        status = str(state.get("status", "")).lower() if isinstance(state, dict) else ""
        timestamp = _part_timestamp(part)
        if status in _IN_FLIGHT_STATUSES:
            pending[call_id or tool] = tool
            return AgentMessage(
                text=format_tool_line(tool, ""),
                role="assistant",
                content_type="tool_use",
                tool_name=tool,
                tool_use_id=call_id or None,
                timestamp=timestamp,
            )
        if status in _TERMINAL_TOOL_STATUSES or not status:
            pending.pop(call_id, None)
            summary = _extract_tool_result_summary(part)
            if not summary:
                summary = f"✓ **{tool.lower()}**"
            return AgentMessage(
                text=summary,
                role="assistant",
                content_type="tool_result",
                tool_name=tool,
                tool_use_id=call_id or None,
                timestamp=timestamp,
            )
        return None

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        if entry.get("type") != "opencode_part":
            return False
        part = entry.get("part")
        return (
            entry.get("role") == "user"
            and isinstance(part, dict)
            and part.get("type") == "text"
        )

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        if entry.get("type") != "opencode_part":
            return None
        role = entry.get("role")
        if role not in ("user", "assistant"):
            return None
        part = entry.get("part")
        if not isinstance(part, dict) or part.get("type") != "text":
            return None
        text = str(part.get("text") or "").strip()
        if not text:
            return None
        return AgentMessage(
            text=text,
            role=role,
            content_type="text",
            timestamp=_part_timestamp(part),
        )

    # ── Status / snapshot ──────────────────────────────────────────────────

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",  # noqa: ARG002
    ) -> StatusUpdate | None:
        """Conservatively detect OpenCode TUI permission prompts.

        Full TUI chrome parsing is not attempted in v1; herdr's native agent
        status covers working/blocked, so only the permission banner that
        requires a keypress is surfaced here.
        """
        if not pane_text:
            return None
        lower = pane_text.lower()
        if "permission" in lower and (
            "requested" in lower or "required" in lower or "approve" in lower
        ):
            lines = [ln.strip() for ln in pane_text.splitlines() if ln.strip()]
            return StatusUpdate(
                raw_text="\n".join(lines[-15:]),
                display_label="PermissionPrompt",
                is_interactive=True,
                ui_type="PermissionPrompt",
            )
        return None

    def build_status_snapshot(
        self,
        transcript_path: str,
        *,
        display_name: str = "",
        session_id: str = "",
        cwd: str = "",
    ) -> str | None:
        """Render a /status snapshot from the session row (or mirror header)."""
        title = ""
        model = ""
        cost = 0.0
        if session_id:
            row = _session_snapshot_row(session_id)
            if row:
                title, model, cost = row
        if not title and not session_id:
            meta = _read_mirror_meta(Path(transcript_path))
            if meta:
                session_id = meta.get("session_id", session_id)
                cwd = meta.get("cwd", cwd)
        short_id = session_id[:8] if session_id else "unknown"
        lines = [
            f"🌀 [{display_name or title or 'OpenCode session'}] OpenCode session active."
        ]
        if cwd:
            lines.append(f"📁 `{cwd}`")
        lines.append(f"🆔 `{short_id}`")
        if model:
            lines.append(f"🤖 `{model}`")
        if cost:
            lines.append(f"💵 `${cost:.4f}`")
        return "\n".join(lines)

    def has_output_since(self, transcript_path: str, offset: int) -> bool:
        """True when the mirror file grew past *offset* (byte semantics).

        The /status probe records ``transcript_path.stat().st_size`` before
        sending; any mirrored OpenCode output after that grows the file.
        """
        try:
            return Path(transcript_path).stat().st_size > offset
        except OSError:
            return False

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:  # noqa: ARG002
        return [
            DiscoveredCommand(name=name, description=desc, source="builtin")
            for name, desc in _OPENCODE_BUILTINS.items()
        ]
