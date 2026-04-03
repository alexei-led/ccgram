"""Claude Code provider — wraps existing modules behind AgentProvider protocol.

Delegates to hook.py, transcript_parser.py, terminal_parser.py, and
cc_commands.py without changing any behavior. This is a thin adapter layer
that translates between the provider protocol and existing module APIs.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, cast

from ccgram.cc_commands import CC_BUILTINS
from ccgram.hook import UUID_RE
from ccgram.providers.base import (
    AgentMessage,
    ContentType,
    DiscoveredCommand,
    MessageRole,
    ProviderCapabilities,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.utils import read_session_metadata_from_jsonl

from ccgram.terminal_parser import (
    extract_bash_output,
    extract_interactive_content,
    format_status_display,
    parse_status_block,
)
from ccgram.transcript_parser import TranscriptParser

_TRANSCRIPT_MAX_AGE_SECS = 120.0
_TAIL_READ_BYTES = 8192
_TAIL_SCAN_LINES = 8
_SESSION_END_COMMANDS = frozenset({"/clear", "/compact", "/exit"})


def _resolve_cwd(cwd: str) -> str:
    try:
        return str(Path(cwd).resolve())
    except OSError:
        return cwd


def _claude_project_dir(cwd: str) -> Path:
    resolved = _resolve_cwd(cwd)
    return Path.home() / ".claude" / "projects" / resolved.replace("/", "-")


def _read_tail_lines(
    file_path: str | Path,
    *,
    read_bytes: int = _TAIL_READ_BYTES,
    max_lines: int = _TAIL_SCAN_LINES,
) -> list[str]:
    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - read_bytes)
            f.seek(start)
            chunk = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []

    lines = chunk.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    return lines[-max_lines:]


def _transcript_indicates_session_end(file_path: str | Path) -> bool:
    """Return True when the transcript tail contains a session-ending command."""
    for line in reversed(_read_tail_lines(file_path)):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "user":
            continue
        message = entry.get("message", {})
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if any(
            f"<command-name>{command}</command-name>" in content
            for command in _SESSION_END_COMMANDS
        ):
            return True
    return False


class ClaudeProvider:
    """AgentProvider implementation for Claude Code CLI."""

    _CAPS = ProviderCapabilities(
        name="claude",
        launch_command="claude",
        supports_hook=True,
        supports_hook_events=True,
        hook_event_types=(
            "Notification",
            "Stop",
            "StopFailure",
            "SessionEnd",
            "SubagentStart",
            "SubagentStop",
            "TeammateIdle",
            "TaskCompleted",
        ),
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        transcript_format="jsonl",
        builtin_commands=tuple(CC_BUILTINS.keys()),
        supports_user_command_discovery=True,
    )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._CAPS

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        """Build Claude Code CLI args string for launching or resuming a session."""
        if resume_id:
            if not UUID_RE.match(resume_id):
                raise ValueError(f"Invalid resume_id: {resume_id!r}")
            return f"--resume {resume_id}"
        if use_continue:
            return "--continue"
        return ""

    def parse_hook_payload(self, payload: dict[str, Any]) -> SessionStartEvent | None:
        """Parse a Claude Code SessionStart hook payload.

        Validates session_id (UUID format), cwd (absolute path), and
        rejects payloads missing required fields.
        """
        session_id = payload.get("session_id", "")
        cwd = payload.get("cwd", "")
        transcript_path = payload.get("transcript_path", "")
        window_key = payload.get("window_key", "")

        if not session_id or not cwd:
            return None

        if not UUID_RE.match(session_id):
            return None

        if not os.path.isabs(cwd):
            return None

        return SessionStartEvent(
            session_id=session_id,
            cwd=cwd,
            transcript_path=transcript_path,
            window_key=window_key,
        )

    def read_transcript_file(
        self, file_path: str, last_offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Claude uses incremental JSONL reading, not whole-file."""
        msg = "ClaudeProvider uses incremental JSONL reading, not whole-file"
        raise NotImplementedError(msg)

    def parse_transcript_line(self, line: str) -> dict[str, Any] | None:
        """Delegate to TranscriptParser.parse_line."""
        return TranscriptParser.parse_line(line)

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
        cwd: str | None = None,
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        """Parse JSONL entries via TranscriptParser and wrap as AgentMessages."""
        parsed, remaining = TranscriptParser.parse_entries(
            entries, pending_tools, cwd=cwd
        )

        messages = [
            AgentMessage(
                text=e.text,
                role=cast(MessageRole, e.role),
                content_type=cast(ContentType, e.content_type),
                tool_use_id=e.tool_use_id,
                tool_name=e.tool_name,
                timestamp=e.timestamp,
            )
            for e in parsed
        ]

        return messages, remaining

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",  # noqa: ARG002
    ) -> StatusUpdate | None:
        """Parse pane text; interactive UI takes precedence over status line."""
        interactive = extract_interactive_content(pane_text)
        if interactive:
            return StatusUpdate(
                raw_text=interactive.content,
                display_label=interactive.name,
                is_interactive=True,
                ui_type=interactive.name,
            )

        raw_status = parse_status_block(pane_text)
        if raw_status:
            headline = raw_status.split("\n", 1)[0]
            return StatusUpdate(
                raw_text=raw_status,
                display_label=format_status_display(headline),
            )

        return None

    def extract_bash_output(self, pane_text: str, command: str) -> str | None:
        return extract_bash_output(pane_text, command)

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        return TranscriptParser.is_user_message(entry)

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        """Parse a single transcript entry for history display."""
        raw_role = entry.get("type", "assistant")
        if raw_role not in ("user", "assistant"):
            return None
        parsed = TranscriptParser.parse_message(entry)
        if parsed is None or not parsed.text:
            return None
        role = cast(MessageRole, raw_role)
        # "user"/"assistant" message_type maps to "text"; others pass through.
        raw_ct = (
            "text"
            if parsed.message_type in ("user", "assistant")
            else parsed.message_type
        )
        content_type = cast(ContentType, raw_ct)
        return AgentMessage(
            text=parsed.text,
            role=role,
            content_type=content_type,
            tool_name=parsed.tool_name,
            timestamp=TranscriptParser.get_timestamp(entry),
        )

    def requires_pane_title_for_detection(
        self,
        pane_current_command: str,  # noqa: ARG002 — protocol signature
    ) -> bool:
        return False

    def detect_from_pane_title(
        self,
        pane_current_command: str,  # noqa: ARG002 — protocol signature
        pane_title: str,  # noqa: ARG002 — protocol signature
    ) -> bool:
        return False

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Discover the newest Claude transcript for a cwd.

        Used only as a guarded recovery path when hook delivery was missed.
        """
        project_dir = _claude_project_dir(cwd)
        if not project_dir.is_dir():
            return None

        age_limit = _TRANSCRIPT_MAX_AGE_SECS if max_age is None else max_age
        now = time.time()
        resolved_cwd = _resolve_cwd(cwd)
        transcripts: list[tuple[float, Path]] = []
        try:
            for fpath in project_dir.glob("*.jsonl"):
                try:
                    transcripts.append((fpath.stat().st_mtime, fpath))
                except OSError:
                    continue
        except OSError:
            return None

        transcripts.sort(reverse=True)
        for mtime, fpath in transcripts[:20]:
            if age_limit > 0 and now - mtime > age_limit:
                break
            session_id = fpath.stem
            if not UUID_RE.match(session_id):
                continue
            file_cwd, _ = read_session_metadata_from_jsonl(fpath)
            if _resolve_cwd(file_cwd) != resolved_cwd:
                continue
            return SessionStartEvent(
                session_id=session_id,
                cwd=file_cwd or cwd,
                transcript_path=str(fpath),
                window_key=window_key,
            )
        return None

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        _ = base_dir
        return [
            DiscoveredCommand(
                name=name,
                description=desc,
                source="builtin",
            )
            for name, desc in CC_BUILTINS.items()
        ]

    def build_status_snapshot(
        self,
        transcript_path: str,
        *,
        display_name: str = "",
        session_id: str = "",
        cwd: str = "",
    ) -> str | None:
        _ = transcript_path, display_name, session_id, cwd
        return None

    def has_output_since(self, transcript_path: str, offset: int) -> bool:
        _ = transcript_path, offset
        return False
