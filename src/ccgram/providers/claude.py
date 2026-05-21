"""Claude Code provider — wraps existing modules behind AgentProvider protocol.

Delegates to hook.py, transcript_parser.py, terminal_parser.py, and
cc_commands.py without changing any behavior. This is a thin adapter layer
that translates between the provider protocol and existing module APIs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

import structlog

from ccgram.cc_commands import CC_BUILTINS
from ccgram.providers.base import UUID_RE
from ccgram.providers.base import (
    AgentMessage,
    ContentType,
    DiscoveredCommand,
    MessageRole,
    ProviderCapabilities,
    SessionStartEvent,
    StatusUpdate,
)

from ccgram.terminal_parser import (
    extract_bash_output,
    extract_interactive_content,
    find_chrome_boundary,
    format_status_display,
    parse_status_block,
)
from ccgram.transcript_parser import TranscriptParser

_log = structlog.get_logger(__name__)

_MODE_MARKERS: tuple[str, ...] = ("\u23f5\u23f5", "\u23f8")
_MODE_HINTS: tuple[str, ...] = (
    "auto mode",
    "auto-accept",
    "accept edits",
    "plan mode",
    "bypass permissions",
    "yolo",
    "auto-approve",
)
_MODE_SHORT_LABELS: tuple[tuple[str, str], ...] = (
    ("accept edits", "Edit"),
    ("auto-accept", "Edit"),
    ("plan", "Plan"),
    ("auto mode", "Auto"),
    ("bypass", "YOLO"),
    ("yolo", "YOLO"),
    ("auto-approve", "YOLO"),
)
_LINE_LIMIT = 80


def _find_mode_line(capture: str) -> str | None:
    lines = capture.splitlines()
    boundary = find_chrome_boundary(lines)
    chrome_lines = lines[boundary + 1 :] if boundary is not None else lines[-20:]
    for line in reversed(chrome_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker in stripped for marker in _MODE_MARKERS):
            return stripped[:_LINE_LIMIT]
    for line in reversed(lines[-25:]):
        stripped = line.strip()
        if not stripped:
            continue
        if any(hint in stripped.lower() for hint in _MODE_HINTS):
            return stripped[:_LINE_LIMIT]
    return None


def _mode_short_label(mode_line: str) -> str | None:
    lower = mode_line.lower()
    for pattern, label in _MODE_SHORT_LABELS:
        if pattern in lower:
            return label
    return None


class ClaudeProvider:
    """AgentProvider implementation for Claude Code CLI."""

    _CAPS = ProviderCapabilities(
        name="claude",
        launch_command="claude",
        supports_hook=True,
        supports_hook_events=True,
        hook_install_managed_by_ccgram=True,
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
        has_yolo_confirmation=True,
        supports_task_tracking=True,
        uses_pyte_status_parsing=True,
        # Commands documented at code.claude.com/docs/en/commands as
        # opening a modal/interactive overlay the user navigates with
        # arrow keys, Enter, and Esc. Sourced from the v2.1.x docs page.
        tui_picker_commands=frozenset(
            {
                "agents",
                "copy",
                "diff",
                "effort",
                "model",
                "permissions",
                "release-notes",
                "rewind",
                "settings",
                "skills",
                "theme",
                "tui",
            }
        ),
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
        cwd: str,  # noqa: ARG002 — protocol signature
        window_key: str,  # noqa: ARG002 — protocol signature
        *,
        max_age: float | None = None,  # noqa: ARG002 — protocol signature
    ) -> SessionStartEvent | None:
        return None  # Claude uses hooks, not transcript discovery

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

    async def scrape_current_mode(
        self,
        window_id: str,
        *,
        capture_fn: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> str | None:
        """Return the short mode label visible in the Claude Code status line.

        ``capture_fn`` is optional and injectable for tests — defaults to
        ``tmux_manager.capture_pane`` so production callers need no changes.
        """
        if capture_fn is not None:
            _fn: Callable[[str], Awaitable[str | None]] = capture_fn
        else:
            # Lazy: tmux_manager imports providers; lazy fallback avoids the
            # cycle when tests inject capture_fn directly.
            from ccgram.tmux_manager import tmux_manager

            _fn = tmux_manager.capture_pane
        try:
            capture = await _fn(window_id)
        except OSError as exc:
            _log.warning("Mode scrape: capture_pane failed %s (%s)", window_id, exc)
            return None
        if not capture:
            return None
        mode_line = _find_mode_line(capture)
        return _mode_short_label(mode_line) if mode_line else None

    async def seed_task_state(
        self,
        window_id: str,
        session_id: str,
        transcript_path: str,
    ) -> None:
        """Seed Claude task-tracking state by reading the full transcript once."""
        # Lazy: aiofiles is heavy; defer until the JSONL reader actually opens a file
        import aiofiles

        # Lazy: claude_task_state ↔ claude provider cycle
        from ccgram.claude_task_state import claude_task_state

        entries: list[dict] = []
        try:
            async with aiofiles.open(transcript_path, "r", encoding="utf-8") as f:
                async for line in f:
                    data = self.parse_transcript_line(line)
                    if data:
                        entries.append(data)
        except OSError:
            _log.exception("seed_task_state: error reading %s", transcript_path)
            return
        claude_task_state.rebuild_from_entries(window_id, session_id, entries)

    def apply_task_entries(
        self,
        window_id: str,
        session_id: str,
        entries: list[dict],
    ) -> None:
        """Apply parsed transcript entries to Claude task-tracking state."""
        # Lazy: claude_task_state ↔ claude provider cycle
        from ccgram.claude_task_state import claude_task_state

        claude_task_state.apply_entries(window_id, session_id, entries)
