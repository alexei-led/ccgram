"""Claude Code provider — terminal and lifecycle adapter behind AgentProvider.

Transcript interpretation is shared in ``session_protocol.messages``.
This provider handles Claude-specific launch, hook, status, and command behavior.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog

from ccgram.cc_commands import CC_BUILTINS
from ccgram.providers._resume import discover_claude_sessions
from ccgram.providers.base import UUID_RE
from ccgram.providers.base import (
    DiscoveredCommand,
    FirstRunPrompt,
    ProviderCapabilities,
    ResumableSession,
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


def _claude_projects_path() -> Path:
    """Resolve Claude's project store only when session discovery is requested."""
    # Lazy: Config requires runtime environment; session discovery is opt-in.
    from ccgram.config import config

    return config.claude_projects_path


class ClaudeProvider:
    """AgentProvider implementation for Claude Code CLI."""

    _CAPS = ProviderCapabilities(
        name="claude",
        launch_command="claude",
        supports_hook=True,
        hook_install_managed_by_ccgram=True,
        supports_resume=True,
        supports_resume_picker=True,
        supports_continue=True,
        supports_structured_transcript=True,
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
        # Folder trust dialog: "Yes, proceed" is the default selection, so
        # Enter clears it. Seeded in ~/.claude.json first; this covers a
        # config path ccgram could not write.
        first_run=FirstRunPrompt(
            needles=("do you trust the files in this folder",),
            keys=("Enter",),
            settled=("bypass permissions", "for shortcuts"),
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

    def discover_resumable_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[ResumableSession]:
        """Discover Claude sessions from the configured projects directory."""
        return discover_claude_sessions(
            _claude_projects_path(),
            cwd=cwd,
            limit=limit,
        )

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
                ui_advisory=interactive.advisory,
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
            from ccgram.multiplexer import multiplexer as tmux_manager

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
        # Lazy: claude_task_state ↔ claude provider cycle
        from ccgram.claude_task_state import claude_task_state

        # Lazy: keep transcript parsing out of the Claude provider import cycle.
        from ccgram.session_protocol.messages import read_entries

        try:
            entries = await asyncio.to_thread(
                read_entries, Path(transcript_path), "claude"
            )
        except OSError:
            _log.exception("seed_task_state: error reading %s", transcript_path)
            return
        claude_task_state.rebuild_from_entries(
            window_id, session_id, [entry.data for entry in entries]
        )

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
