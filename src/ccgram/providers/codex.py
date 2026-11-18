"""Codex CLI provider — OpenAI's terminal agent behind AgentProvider protocol.

Codex CLI uses a similar tmux-based launch model but differs in hook mechanism
(no SessionStart hook) and resume syntax (``resume`` subcommand, not a flag).

Transcript format: JSONL with entries ``{timestamp, type, payload}``.
Entry types: ``session_meta``, ``response_item``, ``input_item``, ``event_msg``,
``turn_context``.

Modern Codex ``response_item`` payloads use typed shapes:
  - ``type=message`` with ``role`` + content blocks
  - ``type=function_call`` with ``name``, ``arguments``, ``call_id``
  - ``type=function_call_output`` with ``call_id``, ``output``
"""

import json
from pathlib import Path
from typing import Any

from ccgram.providers.codex_format import format_codex_interactive_prompt
from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import (
    RESUME_ID_RE,
    FirstRunPrompt,
    ProviderCapabilities,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.providers.codex_tui import parse_codex_menu
from ccgram.terminal_parser import extract_interactive_content

# Codex CLI known slash commands. Refreshed from
# github.com/openai/codex/blob/main/codex-rs/tui/tooltips.txt.
_CODEX_BUILTINS: dict[str, str] = {
    "/clear": "Clear conversation history",
    "/compact": "Summarize history and free up context",
    "/copy": "Copy last response as Markdown",
    "/config": "Configure Codex preferences",
    "/feedback": "Send logs to maintainers",
    "/fork": "Branch chat into a new thread",
    "/init": "Create AGENTS.md project guidance",
    "/keymap": "View and configure keyboard shortcuts",
    "/mcp": "List configured MCP tools",
    "/mention": "Attach files to conversation",
    "/model": "Switch model and reasoning effort",
    "/permissions": "Manage confirmation rules",
    "/plan": "Enter plan mode",
    "/rename": "Rename current thread",
    "/review": "Review current changes",
    "/side": "Start a side conversation in a temporary fork",
    "/skills": "List available skills",
    "/statusline": "Configure status line items",
    "/status": "Show model, approvals, token usage",
    "/theme": "Choose a syntax highlighting theme",
    "/title": "Configure terminal title items",
}

_TOOL_NAME_ALIASES: dict[str, str] = {
    "request_user_input": "AskUserQuestion",
    "apply_patch": "Edit",
}

# Minimum line count to trigger stats + expandable quote for tool results.
_TOOL_RESULT_QUOTE_THRESHOLD = 3


# Transcripts older than this are considered stale and skipped during discovery.
# Prevents matching a finished session when a new Codex window opens in the same cwd.
_TRANSCRIPT_MAX_AGE_SECS = 120.0


def _collect_codex_sessions(sessions_dir: Path) -> list[tuple[float, Path]]:
    """Collect all JSONL files under sessions_dir, sorted newest-first by mtime."""
    result: list[tuple[float, Path]] = []
    for fpath in sessions_dir.rglob("*.jsonl"):
        try:
            result.append((fpath.stat().st_mtime, fpath))
        except OSError:
            continue
    result.sort(reverse=True)
    return result


def _read_codex_session_meta(fpath: Path) -> dict[str, Any] | None:
    """Read the session_meta payload from the first line of a Codex JSONL file."""
    try:
        with open(fpath, encoding="utf-8") as f:
            first_line = f.readline()
    except OSError:
        return None
    if not first_line:
        return None
    try:
        data = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    if data.get("type") != "session_meta":
        return None
    payload = data.get("payload")
    return payload if isinstance(payload, dict) else None


def _is_primary_codex_session(meta: dict[str, Any]) -> bool:
    """Whether session metadata represents a top-level Codex CLI session.

    Hookless discovery should bind a tmux window to the main user-visible
    conversation, not transient guardian/subagent transcripts created for
    approvals or delegated work, nor non-interactive ``codex exec`` runs that
    happen to share a cwd with a real interactive session. All of those write
    JSONL transcripts under the same ``~/.codex/sessions/`` tree and can be
    newer than the main transcript, so they must be filtered out here.

    Filtering precedence:

    * ``originator == "codex_exec"`` -> non-interactive ``codex exec``; never
      bind regardless of ``source``. Some ``codex exec`` runs report
      ``source = "exec"`` and others report no ``source`` field at all, so the
      originator check must happen before the ``source`` branches below.
    * Non-dict ``source`` (string such as ``"cli"`` / ``"exec"``, or missing) ->
      treated as a primary session. This covers ``codex_cli_rs`` interactive
      sessions and Codex Desktop variants that legitimately set
      ``source = "exec"`` but whose ``originator`` is not ``codex_exec``.
    * Dict ``source`` -> filter out anything carrying a ``subagent`` key
      (guardian / delegated work).
    """
    originator = meta.get("originator")
    if isinstance(originator, str) and originator == "codex_exec":
        return False
    source = meta.get("source")
    if not isinstance(source, dict):
        return True
    return "subagent" not in source


class CodexProvider(JsonlProvider):
    """AgentProvider implementation for OpenAI Codex CLI."""

    _CAPS = ProviderCapabilities(
        name="codex",
        launch_command="codex",
        supports_hook=True,
        hook_install_managed_by_ccgram=True,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        builtin_commands=tuple(_CODEX_BUILTINS.keys()),
        supports_user_command_discovery=True,
        supports_status_snapshot=True,
        # Command menus observed in the installed Codex CLI 0.164.0.
        tui_picker_commands=frozenset(
            {
                "config",
                "keymap",
                "model",
                "permissions",
                "review",
                "skills",
                "statusline",
                "theme",
                "title",
            }
        ),
        # Hooks review screen shown when ccgram's hooks entry is new for this
        # project: "1. review hooks / 2. trust all and continue / 3. continue
        # without trust". Option 2 is one Down from the default selection.
        # The project trust entry in config.toml is seeded first, which is
        # what removes this screen in the first place.
        first_run=FirstRunPrompt(
            needles=("trust all and continue",),
            keys=("Down", "Enter"),
            settled=("for shortcuts",),
        ),
    )

    _BUILTINS = _CODEX_BUILTINS

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        """Build Codex CLI args for launching or resuming a session.

        Resume uses ``resume <id>`` subcommand syntax.
        Continue uses ``resume --last`` to pick up the most recent session.
        """
        if resume_id:
            if not RESUME_ID_RE.match(resume_id):
                raise ValueError(f"Invalid resume_id: {resume_id!r}")
            return f"resume {resume_id}"
        if use_continue:
            return "resume --last"
        return ""

    # ── Codex-specific transcript parsing ─────────────────────────────

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",  # noqa: ARG002
    ) -> StatusUpdate | None:
        """Parse Codex command menus before generic interactive prompts."""
        codex_menu = parse_codex_menu(pane_text)
        if codex_menu:
            return codex_menu

        interactive = extract_interactive_content(pane_text)
        if interactive:
            formatted = format_codex_interactive_prompt(
                interactive.content, interactive.name
            )
            return StatusUpdate(
                raw_text=formatted,
                display_label=interactive.name,
                is_interactive=True,
                ui_type=interactive.name,
                ui_advisory=interactive.advisory,
            )
        return None

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Scan ~/.codex/sessions/ for the most recent transcript matching cwd.

        Codex transcript path: ~/.codex/sessions/YYYY/MM/DD/<name>-<ts>-<uuid>.jsonl
        First line: {"type": "session_meta", "payload": {"id": "<uuid>", "cwd": "..."}}

        Args:
            max_age: Maximum transcript age in seconds. ``None`` uses the
                default ``_TRANSCRIPT_MAX_AGE_SECS`` (120s). Pass ``0`` or
                negative to disable the age check entirely.
        """
        sessions_dir = Path.home() / ".codex" / "sessions"
        if not sessions_dir.is_dir():
            return None

        # Lazy: only used inside the polling branch below
        import time

        age_limit = _TRANSCRIPT_MAX_AGE_SECS if max_age is None else max_age

        jsonl_files = _collect_codex_sessions(sessions_dir)
        now = time.time()
        resolved_cwd = str(Path(cwd).resolve())
        for mtime, fpath in jsonl_files[:20]:
            if age_limit > 0 and now - mtime > age_limit:
                break  # sorted newest-first; remaining are all older
            meta = _read_codex_session_meta(fpath)
            if not meta:
                continue
            if not _is_primary_codex_session(meta):
                continue
            file_cwd = meta.get("cwd", "")
            if file_cwd and str(Path(file_cwd).resolve()) == resolved_cwd:
                session_id = meta.get("id", "")
                if session_id:
                    return SessionStartEvent(
                        session_id=session_id,
                        cwd=file_cwd,
                        transcript_path=str(fpath),
                        window_key=window_key,
                    )
        return None

    # ── Status snapshot (Codex-specific) ─────────────────────────────

    def build_status_snapshot(
        self,
        transcript_path: str,
        *,
        display_name: str = "",
        session_id: str = "",
        cwd: str = "",
    ) -> str | None:
        # Lazy: codex_status pulls transcript parsers; defer until polling tick
        from ccgram.providers.codex_status import build_codex_status_snapshot

        return build_codex_status_snapshot(
            transcript_path,
            display_name=display_name,
            session_id=session_id,
            cwd=cwd,
        )

    def has_output_since(self, transcript_path: str, offset: int) -> bool:
        # Lazy: codex_status pulls transcript parsers; defer until polling tick
        from ccgram.providers.codex_status import has_codex_assistant_output_since

        return has_codex_assistant_output_since(transcript_path, offset)
