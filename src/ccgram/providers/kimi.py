"""Kimi Code provider — Moonshot AI's ``kimi`` CLI behind AgentProvider.

Kimi Code is a native (non-Node) binary, so the multiplexer reports ``kimi``
as ``pane_current_command`` directly — no runtime-shim detection needed.

State lives under ``~/.kimi-code``::

    session_index.jsonl                     {sessionId, sessionDir, workDir}
    sessions/<workspace>/<session>/state.json
    sessions/<workspace>/<session>/agents/main/wire.jsonl   ← transcript

Kimi has no Claude-style lifecycle hooks, so it is a hookless provider like
Codex and Gemini: sessions are discovered by scanning ``session_index.jsonl``
for the newest entry whose ``workDir`` matches the window's directory, and the
transcript is the source of truth for relayed messages.

Resume strategy: ``-S <session_id>``.  Passing an explicit id resumes directly
without opening kimi's interactive picker, and kimi *appends* to the existing
``wire.jsonl``, so byte offsets stay valid across a resume.

Terminal chrome (kimi 0.31):
  - Status bar (last two rows): ``[<modes> ]<model> thinking: <effort>  …/cwd``
    and ``context: N% (x/256k)``.  Modes are ``yolo``, ``auto`` and ``plan``.
  - Busy line above the input box: ``<spinner>[ <label>] · Tip: …`` where the
    spinner is a moon phase (thinking) or braille frame (streaming).
  - Interactive prompts and pickers are fenced between full-width ``─`` rules.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import structlog

from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import (
    AgentMessage,
    DiscoveredCommand,
    ProviderCapabilities,
    RESUME_ID_RE,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.providers.kimi_format import (
    LOOP_EVENT,
    TURN_PROMPT,
    Pending,
    normalize_pending,
    parse_loop_event,
    parse_turn_prompt,
)

logger = structlog.get_logger()

# Cap transcript age when adopting an existing pane — guards against picking up
# an unrelated historical session for the same cwd.
_STALE_TRANSCRIPT_MAX_AGE_SECS = 120.0

# How many recent index entries to inspect when searching for a cwd match.
_DISCOVERY_SCAN_LIMIT = 50

_SHORT_SESSION_ID_LEN = 8
_SHORT_SESSION_ID_THRESHOLD = 10


def kimi_home() -> Path:
    """Return kimi's state directory (``~/.kimi-code``)."""
    return Path.home() / ".kimi-code"


def session_index_path() -> Path:
    """Return the path of kimi's append-only session index."""
    return kimi_home() / "session_index.jsonl"


def transcript_for_session_dir(session_dir: str) -> Path:
    """Return the main agent's wire log inside a kimi session directory."""
    return Path(session_dir) / "agents" / "main" / "wire.jsonl"


# ── Terminal chrome ──────────────────────────────────────────────────────

# Full-width horizontal rule fencing an interactive prompt or picker.
_RULE_RE = re.compile(r"^\s*─{20,}\s*$")

# Footer/legend hints kimi renders inside an interactive region.  Matching any
# one of these is what separates a real prompt from decorative rules.
_INTERACTIVE_HINTS = (
    re.compile(r"↑/?↓\s"),  # "↑/↓ select" / "↑↓ navigate"
    re.compile(r"(?i)\bEnter\s+select\b"),
    re.compile(r"(?i)\bEsc\s+cancel\b"),
    re.compile(r"↵\s+confirm"),  # "↵ confirm"
    re.compile(r"(?i)\bchoose\b"),
)

# Selection cursor kimi paints on the highlighted row.
_CURSOR_RE = re.compile(r"^\s*[▶❯]\s+\S")

# A numbered option row ("  2. Approve for this session").
_NUMBERED_OPTION_RE = re.compile(r"^\s*(?:[▶❯]\s+)?\d+\.\s+\S")

_MIN_NUMBERED_OPTIONS = 2

# A fence needs an opening and a closing rule, with at least one content row
# between them, before it can be read as an interactive block.
_MIN_FENCE_RULES = 2
_MIN_FENCE_SPAN = 2

# Question header on an approval prompt ("▶ Run this command?").
_PROMPT_HEADER_RE = re.compile(r"^\s*[▶❯]\s+(.+\?)\s*$")

# Approval prompts carry these; other pickers (model, theme, …) do not.
_PERMISSION_HINT_RE = re.compile(
    r"(?i)\b(approve|reject|allow|deny|permission|proceed)\b"
)

# Busy spinner frames: moon phases U+1F311–U+1F318 (thinking) and braille
# U+2800–U+28FF (streaming), followed by an optional label then " · <tip>".
_SPINNER_RE = re.compile(
    r"^\s*(?:[\U0001f311-\U0001f318]|[⠀-⣿])\s*(?P<label>[^·]*?)\s*·\s"
)

# Status bar: "[<modes> ]<model> thinking: <effort>  …/cwd".
_STATUS_BAR_RE = re.compile(
    r"^\s*(?P<modes>(?:yolo|auto|plan|manual)(?:\s+(?:yolo|auto|plan|manual))*\s+)?\S+ thinking:\s"
)

_MODE_LABELS = {
    "yolo": "YOLO",
    "auto": "Auto",
    "plan": "Plan",
    "manual": "Manual",
}

# How far up from the bottom to look for the status bar / spinner.
_BOTTOM_SCAN_LINES = 6


def _extract_fenced_block(lines: list[str]) -> list[str] | None:
    """Return the content between the last pair of full-width ``─`` rules.

    Conservative against torn captures: a dangling rule below the closing one
    (a block mid-render) yields None so the next poll reads a complete frame.
    """
    rule_rows = [i for i, line in enumerate(lines) if _RULE_RE.match(line)]
    if len(rule_rows) < _MIN_FENCE_RULES:
        return None
    close_idx = rule_rows[-1]
    open_idx = rule_rows[-2]
    if close_idx - open_idx < _MIN_FENCE_SPAN:
        return None
    inner = [line.rstrip() for line in lines[open_idx + 1 : close_idx]]
    while inner and not inner[0].strip():
        inner.pop(0)
    while inner and not inner[-1].strip():
        inner.pop()
    return inner or None


def _block_is_interactive(block: list[str]) -> bool:
    """True when a fenced block carries an interactive affordance.

    A footer hint or a selection cursor is sufficient on its own; a bare
    numbered list needs two entries so prose containing "1." is not misread.
    """
    numbered = 0
    for line in block:
        if any(hint.search(line) for hint in _INTERACTIVE_HINTS):
            return True
        if _CURSOR_RE.match(line):
            return True
        if _NUMBERED_OPTION_RE.match(line):
            numbered += 1
            if numbered >= _MIN_NUMBERED_OPTIONS:
                return True
    return False


def _block_title(block: list[str]) -> tuple[str, bool]:
    """Return ``(title, is_question)`` for a fenced interactive block.

    Approval prompts lead with a cursor-marked question (``▶ Run this
    command?``); menu pickers lead with a plain heading (``Select permission
    mode``).  The flag lets the caller tell the two apart — the option text of
    a picker can mention "approve" without the block being an approval.
    """
    for line in block:
        match = _PROMPT_HEADER_RE.match(line)
        if match:
            return match.group(1).strip(), True

    for line in block:
        stripped = line.strip()
        if not stripped:
            continue
        if any(hint.search(stripped) for hint in _INTERACTIVE_HINTS):
            continue
        if _CURSOR_RE.match(line) or _NUMBERED_OPTION_RE.match(line):
            break
        return stripped, False
    return "", False


def _nonblank_tail(pane_text: str) -> list[str]:
    """Split pane text into lines with trailing blank rows dropped."""
    lines = pane_text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _spinner_label(lines: list[str]) -> str | None:
    """Return the busy-line label when kimi is mid-turn, else None.

    The spinner sits just above the input box, so only the tail of the pane is
    scanned — a spinner glyph inside scrollback must not read as "busy".
    """
    for line in reversed(lines[-_BOTTOM_SCAN_LINES:]):
        match = _SPINNER_RE.match(line)
        if match:
            return match.group("label").strip()
    return None


def parse_status_modes(pane_text: str) -> str | None:
    """Extract the mode badge (``YOLO``, ``Plan``, …) from kimi's status bar."""
    for line in reversed(_nonblank_tail(pane_text)[-_BOTTOM_SCAN_LINES:]):
        match = _STATUS_BAR_RE.match(line)
        if not match:
            continue
        modes = (match.group("modes") or "").split()
        labels = [_MODE_LABELS[m] for m in modes if m in _MODE_LABELS]
        return " ".join(labels) if labels else None
    return None


# ── Commands ─────────────────────────────────────────────────────────────

# Kimi built-ins worth surfacing in Telegram (/commands listing + autocomplete).
# Modal TUI flows are still forwarded — they are driven with the inline toolbar.
#
# /sessions is excluded: it collides with ccgram's own session picker.
# /clear is accepted as a hidden compatibility alias in forward.py; kimi's
# documented reset command is /new.
_KIMI_TELEGRAM_BUILTINS: dict[str, str] = {
    "/add-dir": "Add or list an additional workspace directory",
    "/auto": "Toggle Auto mode — fully autonomous, never asks",
    "/btw": "Ask a forked side agent a question",
    "/compact": "Compact the conversation context",
    "/copy": "Copy the last assistant message to the clipboard",
    "/editor": "Set the external editor for Ctrl-G",
    "/effort": "Switch thinking effort",
    "/experiments": "Manage experimental features",
    "/export-md": "Export current session as a Markdown file",
    "/fork": "Fork the current session",
    "/goal": "Start or manage an autonomous goal",
    "/help": "Show available commands and shortcuts",
    "/init": "Analyze the codebase and generate AGENTS.md",
    "/login": "Select a platform and authenticate",
    "/logout": "Log out of a configured provider",
    "/mcp": "Show MCP server status",
    "/mcp-config": "Configure MCP servers and handle MCP OAuth login",
    "/model": "Switch LLM model",
    "/new": "Start a fresh session in the current workspace",
    "/permission": "Select permission mode",
    "/plan": "Toggle plan mode",
    "/plugins": "Manage plugins",
    "/provider": "Manage AI providers (add / delete / refresh)",
    "/settings": "Open TUI settings",
    "/status": "Show current session and runtime status",
    "/swarm": "Toggle swarm mode or run one task in swarm mode",
    "/tasks": "Browse background tasks",
    "/theme": "Set the terminal UI theme",
    "/title": "Set or show session title",
    "/undo": "Withdraw the last prompt from the transcript",
    "/usage": "Show session tokens, context window and plan quotas",
    "/version": "Show version information",
    "/yolo": "Toggle YOLO mode — auto-approve tool actions",
}

# Commands that open a modal in-TUI picker the user must drive with arrow keys.
# Must stay a subset of the built-ins above (enforced by the picker-drift test).
_KIMI_PICKER_COMMANDS = frozenset(
    {
        "editor",
        "effort",
        "experiments",
        "login",
        "logout",
        "mcp-config",
        "model",
        "permission",
        "plugins",
        "provider",
        "settings",
        "tasks",
        "theme",
    }
)


class KimiProvider(JsonlProvider):
    """AgentProvider implementation for the Kimi Code CLI."""

    _CAPS = ProviderCapabilities(
        name="kimi",
        launch_command="kimi",
        supports_hook=False,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        supports_incremental_read=True,
        builtin_commands=tuple(_KIMI_TELEGRAM_BUILTINS.keys()),
        supports_user_command_discovery=False,
        supports_status_snapshot=True,
        # ``kimi --yolo`` enters permissive mode straight from the CLI flag —
        # there is no in-TUI confirmation dialog to accept afterwards.
        has_yolo_confirmation=False,
        tui_picker_commands=_KIMI_PICKER_COMMANDS,
    )

    _BUILTINS = _KIMI_TELEGRAM_BUILTINS

    # ── Launch ───────────────────────────────────────────────────────────

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        """Build CLI args.  ``-S <id>`` resumes without the interactive picker."""
        if resume_id:
            if not RESUME_ID_RE.match(resume_id):
                raise ValueError(f"Invalid resume_id: {resume_id!r}")
            return f"-S {shlex.quote(resume_id)}"
        if use_continue:
            return "--continue"
        return ""

    # ── Transcript parsing ───────────────────────────────────────────────

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
        cwd: str | None = None,  # noqa: ARG002 — kept for protocol compat
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        """Parse wire records into AgentMessages, pairing tool calls to results."""
        messages: list[AgentMessage] = []
        pending: Pending = normalize_pending(pending_tools)

        for entry in entries:
            record_type = entry.get("type")
            if record_type == TURN_PROMPT:
                message = parse_turn_prompt(entry)
            elif record_type == LOOP_EVENT:
                message = parse_loop_event(entry, pending)
            else:
                # context.append_message duplicates the user turn and carries
                # injected system reminders — never relayed.
                continue
            if message is not None:
                messages.append(message)

        return messages, dict(pending)

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        """Return True for a human turn (``turn.prompt`` with a user origin)."""
        if entry.get("type") != TURN_PROMPT:
            return False
        origin = entry.get("origin")
        return not (
            isinstance(origin, dict) and origin.get("kind") not in (None, "user")
        )

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        """Parse one wire record for history display (user + assistant text)."""
        record_type = entry.get("type")
        if record_type == TURN_PROMPT:
            return parse_turn_prompt(entry)
        if record_type == LOOP_EVENT:
            message = parse_loop_event(entry, {})
            if message is not None and message.content_type == "text":
                return message
        return None

    # ── Discovery ────────────────────────────────────────────────────────

    def _index_entries(self) -> list[dict[str, str]]:
        """Read kimi's session index, newest last.  Empty on any read error."""
        try:
            with open(session_index_path(), encoding="utf-8") as fh:
                raw_lines = fh.readlines()
        except OSError:
            return []

        entries: list[dict[str, str]] = []
        for raw in raw_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            session_id = data.get("sessionId")
            session_dir = data.get("sessionDir")
            work_dir = data.get("workDir")
            if not (
                isinstance(session_id, str)
                and isinstance(session_dir, str)
                and isinstance(work_dir, str)
            ):
                continue
            entries.append(
                {
                    "session_id": session_id,
                    "session_dir": session_dir,
                    "work_dir": work_dir,
                }
            )
        return entries

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Return the newest kimi session whose ``workDir`` matches *cwd*.

        Candidates come from ``session_index.jsonl`` (append-only, so the tail
        is the most recent), and are ranked by transcript mtime rather than
        index order — a resumed session keeps its original index position but
        has the freshest wire log.
        """
        if not cwd:
            return None

        age_limit = (
            _STALE_TRANSCRIPT_MAX_AGE_SECS if max_age is None else float(max_age)
        )
        now = time.time()
        try:
            resolved_target = str(Path(cwd).resolve())
        except OSError:
            return None

        candidates: list[tuple[float, dict[str, str], Path]] = []
        for entry in reversed(self._index_entries()[-_DISCOVERY_SCAN_LIMIT:]):
            try:
                if str(Path(entry["work_dir"]).resolve()) != resolved_target:
                    continue
            except OSError:
                continue
            transcript = transcript_for_session_dir(entry["session_dir"])
            try:
                mtime = transcript.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, entry, transcript))

        if not candidates:
            return None

        mtime, entry, transcript = max(candidates, key=lambda item: item[0])
        if age_limit > 0 and now - mtime > age_limit:
            return None

        return SessionStartEvent(
            session_id=entry["session_id"],
            cwd=entry["work_dir"],
            transcript_path=str(transcript),
            window_key=window_key,
        )

    # ── Terminal status ──────────────────────────────────────────────────

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",  # noqa: ARG002 — kimi's title is a constant
    ) -> StatusUpdate | None:
        """Parse kimi's pane for an interactive prompt or a busy spinner.

        Interactive regions are fenced between full-width ``─`` rules, so the
        last complete fence is extracted and accepted only when it carries an
        interactive affordance (footer hint, selection cursor, or a numbered
        option list).  Otherwise the busy line above the input box decides.
        """
        if not pane_text:
            return None

        lines = _nonblank_tail(pane_text)
        if not lines:
            return None

        block = _extract_fenced_block(lines)
        if block and _block_is_interactive(block):
            title, is_question = _block_title(block)
            body = "\n".join(block)
            is_permission = is_question and bool(_PERMISSION_HINT_RE.search(body))
            ui_type = "PermissionPrompt" if is_permission else "SelectionUI"
            return StatusUpdate(
                raw_text=body,
                display_label=title or ui_type,
                is_interactive=True,
                ui_type=ui_type,
            )

        label = _spinner_label(lines)
        if label is not None:
            text = label or "working"
            return StatusUpdate(raw_text=text, display_label=f"…{text}")

        return None

    async def scrape_current_mode(
        self,
        window_id: str,
        *,
        capture_fn: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> str | None:
        """Return kimi's active mode badge (``YOLO``, ``Plan``, …) or None.

        ``capture_fn`` is injectable for tests — defaults to the multiplexer
        proxy's ``capture_pane`` so production callers need no changes.
        """
        if not window_id:
            return None
        if capture_fn is not None:
            _fn: Callable[[str], Awaitable[str | None]] = capture_fn
        else:
            # Lazy: the multiplexer package imports providers; deferring the
            # import keeps the provider module free of that cycle.
            from ccgram.multiplexer import multiplexer

            _fn = multiplexer.capture_pane
        try:
            capture = await _fn(window_id)
        except OSError as exc:
            logger.warning("Mode scrape: capture_pane failed %s (%s)", window_id, exc)
            return None
        if not capture:
            return None
        return parse_status_modes(capture)

    # ── Status snapshot ──────────────────────────────────────────────────

    def build_status_snapshot(
        self,
        transcript_path: str,
        *,
        display_name: str = "",
        session_id: str = "",
        cwd: str = "",
    ) -> str | None:
        """Build a lightweight status snapshot for a kimi session."""
        try:
            size = os.path.getsize(transcript_path)
        except OSError:
            return None

        # Kimi ids are ``session_<uuid>``; drop the redundant prefix and keep a
        # readable head. Ids that are already short pass through untouched.
        short_id = session_id
        prefix = "session_"
        if short_id.startswith(prefix):
            short_id = short_id[len(prefix) :]
        if len(short_id) > _SHORT_SESSION_ID_THRESHOLD:
            short_id = short_id[:_SHORT_SESSION_ID_LEN]

        return (
            f"\U0001f319 [{display_name}] Kimi session active.\n"
            f"\U0001f4c2 `{cwd}`\n"
            f"\U0001f4c4 `{os.path.basename(transcript_path)}` ({size} bytes)\n"
            f"⭐ ID: `{short_id}`"
        )

    def has_output_since(self, transcript_path: str, offset: int) -> bool:
        """Check whether the wire log grew past *offset*."""
        try:
            return os.path.getsize(transcript_path) > offset
        except OSError:
            return False

    # ── Commands ─────────────────────────────────────────────────────────

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        """Return kimi built-ins plus workspace/user skills."""
        commands = super().discover_commands(base_dir)
        seen = {cmd.name for cmd in commands}
        for skill in _discover_kimi_skills(base_dir):
            if skill.name in seen:
                continue
            commands.append(skill)
            seen.add(skill.name)
        return commands


def _skill_dirs(base_dir: str) -> list[Path]:
    """Return the skill roots kimi loads: user-level then workspace-level."""
    dirs = [kimi_home() / "skills"]
    if base_dir:
        dirs.append(Path(base_dir) / ".kimi" / "skills")
    return dirs


def _discover_kimi_skills(base_dir: str) -> list[DiscoveredCommand]:
    """Discover ``SKILL.md``-backed kimi skills as slash commands."""
    # Lazy: command_catalog pulls provider/config machinery; only needed when
    # a caller actually enumerates commands.
    from ccgram.command_catalog import parse_frontmatter

    found: list[DiscoveredCommand] = []
    for root in _skill_dirs(base_dir):
        try:
            children = sorted(root.iterdir())
        except OSError:
            continue
        for entry in children:
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue
            frontmatter = parse_frontmatter(skill_file)
            name = frontmatter.get("name") or entry.name
            description = frontmatter.get("description", "")
            found.append(
                DiscoveredCommand(
                    name=f"/{name}",
                    description=description or f"Kimi skill: {name}",
                    source="skill",
                )
            )
    return found


__all__ = [
    "KimiProvider",
    "kimi_home",
    "parse_status_modes",
    "session_index_path",
    "transcript_for_session_dir",
]
