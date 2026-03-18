"""Gemini CLI provider — Google's terminal agent behind AgentProvider protocol.

Gemini CLI uses directory-scoped sessions with automatic persistence. Resume
uses ``--resume <index|latest>`` flag syntax (index number or "latest", not
a session UUID). No SessionStart hook — session detection requires external
wrapping.

Terminal UI: Gemini CLI uses ``@inquirer/select`` for interactive prompts.
Permission prompts start with "Action Required" and list numbered options
with a ``●`` (U+25CF) marker on the selected choice.

Transcript format: single JSON file per session (NOT JSONL) with structure:
  ``{sessionId, projectHash, startTime, lastUpdated, messages: [...]}``
Messages use ``type`` field with values ``"user"`` / ``"gemini"`` and can
store content as either a string or a list of ``{text: ...}`` fragments.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import threading
import time
import tomllib
from typing import Any, cast

from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import (
    AgentMessage,
    ContentType,
    DiscoveredCommand,
    MessageRole,
    ProviderCapabilities,
    RESUME_ID_RE,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.terminal_parser import UIPattern, extract_interactive_content
from ccgram.utils import atomic_write_json, ccgram_dir

# Gemini CLI known slash commands
_GEMINI_BUILTINS: dict[str, str] = {
    "/about": "Show version info",
    "/auth": "Manage authentication",
    "/bug": "Submit a bug report",
    "/chat": "Save, resume, list, or delete named sessions",
    "/clear": "Clear screen and chat context",
    "/commands": "Manage custom slash commands",
    "/compress": "Summarize chat context to save tokens",
    "/copy": "Copy last response to clipboard",
    "/diff": "View file changes",
    "/directory": "Manage accessible directories",
    "/directories": "Manage accessible directories",
    "/docs": "Open full Gemini CLI docs",
    "/editor": "Set editor preference",
    "/extensions": "Manage extensions",
    "/help": "Display available commands",
    "/hooks": "Manage hooks",
    "/ide": "Manage IDE integration",
    "/init": "Generate GEMINI.md context",
    "/mcp": "List MCP servers and tools",
    "/memory": "Show or manage GEMINI.md context",
    "/model": "Switch model mid-session",
    "/oncall": "Oncall workflows",
    "/permissions": "Manage trust and permissions",
    "/plan": "Switch to plan mode",
    "/policies": "List active policies",
    "/privacy": "Display privacy notice",
    "/quit": "Exit Gemini CLI",
    "/resume": "Browse and resume auto-saved sessions",
    "/rewind": "Restart from an earlier message",
    "/restore": "List or restore project state checkpoints",
    "/settings": "View and edit Gemini settings",
    "/setup-github": "Set up GitHub Actions",
    "/shells": "Toggle background shells",
    "/shortcuts": "Toggle shortcuts panel",
    "/skills": "Enable, list, or reload agent skills",
    "/stats": "Show session statistics",
    "/terminal-setup": "Configure terminal keybindings",
    "/theme": "Change theme",
    "/tools": "List accessible tools",
    "/vim": "Toggle Vim input mode",
}

# Gemini role → our MessageRole mapping
_GEMINI_ROLE_MAP: dict[str, MessageRole] = {
    "user": "user",
    "gemini": "assistant",
    # Gemini emits informational/error messages as separate record types.
    "info": "assistant",
    "error": "assistant",
}

# ── Gemini CLI UI patterns ──────────────────────────────────────────────
#
# Gemini uses @inquirer/select for permission prompts.  The structure is:
#
#   Action Required
#   ? Shell <command> [current working directory <path>] (<description>…
#   <command>
#   Allow execution of: '<tools>'?
#   ● 1. Allow once
#     2. Allow for this session
#     3. Allow for all future sessions
#     4. No, suggest changes (esc
#
# For file writes: "? WriteFile <path>" instead of "? Shell <command>".
# The ● (U+25CF) marks the selected option; (esc is always on the last line.
#
# We match on structural markers rather than exact wording for resilience
# against prompt text changes.
_GEMINI_BOX_PREFIX = r"[\s│┃║|]*"

GEMINI_UI_PATTERNS: list[UIPattern] = [
    UIPattern(
        name="SelectionUI",
        top=(
            # e.g. "Select Model"
            re.compile(rf"^{_GEMINI_BOX_PREFIX}Select\b"),
        ),
        bottom=(
            # Gemini modal close hint
            re.compile(rf"^{_GEMINI_BOX_PREFIX}\(Press Esc to (close|cancel)\)"),
            # Some selectors show Enter/Tab hints instead of close text.
            re.compile(rf"^{_GEMINI_BOX_PREFIX}\(Press Enter to (confirm|select)\)"),
        ),
        min_gap=1,
    ),
    UIPattern(
        name="PermissionPrompt",
        top=(
            # "Action Required" header (bold in terminal, plain in capture)
            re.compile(rf"^{_GEMINI_BOX_PREFIX}Action Required"),
        ),
        bottom=(
            # Last option always ends with "(esc" (possibly truncated by pane width)
            re.compile(r"(?i)\(esc"),
            # Fallback: a numbered "No" option (the cancel choice)
            re.compile(rf"^{_GEMINI_BOX_PREFIX}\d+\.\s+No\b"),
        ),
    ),
]


# Cache: file_path -> (mtime_ns, size, parsed_messages)
# Bounded to prevent unbounded growth; oldest entries evicted when full.
# Lock required: read_transcript_file runs in asyncio.to_thread() workers.
_TRANSCRIPT_CACHE_MAX = 64
_transcript_cache: dict[str, tuple[int, int, list[dict[str, Any]]]] = {}
_transcript_cache_lock = threading.Lock()
_TRANSCRIPT_MAX_AGE_SECS = 120.0
_MAX_TOOL_SUMMARY = 200
_JSON_READ_ERRORS = (OSError, json.JSONDecodeError)
_TOML_READ_ERRORS = (OSError, tomllib.TOMLDecodeError)
_GEMINI_SYSTEM_SETTINGS_FILE = "gemini-system-settings.json"
_GEMINI_WRAPPER_COMMANDS = frozenset({"bun", "node", "npx"})
_GEMINI_PANE_TITLE_MARKERS = ("\u2726", "\u270b", "\u25c7")


def _runtime_command_basename(pane_current_command: str) -> str:
    cmd = pane_current_command.strip().lower()
    if not cmd:
        return ""
    return os.path.basename(cmd.split()[0])


def needs_pane_title_for_detection(pane_current_command: str) -> bool:
    """Return True when runtime detection needs pane-title context."""
    return _runtime_command_basename(pane_current_command) in _GEMINI_WRAPPER_COMMANDS


def detect_gemini_from_runtime(pane_current_command: str, pane_title: str) -> bool:
    """Detect Gemini when wrapped by runtime shims like bun/node/npx."""
    if not needs_pane_title_for_detection(pane_current_command):
        return False
    if not isinstance(pane_title, str):
        return False
    return any(marker in pane_title for marker in _GEMINI_PANE_TITLE_MARKERS)


def build_hardened_gemini_launch_command(command: str) -> str:
    """Wrap Gemini launch command with ccgram-managed stability settings.

    Gemini reads this path as "system settings", so it overrides workspace/user
    settings and reliably disables interactive-shell PTY mode for ccgram runs.
    If the settings file cannot be written, returns the original command.
    """
    settings_path = ccgram_dir() / _GEMINI_SYSTEM_SETTINGS_FILE
    try:
        atomic_write_json(
            settings_path,
            {
                "tools": {
                    "shell": {
                        # Disable node-pty interactive shell mode in ccgram-managed Gemini runs.
                        # This avoids known EBADF crashes in tmux when shell tools execute.
                        "enableInteractiveShell": False
                    }
                }
            },
        )
    except OSError:
        return command
    quoted_path = shlex.quote(str(settings_path))
    return f"env GEMINI_CLI_SYSTEM_SETTINGS_PATH={quoted_path} {command}"


def _extract_gemini_text(value: Any) -> str:
    """Extract visible text from Gemini content/displayContent payloads."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
            continue
        content = item.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    return "".join(parts)


def _entry_text(entry: dict[str, Any]) -> str:
    """Extract human-visible message text from a Gemini transcript entry."""
    text = _extract_gemini_text(entry.get("content"))
    if text:
        return text
    return _extract_gemini_text(entry.get("displayContent"))


def _summarize_tool_args(args: Any) -> str:
    """Create a short summary from Gemini tool-call args."""
    if not isinstance(args, dict):
        return ""
    preferred = (
        "cmd",
        "command",
        "file_path",
        "dir_path",
        "path",
        "pattern",
        "query",
        "url",
    )
    for key in preferred:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value[:_MAX_TOOL_SUMMARY]
    for value in args.values():
        if isinstance(value, str) and value:
            return value[:_MAX_TOOL_SUMMARY]
    return ""


def _extract_tool_result_text(tool_call: dict[str, Any]) -> str:
    """Extract tool-result text from a Gemini tool-call payload."""
    result_display = tool_call.get("resultDisplay")
    if isinstance(result_display, str) and result_display:
        return result_display

    result = tool_call.get("result")
    if not isinstance(result, list):
        return ""
    for item in result:
        if not isinstance(item, dict):
            continue
        function_response = item.get("functionResponse")
        if not isinstance(function_response, dict):
            continue
        response = function_response.get("response")
        if not isinstance(response, dict):
            continue
        for key in ("output", "error", "result"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _read_project_alias(config_dir: Path, resolved_cwd: str) -> str:
    """Read the project alias for cwd from ~/.gemini/projects.json."""
    projects_path = config_dir / "projects.json"
    try:
        with open(projects_path, encoding="utf-8") as f:
            data = json.load(f)
    except _JSON_READ_ERRORS:
        return ""
    if not isinstance(data, dict):
        return ""
    projects = data.get("projects")
    if not isinstance(projects, dict):
        return ""
    alias = projects.get(resolved_cwd, "")
    return alias if isinstance(alias, str) else ""


def _collect_gemini_sessions(chats_dir: Path) -> list[tuple[float, Path]]:
    """Collect Gemini chat transcripts from a chats directory."""
    result: list[tuple[float, Path]] = []
    try:
        files = sorted(chats_dir.glob("session-*.json"))
    except OSError:
        return result
    for fpath in files:
        try:
            result.append((fpath.stat().st_mtime, fpath))
        except OSError:
            continue
    return result


def _read_gemini_session_meta(fpath: Path) -> tuple[str, str] | None:
    """Read (session_id, project_hash) from a Gemini transcript JSON file."""
    try:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
    except _JSON_READ_ERRORS:
        return None
    if not isinstance(data, dict):
        return None
    session_id = data.get("sessionId", "")
    if not isinstance(session_id, str) or not session_id:
        return None
    project_hash = data.get("projectHash", "")
    project_hash_str = project_hash if isinstance(project_hash, str) else ""
    return session_id, project_hash_str


def _resolve_gemini_commands_dir(base_dir: str) -> Path:
    """Resolve Gemini commands directory from a provider base dir."""
    base = Path(base_dir).expanduser()
    if base.name == ".claude":
        return base.with_name(".gemini") / "commands"
    if base.name == ".gemini":
        return base / "commands"
    return base / ".gemini" / "commands"


def _parse_toml_command_description(cmd_file: Path, *, default: str) -> str:
    """Read command description from Gemini TOML file."""
    try:
        with open(cmd_file, "rb") as f:
            parsed = tomllib.load(f)
    except _TOML_READ_ERRORS:
        return default
    if not isinstance(parsed, dict):
        return default
    raw_description = parsed.get("description")
    if isinstance(raw_description, str) and raw_description:
        return raw_description
    return default


def _discover_gemini_toml_commands(base_dir: str) -> list[DiscoveredCommand]:
    """Discover Gemini custom slash commands from .gemini/commands/*.toml."""
    commands_dir = _resolve_gemini_commands_dir(base_dir)
    if not commands_dir.is_dir():
        return []

    discovered: list[DiscoveredCommand] = []
    try:
        groups = sorted(commands_dir.iterdir())
    except OSError:
        return []

    for group_dir in groups:
        if not group_dir.is_dir() or group_dir.name.startswith("."):
            continue
        try:
            files = sorted(group_dir.glob("*.toml"))
        except OSError:
            continue
        for cmd_file in files:
            if cmd_file.name.startswith("."):
                continue
            name = f"{group_dir.name}:{cmd_file.stem}"
            description = _parse_toml_command_description(
                cmd_file,
                default=f"/{name}",
            )
            discovered.append(
                DiscoveredCommand(
                    name=name,
                    description=description,
                    source="command",
                )
            )
    return discovered


class GeminiProvider(JsonlProvider):
    """AgentProvider implementation for Google Gemini CLI."""

    _CAPS = ProviderCapabilities(
        name="gemini",
        launch_command="gemini",
        supports_hook=False,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        supports_incremental_read=False,
        transcript_format="plain",
        uses_pane_title=True,
        builtin_commands=tuple(_GEMINI_BUILTINS.keys()),
    )

    _BUILTINS = _GEMINI_BUILTINS

    def requires_pane_title_for_detection(self, pane_current_command: str) -> bool:
        """Return True when Gemini runtime detection needs pane-title context."""
        return needs_pane_title_for_detection(pane_current_command)

    def detect_from_pane_title(
        self, pane_current_command: str, pane_title: str
    ) -> bool:
        """Detect Gemini from wrapped command + OSC title markers."""
        return detect_gemini_from_runtime(pane_current_command, pane_title)

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        """Build Gemini CLI args for launching or resuming a session.

        Resume uses ``--resume <index|latest>`` — accepts a numeric index
        or ``"latest"``, NOT a UUID.
        Continue uses ``--resume latest`` to pick up the most recent session.
        """
        if resume_id:
            # Allow numeric indices and "latest" in addition to standard IDs
            if not (resume_id == "latest" or RESUME_ID_RE.match(resume_id)):
                raise ValueError(f"Invalid resume_id: {resume_id!r}")
            return f"--resume {resume_id}"
        if use_continue:
            return "--resume latest"
        return ""

    # ── Gemini-specific transcript parsing ────────────────────────────

    def read_transcript_file(
        self, file_path: str, last_offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Read Gemini's single-JSON transcript and return new messages.

        Gemini transcripts are a single JSON object with a ``messages`` array,
        not JSONL. ``last_offset`` tracks the number of messages already seen.
        Returns (new_message_entries, updated_offset).

        Uses an mtime+size cache to skip re-parsing when the file is unchanged.
        """

        try:
            st = os.stat(file_path)
        except OSError:
            return [], last_offset

        with _transcript_cache_lock:
            cached = _transcript_cache.get(file_path)
        if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            messages = list(cached[2])
        else:
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):  # fmt: skip
                return [], last_offset

            if not isinstance(data, dict):
                return [], last_offset

            messages = data.get("messages", [])
            if not isinstance(messages, list):
                return [], last_offset

            # Store a copy to prevent mutation of cached data
            messages = list(messages)
            with _transcript_cache_lock:
                if len(_transcript_cache) >= _TRANSCRIPT_CACHE_MAX:
                    # Evict first-inserted entry
                    _transcript_cache.pop(next(iter(_transcript_cache)))
                _transcript_cache[file_path] = (
                    st.st_mtime_ns,
                    st.st_size,
                    messages,
                )

        new_entries = messages[last_offset:]
        new_offset = len(messages)
        return [m for m in new_entries if isinstance(m, dict)], new_offset

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        """Parse Gemini transcript entries into AgentMessages.

        Gemini messages use ``type`` field ("user"/"gemini"/"info"/"error")
        instead of ``role`` and support mixed content formats (string or list
        of text fragments). Tool calls are emitted as ``tool_use`` and
        ``tool_result`` messages when possible.
        """
        messages: list[AgentMessage] = []
        pending = dict(pending_tools)

        for entry in entries:
            msg_type = entry.get("type", "")
            role = _GEMINI_ROLE_MAP.get(msg_type)
            if not role:
                continue

            # Gemini tool calls are attached to a gemini turn and may contain
            # both input args and immediate result payloads.
            tool_calls = entry.get("toolCalls", [])
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    raw_name = tc.get("displayName") or tc.get("name") or "unknown"
                    tool_name = raw_name if isinstance(raw_name, str) else "unknown"
                    call_id = tc.get("id")
                    tool_use_id = (
                        call_id if isinstance(call_id, str) and call_id else None
                    )
                    if tool_use_id:
                        pending[tool_use_id] = tool_name
                    summary = _summarize_tool_args(tc.get("args"))
                    tool_use_text = (
                        f"**{tool_name}** `{summary}`"
                        if summary
                        else f"**{tool_name}**"
                    )
                    messages.append(
                        AgentMessage(
                            text=tool_use_text,
                            role="assistant",
                            content_type="tool_use",
                            tool_use_id=tool_use_id,
                            tool_name=tool_name,
                            timestamp=entry.get("timestamp"),
                        )
                    )
                    result_text = _extract_tool_result_text(tc)
                    if result_text:
                        messages.append(
                            AgentMessage(
                                text=result_text,
                                role="assistant",
                                content_type="tool_result",
                                tool_use_id=tool_use_id,
                                tool_name=tool_name,
                                timestamp=entry.get("timestamp"),
                            )
                        )
                        if tool_use_id:
                            pending.pop(tool_use_id, None)

            text = _entry_text(entry)
            if text:
                messages.append(
                    AgentMessage(
                        text=text,
                        role=cast(MessageRole, role),
                        content_type=cast(ContentType, "text"),
                        timestamp=entry.get("timestamp"),
                    )
                )

        return messages, pending

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        """Check if this Gemini entry is a human turn."""
        return entry.get("type") == "user"

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        """Parse a single Gemini transcript entry for history display."""
        msg_type = entry.get("type", "")
        role = _GEMINI_ROLE_MAP.get(msg_type)
        if not role:
            return None
        text = _entry_text(entry)
        if not text:
            return None
        return AgentMessage(
            text=text,
            role=cast(MessageRole, role),
            content_type="text",
            timestamp=entry.get("timestamp"),
        )

    @staticmethod
    def _candidate_chats_dirs(
        sessions_root: Path,
        config_dir: Path,
        resolved_cwd: str,
        expected_hash: str,
    ) -> list[Path]:
        """Build candidate chats directories for a cwd."""
        dirs: list[Path] = [sessions_root / expected_hash / "chats"]
        alias = _read_project_alias(config_dir, resolved_cwd)
        if alias:
            dirs.append(sessions_root / alias / "chats")
        return dirs

    @staticmethod
    def _collect_candidate_sessions(
        candidate_dirs: list[Path],
    ) -> list[tuple[float, Path]]:
        """Collect session files from expected candidate directories."""
        sessions: list[tuple[float, Path]] = []
        seen_dirs: set[Path] = set()
        for chats_dir in candidate_dirs:
            if chats_dir in seen_dirs or not chats_dir.is_dir():
                continue
            seen_dirs.add(chats_dir)
            sessions.extend(_collect_gemini_sessions(chats_dir))
        return sessions

    @staticmethod
    def _match_session_event(
        sessions: list[tuple[float, Path]],
        expected_hash: str,
        resolved_cwd: str,
        window_key: str,
        *,
        age_limit: float,
        now: float,
    ) -> SessionStartEvent | None:
        """Return the newest matching session event from candidate files."""
        sessions.sort(reverse=True)
        for mtime, fpath in sessions[:50]:
            if age_limit > 0 and now - mtime > age_limit:
                break
            meta = _read_gemini_session_meta(fpath)
            if not meta:
                continue
            session_id, project_hash = meta
            # Match project strictly to avoid cross-project false positives.
            if project_hash != expected_hash:
                continue
            return SessionStartEvent(
                session_id=session_id,
                cwd=resolved_cwd,
                transcript_path=str(fpath),
                window_key=window_key,
            )
        return None

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Discover latest Gemini transcript matching cwd.

        Gemini stores chats under ``~/.gemini/tmp/<project>/chats/session-*.json``
        (project alias) and older versions may use ``<projectHash>`` directory
        names. We match by ``projectHash`` (sha256 of resolved cwd).
        """
        config_dir = Path.home() / ".gemini"
        sessions_root = config_dir / "tmp"
        if not sessions_root.is_dir():
            return None

        resolved_cwd = str(Path(cwd).resolve())
        expected_hash = hashlib.sha256(resolved_cwd.encode()).hexdigest()
        age_limit = _TRANSCRIPT_MAX_AGE_SECS if max_age is None else max_age
        now = time.time()
        candidate_dirs = self._candidate_chats_dirs(
            sessions_root,
            config_dir,
            resolved_cwd,
            expected_hash,
        )
        sessions = self._collect_candidate_sessions(candidate_dirs)
        return self._match_session_event(
            sessions,
            expected_hash,
            resolved_cwd,
            window_key,
            age_limit=age_limit,
            now=now,
        )

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        """Discover built-ins plus workspace Gemini TOML commands."""
        commands = super().discover_commands(base_dir)
        commands.extend(_discover_gemini_toml_commands(base_dir))
        deduped: list[DiscoveredCommand] = []
        seen: set[str] = set()
        for cmd in commands:
            if not cmd.name or cmd.name in seen:
                continue
            deduped.append(cmd)
            seen.add(cmd.name)
        return deduped

    def parse_terminal_status(
        self, pane_text: str, *, pane_title: str = ""
    ) -> StatusUpdate | None:
        """Parse Gemini CLI pane for status via title and interactive UI.

        Gemini CLI sets pane title via OSC escape sequences:
          - ``Working: ✦`` (U+2726) — agent is processing
          - ``Action Required: ✋`` (U+270B) — needs user input
          - ``Ready: ◇`` (U+25C7) — idle / waiting for input

        Title-based detection is checked first (most reliable), then
        pane content is scanned for interactive UI patterns.
        """
        # 1. Working title → non-interactive status
        # Accept both the emoji marker and plain text for robustness.
        if "\u2726" in pane_title or "Working" in pane_title:  # ✦
            return StatusUpdate(raw_text="working", display_label="\u2026working")

        # 2. Action Required title → check content for specific UI
        action_required = (
            "\u270b" in pane_title or "Action Required" in pane_title
        )  # ✋

        # 3. Pane content for interactive UI details
        interactive = extract_interactive_content(pane_text, GEMINI_UI_PATTERNS)
        if interactive:
            return StatusUpdate(
                raw_text=interactive.content,
                display_label=interactive.name,
                is_interactive=True,
                ui_type=interactive.name,
            )

        # 4. Title says action required but content didn't match patterns
        if action_required:
            return StatusUpdate(
                raw_text="Action Required",
                display_label="PermissionPrompt",
                is_interactive=True,
                ui_type="PermissionPrompt",
            )

        # 5. Ready title or unknown — no status (let activity heuristic handle)
        return None
