"""Antigravity CLI provider — wraps Gemini/Antigravity backend behind AgentProvider protocol.

Antigravity CLI uses directory-scoped sessions with automatic persistence. Resume
uses ``--resume <index|latest>`` flag syntax (index number or "latest", not
a session UUID). No SessionStart hook — session detection requires external
wrapping.

Terminal UI: Antigravity CLI uses ``@inquirer/select`` for interactive prompts.
Permission prompts start with "Action Required" and list numbered options
with a ``●`` (U+25CF) marker on the selected choice.

Transcript format: incremental JSONL files per session with structure:
  - Header: ``{"sessionId", "projectHash", "startTime", ...}``
  - Entries: ``{"id", "timestamp", "type", "content": [...]}``
  - Updates: ``{"$set": {"lastUpdated": "..."}}``

Messages use ``type`` field with values ``"user"`` / ``"gemini"`` (or "antigravity" equivalents)
and can store content as either a string or a list of ``{text: ...}`` fragments.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
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

# Antigravity CLI known slash commands
_ANTIGRAVITY_BUILTINS: dict[str, str] = {
    "/about": "Show version info",
    "/agents": "Manage agent configurations",
    "/auth": "Manage authentication",
    "/bug": "Submit a bug report",
    "/chat": "Save, resume, list, or delete named sessions",
    "/clear": "Clear screen and chat context",
    "/commands": "Manage custom slash commands",
    "/compress": "Summarize chat context to save tokens",
    "/copy": "Copy last response to clipboard",
    "/directory": "Manage accessible directories",
    "/directories": "Manage accessible directories",
    "/docs": "Open full documentation",
    "/editor": "Set editor preference",
    "/extensions": "Manage extensions",
    "/help": "Display available commands",
    "/hooks": "Manage hooks",
    "/ide": "Manage IDE integration",
    "/init": "Generate context document",
    "/mcp": "List MCP servers and tools",
    "/memory": "Show or manage memory/context",
    "/model": "Switch model mid-session",
    "/oncall": "Oncall workflows",
    "/permissions": "Manage trust and permissions",
    "/plan": "Switch to plan mode",
    "/policies": "List active policies",
    "/privacy": "Display privacy notice",
    "/quit": "Exit chat",
    "/resume": "Browse and resume auto-saved sessions",
    "/rewind": "Restart from an earlier message",
    "/restore": "List or restore project state checkpoints",
    "/settings": "View and edit settings",
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

# Role/type → our MessageRole mapping
_ANTIGRAVITY_ROLE_MAP: dict[str, MessageRole] = {
    "user": "user",
    "gemini": "assistant",
    "antigravity": "assistant",
    "info": "assistant",
    "error": "assistant",
}

# ── Antigravity CLI UI patterns ──────────────────────────────────────────
#
# Antigravity uses @inquirer/select for permission prompts. The structure is:
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
_ANTIGRAVITY_BOX_PREFIX = r"[\s│┃║|]*"

ANTIGRAVITY_UI_PATTERNS: list[UIPattern] = [
    UIPattern(
        name="SelectionUI",
        top=(
            re.compile(rf"^{_ANTIGRAVITY_BOX_PREFIX}Select\b"),
        ),
        bottom=(
            re.compile(rf"^{_ANTIGRAVITY_BOX_PREFIX}\(Press Esc to (close|cancel)\)"),
            re.compile(rf"^{_ANTIGRAVITY_BOX_PREFIX}\(Press Enter to (confirm|select)\)"),
        ),
        min_gap=1,
    ),
    UIPattern(
        name="PermissionPrompt",
        top=(
            re.compile(rf"^{_ANTIGRAVITY_BOX_PREFIX}Action Required"),
        ),
        bottom=(
            re.compile(r"(?i)\(esc"),
            re.compile(rf"^{_ANTIGRAVITY_BOX_PREFIX}\d+\.\s+No\b"),
        ),
    ),
]

_TRANSCRIPT_MAX_AGE_SECS = 120.0
_MAX_TOOL_SUMMARY = 200
_JSON_READ_ERRORS = (OSError, json.JSONDecodeError)
_TOML_READ_ERRORS = (OSError, tomllib.TOMLDecodeError)
_ANTIGRAVITY_SYSTEM_SETTINGS_FILE = "antigravity-system-settings.json"
_ANTIGRAVITY_WRAPPER_COMMANDS = frozenset({"bun", "node", "npx"})
_ANTIGRAVITY_PANE_TITLE_MARKERS = ("🛸",)


def _runtime_command_basename(pane_current_command: str) -> str:
    cmd = pane_current_command.strip().lower()
    if not cmd:
        return ""
    return os.path.basename(cmd.split()[0])


def needs_pane_title_for_detection(pane_current_command: str) -> bool:
    """Return True when runtime detection needs pane-title context."""
    return _runtime_command_basename(pane_current_command) in _ANTIGRAVITY_WRAPPER_COMMANDS


def detect_antigravity_from_runtime(pane_current_command: str, pane_title: str) -> bool:
    """Detect Antigravity when wrapped by runtime shims like bun/node/npx."""
    if not needs_pane_title_for_detection(pane_current_command):
        return False
    if not isinstance(pane_title, str):
        return False
    return any(marker in pane_title for marker in _ANTIGRAVITY_PANE_TITLE_MARKERS)


def build_hardened_antigravity_launch_command(command: str) -> str:
    """Wrap Antigravity launch command with ccgram-managed stability settings."""
    settings_path = ccgram_dir() / _ANTIGRAVITY_SYSTEM_SETTINGS_FILE
    try:
        atomic_write_json(
            settings_path,
            {
                "tools": {
                    "shell": {
                        "enableInteractiveShell": False
                    }
                }
            },
        )
    except OSError:
        return command
    quoted_path = shlex.quote(str(settings_path))
    # Note: Antigravity CLI shares GEMINI_CLI_SYSTEM_SETTINGS_PATH or its own environment settings
    return f"env GEMINI_CLI_SYSTEM_SETTINGS_PATH={quoted_path} ANTIGRAVITY_CLI_SYSTEM_SETTINGS_PATH={quoted_path} {command}"


def _extract_antigravity_text(value: Any) -> str:
    """Extract visible text from Antigravity content/displayContent payloads."""
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
    """Extract human-visible message text from an Antigravity transcript entry."""
    text = _extract_antigravity_text(entry.get("content"))
    if text:
        return text
    return _extract_antigravity_text(entry.get("displayContent"))


def _clean_user_content(text: str) -> str:
    """Extract clean user request without additional metadata tags."""
    if not text:
        return ""
    match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _find_matching_pending_tool(entry_type: str, pending: dict[str, str]) -> str | None:
    """Find a pending tool that matches the entry_type name (e.g. LIST_DIRECTORY -> list_dir)."""
    if not entry_type or not pending:
        return None
    normalized_type = entry_type.replace("_", "").lower()
    for tool_use_id, tool_name in pending.items():
        normalized_name = tool_name.replace("_", "").lower()
        if (normalized_name in normalized_type) or (normalized_type in normalized_name) or (normalized_name[:4] == normalized_type[:4]):
            return tool_use_id
    # Fallback to the oldest pending tool if no match
    return next(iter(pending.keys()), None)



def _summarize_tool_args(args: Any) -> str:
    """Create a short summary from Antigravity tool-call args."""
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
    """Extract tool-result text from an Antigravity tool-call payload."""
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


def _collect_antigravity_sessions(chats_dir: Path) -> list[tuple[float, Path]]:
    """Collect Antigravity chat transcripts from a chats directory."""
    result: list[tuple[float, Path]] = []
    try:
        files = sorted(chats_dir.glob("session-*.jsonl"))
    except OSError:
        return result
    for fpath in files:
        try:
            result.append((fpath.stat().st_mtime, fpath))
        except OSError:
            continue
    return result


def _read_antigravity_session_meta(fpath: Path) -> tuple[str, str] | None:
    """Read (session_id, project_hash) from an Antigravity transcript JSONL file."""
    try:
        with open(fpath, encoding="utf-8") as f:
            line = f.readline()
            if not line:
                return None
            data = json.loads(line)
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


def _resolve_antigravity_commands_dir(base_dir: str) -> Path:
    """Resolve Antigravity commands directory from a provider base dir."""
    base = Path(base_dir).expanduser()
    if base.name == ".claude":
        return base.with_name(".gemini") / "commands"
    if base.name == ".gemini":
        return base / "commands"
    return base / ".gemini" / "commands"


def _parse_toml_command_description(cmd_file: Path, *, default: str) -> str:
    """Read command description from Antigravity TOML file."""
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


def _discover_antigravity_toml_commands(base_dir: str) -> list[DiscoveredCommand]:
    """Discover custom slash commands from .gemini/commands/*.toml."""
    commands_dir = _resolve_antigravity_commands_dir(base_dir)
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


class AntigravityProvider(JsonlProvider):
    """AgentProvider implementation for Antigravity CLI."""

    _CAPS = ProviderCapabilities(
        name="antigravity",
        launch_command="antigravity chat",
        supports_hook=False,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        supports_incremental_read=True,
        transcript_format="jsonl",
        uses_pane_title=True,
        builtin_commands=tuple(_ANTIGRAVITY_BUILTINS.keys()),
        supports_status_snapshot=True,
    )

    _BUILTINS = _ANTIGRAVITY_BUILTINS

    def requires_pane_title_for_detection(self, pane_current_command: str) -> bool:
        """Return True when Antigravity runtime detection needs pane-title context."""
        return needs_pane_title_for_detection(pane_current_command)

    def detect_from_pane_title(
        self, pane_current_command: str, pane_title: str
    ) -> bool:
        """Detect Antigravity from wrapped command + OSC title markers."""
        return detect_antigravity_from_runtime(pane_current_command, pane_title)

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        """Build Antigravity CLI args for launching or resuming a session."""
        if resume_id:
            if not (resume_id == "latest" or RESUME_ID_RE.match(resume_id)):
                raise ValueError(f"Invalid resume_id: {resume_id!r}")
            return f"--resume {resume_id}"
        if use_continue:
            return "--resume latest"
        return ""

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
        cwd: str | None = None,  # noqa: ARG002
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        """Parse Antigravity transcript entries into AgentMessages."""
        messages: list[AgentMessage] = []
        pending = dict(pending_tools)

        for entry in entries:
            msg_type = entry.get("type", "")
            source = entry.get("source", "")

            # 1. User turn (new & old formats)
            if msg_type == "USER_INPUT" or source == "USER_EXPLICIT" or msg_type == "user":
                content_text = _clean_user_content(_entry_text(entry))
                if content_text:
                    messages.append(
                        AgentMessage(
                            text=content_text,
                            role="user",
                            content_type="text",
                            timestamp=entry.get("created_at") or entry.get("timestamp"),
                        )
                    )
                continue

            # Check if this is an assistant or planner turn
            is_assistant = (
                msg_type == "PLANNER_RESPONSE"
                or (source == "MODEL" and msg_type in ("PLANNER_RESPONSE", "gemini", "antigravity", "info", "error", ""))
                or (msg_type in _ANTIGRAVITY_ROLE_MAP and msg_type != "user")
            )


            if is_assistant:
                # A. Parse old-format tool calls
                tool_calls_old = entry.get("toolCalls", [])
                if isinstance(tool_calls_old, list) and tool_calls_old:
                    for tc in tool_calls_old:
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
                                timestamp=entry.get("created_at") or entry.get("timestamp"),
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
                                    timestamp=entry.get("created_at") or entry.get("timestamp"),
                                )
                            )
                            if tool_use_id:
                                pending.pop(tool_use_id, None)

                # B. Parse new-format tool calls
                tool_calls_new = entry.get("tool_calls", [])
                if isinstance(tool_calls_new, list) and tool_calls_new:
                    for tc in tool_calls_new:
                        if not isinstance(tc, dict):
                            continue
                        tool_name = tc.get("name") or "unknown"
                        step_idx = entry.get("step_index", "0")
                        tool_use_id = f"step-{step_idx}-{tool_name}"
                        
                        summary = _summarize_tool_args(tc.get("args"))
                        tool_use_text = (
                            f"**{tool_name}** `{summary}`"
                            if summary
                            else f"**{tool_name}**"
                        )
                        pending[tool_use_id] = tool_name
                        messages.append(
                            AgentMessage(
                                text=tool_use_text,
                                role="assistant",
                                content_type="tool_use",
                                tool_use_id=tool_use_id,
                                tool_name=tool_name,
                                timestamp=entry.get("created_at") or entry.get("timestamp"),
                            )
                        )

                # C. Parse standard/fallback text content
                text = _entry_text(entry)
                if text:
                    messages.append(
                        AgentMessage(
                            text=text,
                            role="assistant",
                            content_type="text",
                            timestamp=entry.get("created_at") or entry.get("timestamp"),
                        )
                    )
                continue

            # 2. Tool result (new format: msg_type represents the tool name in uppercase, e.g. LIST_DIRECTORY)
            if (source in ("MODEL", "SYSTEM") or msg_type) and msg_type not in ("CONVERSATION_HISTORY", ""):
                tool_use_id = _find_matching_pending_tool(msg_type, pending)
                if tool_use_id:
                    tool_name = pending.pop(tool_use_id)
                    text = _entry_text(entry)
                    if text:
                        messages.append(
                            AgentMessage(
                                text=text,
                                role="assistant",
                                content_type="tool_result",
                                tool_use_id=tool_use_id,
                                tool_name=tool_name,
                                timestamp=entry.get("created_at") or entry.get("timestamp"),
                            )
                        )
                    continue

        return messages, pending

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        """Check if this entry is a human turn."""
        msg_type = entry.get("type", "")
        source = entry.get("source", "")
        return msg_type == "USER_INPUT" or source == "USER_EXPLICIT" or msg_type == "user"

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        """Parse a single transcript entry for history display."""
        msg_type = entry.get("type", "")
        source = entry.get("source", "")

        if msg_type == "USER_INPUT" or source == "USER_EXPLICIT" or msg_type == "user":
            text = _clean_user_content(_entry_text(entry))
            if not text:
                return None
            return AgentMessage(
                text=text,
                role="user",
                content_type="text",
                timestamp=entry.get("created_at") or entry.get("timestamp"),
            )

        is_assistant = (
            msg_type == "PLANNER_RESPONSE"
            or (source == "MODEL" and msg_type in ("PLANNER_RESPONSE", "gemini", "antigravity", "info", "error", ""))
            or (msg_type in _ANTIGRAVITY_ROLE_MAP and msg_type != "user")
        )
        if is_assistant:
            text = _entry_text(entry)
            if not text:
                return None
            return AgentMessage(
                text=text,
                role="assistant",
                content_type="text",
                timestamp=entry.get("created_at") or entry.get("timestamp"),
            )

        return None

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
            sessions.extend(_collect_antigravity_sessions(chats_dir))
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
            meta = _read_antigravity_session_meta(fpath)
            if not meta:
                continue
            session_id, project_hash = meta
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
        """Discover latest Antigravity transcript matching cwd."""
        resolved_cwd = str(Path(cwd).resolve())

        # 1. Try discovering via history.jsonl
        history_path = Path.home() / ".gemini" / "antigravity-cli" / "history.jsonl"
        if history_path.is_file():
            try:
                # Read all lines from history.jsonl and reverse them (newest first)
                with open(history_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    line_workspace = data.get("workspace")
                    if not line_workspace:
                        continue
                    
                    try:
                        if Path(line_workspace).resolve() == Path(resolved_cwd).resolve():
                            conversation_id = data.get("conversationId")
                            if conversation_id:
                                transcript_path = (
                                    Path.home()
                                    / ".gemini"
                                    / "antigravity-cli"
                                    / "brain"
                                    / conversation_id
                                    / ".system_generated"
                                    / "logs"
                                    / "transcript.jsonl"
                                )
                                if transcript_path.is_file():
                                    # Check age limit if applicable
                                    if max_age is not None and max_age > 0:
                                        mtime = transcript_path.stat().st_mtime
                                        if time.time() - mtime > max_age:
                                            continue
                                    
                                    return SessionStartEvent(
                                        session_id=conversation_id,
                                        cwd=resolved_cwd,
                                        transcript_path=str(transcript_path),
                                        window_key=window_key,
                                    )
                    except Exception:
                        continue
            except OSError:
                pass

        # 2. Fallback to standard/old discovery logic for backward compatibility/tests
        config_dir = Path.home() / ".gemini"
        sessions_root = config_dir / "tmp"
        if not sessions_root.is_dir():
            return None

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
        """Discover built-ins plus workspace custom commands."""
        commands = super().discover_commands(base_dir)
        commands.extend(_discover_antigravity_toml_commands(base_dir))
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
        """Parse captured pane text and title into a StatusUpdate."""
        action_required = False

        # 1. Match against known interactive UI patterns (Permission Prompt, Selection UI)
        interactive = extract_interactive_content(pane_text, ANTIGRAVITY_UI_PATTERNS)
        if interactive:
            return StatusUpdate(
                raw_text=interactive.content,
                display_label=interactive.name,
                is_interactive=True,
                ui_type=interactive.name,
            )

        # 2. Check title for specific interactive state hints
        if pane_title:
            if "\u270b" in pane_title or "Action Required" in pane_title:
                action_required = True

        # 3. Check raw pane content for "Action Required" text
        if not action_required and "Action Required" in pane_text:
            action_required = True

        # 4. Action Required fallback
        if action_required:
            return StatusUpdate(
                raw_text="Action Required",
                display_label="PermissionPrompt",
                is_interactive=True,
                ui_type="PermissionPrompt",
            )

        return None

    def build_status_snapshot(
        self,
        transcript_path: str,
        *,
        display_name: str,
        session_id: str = "",
        cwd: str = "",
    ) -> str | None:
        """Build a basic status snapshot for Antigravity sessions."""
        try:
            size = os.path.getsize(transcript_path)
        except OSError:
            return None

        short_id = session_id[:8] if len(session_id) > 10 else session_id

        return (
            f"🛸 [{display_name}] Antigravity session active.\n"
            f"📁 `{cwd}`\n"
            f"📄 `{os.path.basename(transcript_path)}` ({size} bytes)\n"
            f"⭐ ID: `{short_id}`"
        )

    def has_output_since(self, transcript_path: str, offset: int) -> bool:
        """Check if any assistant output appeared after *offset*."""
        try:
            return os.path.getsize(transcript_path) > offset
        except OSError:
            return False
