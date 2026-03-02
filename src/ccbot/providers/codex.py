"""Codex CLI provider — OpenAI's terminal agent behind AgentProvider protocol.

Codex CLI uses a similar tmux-based launch model but differs in hook mechanism
(no SessionStart hook) and resume syntax (``resume`` subcommand, not a flag).
No interactive UI detection — terminal_ui_patterns is empty.

Transcript format: JSONL with entries ``{timestamp, type, payload}``.
Entry types: ``session_meta``, ``response_item``, ``input_item``, ``event_msg``,
``turn_context``.  Roles in payload: ``developer``, ``user``, ``assistant``.
Content blocks: ``input_text`` (user), ``output_text`` (assistant),
``function_call`` / ``function_call_output`` (tool use).
"""

import json
from pathlib import Path
from typing import Any, cast

from ccbot.providers.base import (
    RESUME_ID_RE,
    AgentMessage,
    ContentType,
    MessageRole,
    ProviderCapabilities,
    SessionStartEvent,
)
from ccbot.providers._jsonl import JsonlProvider

# Codex CLI known slash commands
# NOTE: /new excluded — collides with bot-native /new (create session)
_CODEX_BUILTINS: dict[str, str] = {
    "/model": "Switch model",
    "/mode": "Switch approval mode (suggest/auto-edit/full-auto)",
    "/status": "Show session config and token usage",
    "/permissions": "Adjust approval requirements",
    "/diff": "Show git changes",
    "/compact": "Summarize context to save tokens",
    "/mcp": "List MCP tools",
    "/mention": "Attach files to conversation",
}


def _extract_codex_content(
    content: Any, pending: dict[str, Any]
) -> tuple[str, ContentType, dict[str, Any]]:
    """Extract text and track tool use from Codex content blocks.

    Codex uses ``output_text`` (assistant), ``input_text`` (user),
    ``function_call`` and ``function_call_output`` (tool use).
    """
    if isinstance(content, str):
        return content, "text", pending
    if not isinstance(content, list):
        return "", "text", pending

    text = ""
    content_type: ContentType = "text"
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype in ("output_text", "input_text"):
            text += block.get("text", "")
        elif btype == "function_call" and block.get("call_id"):
            pending[block["call_id"]] = block.get("name", "unknown")
            content_type = "tool_use"
        elif btype == "function_call_output":
            call_id = block.get("call_id")
            if call_id:
                pending.pop(call_id, None)
            content_type = "tool_result"
    return text, content_type, pending


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


class CodexProvider(JsonlProvider):
    """AgentProvider implementation for OpenAI Codex CLI."""

    _CAPS = ProviderCapabilities(
        name="codex",
        launch_command="codex",
        supports_hook=False,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        transcript_format="jsonl",
        terminal_ui_patterns=(),
        builtin_commands=tuple(_CODEX_BUILTINS.keys()),
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

    def parse_transcript_line(self, line: str) -> dict[str, Any] | None:
        """Parse a Codex JSONL line.

        Codex entries are ``{timestamp, type, payload}``.  We normalize to
        a flat dict with ``type`` set to a role-like value for downstream
        compatibility, and ``_codex_type`` preserving the original type.
        """
        if not line or not line.strip():
            return None
        try:
            result = json.loads(line)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        """Parse Codex JSONL entries into AgentMessages.

        Maps ``response_item`` with ``role=assistant`` to assistant messages
        and ``role=user`` (non-system) to user messages.  Skips developer
        (system prompt) and event_msg entries.
        """
        messages: list[AgentMessage] = []
        pending = dict(pending_tools)

        for entry in entries:
            entry_type = entry.get("type", "")
            payload = entry.get("payload", {})

            if entry_type == "response_item":
                role = payload.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                content = payload.get("content", "")
                text, content_type, pending = _extract_codex_content(content, pending)
                if text:
                    messages.append(
                        AgentMessage(
                            text=text,
                            role=cast(MessageRole, role),
                            content_type=content_type,
                        )
                    )
            elif entry_type == "input_item":
                role = payload.get("role", "")
                if role != "user":
                    continue
                content = payload.get("content", "")
                if isinstance(content, str) and content:
                    messages.append(
                        AgentMessage(text=content, role="user", content_type="text")
                    )

        return messages, pending

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        """Check if this Codex entry is a human turn."""
        entry_type = entry.get("type", "")
        payload = entry.get("payload", {})
        if entry_type == "response_item" and payload.get("role") == "user":
            # Skip system/developer messages that look like user
            content = payload.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "input_text":
                        text = block.get("text", "")
                        if text.startswith(("<permissions", "<environment_context")):
                            return False
            return True
        return entry_type == "input_item" and payload.get("role") == "user"

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        """Parse a single Codex transcript entry for history display."""
        entry_type = entry.get("type", "")
        payload = entry.get("payload", {})

        if entry_type == "response_item":
            role = payload.get("role", "")
            if role not in ("user", "assistant"):
                return None
            content = payload.get("content", "")
            text, _, _ = _extract_codex_content(content, {})
            if not text:
                return None
            return AgentMessage(
                text=text,
                role=cast(MessageRole, role),
                content_type="text",
            )
        if entry_type == "input_item" and payload.get("role") == "user":
            content = payload.get("content", "")
            text = content if isinstance(content, str) else ""
            if not text:
                return None
            return AgentMessage(text=text, role="user", content_type="text")

        return None

    def discover_transcript(
        self, cwd: str, window_key: str
    ) -> SessionStartEvent | None:
        """Scan ~/.codex/sessions/ for the most recent transcript matching cwd.

        Codex transcript path: ~/.codex/sessions/YYYY/MM/DD/<name>-<ts>-<uuid>.jsonl
        First line: {"type": "session_meta", "payload": {"id": "<uuid>", "cwd": "..."}}
        """
        sessions_dir = Path.home() / ".codex" / "sessions"
        if not sessions_dir.is_dir():
            return None

        import time

        jsonl_files = _collect_codex_sessions(sessions_dir)
        now = time.time()
        resolved_cwd = str(Path(cwd).resolve())
        for mtime, fpath in jsonl_files[:20]:
            if now - mtime > _TRANSCRIPT_MAX_AGE_SECS:
                break  # sorted newest-first; remaining are all older
            meta = _read_codex_session_meta(fpath)
            if not meta:
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

    def parse_hook_payload(
        self,
        payload: dict[str, Any],  # noqa: ARG002 — protocol signature
    ) -> SessionStartEvent | None:
        return None
