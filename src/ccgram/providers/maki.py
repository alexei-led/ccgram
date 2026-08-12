from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar, cast

from ccgram.expandable_quote import format_expandable_quote
from ccgram.providers._jsonl import JsonlProvider, parse_jsonl_line
from ccgram.providers.base import (
    AgentMessage,
    MessageRole,
    ProviderCapabilities,
    RESUME_ID_RE,
    ResumableSession,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.tool_format import format_tool_line

_TOOL_RESULT_QUOTE_THRESHOLD = 3
_MAKI_BUILTINS: dict[str, str] = {
    "/tasks": "Browse and search tasks",
    "/compact": "Summarize and compact conversation history",
    "/new": "Start a new session",
    "/help": "Show keybindings",
    "/usage": "Show token usage breakdown",
    "/queue": "Remove items from queue",
    "/model": "Switch model",
    "/theme": "Switch color theme",
    "/mcp": "Configure MCP servers",
    "/login": "Authenticate with an LLM provider",
    "/cd": "Change working directory",
    "/btw": "Ask a quick question",
    "/yolo": "Toggle YOLO mode",
    "/thinking": "Toggle extended thinking",
    "/fast": "Toggle fast mode",
    "/workflow": "Toggle workflow mode",
    "/exit": "Exit the application",
    "/reload": "Reload plugins and config",
    "/memory": "View, edit, and delete memory files",
    "/rename": "Rename the current session",
    "/sessions": "Browse and switch sessions",
    "/skill": "Insert a skill prompt",
}
_TUI_PICKERS = frozenset(
    {
        "tasks",
        "model",
        "theme",
        "mcp",
        "login",
        "memory",
        "sessions",
        "skill",
    }
)


def _maki_state_dirs() -> list[Path]:
    candidates = [
        Path.home() / ".maki",
        Path.home() / ".local" / "state" / "maki",
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen and candidate.exists():
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _maki_sessions_dirs() -> list[Path]:
    return [
        state_dir / "sessions"
        for state_dir in _maki_state_dirs()
        if (state_dir / "sessions").is_dir()
    ]


def _normalize_cwd(cwd: str | None) -> str | None:
    if not cwd:
        return None
    try:
        return str(Path(cwd).expanduser().resolve())
    except OSError:
        return None


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _first_nonempty_string(data: dict[str, Any]) -> str:
    for value in data.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _summarize_tool_input(tool_name: str, payload: dict[str, Any]) -> str:
    if tool_name == "bash":
        command = payload.get("command")
        return command if isinstance(command, str) else ""
    if tool_name in {"read", "write", "edit", "multiedit"}:
        path = payload.get("path")
        return path if isinstance(path, str) else ""
    if tool_name == "glob":
        pattern = payload.get("pattern")
        return pattern if isinstance(pattern, str) else ""
    if tool_name == "grep":
        pattern = payload.get("pattern")
        return pattern if isinstance(pattern, str) else ""
    if tool_name == "task":
        return _first_nonempty_string(payload)
    if tool_name == "todo_write":
        todos = payload.get("todos")
        if isinstance(todos, list):
            return f"{len(todos)} task(s)"
    if tool_name == "memory":
        command = payload.get("command")
        return command if isinstance(command, str) else ""
    return _first_nonempty_string(payload)


def _format_tool_result_text(text: str) -> str:
    if not text:
        return "Done"
    line_count = text.count("\n") + 1
    if line_count > _TOOL_RESULT_QUOTE_THRESHOLD:
        return f"  ⏿  {line_count} lines\n" + format_expandable_quote(text)
    return text


def _load_session_header(path: Path) -> tuple[str, str, str] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline()
    except OSError:
        return None
    header = parse_jsonl_line(first)
    if not isinstance(header, dict) or header.get("t") != "header":
        return None
    session_id = header.get("id")
    cwd = header.get("cwd")
    model = header.get("model")
    if not isinstance(session_id, str) or not isinstance(cwd, str):
        return None
    return session_id, cwd, model if isinstance(model, str) else ""


def _iter_session_files() -> list[Path]:
    files: list[Path] = []
    for sessions_dir in _maki_sessions_dirs():
        try:
            files.extend(
                path
                for path in sessions_dir.iterdir()
                if path.suffix == ".jsonl" and path.is_file()
            )
        except OSError:
            continue
    return files


def _message_title_text(entry: dict[str, Any]) -> str:
    if entry.get("t") != "msg":
        return ""
    payload = entry.get("d")
    if not isinstance(payload, dict) or payload.get("role") != "user":
        return ""
    text = _extract_text(payload.get("content"))
    return " ".join(text.split())[:80]


def _session_summary(path: Path, session_id: str) -> str:
    summary = session_id[:12]
    try:
        with path.open("r", encoding="utf-8") as f:
            _ = f.readline()
            for _idx, line in zip(range(8), f):
                entry = parse_jsonl_line(line)
                if not isinstance(entry, dict):
                    continue
                title = _message_title_text(entry)
                if title:
                    return title
    except OSError:
        return summary
    return summary


def _append_text_message(
    messages: list[AgentMessage],
    role: MessageRole,
    text: str,
) -> None:
    cleaned = text.strip()
    if not cleaned:
        return
    messages.append(
        AgentMessage(
            text=cleaned,
            role=role,
            content_type="text",
        )
    )


def _append_tool_use_message(
    messages: list[AgentMessage],
    pending: dict[str, Any],
    *,
    tool_id: Any,
    tool_name: Any,
    tool_input: Any,
) -> None:
    if not isinstance(tool_id, str) or not isinstance(tool_name, str):
        return
    pending[tool_id] = tool_name
    summary = _summarize_tool_input(tool_name, tool_input) if isinstance(tool_input, dict) else ""
    messages.append(
        AgentMessage(
            text=format_tool_line(tool_name, summary),
            role="assistant",
            content_type="tool_use",
            tool_use_id=tool_id,
            tool_name=tool_name,
        )
    )


def _append_tool_result_message(
    messages: list[AgentMessage],
    *,
    text: str,
    tool_use_id: str | None,
    tool_name: str | None,
) -> None:
    if not text and not tool_name:
        return
    messages.append(
        AgentMessage(
            text=_format_tool_result_text(text),
            role="user",
            content_type="tool_result",
            tool_use_id=tool_use_id,
            tool_name=tool_name,
        )
    )


def _parse_out_payload(payload: dict[str, Any]) -> str:
    if "Plain" in payload and isinstance(payload["Plain"], dict):
        plain = payload["Plain"].get("text")
        return plain if isinstance(plain, str) else ""
    if "Diff" not in payload or not isinstance(payload["Diff"], dict):
        return json.dumps(payload, ensure_ascii=False)
    diff = payload["Diff"]
    summary = diff.get("summary")
    path = diff.get("path")
    parts = [
        part
        for part in (
            summary if isinstance(summary, str) else "",
            path if isinstance(path, str) else "",
        )
        if part
    ]
    return " — ".join(parts)


class MakiProvider(JsonlProvider):
    _CAPS: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        name="maki",
        launch_command="maki",
        supports_hook=False,
        supports_resume=True,
        supports_resume_picker=True,
        supports_continue=True,
        supports_structured_transcript=True,
        supports_incremental_read=True,
        supports_user_command_discovery=False,
        supports_status_snapshot=False,
        has_yolo_confirmation=True,
        supports_task_tracking=False,
        builtin_commands=tuple(sorted(_MAKI_BUILTINS.keys())),
        tui_picker_commands=_TUI_PICKERS,
    )
    _BUILTINS = _MAKI_BUILTINS

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._CAPS

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

    def parse_transcript_line(self, line: str) -> dict[str, Any] | None:
        return parse_jsonl_line(line)

    def discover_resumable_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[ResumableSession]:
        target_cwd = _normalize_cwd(cwd)
        sessions: list[tuple[float, ResumableSession]] = []
        seen_ids: set[str] = set()
        for path in _iter_session_files():
            session = self._build_resumable_session(path, target_cwd, seen_ids)
            if session is not None:
                sessions.append(session)
        sessions.sort(key=lambda item: item[0], reverse=True)
        result = [session for _, session in sessions]
        return result[:limit] if limit is not None else result

    def _build_resumable_session(
        self,
        path: Path,
        target_cwd: str | None,
        seen_ids: set[str],
    ) -> tuple[float, ResumableSession] | None:
        header = _load_session_header(path)
        if header is None:
            return None
        session_id, session_cwd, _model = header
        resolved_cwd = _normalize_cwd(session_cwd)
        if resolved_cwd is None or (target_cwd and resolved_cwd != target_cwd):
            return None
        if session_id in seen_ids:
            return None
        seen_ids.add(session_id)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        return (
            mtime,
            ResumableSession(
                session_id=session_id,
                summary=_session_summary(path, session_id),
                cwd=resolved_cwd,
                provider_name="maki",
                mtime=mtime,
            ),
        )

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        target_cwd = _normalize_cwd(cwd)
        if target_cwd is None:
            return None
        best: tuple[float, Path, str] | None = None
        now = time.time()
        age_limit = max_age if max_age is not None else 120.0
        for path in _iter_session_files():
            header = _load_session_header(path)
            if header is None:
                continue
            session_id, session_cwd, _model = header
            resolved_cwd = _normalize_cwd(session_cwd)
            if resolved_cwd != target_cwd:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if age_limit > 0 and now - mtime > age_limit:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, path, session_id)
        if best is None:
            return None
        _, transcript_path, session_id = best
        return SessionStartEvent(
            session_id=session_id,
            cwd=target_cwd,
            transcript_path=str(transcript_path),
            window_key=window_key,
        )

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
        cwd: str | None = None,
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        del cwd
        messages: list[AgentMessage] = []
        pending = dict(pending_tools)
        for entry in entries:
            kind = entry.get("t")
            if kind == "msg":
                self._parse_message_entry(entry, messages, pending)
                continue
            if kind == "out":
                self._parse_output_entry(entry, messages, pending)
        return messages, pending

    def _parse_message_entry(
        self,
        entry: dict[str, Any],
        messages: list[AgentMessage],
        pending: dict[str, Any],
    ) -> None:
        payload = entry.get("d")
        if not isinstance(payload, dict):
            return
        role = payload.get("role")
        if role not in ("user", "assistant"):
            return
        typed_role = cast(MessageRole, role)
        content = payload.get("content")
        if isinstance(content, str):
            _append_text_message(messages, typed_role, content)
            return
        if not isinstance(content, list):
            return
        for block in content:
            self._parse_message_block(block, typed_role, messages, pending)

    def _parse_message_block(
        self,
        block: Any,
        role: MessageRole,
        messages: list[AgentMessage],
        pending: dict[str, Any],
    ) -> None:
        if not isinstance(block, dict):
            return
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                _append_text_message(messages, role, text)
            return
        if block_type == "tool_use" and role == "assistant":
            _append_tool_use_message(
                messages,
                pending,
                tool_id=block.get("id"),
                tool_name=block.get("name"),
                tool_input=block.get("input"),
            )
            return
        if block_type != "tool_result":
            return
        tool_use_id = block.get("tool_use_id")
        raw_text = _extract_text(block.get("content"))
        if not raw_text and isinstance(block.get("content"), str):
            raw_text = block["content"]
        tool_name = pending.pop(tool_use_id, None) if isinstance(tool_use_id, str) else None
        _append_tool_result_message(
            messages,
            text=raw_text,
            tool_use_id=tool_use_id if isinstance(tool_use_id, str) else None,
            tool_name=tool_name if isinstance(tool_name, str) else None,
        )

    def _parse_output_entry(
        self,
        entry: dict[str, Any],
        messages: list[AgentMessage],
        pending: dict[str, Any],
    ) -> None:
        tool_id = entry.get("id")
        payload = entry.get("d")
        tool_use_id = tool_id if isinstance(tool_id, str) else None
        tool_name = pending.pop(tool_use_id, None) if tool_use_id else None
        text = _parse_out_payload(payload) if isinstance(payload, dict) else ""
        _append_tool_result_message(
            messages,
            text=text,
            tool_use_id=tool_use_id,
            tool_name=tool_name if isinstance(tool_name, str) else None,
        )

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        return entry.get("t") == "msg" and isinstance(entry.get("d"), dict) and entry["d"].get("role") == "user"

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        kind = entry.get("t")
        if kind != "msg":
            return None
        payload = entry.get("d")
        if not isinstance(payload, dict):
            return None
        role = payload.get("role")
        if role not in ("user", "assistant"):
            return None
        text = _extract_text(payload.get("content"))
        if not text:
            return None
        return AgentMessage(
            text=text,
            role=cast(MessageRole, role),
            content_type="text",
        )

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",
    ) -> StatusUpdate | None:
        del pane_title
        text = pane_text.strip()
        if not text:
            return None
        lowered = text.lower()
        if "/sessions" in text or "/model" in text or "/memory" in text:
            marker = "/sessions" if "/sessions" in text else "/model" if "/model" in text else "/memory"
            return StatusUpdate(raw_text=text, display_label=marker, is_interactive=True, ui_type=marker)
        if "waiting for your response" in lowered or "press enter to" in lowered:
            return StatusUpdate(raw_text=text, display_label="needs input")
        return None

    def build_status_snapshot(
        self,
        transcript_path: str,
        *,
        display_name: str = "",
        session_id: str = "",
        cwd: str = "",
    ) -> str | None:
        try:
            size = os.path.getsize(transcript_path)
        except OSError:
            return None
        return (
            f"🪶 [{display_name}] Maki session active.\n"
            f"📁 `{cwd}`\n"
            f"📄 `{os.path.basename(transcript_path)}` ({size} bytes)\n"
            f"⭐ ID: `{session_id[:8]}`"
        )

    def has_output_since(self, transcript_path: str, offset: int) -> bool:
        try:
            return os.path.getsize(transcript_path) > offset
        except OSError:
            return False
