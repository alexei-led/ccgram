"""Codex status snapshot helpers.

Builds a Telegram-friendly status message from Codex JSONL transcripts.
Codex slash commands like ``/status`` may not emit assistant transcript
messages; this snapshot provides a reliable fallback.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    """Return *value* as dict when possible."""
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int | None:
    """Convert ints/floats to int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _fmt_int(value: Any) -> str:
    """Format a numeric value with grouping, fallback to '?'."""
    parsed = _as_int(value)
    return f"{parsed:,}" if parsed is not None else "?"


def _fmt_epoch_utc(value: Any) -> str:
    """Format UNIX epoch seconds as UTC timestamp."""
    parsed = _as_int(value)
    if parsed is None:
        return "?"
    return datetime.fromtimestamp(parsed, UTC).strftime("%Y-%m-%d %H:%M UTC")


def _display_cwd(cwd: str) -> str:
    """Compress home path for display."""
    home = str(Path.home())
    return cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd


def _parse_json_object(raw_line: str) -> dict[str, Any] | None:
    """Parse one JSON object line."""
    line = raw_line.strip()
    if not line:
        return None
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None
    return entry if isinstance(entry, dict) else None


def _iter_json_entries(
    path: Path,
    *,
    start_offset: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON object entries from transcript."""
    with path.open(encoding="utf-8") as handle:
        if start_offset > 0:
            handle.seek(start_offset)
        for raw_line in handle:
            parsed = _parse_json_object(raw_line)
            if parsed is not None:
                yield parsed


def _entry_has_assistant_output(entry: dict[str, Any]) -> bool:
    """True when entry contains assistant-visible output."""
    entry_type = entry.get("type")
    payload = _as_dict(entry.get("payload"))

    if entry_type == "event_msg" and payload.get("type") == "agent_message":
        message = payload.get("message")
        return isinstance(message, str) and bool(message.strip())

    if entry_type != "response_item":
        return False
    if payload.get("type") != "message" or payload.get("role") != "assistant":
        return False

    content = payload.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "output_text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            return True
    return False


def has_codex_assistant_output_since(
    transcript_path: str,
    offset: int,
) -> bool:
    """Check whether transcript contains assistant output after offset."""
    path = Path(transcript_path)
    if not path.exists():
        return False
    try:
        for entry in _iter_json_entries(path, start_offset=offset):
            if _entry_has_assistant_output(entry):
                return True
    except OSError:
        return False
    return False


def _format_token_lines(last_token_info: dict[str, Any]) -> list[str]:
    """Format token and rate-limit lines."""
    total_usage = _as_dict(last_token_info.get("total_token_usage"))
    if not total_usage:
        return ["- token stats: unavailable (no `token_count` event yet)"]

    lines = [
        "- tokens (total): "
        f"in `{_fmt_int(total_usage.get('input_tokens'))}`, "
        f"cached `{_fmt_int(total_usage.get('cached_input_tokens'))}`, "
        f"out `{_fmt_int(total_usage.get('output_tokens'))}`, "
        f"reason `{_fmt_int(total_usage.get('reasoning_output_tokens'))}`, "
        f"total `{_fmt_int(total_usage.get('total_tokens'))}`"
    ]

    total_tokens = _as_int(total_usage.get("total_tokens"))
    context_window = _as_int(last_token_info.get("model_context_window"))
    if total_tokens is not None and context_window is not None and context_window > 0:
        usage_pct = (total_tokens / context_window) * 100
        lines.append(
            "- context window: "
            f"`{total_tokens:,}` / `{context_window:,}` ({usage_pct:.1f}%)"
        )

    rate_limits = _as_dict(last_token_info.get("rate_limits"))
    primary = _as_dict(rate_limits.get("primary"))
    secondary = _as_dict(rate_limits.get("secondary"))
    if primary:
        lines.append(
            "- primary limit: "
            f"`{primary.get('used_percent', '?')}%` used, "
            f"reset `{_fmt_epoch_utc(primary.get('resets_at'))}`"
        )
    if secondary:
        lines.append(
            "- secondary limit: "
            f"`{secondary.get('used_percent', '?')}%` used, "
            f"reset `{_fmt_epoch_utc(secondary.get('resets_at'))}`"
        )
    return lines


def build_codex_status_snapshot(
    transcript_path: str,
    *,
    display_name: str,
    session_id: str = "",
    cwd: str = "",
) -> str | None:
    """Build a status snapshot from a Codex transcript.

    Returns None when the transcript cannot be read or has no JSON entries.
    """
    path = Path(transcript_path)
    if not path.exists():
        return None

    entry_count = 0
    last_timestamp = ""
    session_meta: dict[str, Any] = {}
    last_token_info: dict[str, Any] = {}

    try:
        for entry in _iter_json_entries(path):
            entry_count += 1
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, str) and timestamp:
                last_timestamp = timestamp

            payload = _as_dict(entry.get("payload"))
            entry_type = entry.get("type")
            if entry_type == "session_meta" and payload:
                session_meta = payload
                continue
            if entry_type == "event_msg" and payload.get("type") == "token_count":
                info = _as_dict(payload.get("info"))
                if info:
                    last_token_info = info
    except OSError:
        return None

    if entry_count == 0:
        return None

    resolved_session = (
        session_id
        or (session_meta.get("id") if isinstance(session_meta.get("id"), str) else "")
        or "unknown"
    )
    resolved_cwd = (
        cwd
        or (session_meta.get("cwd") if isinstance(session_meta.get("cwd"), str) else "")
        or "unknown"
    )
    cli_version = (
        session_meta.get("cli_version")
        if isinstance(session_meta.get("cli_version"), str)
        else "unknown"
    )

    lines = [
        f"[{display_name}] Codex status snapshot",
        f"- session: `{resolved_session}`",
        f"- cwd: `{_display_cwd(resolved_cwd)}`",
        f"- cli: `{cli_version}`",
        f"- transcript entries: `{entry_count:,}`",
    ]
    if last_timestamp:
        lines.append(f"- last transcript event: `{last_timestamp}`")
    lines.extend(_format_token_lines(last_token_info))

    lines.append(
        "_Note: Codex `/status` may not emit transcript messages; this is a transcript snapshot._"
    )
    return "\n".join(lines)
