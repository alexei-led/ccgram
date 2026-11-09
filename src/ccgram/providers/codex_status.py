"""Codex status snapshot helpers.

Builds a Telegram-friendly status message from Codex JSONL transcripts.
Codex slash commands like ``/status`` may not emit assistant transcript
messages; this snapshot provides a reliable fallback.
"""

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# /status must not parse a whole rollout: Codex rollouts reach hundreds of MB.
# The snapshot reads session_meta from the head, the latest token_count and
# timestamp from a bounded tail, and counts lines with a byte scan. Upgrade
# trigger: if one turn pushes the last token_count past the tail window, the
# snapshot shows "token stats: unavailable"; raise _TAIL_MAX_BYTES then.
# session_meta shares the cap: a longer first line reads as "unknown".
_TAIL_MAX_BYTES = 8 * 1024 * 1024
_READ_BLOCK_BYTES = 1024 * 1024
_HEAD_MAX_BYTES = _TAIL_MAX_BYTES


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


def _count_lines(path: Path) -> int:
    """Count transcript lines with a byte scan; no JSON parsing.

    Matches the parsed-entry count for well-formed rollouts. Blank or
    unparsable lines are counted too, which the old full parse skipped.
    """
    count = 0
    last = b""
    with path.open("rb") as handle:
        while block := handle.read(_READ_BLOCK_BYTES):
            count += block.count(b"\n")
            last = block[-1:]
    return count + (1 if last and last != b"\n" else 0)


def _read_session_meta(path: Path) -> dict[str, Any]:
    """Return the first-line ``session_meta`` payload, as the provider reads it."""
    with path.open("rb") as handle:
        first = handle.readline(_HEAD_MAX_BYTES)
    entry = _parse_json_object(first.decode("utf-8", errors="replace"))
    if entry is None or entry.get("type") != "session_meta":
        return {}
    return _as_dict(entry.get("payload"))


def _read_tail_state(
    path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], bool]:
    """Find the newest timestamp, token info and rate limits, scanning backwards.

    ``info`` and ``rate_limits`` are independent: Codex can write a
    ``token_count`` with ``info: null`` that still carries fresh limits. The
    limits come from the newest ``token_count`` (none when it has none, so an
    older, possibly reset window is never shown as current); ``info`` comes
    from the newest event that has it. Reads blocks from the end until the
    timestamp and info are found or ``_TAIL_MAX_BYTES`` of the file were
    scanned. Returns ``(timestamp, token_info, rate_limits, saw_entry)``.
    """
    last_timestamp = ""
    token_info: dict[str, Any] = {}
    rate_limits: dict[str, Any] = {}
    seen_token_count = False
    saw_entry = False
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        floor = max(0, size - _TAIL_MAX_BYTES)
        pos = size
        carry = b""
        while pos > floor and not (last_timestamp and token_info):
            start = max(floor, pos - _READ_BLOCK_BYTES)
            handle.seek(start)
            lines = (handle.read(pos - start) + carry).split(b"\n")
            pos = start
            # Unless the file start is in view, the first piece may be cut mid-line.
            carry = lines.pop(0) if pos > 0 else b""
            for raw in reversed(lines):
                entry = _parse_json_object(raw.decode("utf-8", errors="replace"))
                if entry is None:
                    continue
                saw_entry = True
                timestamp = entry.get("timestamp")
                if not last_timestamp and isinstance(timestamp, str) and timestamp:
                    last_timestamp = timestamp
                payload = _as_dict(entry.get("payload"))
                if entry.get("type") == "event_msg" and payload.get("type") == (
                    "token_count"
                ):
                    if not seen_token_count:
                        seen_token_count = True
                        rate_limits = _as_dict(payload.get("rate_limits"))
                    token_info = token_info or _as_dict(payload.get("info"))
                if last_timestamp and token_info:
                    break
    return last_timestamp, token_info, rate_limits, saw_entry


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


def _format_token_lines(token_info: dict[str, Any]) -> list[str]:
    """Format token totals and the current context-window use."""
    total_usage = _as_dict(token_info.get("total_token_usage"))
    if not total_usage:
        return ["- token stats: unavailable (no recent `token_count` event)"]

    lines = [
        "- tokens (total): "
        f"in `{_fmt_int(total_usage.get('input_tokens'))}`, "
        f"cached `{_fmt_int(total_usage.get('cached_input_tokens'))}`, "
        f"out `{_fmt_int(total_usage.get('output_tokens'))}`, "
        f"reason `{_fmt_int(total_usage.get('reasoning_output_tokens'))}`, "
        f"total `{_fmt_int(total_usage.get('total_tokens'))}`"
    ]

    # The window holds the last request's context, not the session's running
    # total, which passes the window on long sessions.
    context_tokens = _as_int(
        _as_dict(token_info.get("last_token_usage")).get("total_tokens")
    )
    context_window = _as_int(token_info.get("model_context_window"))
    if context_tokens is not None and context_window is not None and context_window > 0:
        usage_pct = (context_tokens / context_window) * 100
        lines.append(
            "- context window: "
            f"`{context_tokens:,}` / `{context_window:,}` ({usage_pct:.1f}%)"
        )
    return lines


_MINUTES_PER_HOUR = 60
_MINUTES_PER_DAY = 24 * _MINUTES_PER_HOUR
_MINUTES_PER_WEEK = 7 * _MINUTES_PER_DAY


def _limit_label(limit: dict[str, Any], fallback: str) -> str:
    """Name a rate-limit window from ``window_minutes``, else *fallback*."""
    minutes = _as_int(limit.get("window_minutes"))
    if minutes is None or minutes <= 0:
        return fallback
    if minutes == _MINUTES_PER_WEEK:
        return "weekly"
    if minutes % _MINUTES_PER_DAY == 0:
        return f"{minutes // _MINUTES_PER_DAY}d"
    if minutes % _MINUTES_PER_HOUR == 0:
        return f"{minutes // _MINUTES_PER_HOUR}h"
    return f"{minutes}m"


def _format_rate_limit_lines(rate_limits: dict[str, Any]) -> list[str]:
    """Format one line per rate-limit window present."""
    lines = []
    for key in ("primary", "secondary"):
        limit = _as_dict(rate_limits.get(key))
        if limit:
            lines.append(
                f"- {_limit_label(limit, key)} limit: "
                f"`{'?' if limit.get('used_percent') is None else limit['used_percent']}%` used, "
                f"reset `{_fmt_epoch_utc(limit.get('resets_at'))}`"
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

    try:
        entry_count = _count_lines(path)
        session_meta = _read_session_meta(path)
        last_timestamp, token_info, rate_limits, saw_entry = _read_tail_state(path)
    except OSError:
        return None

    if not (saw_entry or session_meta):
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
    lines.extend(_format_token_lines(token_info))
    lines.extend(_format_rate_limit_lines(rate_limits))

    lines.append(
        "_Note: Codex `/status` may not emit transcript messages; this is a transcript snapshot._"
    )
    return "\n".join(lines)
