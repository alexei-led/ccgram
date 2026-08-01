"""Kimi Code transcript formatting — parse kimi ``wire.jsonl`` into AgentMessages.

Kimi Code records each session as a newline-delimited "wire" log at::

    ~/.kimi-code/sessions/<workspace_id>/<session_id>/agents/main/wire.jsonl

Every line is one record discriminated by ``type``.  The records that carry
conversation content are:

``turn.prompt``
    A human turn — ``{"input": [{"type": "text", "text": ...}],
    "origin": {"kind": "user"}, "time": <epoch_ms>}``.

``context.append_loop_event``
    Wraps the agent loop under ``event.type``:
      - ``content.part`` — ``part.type`` is ``text`` (assistant prose) or
        ``think`` (reasoning)
      - ``tool.call`` — ``uuid``/``toolCallId``, ``name``, ``args``,
        ``description``
      - ``tool.result`` — ``parentUuid``/``toolCallId``, ``result.output``
        (plus an optional ``result.note`` ``<system>`` annotation)
      - ``step.begin`` / ``step.end`` — turn bookkeeping, no relay output

``context.append_message`` re-appends the same user turn *and* injected system
reminders to the model context, so it is deliberately ignored: relaying it
would duplicate every prompt and leak ``<system-reminder>`` text into Telegram.

Timestamps are epoch milliseconds; they are normalised to ISO-8601 UTC so
history rendering matches the other providers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ccgram.expandable_quote import format_expandable_quote
from ccgram.providers.base import AgentMessage, ContentType
from ccgram.tool_format import format_tool_line

# Record type discriminators.
TURN_PROMPT = "turn.prompt"
LOOP_EVENT = "context.append_loop_event"
APPEND_MESSAGE = "context.append_message"
METADATA = "metadata"

# ``event.type`` values inside a ``context.append_loop_event`` record.
EVENT_CONTENT_PART = "content.part"
EVENT_TOOL_CALL = "tool.call"
EVENT_TOOL_RESULT = "tool.result"

# Outputs longer than this render as a line count + expandable quote.
_TOOL_RESULT_QUOTE_THRESHOLD = 3

# Pending value: toolCallId -> tool name.
Pending = dict[str, str]

# Milliseconds-since-epoch guard: kimi always emits ms, but a seconds-based
# value would silently render as 1970, so only convert plausible timestamps.
_MIN_EPOCH_MS = 10**11


def epoch_ms_to_iso(value: Any) -> str | None:
    """Convert kimi's epoch-millisecond ``time`` field to an ISO-8601 UTC string."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value < _MIN_EPOCH_MS:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()
    except OverflowError, OSError, ValueError:
        return None


def extract_text(content: Any) -> str:
    """Collect visible text from a kimi content field (string or block list)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


# Preferred ``args`` keys per tool, used to build a short call summary.
_TOOL_SUMMARY_ARG_KEYS: dict[str, tuple[str, ...]] = {
    "bash": ("command",),
    "read": ("path", "file_path"),
    "readmediafile": ("path", "file_path"),
    "write": ("path", "file_path"),
    "edit": ("path", "file_path"),
    "grep": ("pattern",),
    "glob": ("pattern",),
    "websearch": ("query",),
    "fetchurl": ("url",),
    "skill": ("name", "command"),
    "agent": ("description", "prompt"),
    "agentswarm": ("description", "prompt"),
    "askuserquestion": ("question",),
}


def _first_string_arg(args: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty string value among *keys*."""
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def tool_call_summary(name: str, args: dict[str, Any], description: str = "") -> str:
    """Build the one-line tool-call display for a kimi ``tool.call`` event.

    Prefers the tool's own argument (command / path / pattern) so the summary
    matches the other providers, then kimi's prewritten ``description``
    ("Running: ls -la"), then any string argument.
    """
    preferred = _first_string_arg(args, _TOOL_SUMMARY_ARG_KEYS.get(name.lower(), ()))
    if preferred:
        return format_tool_line(name, preferred)
    if description:
        return format_tool_line(name, description)
    for value in args.values():
        if isinstance(value, str) and value:
            return format_tool_line(name, value)
    return format_tool_line(name, "")


def format_tool_result_text(output: str) -> str:
    """Render long outputs as ``N lines`` + expandable quote, short ones inline."""
    if not output:
        return "Done"
    line_count = output.count("\n") + 1
    if line_count > _TOOL_RESULT_QUOTE_THRESHOLD:
        unit = "line" if line_count == 1 else "lines"
        return f"  ⎿  {line_count} {unit}\n" + format_expandable_quote(output)
    return output


def parse_turn_prompt(record: dict[str, Any]) -> AgentMessage | None:
    """Parse a ``turn.prompt`` record into the user's AgentMessage.

    Only ``origin.kind == "user"`` turns are relayed — kimi replays queued or
    synthesised prompts through the same record type.
    """
    origin = record.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "user"):
        return None
    text = extract_text(record.get("input", "")).strip()
    if not text:
        return None
    return AgentMessage(
        text=text,
        role="user",
        content_type="text",
        timestamp=epoch_ms_to_iso(record.get("time")),
    )


def parse_content_part(
    event: dict[str, Any], timestamp: str | None = None
) -> AgentMessage | None:
    """Parse a ``content.part`` loop event into assistant text or thinking."""
    part = event.get("part")
    if not isinstance(part, dict):
        return None
    part_type = part.get("type")
    content_type: ContentType
    if part_type == "text":
        text = part.get("text")
        content_type = "text"
    elif part_type == "think":
        text = part.get("think")
        content_type = "thinking"
    else:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return AgentMessage(
        text=text.strip(),
        role="assistant",
        content_type=content_type,
        timestamp=timestamp,
    )


def _call_id(event: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string id among *keys*."""
    for key in keys:
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def parse_tool_call(
    event: dict[str, Any], pending: Pending, timestamp: str | None = None
) -> AgentMessage | None:
    """Parse a ``tool.call`` loop event, recording the call in *pending*."""
    name = event.get("name")
    if not isinstance(name, str) or not name:
        name = "unknown"
    args = event.get("args")
    if not isinstance(args, dict):
        args = {}
    description = event.get("description")
    if not isinstance(description, str):
        description = ""

    call_id = _call_id(event, "toolCallId", "uuid")
    if call_id:
        pending[call_id] = name

    return AgentMessage(
        text=tool_call_summary(name, args, description),
        role="assistant",
        content_type="tool_use",
        tool_use_id=call_id or None,
        tool_name=name,
        timestamp=timestamp,
    )


def parse_tool_result(
    event: dict[str, Any], pending: Pending, timestamp: str | None = None
) -> AgentMessage | None:
    """Parse a ``tool.result`` loop event, pairing it back to its call.

    ``result.note`` is a ``<system>`` annotation kimi feeds back to the model
    (line counts, truncation hints); it is metadata, not output, so it is
    dropped rather than relayed.
    """
    call_id = _call_id(event, "toolCallId", "parentUuid")
    name = pending.pop(call_id, "") if call_id else ""
    if not name:
        native = event.get("toolName")
        name = native if isinstance(native, str) and native else "unknown"

    result = event.get("result")
    if not isinstance(result, dict):
        result = {}

    error = result.get("error")
    if isinstance(error, str) and error:
        text = f"Error: {error}"
    else:
        output = result.get("output")
        text = format_tool_result_text(output if isinstance(output, str) else "")

    return AgentMessage(
        text=text,
        role="assistant",
        content_type="tool_result",
        tool_use_id=call_id or None,
        tool_name=name,
        timestamp=timestamp,
    )


def parse_loop_event(record: dict[str, Any], pending: Pending) -> AgentMessage | None:
    """Dispatch one ``context.append_loop_event`` record to its parser."""
    event = record.get("event")
    if not isinstance(event, dict):
        return None
    timestamp = epoch_ms_to_iso(record.get("time"))
    event_type = event.get("type")
    if event_type == EVENT_CONTENT_PART:
        return parse_content_part(event, timestamp)
    if event_type == EVENT_TOOL_CALL:
        return parse_tool_call(event, pending, timestamp)
    if event_type == EVENT_TOOL_RESULT:
        return parse_tool_result(event, pending, timestamp)
    return None


def normalize_pending(value: Any) -> Pending:
    """Coerce the cross-batch pending dict into ``{call_id: tool_name}``."""
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }
