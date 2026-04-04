"""LLM-powered completion summary for agent sessions.

Reads recent transcript entries and produces a single-line summary of what
the agent accomplished. Used by the Stop hook to enhance the "Ready" message.
Returns None gracefully when LLM is not configured or on any failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = """\
You are a development assistant summarizing what a coding agent accomplished.
Given the recent activity log, write a single-line summary (max 120 chars).
Be specific: mention file names, test counts, command outcomes.
Examples:
- "Fixed auth bug in login.py, all 23 tests pass"
- "Added 3 API endpoints in src/api/, updated OpenAPI spec"
- "Refactored database module — 2 tests failing (test_connection, test_pool)"
Return ONLY the summary line, no quotes or formatting."""

_MAX_ENTRIES = 30
_MAX_RESULT_CHARS = 200
_MAX_ASSISTANT_CHARS = 500


def _read_tail_lines(path: Path, max_lines: int) -> list[str]:
    """Read the last N lines from a file via reverse seek. Runs in a thread."""
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        # Read a chunk from the end — 200 bytes per line is generous for JSONL
        chunk_size = min(size, max_lines * 200)
        with path.open("rb") as f:
            f.seek(max(0, size - chunk_size))
            tail = f.read().decode(errors="replace")
        lines = tail.splitlines()
        # Drop the first partial line if we didn't read from the start
        if chunk_size < size:
            lines = lines[1:]
        return lines[-max_lines:]
    except OSError:
        return []


def _extract_tool_summary(block: dict) -> str | None:
    """Extract a compact tool summary from a tool_use content block."""
    name = block.get("name", "")
    input_data = block.get("input", {})
    if not isinstance(input_data, dict):
        return f"{name}"

    if name in ("Read", "Glob"):
        return f"{name} {input_data.get('file_path') or input_data.get('pattern', '')}"
    if name == "Edit":
        return f"Edit {input_data.get('file_path', '')}"
    if name == "Write":
        return f"Write {input_data.get('file_path', '')}"
    if name == "Bash":
        cmd = str(input_data.get("command", ""))[:100]
        return f"Bash: {cmd}"
    if name == "Grep":
        return f"Grep {input_data.get('pattern', '')}"
    if name in ("TaskCreate", "TaskUpdate"):
        return f"{name} {input_data.get('subject', '')}"
    return f"{name}"


def _extract_result_snippet(block: dict) -> str:
    """Extract a short snippet from a tool_result content block."""
    content = block.get("content", "")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        content = "\n".join(parts)
    if not isinstance(content, str):
        content = str(content)
    return content[:_MAX_RESULT_CHARS]


def _parse_entry(raw_line: str) -> tuple[str, list[dict]] | None:
    """Parse a JSONL line into (entry_type, content_blocks) or None."""
    raw_line = raw_line.strip()
    if not raw_line:
        return None
    try:
        entry = json.loads(raw_line)
    except json.JSONDecodeError:
        return None

    entry_type = entry.get("type")
    message = entry.get("message", {})
    if not isinstance(message, dict):
        return None
    content_blocks = message.get("content", [])
    if not isinstance(content_blocks, list):
        return None
    return entry_type, [b for b in content_blocks if isinstance(b, dict)]


def _process_assistant_blocks(blocks: list[dict], context_parts: list[str]) -> str:
    """Process assistant content blocks, return last text seen."""
    last_text = ""
    for block in blocks:
        btype = block.get("type")
        if btype == "tool_use":
            summary = _extract_tool_summary(block)
            if summary:
                context_parts.append(f"\u2192 {summary}")
        elif btype == "text":
            text = block.get("text", "")
            if text:
                last_text = text
    return last_text


def _process_user_blocks(blocks: list[dict], context_parts: list[str]) -> None:
    """Process user content blocks (tool results)."""
    for block in blocks:
        if block.get("type") == "tool_result":
            snippet = _extract_result_snippet(block)
            if snippet:
                first_line = snippet.split("\n", 1)[0][:120]
                context_parts.append(f"  = {first_line}")


def _build_summary_context(lines: list[str]) -> str:
    """Build a compact context string from JSONL transcript lines."""
    context_parts: list[str] = []
    last_assistant_text = ""

    for line in lines:
        parsed = _parse_entry(line)
        if parsed is None:
            continue
        entry_type, blocks = parsed

        if entry_type == "assistant":
            text = _process_assistant_blocks(blocks, context_parts)
            if text:
                last_assistant_text = text
        elif entry_type == "user":
            _process_user_blocks(blocks, context_parts)

    if last_assistant_text:
        trimmed = last_assistant_text[-_MAX_ASSISTANT_CHARS:]
        context_parts.append(f"\nFinal response:\n{trimmed}")

    return "\n".join(context_parts)


async def summarize_completion(transcript_path: str) -> str | None:
    """Produce a single-line completion summary via the configured LLM.

    Returns None if:
    - transcript_path is empty or file doesn't exist
    - LLM is not configured
    - LLM call fails for any reason
    """
    if not transcript_path:
        return None

    path = Path(transcript_path)
    if not path.exists():
        return None

    from . import get_text_completer

    completer = get_text_completer()
    if completer is None:
        return None

    try:
        lines = await asyncio.to_thread(_read_tail_lines, path, _MAX_ENTRIES)
        if not lines:
            return None

        context = _build_summary_context(lines)
        if not context.strip():
            return None

        result = await completer.complete(_SUMMARY_SYSTEM_PROMPT, context)
        return result.strip()[:150] if result else None
    except RuntimeError:
        logger.warning("LLM summary failed", exc_info=True)
        return None
