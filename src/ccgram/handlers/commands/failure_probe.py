"""Post-send command failure detection.

After a slash command is forwarded to the provider session, this module
probes for "command failed" signals from two sources:

  1. transcript delta — incremental read of the JSONL since the offset
     captured before the send, looking for assistant messages that
     match common "unknown command" / "unrecognized command" lines.
  2. pane delta — diff of capture_pane output before/after the send,
     used as a fallback when the provider has no transcript or the
     transcript has no error line.

If either source yields an error line, a one-shot Telegram reply is
posted in the topic.
"""

from __future__ import annotations


import asyncio
from pathlib import Path
import re

import structlog
from telegram import Message

from ...providers import AgentProvider
from ... import window_query
from ...multiplexer import multiplexer as tmux_manager
from ...utils import task_done_callback
from ..messaging_pipeline.message_sender import safe_reply

logger = structlog.get_logger()

_COMMAND_ERROR_PROBE_DELAY_SECONDS = 1.0
_COMMAND_ERROR_RE = re.compile(
    r"(?i)\b(?:"
    r"unrecognized command|"
    r"unknown command|"
    r"invalid command|"
    r"unsupported command|"
    r"no such command|"
    r"command not found|"
    r"not recognized"
    r")\b"
)


def _extract_probe_error_line(text: str, command: str | None = None) -> str | None:
    """First command-error line in *text* that names *command*, if given.

    The pane fallback diffs whole terminal captures, and a TUI redraw
    (skill expansion, spinner) breaks the line-overlap heuristic, so the
    delta can include scrollback. An error line that does not mention
    the dispatched command is a STALE error from an earlier attempt and
    must not fail the current one (2026-10-04: a successful
    /pa:research was reported failed off the leftover "Unknown command:
    /names" line).
    """
    stem = _command_stem(command)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not (_COMMAND_ERROR_RE.search(line) or (
            "error" in line.lower() and "command" in line.lower()
        )):
            continue
        # The line must name the dispatched command as a whole token:
        # anywhere-in-line substring matching would also match the
        # suggestion ("Did you mean /pa:research?") and let the
        # suggested command fail the suggested one. Accept any provider
        # phrasing (unknown / unrecognized / command not found, quoted
        # or slashed) before the name, and require a non-name boundary
        # after it so a shorter dispatched command cannot match a
        # longer colon-namespaced one (/spec vs /spec:work).
        if stem:
            escaped = re.escape(stem.lstrip("/"))
            pattern = (
                "(?i)(?:unknown|unrecognized) command[: ]+['/]*"
                + escaped
                + "(?![\\w:-])"
            )
            fallback = (
                "(?i)command not found[: ]+['/]*" + escaped + "(?![\\w:-])"
            )
            if not (re.search(pattern, line) or re.search(fallback, line)):
                continue
        return line
    return None


def _command_stem(command: str | None) -> str:
    """The bare command token ("/pa:research args" -> "pa:research")."""
    if not command:
        return ""
    return command.strip().split()[0].lstrip("/")


def _extract_pane_delta(before: str | None, after: str | None) -> str:
    """Return the likely newly-added pane text after a command send."""
    if not after:
        return ""
    if not before:
        return after
    if before == after:
        return ""

    before_lines = before.splitlines()
    after_lines = after.splitlines()
    max_overlap = min(len(before_lines), len(after_lines))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if before_lines[-size:] == after_lines[:size]:
            overlap = size
            break
    return "\n".join(after_lines[overlap:]).strip()


async def _capture_command_probe_context(
    window_id: str,
    provider: AgentProvider,
) -> tuple[str | None, int | None, str | None]:
    """Capture transcript offset + pane snapshot before sending a command."""
    view = window_query.view_window(window_id)
    transcript_path: str | None = (
        str(view.transcript_path) if view and view.transcript_path else None
    )
    since_offset: int | None = None
    if transcript_path:
        try:
            if provider.capabilities.supports_incremental_read:
                since_offset = Path(transcript_path).stat().st_size
            else:
                _, since_offset = await asyncio.to_thread(
                    provider.read_transcript_file,
                    transcript_path,
                    0,
                )
        except OSError:
            since_offset = None
    pane_before = await tmux_manager.capture_pane(window_id)
    return transcript_path, since_offset, pane_before


async def _probe_transcript_command_error(
    provider: AgentProvider,
    transcript_path: str | None,
    since_offset: int | None,
    command: str | None = None,
) -> str | None:
    """Return first command-like error line found in transcript delta."""
    if not transcript_path or since_offset is None:
        return None

    def _read_incremental_entries(path: str, offset: int) -> list[dict]:
        entries: list[dict] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            fh.seek(offset)
            for line in fh:
                parsed = provider.parse_transcript_line(line)
                if parsed:
                    entries.append(parsed)
        return entries

    try:
        if provider.capabilities.supports_incremental_read:
            entries = await asyncio.to_thread(
                _read_incremental_entries,
                transcript_path,
                since_offset,
            )
        else:
            entries, _ = await asyncio.to_thread(
                provider.read_transcript_file,
                transcript_path,
                since_offset,
            )
    except OSError, NotImplementedError:
        return None

    messages, _ = provider.parse_transcript_entries(entries, pending_tools={})
    for msg in messages:
        if msg.role != "assistant":
            continue
        found = _extract_probe_error_line(msg.text, command)
        if found:
            return found
    return None


async def _maybe_send_command_failure_message(
    message: Message,
    window_id: str,
    display: str,
    cc_slash: str,
    *,
    provider: AgentProvider,
    transcript_path: str | None,
    since_offset: int | None,
    pane_before: str | None,
) -> None:
    """Probe transcript/pane for quick command failures and surface them."""
    await asyncio.sleep(_COMMAND_ERROR_PROBE_DELAY_SECONDS)

    error_line = await _probe_transcript_command_error(
        provider,
        transcript_path,
        since_offset,
        command=cc_slash,
    )
    if not error_line:
        pane_after = await tmux_manager.capture_pane(window_id)
        pane_delta = _extract_pane_delta(pane_before, pane_after)
        error_line = _extract_probe_error_line(pane_delta, cc_slash)
    if error_line:
        await safe_reply(
            message,
            f"❌ [{display}] `{cc_slash}` failed\n> {error_line}",
        )


def _spawn_command_failure_probe(
    message: Message,
    window_id: str,
    display: str,
    cc_slash: str,
    *,
    provider: AgentProvider,
    transcript_path: str | None,
    since_offset: int | None,
    pane_before: str | None,
) -> None:
    async def _run() -> None:
        await _maybe_send_command_failure_message(
            message,
            window_id,
            display,
            cc_slash,
            provider=provider,
            transcript_path=transcript_path,
            since_offset=since_offset,
            pane_before=pane_before,
        )

    task = asyncio.create_task(_run())
    task.add_done_callback(task_done_callback)
