"""Terminal output capture and relay for shell provider sessions.

Passive monitoring detects commands run in tmux (both via Telegram and typed
directly) and relays output to the corresponding Telegram topic.  Uses prompt
markers (ccgram:N❯) for output isolation and exit code detection.

When a command originates from Telegram, the monitor state is annotated with
``mark_telegram_command`` so that non-zero exit codes trigger LLM-based error
suggestions.

Key components:
  - check_passive_shell_output: Poll-driven passive output relay
  - mark_telegram_command: Annotate monitor for Telegram-initiated commands
  - _extract_command_output: Prompt-marker-based output extraction
  - _extract_passive_output: Extract output for passive monitoring
  - strip_terminal_glyphs: Remove Nerd Font / PUA characters
"""

import asyncio
import re
import structlog
from dataclasses import dataclass

from telegram import Bot

from ..providers.shell import get_prompt_re
from ..session import session_manager
from .message_sender import edit_with_fallback, rate_limit_send_message

logger = structlog.get_logger()

# Maximum characters per message (fits Telegram 4096-char limit with margin)
_OUTPUT_LIMIT = 3800

_MAX_FIX_OUTPUT_CHARS = 800

# Unicode ranges for Nerd Font / Private Use Area glyphs
# BMP PUA: U+E000–U+F8FF, Supplement PUA-A: U+F0000–U+FFFFD
_GLYPH_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")

_SCROLLBACK_LINES = 200


async def _capture_with_scrollback(
    window_id: str, history: int = _SCROLLBACK_LINES
) -> str | None:
    """Capture pane text including scrollback history.

    Uses ``tmux capture-pane -p -J -S -{history}`` to get *history* lines of
    scrollback.  ``-J`` joins wrapped lines so prompt markers are never
    split across two lines on narrow terminals.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "tmux",
            "capture-pane",
            "-p",
            "-J",
            "-S",
            f"-{history}",
            "-t",
            window_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(5.0):
            stdout, _ = await proc.communicate()
        text = stdout.decode("utf-8", errors="replace").rstrip()
        return text if text else None
    except (TimeoutError, OSError) as exc:
        logger.debug(
            "capture_with_scrollback failed", window_id=window_id, error=str(exc)
        )
        return None


@dataclass
class _CommandOutput:
    """Result of output extraction with optional exit code."""

    text: str
    exit_code: int | None = None


@dataclass
class _PassiveOutput:
    """Result of passive output extraction for tmux-direct commands."""

    command_echo: str
    echo_index: int  # line index in the pane — distinguishes re-runs of same command
    text: str
    exit_code: int | None = None


@dataclass
class _ShellMonitorState:
    """Per-window state for shell output monitoring."""

    last_text_hash: int = (
        0  # hash(rendered_text) — best-effort dedup, skip unchanged polls
    )
    last_command_echo: str = ""  # echo line text of last relayed command
    last_echo_index: int = (
        -1
    )  # line index of echo — distinguishes re-runs of same command
    msg_id: int | None = None  # Telegram message ID for in-place editing
    last_output: str = ""  # last relayed output text
    exit_code_sent: bool = False  # already showed error indicator for this command
    telegram_command: str = ""  # command sent via Telegram (for error suggestions)
    telegram_user_id: int = 0
    telegram_thread_id: int = 0
    telegram_generation: int = 0  # monotonic counter to discard stale fix suggestions


_shell_monitor_state: dict[str, _ShellMonitorState] = {}
_fix_generation: int = 0


def strip_terminal_glyphs(text: str) -> str:
    """Strip Nerd Font and Private Use Area glyphs from terminal output."""
    return _GLYPH_RE.sub("", text)


def _extract_command_output(current: str) -> _CommandOutput:
    """Extract command output and exit code from terminal capture.

    Uses prompt-marker extraction (ccgram:N❯ lines).  Returns empty
    output when no markers are found.
    """
    lines = current.rstrip().splitlines()
    if not lines:
        return _CommandOutput(text="")

    # Scan from bottom (last 10 lines only) for bare prompt: ccgram:N❯
    scan_start = max(0, len(lines) - 10)
    end_idx = None
    exit_code = None
    for i in range(len(lines) - 1, scan_start - 1, -1):
        m = get_prompt_re().match(lines[i])
        if m and not m.group(2).strip():
            end_idx = i
            exit_code = int(m.group(1))
            break

    if end_idx is None:
        return _CommandOutput(text="")

    # Scan upward for command echo: ccgram:N❯ <command text>
    start_idx = None
    for i in range(end_idx - 1, -1, -1):
        m = get_prompt_re().match(lines[i])
        if m and m.group(2).strip():
            start_idx = i
            break

    if start_idx is None:
        return _CommandOutput(text="", exit_code=exit_code)

    output_lines = lines[start_idx + 1 : end_idx]
    return _CommandOutput(text="\n".join(output_lines), exit_code=exit_code)


def _find_command_echo(lines: list[str]) -> tuple[str, int] | None:
    """Find the command echo line above the last bare prompt.

    Scans from bottom for a bare prompt, then upward for the command echo.
    Returns ``(echo_text, line_index)`` or None if idle.
    """
    scan_start = max(0, len(lines) - 10)
    for i in range(len(lines) - 1, scan_start - 1, -1):
        m = get_prompt_re().match(lines[i])
        if m and not m.group(2).strip():
            for j in range(i - 1, -1, -1):
                mj = get_prompt_re().match(lines[j])
                if mj and mj.group(2).strip():
                    return (lines[j], j)
            return None
    return None


def _find_in_progress(lines: list[str]) -> _PassiveOutput | None:
    """Find in-progress command output (no bare prompt at bottom)."""
    for i in range(len(lines) - 1, -1, -1):
        m = get_prompt_re().match(lines[i])
        if m and m.group(2).strip():
            output_lines = lines[i + 1 :]
            while output_lines and not output_lines[-1].strip():
                output_lines.pop()
            return _PassiveOutput(
                command_echo=lines[i],
                echo_index=i,
                text="\n".join(output_lines),
            )
    return None


def _extract_passive_output(text: str) -> _PassiveOutput | None:
    """Extract command output for passive monitoring.

    Returns None for idle shell (bare prompt only) or no markers.
    For completed commands: returns output with exit_code (int).
    For in-progress commands: returns partial output with exit_code=None.
    """
    lines = text.rstrip().splitlines()
    if not lines:
        return None

    # Check bottom 10 lines for any prompt marker
    tail = lines[max(0, len(lines) - 10) :]
    if not any(get_prompt_re().match(line) for line in tail):
        return None

    # Try completed-command extraction (bare prompt at bottom)
    result = _extract_command_output(text)
    if result.exit_code is not None:
        found = _find_command_echo(lines)
        if found is None:
            return None  # idle — bare prompt with no command above
        echo_text, echo_idx = found
        return _PassiveOutput(
            command_echo=echo_text,
            echo_index=echo_idx,
            text=result.text,
            exit_code=result.exit_code,
        )

    # No bare prompt — check for in-progress command
    return _find_in_progress(lines)


def mark_telegram_command(
    window_id: str, command: str, user_id: int, thread_id: int
) -> None:
    """Annotate the monitor state with a Telegram-initiated command.

    When the passive monitor detects this command completed with a non-zero
    exit code, it will trigger LLM-based error suggestions.
    """
    global _fix_generation  # noqa: PLW0603
    _fix_generation += 1
    state = _shell_monitor_state.setdefault(window_id, _ShellMonitorState())
    state.telegram_command = command
    state.telegram_user_id = user_id
    state.telegram_thread_id = thread_id
    state.telegram_generation = _fix_generation


async def _relay_output(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    output: str,
    *,
    msg_id: int | None = None,
) -> int | None:
    """Send or edit the output message in Telegram (monospace formatted).

    Returns the Telegram message ID (new or existing) so callers can
    track it for subsequent edits.  Returns None when the initial send
    fails (rate limit, network), in which case the next call with
    ``msg_id=None`` will attempt a fresh send.
    """
    display = strip_terminal_glyphs(output)
    if len(display) > _OUTPUT_LIMIT:
        display = "\u2026 " + display[-_OUTPUT_LIMIT:]

    if not display.strip():
        return msg_id

    # Wrap in code fence for monospace rendering on Telegram
    display = display.replace("```", "` ` `")
    formatted = f"```\n{display}\n```"

    if msg_id is None:
        sent = await rate_limit_send_message(
            bot,
            chat_id,
            formatted,
            message_thread_id=thread_id,
        )
        if sent:
            return sent.message_id
        return None
    else:
        await edit_with_fallback(bot, chat_id, msg_id, formatted)
        return msg_id


async def _update_error_message(
    bot: Bot, chat_id: int, msg_id: int, exit_code: int, output: str
) -> None:
    """Edit the output message to prepend an error indicator (monospace)."""
    error_prefix = f"\u274c exit {exit_code}\n"
    display = strip_terminal_glyphs(output)
    fence_overhead = 8  # ```\n ... \n```
    max_body = _OUTPUT_LIMIT - len(error_prefix) - fence_overhead
    if len(display) > max_body:
        display = display[-max_body:]
    display = display.replace("```", "` ` `")
    formatted = f"{error_prefix}```\n{display}\n```"
    await edit_with_fallback(bot, chat_id, msg_id, formatted)


async def _maybe_suggest_fix(
    bot: Bot,
    user_id: int,
    chat_id: int,
    thread_id: int,
    window_id: str,
    *,
    command: str,
    exit_code: int,
    msg_id: int | None,
    output: str,
    generation: int = 0,
) -> None:
    """If exit code is non-zero, show error indicator and ask LLM for a fix."""
    if msg_id:
        await _update_error_message(bot, chat_id, msg_id, exit_code, output)

    try:
        from ..llm import get_completer

        completer = get_completer()
    except (ValueError, ImportError):  # fmt: skip
        completer = None

    if not completer:
        return

    from .shell_commands import gather_llm_context

    ctx = await gather_llm_context(window_id)

    from .shell_commands import redact_for_llm

    trimmed = redact_for_llm(output or "")
    if len(trimmed) > _MAX_FIX_OUTPUT_CHARS:
        trimmed = f"\u2026{trimmed[-_MAX_FIX_OUTPUT_CHARS:]}"

    fix_description = (
        f"The command `{command}` failed (exit {exit_code}):\n{trimmed}\n\n"
        "Generate a corrected command."
    )

    try:
        result = await completer.generate_command(
            fix_description,
            cwd=ctx["cwd"],
            shell=ctx["shell"],
            shell_tools=ctx["shell_tools"],
        )
    except RuntimeError:
        logger.debug("LLM fix suggestion failed")
        return

    # Discard stale fix if a newer command was sent while LLM was thinking
    if generation and _fix_generation != generation:
        return

    if not result.command or result.command == command:
        return

    from .shell_commands import show_command_approval

    await show_command_approval(bot, chat_id, thread_id, window_id, result, user_id)


# ── Shell output monitoring ───────────────────────────────────────────
# Detects and relays output from commands in tmux (both Telegram-initiated
# and typed directly). Requires prompt markers for reliable extraction.


def clear_shell_monitor_state(window_id: str) -> None:
    """Remove monitor state for a window (cleanup / provider switch)."""
    _shell_monitor_state.pop(window_id, None)


def reset_shell_monitor_state() -> None:
    """Reset all monitor state (for testing)."""
    _shell_monitor_state.clear()


def _reset_monitor(state: _ShellMonitorState) -> None:
    """Reset monitor state to idle."""
    state.last_command_echo = ""
    state.last_echo_index = -1
    state.msg_id = None
    state.last_output = ""
    state.exit_code_sent = False


def _has_markers_in_tail(rendered_text: str) -> bool:
    """Quick check for prompt markers in the last 10 visible lines.

    Strips leading whitespace because pyte may pad lines when the terminal
    wraps long output (the prompt ends up indented on a continuation line).
    """
    lines = rendered_text.rstrip().splitlines()
    tail = lines[max(0, len(lines) - 10) :]
    return any(get_prompt_re().match(line.lstrip()) for line in tail)


async def check_passive_shell_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    rendered_text: str,
) -> None:
    """Check for new shell output.

    Called every poll cycle from status_polling for shell provider windows.
    Uses ``rendered_text`` (cheap, from pyte) for change detection, then
    ``_capture_with_scrollback`` for reliable output extraction so that
    command echoes scrolled off the visible pane are still found.
    """
    text_hash = hash(rendered_text)
    state = _shell_monitor_state.setdefault(window_id, _ShellMonitorState())
    changed = text_hash != state.last_text_hash
    if not changed:
        return
    state.last_text_hash = text_hash

    if not _has_markers_in_tail(rendered_text):
        if not (state.last_command_echo and state.msg_id is not None):
            _reset_monitor(state)
        return

    # Capture with scrollback for reliable command echo finding
    scrollback = await _capture_with_scrollback(window_id)
    if not scrollback:
        return

    passive = _extract_passive_output(scrollback)
    if passive is None:
        if not (state.last_command_echo and state.msg_id is not None):
            _reset_monitor(state)
        return

    if (
        passive.command_echo != state.last_command_echo
        or passive.echo_index != state.last_echo_index
    ):
        state.last_command_echo = passive.command_echo
        state.last_echo_index = passive.echo_index
        state.msg_id = None
        state.last_output = ""
        state.exit_code_sent = False

    await _relay_passive_output(bot, user_id, thread_id, state, passive, window_id)


def _command_from_echo(echo: str) -> str:
    """Extract the command text from a prompt echo line.

    ``"ccgram:0❯ ls -al"`` → ``"ls -al"``
    """
    m = get_prompt_re().match(echo)
    return m.group(2).strip() if m else echo


async def _relay_passive_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    state: _ShellMonitorState,
    passive: _PassiveOutput,
    window_id: str,
) -> None:
    """Relay extracted output to Telegram.

    Formats as: ``❯ <command>`` header followed by output in a code block.
    """
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)

    if passive.text != state.last_output:
        state.last_output = passive.text
        cmd = _command_from_echo(passive.command_echo)
        combined = f"❯ {cmd}\n{passive.text}" if cmd else passive.text
        state.msg_id = await _relay_output(
            bot, chat_id, thread_id, combined, msg_id=state.msg_id
        )

    if (
        passive.exit_code is not None
        and passive.exit_code != 0
        and not state.exit_code_sent
        and state.msg_id
    ):
        state.exit_code_sent = True
        await _update_error_message(
            bot, chat_id, state.msg_id, passive.exit_code, passive.text
        )

    # If this was a Telegram-initiated command, suggest a fix via LLM
    if (
        passive.exit_code is not None
        and passive.exit_code != 0
        and state.telegram_command
    ):
        tg_cmd = state.telegram_command
        tg_uid = state.telegram_user_id
        tg_tid = state.telegram_thread_id
        tg_gen = state.telegram_generation
        state.telegram_command = ""
        state.telegram_user_id = 0
        state.telegram_thread_id = 0
        state.telegram_generation = 0
        await _maybe_suggest_fix(
            bot,
            tg_uid,
            chat_id,
            tg_tid,
            window_id,
            command=tg_cmd,
            exit_code=passive.exit_code,
            msg_id=state.msg_id,
            output=passive.text,
            generation=tg_gen,
        )
