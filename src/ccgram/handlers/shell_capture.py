"""Terminal output capture and relay for shell provider sessions.

Captures raw shell output from tmux panes and relays it to Telegram,
using in-place message editing for streaming updates. Uses prompt markers
(ccgram:N❯) for output isolation and exit code detection, falling back
to baseline diffing when markers are absent.
Strips Nerd Font glyphs for clean Telegram display.

Key components:
  - start_shell_capture: Launch a background capture task
  - _extract_command_output: Prompt-marker-based output extraction
  - _extract_new_output: Baseline-diff fallback for unmarked shells
  - strip_terminal_glyphs: Remove Nerd Font / PUA characters
"""

import asyncio
import re
import structlog
from dataclasses import dataclass, field

from telegram import Bot

from ..providers.shell import PROMPT_RE
from ..session import session_manager
from ..tmux_manager import tmux_manager
from ..utils import task_done_callback
from .message_sender import edit_with_fallback, rate_limit_send_message

logger = structlog.get_logger()

# Maximum characters per message (fits Telegram 4096-char limit with margin)
_OUTPUT_LIMIT = 3800

# Maximum capture duration in seconds
_CAPTURE_TIMEOUT = 60

# Consecutive stable polls required before considering output done
_STABLE_THRESHOLD = 2
_MAX_FIX_OUTPUT_CHARS = 800

# A bare prompt (e.g. "$ " or "❮") is shorter than this
_MAX_BARE_PROMPT_LEN = 5

_shell_capture_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}

# Unicode ranges for Nerd Font / Private Use Area glyphs
# BMP PUA: U+E000–U+F8FF, Supplement PUA-A: U+F0000–U+FFFFD
_GLYPH_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")


@dataclass
class _CommandOutput:
    """Result of output extraction with optional exit code."""

    text: str
    exit_code: int | None = None


@dataclass
class _CaptureState:
    """Mutable state for a shell capture session."""

    msg_id: int | None = None
    last_output: str = ""
    stable_count: int = 0
    baseline_lines: list[str] = field(default_factory=list)
    command: str = ""
    exit_code: int | None = None


def strip_terminal_glyphs(text: str) -> str:
    """Strip Nerd Font and Private Use Area glyphs from terminal output."""
    return _GLYPH_RE.sub("", text)


def _extract_command_output(baseline_lines: list[str], current: str) -> _CommandOutput:
    """Extract command output and exit code from terminal capture.

    First tries prompt-marker extraction (ccgram:N❯ lines).
    Falls back to baseline diffing when no markers are found.
    """
    lines = current.rstrip().splitlines()
    if not lines:
        return _CommandOutput(text="")

    # Scan from bottom for bare prompt: ccgram:N❯ (no command after ❯)
    end_idx = None
    exit_code = None
    for i in range(len(lines) - 1, -1, -1):
        m = PROMPT_RE.match(lines[i])
        if m and not m.group(2).strip():
            end_idx = i
            exit_code = int(m.group(1))
            break

    if end_idx is not None:
        # Scan upward for command echo: ccgram:N❯ <command text>
        start_idx = None
        for i in range(end_idx - 1, -1, -1):
            m = PROMPT_RE.match(lines[i])
            if m and m.group(2).strip():
                start_idx = i
                break

        if start_idx is None:
            return _CommandOutput(text="", exit_code=exit_code)

        output_lines = lines[start_idx + 1 : end_idx]
        return _CommandOutput(text="\n".join(output_lines), exit_code=exit_code)

    # No markers — fall back to baseline diffing
    return _CommandOutput(text=_extract_new_output(baseline_lines, current))


def _extract_new_output(baseline_lines: list[str], current: str) -> str:
    """Extract only new command output by removing baseline content.

    Finds the longest suffix of baseline_lines that appears as a prefix
    of the current capture (accounting for terminal scrolling), then
    returns only the lines after that overlap — stripping the trailing
    prompt block.
    """
    current_lines = current.rstrip().splitlines()
    if not current_lines:
        return ""
    if not baseline_lines:
        return current.rstrip()

    # Drop the last line of baseline (it's the prompt where the command
    # was typed, which changes once the command is entered).
    match_lines = baseline_lines[:-1]
    if not match_lines:
        return current.rstrip()

    # Find longest suffix of match_lines that equals a prefix of current_lines
    max_overlap = min(len(match_lines), len(current_lines))
    best = 0
    for length in range(1, max_overlap + 1):
        if match_lines[-length:] == current_lines[:length]:
            best = length

    new_lines = current_lines[best:]

    # Strip trailing prompt block
    while new_lines and not new_lines[-1].strip():
        new_lines.pop()

    stripped = _strip_trailing_prompt(new_lines)
    return "\n".join(stripped)


def _strip_trailing_prompt(lines: list[str]) -> list[str]:
    """Remove trailing shell prompt lines from output.

    Recognises common prompt-ending characters at start-of-line.
    Handles multi-line prompts (info bar + prompt char) by also
    removing a preceding info line when the prompt line is stripped.
    """
    if not lines:
        return lines

    last = lines[-1].strip()
    clean_last = strip_terminal_glyphs(last)
    prompt_chars = ("❮", "$", "%", ">", "#")
    if (
        any(clean_last.startswith(ch) for ch in prompt_chars)
        and len(clean_last) < _MAX_BARE_PROMPT_LEN
    ):
        lines = lines[:-1]
        if lines and not lines[-1].strip():
            lines = lines[:-1]
        if lines and _looks_like_info_bar(lines[-1]):
            lines = lines[:-1]

    return lines


def _looks_like_info_bar(line: str) -> bool:
    """Heuristic: check if a line looks like a prompt info bar."""
    return "~/" in line or "·" in line or _GLYPH_RE.search(line) is not None


def start_shell_capture(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    *,
    baseline: str = "",
    command: str = "",
) -> None:
    """Launch a background task to capture shell output."""
    key = (user_id, thread_id)
    existing = _shell_capture_tasks.pop(key, None)
    if existing and not existing.done():
        existing.cancel()

    task = asyncio.create_task(
        _capture_shell_output(
            bot, user_id, thread_id, window_id, baseline=baseline, command=command
        )
    )
    task.add_done_callback(task_done_callback)
    _shell_capture_tasks[key] = task


def cancel_shell_capture(user_id: int, thread_id: int) -> None:
    """Cancel any running shell capture for this topic."""
    key = (user_id, thread_id)
    task = _shell_capture_tasks.pop(key, None)
    if task and not task.done():
        task.cancel()


async def _relay_output(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    output: str,
    state: _CaptureState,
) -> None:
    """Send or edit the output message in Telegram (monospace formatted)."""
    display = strip_terminal_glyphs(output)
    if len(display) > _OUTPUT_LIMIT:
        display = "\u2026 " + display[-_OUTPUT_LIMIT:]

    if not display.strip():
        return

    # Wrap in code fence for monospace rendering on Telegram
    display = display.replace("```", "` ` `")
    formatted = f"```\n{display}\n```"

    if state.msg_id is None:
        sent = await rate_limit_send_message(
            bot,
            chat_id,
            formatted,
            message_thread_id=thread_id,
        )
        if sent:
            state.msg_id = sent.message_id
    else:
        await edit_with_fallback(bot, chat_id, state.msg_id, formatted)


async def _poll_once(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    window_id: str,
    state: _CaptureState,
) -> bool:
    """Single poll iteration. Returns True when capture should stop."""
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        return True

    raw = await tmux_manager.capture_pane(window_id)
    if raw is None:
        return False

    result = _extract_command_output(state.baseline_lines, raw)
    new_output = result.text

    changed = new_output != state.last_output
    if changed:
        state.stable_count = 0
        state.last_output = new_output
        await _relay_output(bot, chat_id, thread_id, new_output, state)

    # Prompt marker detected = command finished
    if result.exit_code is not None:
        state.exit_code = result.exit_code
        return True

    # No marker — stable poll heuristic (increment even when empty so
    # no-output commands don't spin for the full capture timeout)
    if not changed:
        state.stable_count += 1
        if state.stable_count >= _STABLE_THRESHOLD:
            return True

    return False


async def _capture_shell_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    *,
    baseline: str = "",
    command: str = "",
) -> None:
    """Background task: capture shell command output from tmux pane.

    Sends the first captured output as a new message, then edits it
    in-place as more output appears.  Only shows new output since the
    baseline (pre-command state).  Stops when a prompt marker appears
    (exit code detected) or pane content stabilizes (2 consecutive polls
    with identical content).

    After capture completes, checks for errors and optionally suggests
    a fix via the LLM.
    """
    try:
        await asyncio.sleep(1.0)

        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        baseline_lines = baseline.rstrip().splitlines() if baseline else []
        state = _CaptureState(
            baseline_lines=baseline_lines,
            command=command,
        )

        for _ in range(_CAPTURE_TIMEOUT):
            should_stop = await _poll_once(
                bot,
                chat_id,
                thread_id,
                window_id,
                state,
            )
            if should_stop:
                break
            await asyncio.sleep(1.0)

        # After capture completes, check for errors and suggest fix
        if state.last_output and state.command:
            await _maybe_suggest_fix(bot, user_id, chat_id, thread_id, window_id, state)
    except asyncio.CancelledError:
        return
    finally:
        key = (user_id, thread_id)
        if _shell_capture_tasks.get(key) is asyncio.current_task():
            _shell_capture_tasks.pop(key, None)


_EXIT_COMMAND_NOT_FOUND = 127
_EXIT_PERMISSION_DENIED = 126


def _classify_error(exit_code: int | None, output: str) -> str:
    """Classify error type from exit code and output for better LLM fix prompts."""
    lower = output.lower()
    if exit_code == _EXIT_COMMAND_NOT_FOUND or "command not found" in lower:
        return "command not found \u2014 check spelling, PATH, or install the tool"
    if exit_code == _EXIT_PERMISSION_DENIED or "permission denied" in lower:
        return "permission denied \u2014 may need sudo or chmod"
    if "syntax error" in lower or "unexpected token" in lower or "parse error" in lower:
        return "shell syntax error \u2014 check shell-specific syntax"
    if "no such file or directory" in lower:
        return "file/directory not found \u2014 check the path"
    if "invalid option" in lower or "unrecognized option" in lower:
        return "invalid option \u2014 check the command flags"
    return ""


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
    state: _CaptureState,
) -> None:
    """If exit code is non-zero, show error indicator and ask LLM for a fix."""
    if state.exit_code is None or state.exit_code == 0:
        return

    if state.msg_id:
        await _update_error_message(
            bot, chat_id, state.msg_id, state.exit_code, state.last_output
        )

    try:
        from ..llm import get_completer

        completer = get_completer()
    except ValueError, ImportError:
        completer = None

    if not completer:
        return

    from .shell_commands import gather_llm_context

    ctx = await gather_llm_context(window_id)

    output = state.last_output or ""
    if len(output) > _MAX_FIX_OUTPUT_CHARS:
        output = f"…{output[-_MAX_FIX_OUTPUT_CHARS:]}"

    error_hint = _classify_error(state.exit_code, output)
    fix_description = (
        f"The command `{state.command}` failed (exit {state.exit_code}):\n{output}\n\n"
    )
    if error_hint:
        fix_description += f"Error type: {error_hint}\n"
    fix_description += "Generate a corrected command."

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

    if not result.command or result.command == state.command:
        return

    from .shell_commands import has_shell_pending, show_command_approval

    if has_shell_pending(chat_id, thread_id):
        return
    await show_command_approval(bot, chat_id, thread_id, window_id, result, user_id)
