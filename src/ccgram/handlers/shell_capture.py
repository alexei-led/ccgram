"""Terminal output capture and relay for shell provider sessions.

Two capture modes:
  - Active capture: triggered when commands are sent via Telegram
    (start_shell_capture → background polling → relay to Telegram)
  - Passive monitoring: detects commands typed directly in tmux
    (check_passive_shell_output, called from status polling loop)

Both use prompt markers (ccgram:N❯) for output isolation and exit code
detection. Active capture falls back to baseline diffing when markers
are absent. Passive monitoring requires markers.

Key components:
  - start_shell_capture: Launch active background capture task
  - check_passive_shell_output: Poll-driven passive output relay
  - _extract_command_output: Prompt-marker-based output extraction
  - _extract_passive_output: Extract output for passive monitoring
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
_STABLE_THRESHOLD_EMPTY = 5
_MAX_FIX_OUTPUT_CHARS = 800

# A bare prompt (e.g. "$ " or "❮") is shorter than this
_MAX_BARE_PROMPT_LEN = 3

_shell_capture_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}

# Unicode ranges for Nerd Font / Private Use Area glyphs
# BMP PUA: U+E000–U+F8FF, Supplement PUA-A: U+F0000–U+FFFFD
_GLYPH_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")

_SCROLLBACK_LINES = 200


async def _capture_with_scrollback(
    window_id: str, history: int = _SCROLLBACK_LINES
) -> str | None:
    """Capture pane text including scrollback history.

    Uses ``tmux capture-pane -p -J -S -N`` to get *history* lines of
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
    except TimeoutError, OSError:
        return None


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


@dataclass
class _PassiveOutput:
    """Result of passive output extraction for tmux-direct commands."""

    command_echo: str
    echo_index: int  # line index in the pane — distinguishes re-runs of same command
    text: str
    exit_code: int | None = None


@dataclass
class _ShellMonitorState:
    """Per-window state for passive shell output monitoring."""

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


_shell_monitor_state: dict[str, _ShellMonitorState] = {}


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

    # Scan from bottom (last 10 lines only) for bare prompt: ccgram:N❯
    scan_start = max(0, len(lines) - 10)
    end_idx = None
    exit_code = None
    for i in range(len(lines) - 1, scan_start - 1, -1):
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


def _find_command_echo(lines: list[str]) -> tuple[str, int] | None:
    """Find the command echo line above the last bare prompt.

    Scans from bottom for a bare prompt, then upward for the command echo.
    Returns ``(echo_text, line_index)`` or None if idle.
    """
    scan_start = max(0, len(lines) - 10)
    for i in range(len(lines) - 1, scan_start - 1, -1):
        m = PROMPT_RE.match(lines[i])
        if m and not m.group(2).strip():
            for j in range(i - 1, -1, -1):
                mj = PROMPT_RE.match(lines[j])
                if mj and mj.group(2).strip():
                    return (lines[j], j)
            return None
    return None


def _find_in_progress(lines: list[str]) -> _PassiveOutput | None:
    """Find in-progress command output (no bare prompt at bottom)."""
    for i in range(len(lines) - 1, -1, -1):
        m = PROMPT_RE.match(lines[i])
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
    """Extract command output for passive monitoring (tmux-direct commands).

    Returns None for idle shell (bare prompt only) or no markers.
    For completed commands: returns output with exit_code (int).
    For in-progress commands: returns partial output with exit_code=None.
    """
    lines = text.rstrip().splitlines()
    if not lines:
        return None

    # Check bottom 10 lines for any prompt marker
    tail = lines[max(0, len(lines) - 10) :]
    if not any(PROMPT_RE.match(line) for line in tail):
        return None

    # Try completed-command extraction (bare prompt at bottom)
    result = _extract_command_output([], text)
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


async def _fast_forward_passive_state(window_id: str) -> None:
    """Snapshot current pane into passive monitor state.

    Called after active capture completes so the next poll cycle's
    passive monitor does not re-relay the same command output.
    """
    state = _shell_monitor_state.get(window_id)
    if state is None:
        state = _ShellMonitorState()
        _shell_monitor_state[window_id] = state

    raw = await _capture_with_scrollback(window_id)
    if not raw:
        return
    passive = _extract_passive_output(raw)
    if passive is not None:
        state.last_command_echo = passive.command_echo
        state.last_echo_index = passive.echo_index
        state.last_output = passive.text
        state.exit_code_sent = passive.exit_code is not None and passive.exit_code != 0
    state.last_text_hash = hash(raw)


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

    # Reset passive monitor so it doesn't carry stale error flags
    # or re-relay the same command after active capture finishes.
    clear_shell_monitor_state(window_id)

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

    raw = await _capture_with_scrollback(window_id)
    if raw is None:
        return False

    result = _extract_command_output(state.baseline_lines, raw)
    new_output = result.text

    changed = new_output != state.last_output
    if changed:
        state.stable_count = 0
        state.last_output = new_output
        state.msg_id = await _relay_output(
            bot, chat_id, thread_id, new_output, msg_id=state.msg_id
        )

    # Prompt marker detected = command finished
    if result.exit_code is not None:
        state.exit_code = result.exit_code
        return True

    # No marker — stable poll heuristic (increment even when empty so
    # no-output commands don't spin for the full capture timeout)
    if not changed:
        state.stable_count += 1
        threshold = (
            _STABLE_THRESHOLD_EMPTY if not state.last_output else _STABLE_THRESHOLD
        )
        if state.stable_count >= threshold:
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

        # Fast-forward passive monitor past this command so it doesn't
        # re-relay the same output on the next poll cycle.
        await _fast_forward_passive_state(window_id)
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

    from .shell_commands import show_command_approval

    await show_command_approval(bot, chat_id, thread_id, window_id, result, user_id)


# ── Passive shell output monitoring ───────────────────────────────────
# Detects and relays output from commands typed directly in tmux
# (not via Telegram). Requires prompt markers for reliable extraction.


def clear_shell_monitor_state(window_id: str) -> None:
    """Remove passive monitor state for a window (cleanup / provider switch)."""
    _shell_monitor_state.pop(window_id, None)


def reset_shell_monitor_state() -> None:
    """Reset all passive monitor state (for testing)."""
    _shell_monitor_state.clear()


def _reset_monitor(state: _ShellMonitorState) -> None:
    """Reset passive monitor state to idle."""
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
    return any(PROMPT_RE.match(line.lstrip()) for line in tail)


async def check_passive_shell_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    rendered_text: str,
) -> None:
    """Check for new shell output from direct tmux interaction.

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

    # Skip if active Telegram-initiated capture is running for this topic
    key = (user_id, thread_id)
    task = _shell_capture_tasks.get(key)
    if task and not task.done():
        return

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

    await _relay_passive_output(bot, user_id, thread_id, state, passive)


def _command_from_echo(echo: str) -> str:
    """Extract the command text from a prompt echo line.

    ``"ccgram:0❯ ls -al"`` → ``"ls -al"``
    """
    m = PROMPT_RE.match(echo)
    return m.group(2).strip() if m else echo


async def _relay_passive_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    state: _ShellMonitorState,
    passive: _PassiveOutput,
) -> None:
    """Relay extracted passive output to Telegram.

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
