"""Terminal output capture and relay for shell provider sessions.

Passive monitoring detects commands run in tmux (both via Telegram and typed
directly) and relays output to the corresponding Telegram topic.  Uses prompt
markers for output isolation and exit code detection.  In *wrap* mode the
marker is ``⌘N⌘`` appended to the user's existing prompt; in *replace* mode
it is ``{prefix}:N❯`` replacing the entire prompt.

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

import re
import structlog
from dataclasses import dataclass
from typing import Any, Protocol

from ...telegram_client import TelegramClient

from ...providers.shell import match_prompt
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...multiplexer.base import CaptureResult
from ..messaging_pipeline.message_sender import (
    REACT_DONE,
    REACT_FAIL,
    edit_with_fallback,
    rate_limit_send_message,
    react,
)
from ...topic_state_registry import topic_state

logger = structlog.get_logger()


async def _react_exit(
    client: TelegramClient, chat_id: int, message_id: int, exit_code: int
) -> None:
    """Persistent ✅/❌ ack on the user's command message after completion."""
    if not message_id:
        return
    emoji = REACT_DONE if exit_code == 0 else REACT_FAIL
    await react(client, chat_id, message_id, emoji)


class CommandApprovalCallback(Protocol):
    """Callable that shows a command-approval keyboard in a Telegram topic.

    Matches the signature of ``shell_commands.show_command_approval``.
    """

    async def __call__(
        self,
        client: TelegramClient,
        chat_id: int,
        thread_id: int,
        window_id: str,
        result: Any,
        user_id: int,
    ) -> bool: ...


async def _approval_unwired(  # type: ignore[misc]
    _bot: Any,
    _chat_id: Any,
    _thread_id: Any,
    _window_id: Any,
    _result: Any,
    _user_id: Any,
) -> bool:
    raise RuntimeError(
        "register_approval_callback not wired — call register_approval_callback() at startup"
    )


_approval_callback: CommandApprovalCallback = _approval_unwired  # type: ignore[assignment]
_approval_callback_registered: bool = False


def register_approval_callback(fn: CommandApprovalCallback) -> None:
    """Wire show_command_approval from shell_commands (called once from bot.py setup).

    Avoids the shell_capture ↔ shell_commands runtime circular import.

    Raises RuntimeError if called more than once — wiring should happen exactly
    once at startup; double registration is a programming error.
    """
    global _approval_callback, _approval_callback_registered
    if _approval_callback_registered:
        raise RuntimeError("register_approval_callback already registered")
    _approval_callback = fn
    _approval_callback_registered = True


def _reset_approval_callback_for_testing() -> None:
    """Restore the unwired default — only for tests."""
    global _approval_callback, _approval_callback_registered
    _approval_callback = _approval_unwired  # type: ignore[assignment]
    _approval_callback_registered = False


# Maximum characters per message (fits Telegram 4096-char limit with margin)
_OUTPUT_LIMIT = 3800

_MAX_FIX_OUTPUT_CHARS = 800

# Unicode ranges for Nerd Font / Private Use Area glyphs
# BMP PUA: U+E000–U+F8FF, Supplement PUA-A: U+F0000–U+FFFFD
_GLYPH_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")

_SCROLLBACK_LINES = 200

# Prepended to relayed output when the backend clamped the scrollback capture
# (e.g. herdr's 1000-line read cap) so the tail was dropped — a clipped
# capture must not be mistaken for the command's full output.
_TRUNCATION_NOTICE = "… [earlier output truncated]"


def _passive_scrollback_lines() -> int:
    """Scrollback depth to request for the passive monitor.

    Capped backends (herdr caps ``read`` at 1000 lines) get the full cap so a
    command's echo stays inside the readable window for as long as the backend
    allows — a 200-line request would drop the echo (and thus all output) for
    any command longer than ~200 lines that herdr could still return. Uncapped
    backends (tmux) keep the lighter default.
    """
    cap = tmux_manager.capabilities.read_max_lines
    return cap if cap is not None else _SCROLLBACK_LINES


async def _capture_with_scrollback(
    window_id: str, history: int = _SCROLLBACK_LINES
) -> CaptureResult | None:
    """Capture pane scrollback as a ``CaptureResult`` via the multiplexer.

    The backend clamps ``history`` to its ``read_max_lines`` (herdr caps at
    1000; tmux is unlimited) and sets ``truncated`` when it dropped the tail,
    so the monitor can flag a clipped capture instead of treating it as the
    command's full output.
    """
    return await tmux_manager.capture_scrollback(window_id, lines=history)


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
    tracked_done: bool = (
        False  # current command's completion (exit code) already relayed
    )
    telegram_command: str = ""  # command sent via Telegram (for error suggestions)
    telegram_user_id: int = 0
    telegram_thread_id: int = 0
    telegram_message_id: int = 0  # original user msg id — target for ✅/❌ reaction
    telegram_generation: int = 0  # monotonic counter to discard stale fix suggestions


_shell_monitor_state: dict[str, _ShellMonitorState] = {}
_fix_generation: int = 0


def strip_terminal_glyphs(text: str) -> str:
    """Strip Nerd Font and Private Use Area glyphs from terminal output."""
    return _GLYPH_RE.sub("", text)


def _extract_command_output(current: str) -> _CommandOutput:
    """Extract command output and exit code from terminal capture.

    Uses prompt-marker extraction via ``match_prompt()``.  Returns empty
    output when no markers are found.
    """
    lines = current.rstrip().splitlines()
    if not lines:
        return _CommandOutput(text="")

    # Scan from bottom (last 10 lines only) for bare prompt (no command text)
    scan_start = max(0, len(lines) - 10)
    end_idx = None
    exit_code = None
    for i in range(len(lines) - 1, scan_start - 1, -1):
        m = match_prompt(lines[i])
        if m and not m.trailing_text.strip():
            end_idx = i
            exit_code = m.exit_code
            break

    if end_idx is None:
        return _CommandOutput(text="")

    # Scan upward for command echo (prompt marker with command text)
    start_idx = None
    for i in range(end_idx - 1, -1, -1):
        m = match_prompt(lines[i])
        if m and m.trailing_text.strip():
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
        m = match_prompt(lines[i])
        if m and not m.trailing_text.strip():
            for j in range(i - 1, -1, -1):
                mj = match_prompt(lines[j])
                if mj and mj.trailing_text.strip():
                    return (lines[j], j)
            return None
    return None


def _find_in_progress(lines: list[str]) -> _PassiveOutput | None:
    """Find in-progress command output (no bare prompt at bottom)."""
    for i in range(len(lines) - 1, -1, -1):
        m = match_prompt(lines[i])
        if m and m.trailing_text.strip():
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
    if not any(match_prompt(line) for line in tail):
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


def _truncated_tail_output(text: str) -> _PassiveOutput | None:
    """Build truncated output when a completed command's echo scrolled beyond
    the backend's readable history.

    Reached only when ``_extract_passive_output`` found no command echo, so no
    prompt-with-command exists in the captured window: the running command's
    echo fell outside it. The capture still ends in a completed-command bare
    prompt, so everything above that prompt is the command's visible tail.
    Relay it (the caller prepends ``_TRUNCATION_NOTICE``) instead of dropping
    the output entirely.

    Returns ``None`` when there is no completed prompt (in-progress) or nothing
    but prompts above it (genuinely idle), so the caller relays nothing then.
    """
    lines = text.rstrip().splitlines()
    tail_start = max(0, len(lines) - 10)
    end_idx = None
    exit_code = None
    for i in range(len(lines) - 1, tail_start - 1, -1):
        m = match_prompt(lines[i])
        if m and not m.trailing_text.strip():
            end_idx = i
            exit_code = m.exit_code
            break
    if end_idx is None:
        return None

    output_lines = lines[:end_idx]
    # A stale bare prompt from a prior command may sit at the top of the
    # readable window; drop leading prompt lines so only real output remains.
    while output_lines:
        m = match_prompt(output_lines[0])
        if m and not m.trailing_text.strip():
            output_lines.pop(0)
        else:
            break
    tail = "\n".join(output_lines).rstrip()
    if not tail.strip():
        return None
    return _PassiveOutput(
        command_echo="",
        echo_index=-1,
        text=tail,
        exit_code=exit_code,
    )


def _extract_passive_with_truncation(
    capture: CaptureResult, requested: int, *, allow_echoless_recovery: bool
) -> tuple[_PassiveOutput, bool] | None:
    """Extract passive output from a capture, recovering an over-long tail.

    Returns ``(passive, truncated)`` or ``None`` when there is nothing to relay
    (idle / no markers). On capped backends (herdr) a command whose echo
    scrolled beyond the full readable window is recovered via
    ``_truncated_tail_output`` and flagged truncated; uncapped backends (tmux,
    ``read_max_lines`` is ``None``) keep the echo-anchored behavior and never
    relay an ambiguous tail.

    ``allow_echoless_recovery`` gates that echoless tail: it is only honored
    when the monitor has evidence a command actually ran in the current prompt
    session. Without it, stale scrollback left above a freshly set-up prompt
    (an existing shell bound with ``clear=False``) is structurally identical to
    a truncated long command and would be relayed as new output.
    """
    passive = _extract_passive_output(capture.text)
    if passive is not None:
        return passive, capture.truncated

    if not allow_echoless_recovery:
        return None

    cap = tmux_manager.capabilities.read_max_lines
    if cap is not None and requested >= cap:
        tail = _truncated_tail_output(capture.text)
        if tail is not None:
            return tail, True
    return None


def _apply_echoless_identity(
    state: _ShellMonitorState, passive: _PassiveOutput
) -> None:
    """Give an echoless recovered tail a stable command discriminator.

    ``_truncated_tail_output`` returns no echo (``command_echo=""``,
    ``echo_index=-1``), so the caller's state comparison can't tell one echoless
    command from the next. Synthesize one in place:
      - tail of the command still on screen (not yet completed) -> reuse its
        tracked identity so it edits the existing message, not a duplicate;
      - a fresh echoless command -> key on ``telegram_generation`` (the
        per-command token), so consecutive Telegram commands each get their own
        message instead of the second overwriting the first.

    Two consecutive echoless *typed* commands still collide — no per-command
    token exists for them, so they are unrecoverable without a visible echo.
    """
    if passive.echo_index != -1 or passive.command_echo:
        return
    if not state.tracked_done and (state.last_command_echo or state.msg_id is not None):
        passive.command_echo = state.last_command_echo
        passive.echo_index = state.last_echo_index
    else:
        passive.echo_index = state.telegram_generation


def mark_telegram_command(
    window_id: str,
    command: str,
    user_id: int,
    thread_id: int,
    message_id: int = 0,
) -> None:
    """Annotate the monitor state with a Telegram-initiated command.

    When the passive monitor detects this command completed with a non-zero
    exit code, it will trigger LLM-based error suggestions. ``message_id``
    is the user's source message — used as the target for ✅/❌ reaction
    feedback once the command completes.
    """
    global _fix_generation  # noqa: PLW0603
    _fix_generation += 1
    state = _shell_monitor_state.setdefault(window_id, _ShellMonitorState())
    state.telegram_command = command
    state.telegram_user_id = user_id
    state.telegram_thread_id = thread_id
    state.telegram_message_id = message_id
    state.telegram_generation = _fix_generation


async def _relay_output(
    client: TelegramClient,
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
            client,
            chat_id,
            formatted,
            message_thread_id=thread_id,
        )
        if sent:
            return sent.message_id
        return None
    else:
        await edit_with_fallback(client, chat_id, msg_id, formatted)
        return msg_id


async def _update_error_message(
    client: TelegramClient, chat_id: int, msg_id: int, exit_code: int, output: str
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
    await edit_with_fallback(client, chat_id, msg_id, formatted)


async def _maybe_suggest_fix(
    client: TelegramClient,
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
        await _update_error_message(client, chat_id, msg_id, exit_code, output)

    try:
        # Lazy: llm wires httpx + provider configs; only loaded when an
        # error suggestion is actually needed.
        from ...llm import get_completer

        completer = get_completer()
    except (ValueError, ImportError):  # fmt: skip
        completer = None

    if not completer:
        return

    # Lazy: sibling cycle — shell_context imports providers.shell which
    # touches shell_capture state.
    # Lazy: shell_capture ↔ shell_context cycle
    from .shell_context import gather_llm_context, redact_for_llm

    ctx = await gather_llm_context(window_id)
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

    await _approval_callback(client, chat_id, thread_id, window_id, result, user_id)


# ── Shell output monitoring ───────────────────────────────────────────
# Detects and relays output from commands in tmux (both Telegram-initiated
# and typed directly). Requires prompt markers for reliable extraction.


@topic_state.register("window")
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
    state.tracked_done = False


def _has_markers_in_tail(rendered_text: str) -> bool:
    """Quick check for prompt markers in the last 10 visible lines.

    Strips leading whitespace because pyte may pad lines when the terminal
    wraps long output (the prompt ends up indented on a continuation line).
    """
    lines = rendered_text.rstrip().splitlines()
    tail = lines[max(0, len(lines) - 10) :]
    return any(match_prompt(line.lstrip()) for line in tail)


async def check_passive_shell_output(
    client: TelegramClient,
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

    # Capture with scrollback for reliable command echo finding. On capped
    # backends (herdr) request the full readable history so the echo stays in
    # the window for as long as the backend allows.
    requested = _passive_scrollback_lines()
    capture = await _capture_with_scrollback(window_id, history=requested)
    if capture is None or not capture.text:
        return

    # Echoless tail recovery is only safe when a command actually ran in this
    # prompt session: a Telegram-issued command, or an echo/output the monitor
    # already tracked. Fresh state (existing shell just bound, provider switch)
    # has none, so stale scrollback above the new prompt is not relayed.
    command_in_flight = bool(
        state.last_command_echo or state.msg_id is not None or state.telegram_command
    )
    extracted = _extract_passive_with_truncation(
        capture, requested, allow_echoless_recovery=command_in_flight
    )
    if extracted is None:
        if not (state.last_command_echo and state.msg_id is not None):
            _reset_monitor(state)
        return
    passive, truncated = extracted
    _apply_echoless_identity(state, passive)

    # The backend clamped the scrollback (herdr's 1000-line cap) and dropped
    # the tail: flag it so the relayed output is not read as the full command
    # output. The notice rides the relayed text and stays stable across polls.
    if truncated and passive.text:
        passive.text = f"{_TRUNCATION_NOTICE}\n{passive.text}"

    if (
        passive.command_echo != state.last_command_echo
        or passive.echo_index != state.last_echo_index
    ):
        state.last_command_echo = passive.command_echo
        state.last_echo_index = passive.echo_index
        state.msg_id = None
        state.last_output = ""
        state.exit_code_sent = False
        state.tracked_done = False

    await _relay_passive_output(client, user_id, thread_id, state, passive, window_id)
    # A completed command's exit code has now been relayed; an echoless tail
    # seen after this marks a new command, not a continuation of this one.
    if passive.exit_code is not None:
        state.tracked_done = True


def _command_from_echo(echo: str) -> str:
    """Extract the command text from a prompt echo line.

    ``"~/code ⌘0⌘ ls -al"`` → ``"ls -al"`` (wrap mode).
    """
    m = match_prompt(echo)
    return m.trailing_text.strip() if m else echo


async def _relay_passive_output(
    client: TelegramClient,
    user_id: int,
    thread_id: int,
    state: _ShellMonitorState,
    passive: _PassiveOutput,
    window_id: str,
) -> None:
    """Relay extracted output to Telegram.

    Formats as: ``❯ <command>`` header followed by output in a code block.
    """
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)

    if passive.text != state.last_output:
        state.last_output = passive.text
        cmd = _command_from_echo(passive.command_echo)
        combined = f"❯ {cmd}\n{passive.text}" if cmd else passive.text
        state.msg_id = await _relay_output(
            client, chat_id, thread_id, combined, msg_id=state.msg_id
        )

    if (
        passive.exit_code is not None
        and passive.exit_code != 0
        and not state.exit_code_sent
        and state.msg_id
    ):
        state.exit_code_sent = True
        await _update_error_message(
            client, chat_id, state.msg_id, passive.exit_code, passive.text
        )

    # ✅/❌ reaction on the user's source message once exit is known.
    # Skips when the message was already reacted (dedupe in react()).
    if passive.exit_code is not None and state.telegram_message_id:
        await _react_exit(client, chat_id, state.telegram_message_id, passive.exit_code)
        if passive.exit_code == 0:
            state.telegram_message_id = 0

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
        state.telegram_message_id = 0
        state.telegram_generation = 0
        await _maybe_suggest_fix(
            client,
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
