"""Terminal output capture and relay for shell provider sessions.

Captures raw shell output from tmux panes and relays it to Telegram,
using in-place message editing for streaming updates.

Key components:
  - start_shell_capture: Launch a background capture task
  - _capture_shell_output: Background loop that polls and relays output
"""

import asyncio
import structlog
from dataclasses import dataclass

from telegram import Bot

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

# Active shell capture tasks: (user_id, thread_id) -> asyncio.Task
_shell_capture_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}


@dataclass
class _CaptureState:
    """Mutable state for a shell capture session."""

    msg_id: int | None = None
    last_output: str = ""
    stable_count: int = 0
    baseline_hash: int = 0


def start_shell_capture(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
) -> None:
    """Launch a background task to capture shell output."""
    key = (user_id, thread_id)
    existing = _shell_capture_tasks.pop(key, None)
    if existing and not existing.done():
        existing.cancel()

    task = asyncio.create_task(
        _capture_shell_output(bot, user_id, thread_id, window_id)
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
    """Send or edit the output message in Telegram."""
    display = output
    if len(display) > _OUTPUT_LIMIT:
        display = "\u2026 " + display[-_OUTPUT_LIMIT:]

    if state.msg_id is None:
        sent = await rate_limit_send_message(
            bot,
            chat_id,
            display,
            message_thread_id=thread_id,
        )
        if sent:
            state.msg_id = sent.message_id
    else:
        await edit_with_fallback(bot, chat_id, state.msg_id, display)


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

    output = raw.rstrip()
    content_hash = hash(output)

    if output == state.last_output:
        if content_hash != state.baseline_hash and output:
            state.stable_count += 1
            if state.stable_count >= _STABLE_THRESHOLD:
                return True
        return False

    state.stable_count = 0
    state.last_output = output
    await _relay_output(bot, chat_id, thread_id, output, state)
    return False


async def _capture_shell_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
) -> None:
    """Background task: capture shell command output from tmux pane.

    Sends the first captured output as a new message, then edits it
    in-place as more output appears.  Stops when pane content stabilizes
    (2 consecutive polls with identical content after output has changed)
    or timeout is reached.
    """
    try:
        await asyncio.sleep(1.0)

        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        state = _CaptureState()

        baseline_raw = await tmux_manager.capture_pane(window_id)
        state.baseline_hash = hash(baseline_raw.rstrip()) if baseline_raw else 0

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
    except asyncio.CancelledError:
        return
    finally:
        key = (user_id, thread_id)
        if _shell_capture_tasks.get(key) is asyncio.current_task():
            _shell_capture_tasks.pop(key, None)
