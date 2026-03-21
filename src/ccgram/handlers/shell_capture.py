"""Terminal output capture and relay for shell provider sessions.

Captures raw shell output from tmux panes and relays it to Telegram,
using in-place message editing for streaming updates.

Key components:
  - start_shell_capture: Launch a background capture task
  - _capture_shell_output: Background loop that polls and relays output
"""

import asyncio
import structlog

from telegram import Bot

from ..session import session_manager
from ..tmux_manager import tmux_manager
from ..utils import task_done_callback
from .message_sender import edit_with_fallback, rate_limit_send_message
from .status_polling import is_shell_prompt

logger = structlog.get_logger()

# Maximum characters per message (fits Telegram 4096-char limit with margin)
_OUTPUT_LIMIT = 3800

# Maximum capture duration in seconds
_CAPTURE_TIMEOUT = 60

# Active shell capture tasks: (user_id, thread_id) -> asyncio.Task
_shell_capture_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}


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


async def _poll_once(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    window_id: str,
    msg_id: int | None,
    last_output: str,
) -> tuple[int | None, str, bool]:
    """Single poll iteration. Returns (msg_id, last_output, should_stop)."""
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        return msg_id, last_output, True

    raw = await tmux_manager.capture_pane(window_id)
    if raw is None:
        return msg_id, last_output, False

    output = raw.rstrip()
    if not output or output == last_output:
        if last_output and is_shell_prompt(w.pane_current_command):
            return msg_id, last_output, True
        return msg_id, last_output, False

    last_output = output

    if len(output) > _OUTPUT_LIMIT:
        output = "\u2026 " + output[-_OUTPUT_LIMIT:]

    if msg_id is None:
        sent = await rate_limit_send_message(
            bot,
            chat_id,
            output,
            message_thread_id=thread_id,
        )
        if sent:
            msg_id = sent.message_id
    else:
        await edit_with_fallback(bot, chat_id, msg_id, output)

    if is_shell_prompt(w.pane_current_command):
        return msg_id, last_output, True

    return msg_id, last_output, False


async def _capture_shell_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
) -> None:
    """Background task: capture shell command output from tmux pane.

    Sends the first captured output as a new message, then edits it
    in-place as more output appears. Stops when shell prompt returns
    or timeout is reached.
    """
    try:
        await asyncio.sleep(1.0)

        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        msg_id: int | None = None
        last_output: str = ""

        for _ in range(_CAPTURE_TIMEOUT):
            msg_id, last_output, should_stop = await _poll_once(
                bot,
                chat_id,
                thread_id,
                window_id,
                msg_id,
                last_output,
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
