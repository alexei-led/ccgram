"""Text message handling — step functions for the text_handler orchestrator.

Routes incoming text messages through a bool early-return chain:
UI guards → unbound topic → dead window recovery → message forwarding.

Each step returns True if it handled the request (stop) or False to continue.
The orchestrator (handle_text_message) calls steps in sequence.
"""

import asyncio
import structlog
from pathlib import Path
from telegram import Bot, Message, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from .callback_helpers import get_thread_id as _get_thread_id
from .directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    STATE_SELECTING_WINDOW,
    UNBOUND_WINDOWS_KEY,
    build_directory_browser,
    build_window_picker,
    clear_browse_state,
    clear_window_picker_state,
)
from .interactive_ui import get_interactive_window, handle_interactive_ui
from .message_queue import enqueue_status_update
from .message_sender import (
    ack_reaction,
    edit_with_fallback,
    rate_limit_send_message,
    safe_reply,
)
from .recovery_callbacks import build_recovery_keyboard
from .polling_strategies import clear_probe_failures
from .user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT, RECOVERY_WINDOW_ID
from ..session import session_manager
from ..thread_router import thread_router
from ..providers import get_provider_for_window
from ..tmux_manager import tmux_manager
from ..utils import handle_general_topic_message, is_general_topic, task_done_callback

logger = structlog.get_logger()

# Maximum characters for bash output before truncation (fits Telegram 4096-char limit)
_BASH_OUTPUT_LIMIT = 3800

# Active bash capture tasks: (user_id, thread_id) -> asyncio.Task
_bash_capture_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}


def _cancel_bash_capture(user_id: int, thread_id: int) -> None:
    """Cancel any running bash capture for this topic."""
    key = (user_id, thread_id)
    task = _bash_capture_tasks.pop(key, None)
    if task and not task.done():
        task.cancel()


async def _edit_bash_message(bot: Bot, chat_id: int, msg_id: int, output: str) -> None:
    """Edit an existing bash-output message with entity-based formatting fallback."""
    await edit_with_fallback(bot, chat_id, msg_id, output)


async def _capture_bash_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    command: str,
) -> None:
    """Background task: capture ``!`` bash command output from tmux pane.

    Sends the first captured output as a new message, then edits it
    in-place as more output appears.  Stops after 30 s or when cancelled
    (e.g. user sends a new message, which pushes content down).
    """
    try:
        # Wait for the command to start producing output
        await asyncio.sleep(2.0)

        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        msg_id: int | None = None
        last_output: str = ""

        for _ in range(30):
            raw = await tmux_manager.capture_pane(window_id)
            if raw is None:
                return

            output = get_provider_for_window(window_id).extract_bash_output(
                raw, command
            )
            if not output or output == last_output:
                await asyncio.sleep(1.0)
                continue

            last_output = output

            # Truncate to fit Telegram's 4096-char limit
            if len(output) > _BASH_OUTPUT_LIMIT:
                output = "\u2026 " + output[-_BASH_OUTPUT_LIMIT:]

            if msg_id is None:
                # First capture — send a new message
                sent = await rate_limit_send_message(
                    bot,
                    chat_id,
                    output,
                    message_thread_id=thread_id,
                )
                if sent:
                    msg_id = sent.message_id
            else:
                await _edit_bash_message(bot, chat_id, msg_id, output)

            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return
    finally:
        key = (user_id, thread_id)
        if _bash_capture_tasks.get(key) is asyncio.current_task():
            _bash_capture_tasks.pop(key, None)


async def _check_ui_guards(
    user_data: dict | None, thread_id: int | None, message: Message
) -> bool:
    """Block text while a window picker or directory browser is active.

    Returns True if the message was handled (blocked), False to continue.
    """
    if not user_data:
        return False

    # Window picker guard
    if user_data.get(STATE_KEY) == STATE_SELECTING_WINDOW:
        pending_tid = user_data.get(PENDING_THREAD_ID)
        if pending_tid == thread_id:
            await safe_reply(
                message,
                "Please use the window picker above, or tap Cancel.",
            )
            return True
        # Stale picker state from a different thread — clear it
        clear_window_picker_state(user_data)
        user_data.pop(PENDING_THREAD_ID, None)
        user_data.pop(PENDING_THREAD_TEXT, None)

    # Directory browser guard
    if user_data.get(STATE_KEY) == STATE_BROWSING_DIRECTORY:
        pending_tid = user_data.get(PENDING_THREAD_ID)
        if pending_tid == thread_id:
            await safe_reply(
                message,
                "Please use the directory browser above, or tap Cancel.",
            )
            return True
        # Stale browsing state from a different thread — clear it
        clear_browse_state(user_data)
        user_data.pop(PENDING_THREAD_ID, None)
        user_data.pop(PENDING_THREAD_TEXT, None)

    return False


async def _handle_unbound_topic(
    user_id: int,
    thread_id: int,
    text: str,
    user_data: dict | None,
    message: Message,
) -> bool:
    """Show window picker or directory browser for an unbound topic.

    Returns True if the topic is unbound (handled), False if already bound.
    """
    window_id = thread_router.get_window_for_thread(user_id, thread_id)
    if window_id is not None:
        return False

    all_windows = await tmux_manager.list_windows()
    external_windows = await tmux_manager.discover_external_sessions()
    all_windows.extend(external_windows)
    bound_ids = {bound_wid for _, _, bound_wid in thread_router.iter_thread_bindings()}
    unbound = [
        (w.window_id, w.window_name, w.cwd)
        for w in all_windows
        if w.window_id not in bound_ids
    ]
    logger.debug(
        "Window picker check: all=%s, bound=%s, unbound=%s",
        [w.window_name for w in all_windows],
        bound_ids,
        [name for _, name, _ in unbound],
    )

    if unbound:
        logger.info(
            "Unbound topic: showing window picker (%d unbound windows, user=%d, thread=%d)",
            len(unbound),
            user_id,
            thread_id,
        )
        msg_text, keyboard, win_ids = build_window_picker(unbound)
        if user_data is not None:
            user_data[STATE_KEY] = STATE_SELECTING_WINDOW
            user_data[UNBOUND_WINDOWS_KEY] = win_ids
            user_data[PENDING_THREAD_ID] = thread_id
            user_data[PENDING_THREAD_TEXT] = text
        await safe_reply(message, msg_text, reply_markup=keyboard)
        return True

    # No unbound windows — show directory browser to create a new session
    logger.info(
        "Unbound topic: showing directory browser (user=%d, thread=%d)",
        user_id,
        thread_id,
    )
    start_path = str(Path.cwd())
    msg_text, keyboard, subdirs = build_directory_browser(start_path, user_id=user_id)
    if user_data is not None:
        user_data[STATE_KEY] = STATE_BROWSING_DIRECTORY
        user_data[BROWSE_PATH_KEY] = start_path
        user_data[BROWSE_PAGE_KEY] = 0
        user_data[BROWSE_DIRS_KEY] = subdirs
        user_data[PENDING_THREAD_ID] = thread_id
        user_data[PENDING_THREAD_TEXT] = text
    await safe_reply(message, msg_text, reply_markup=keyboard)
    return True


async def _handle_dead_window(
    window_id: str,
    user_id: int,
    thread_id: int,
    text: str,
    user_data: dict | None,
    message: Message,
) -> bool:
    """Show recovery UI or directory browser for a dead (killed) window.

    Returns True if the window is dead (handled), False if still alive.
    """
    w = await tmux_manager.find_window_by_id(window_id)
    if w:
        return False

    display = thread_router.get_display_name(window_id)
    window_state = session_manager.get_window_state(window_id)
    cwd = window_state.cwd if window_state.cwd else ""

    if not cwd or not Path(cwd).is_dir():
        # No valid cwd — unbind and fall back to directory browser
        logger.info(
            "Dead window %s (no valid cwd), falling back to directory browser"
            " (user=%d, thread=%d)",
            window_id,
            user_id,
            thread_id,
        )
        thread_router.unbind_thread(user_id, thread_id)
        from .polling_strategies import clear_dead_notification

        clear_dead_notification(user_id, thread_id)
        start_path = str(Path.cwd())
        msg_text, keyboard, subdirs = build_directory_browser(
            start_path, user_id=user_id
        )
        if user_data is not None:
            user_data[STATE_KEY] = STATE_BROWSING_DIRECTORY
            user_data[BROWSE_PATH_KEY] = start_path
            user_data[BROWSE_PAGE_KEY] = 0
            user_data[BROWSE_DIRS_KEY] = subdirs
            user_data[PENDING_THREAD_ID] = thread_id
            user_data[PENDING_THREAD_TEXT] = text
        await safe_reply(message, msg_text, reply_markup=keyboard)
        return True

    # Show recovery UI
    logger.info(
        "Dead window %s (%s), showing recovery UI (user=%d, thread=%d)",
        window_id,
        display,
        user_id,
        thread_id,
    )
    if user_data is not None:
        user_data[PENDING_THREAD_ID] = thread_id
        user_data[PENDING_THREAD_TEXT] = text
        user_data[RECOVERY_WINDOW_ID] = window_id
    keyboard = build_recovery_keyboard(window_id)
    await safe_reply(
        message,
        f"\u26a0 Window `{display}` is no longer running.\n"
        f"\U0001f4c2 `{cwd}`\n\n"
        "How would you like to recover?",
        reply_markup=keyboard,
    )
    return True


async def _forward_message(
    window_id: str,
    user_id: int,
    thread_id: int,
    text: str,
    bot: Bot,
    message: Message,
) -> None:
    """Forward a text message to the bound tmux window."""
    await message.chat.send_action(ChatAction.TYPING)  # type: ignore[union-attr]
    # Enqueue a status clear to actually delete the Telegram message
    # (clear_status_msg_info only clears the tracking dict, leaving a ghost)
    await enqueue_status_update(bot, user_id, window_id, None, thread_id)

    # Cancel any running bash capture — new message pushes pane content down
    _cancel_bash_capture(user_id, thread_id)

    clear_probe_failures(window_id)

    success, err_message = await session_manager.send_to_window(window_id, text)
    if not success:
        await safe_reply(message, f"\u274c {err_message}")
        return

    await ack_reaction(bot, message.chat.id, message.message_id)

    from .command_history import record_command

    record_command(user_id, thread_id, text)

    # Start background capture for ! bash command output
    if text.startswith("!") and len(text) > 1:
        bash_cmd = text[1:]  # strip leading "!"
        task = asyncio.create_task(
            _capture_bash_output(bot, user_id, thread_id, window_id, bash_cmd)
        )
        task.add_done_callback(task_done_callback)
        _bash_capture_tasks[(user_id, thread_id)] = task

    # If in interactive mode, refresh the UI after sending text
    interactive_window = get_interactive_window(user_id, thread_id)
    if interactive_window and interactive_window == window_id:
        await asyncio.sleep(0.2)
        await handle_interactive_ui(bot, user_id, window_id, thread_id)


async def handle_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Orchestrate text message handling via bool early-return chain.

    Called after auth validation in bot.py's text_handler.
    """
    user = update.effective_user
    message = update.message
    assert user is not None  # guaranteed by caller
    assert message is not None and message.text  # guaranteed by caller

    text = message.text
    thread_id = _get_thread_id(update)

    # Store group chat_id for forum topic message routing
    chat = message.chat
    if chat.type in ("group", "supergroup") and thread_id is not None:
        thread_router.set_group_chat_id(user.id, thread_id, chat.id)

    # UI guards (window picker / directory browser active)
    if await _check_ui_guards(context.user_data, thread_id, message):
        return

    # Must be in a named topic
    if thread_id is None:
        if message and update.effective_chat and is_general_topic(message):
            await handle_general_topic_message(
                context.bot, message, update.effective_chat.id
            )
        else:
            await safe_reply(
                message,
                "\u274c Please use a named topic. Create a new topic to start a session.",
            )
        return

    # Unbound topic — show picker or browser
    if await _handle_unbound_topic(
        user.id, thread_id, text, context.user_data, message
    ):
        return

    # Bound topic — check if window is still alive
    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    assert window_id is not None  # _handle_unbound_topic returned False

    if await _handle_dead_window(
        window_id, user.id, thread_id, text, context.user_data, message
    ):
        return

    # Shell provider: route through LLM or raw execution
    provider = get_provider_for_window(window_id)
    if provider.capabilities.name == "shell":
        from .shell_commands import handle_shell_message

        await handle_shell_message(
            context.bot, user.id, thread_id, window_id, text, message
        )
        return

    # Forward message to window
    await _forward_message(window_id, user.id, thread_id, text, context.bot, message)
