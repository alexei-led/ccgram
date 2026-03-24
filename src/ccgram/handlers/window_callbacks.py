"""Window picker callback handlers.

Handles inline keyboard callbacks for the window picker UI:
  - CB_WIN_BIND: Bind an existing unbound tmux window to the current topic
  - CB_WIN_NEW: Transition from window picker to directory browser for new session
  - CB_WIN_CANCEL: Cancel the window picker

Key function: handle_window_callback (uniform callback handler signature).
"""

import structlog
from pathlib import Path

from telegram import Bot, CallbackQuery, Chat, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..session import session_manager
from ..tmux_manager import tmux_manager
from .callback_data import CB_WIN_BIND, CB_WIN_CANCEL, CB_WIN_NEW
from .callback_helpers import get_thread_id
from .directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    UNBOUND_WINDOWS_KEY,
    build_directory_browser,
    clear_window_picker_state,
)
from .message_sender import safe_edit, safe_send
from .topic_emoji import format_topic_name_for_mode
from .user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT

logger = structlog.get_logger()


def _get_topic_chat(update: Update, query: CallbackQuery) -> Chat | None:
    """Resolve the chat object for the current callback topic, if available."""
    query_message = (
        update.callback_query.message if update.callback_query else None
    ) or query.message
    return query_message.chat if query_message else None


def _store_group_chat_id(
    user_id: int, thread_id: int, update: Update, query: CallbackQuery
) -> None:
    """Persist group chat routing for a topic thread (best-effort)."""
    chat = _get_topic_chat(update, query)
    if chat and chat.type in ("group", "supergroup"):
        session_manager.set_group_chat_id(user_id, thread_id, chat.id)


async def handle_window_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle window picker callbacks.

    Dispatches to the appropriate sub-handler based on callback data prefix.
    """
    if data.startswith(CB_WIN_BIND):
        await _handle_bind(query, user_id, data, update, context)
    elif data == CB_WIN_NEW:
        await _handle_new(query, user_id, update, context)
    elif data == CB_WIN_CANCEL:
        await _handle_cancel(query, update, context)


async def _detect_and_setup_provider(
    window_id: str,
    pane_current_command: str | None,
    *,
    pane_tty: str = "",
    bot: "Bot | None" = None,  # noqa: ARG001
    user_id: int = 0,  # noqa: ARG001
    thread_id: int = 0,  # noqa: ARG001
) -> str:
    """Detect provider from pane process and set up prompt if shell.

    Uses TTY-based detection (ps foreground process) when available,
    falling back to basename-only matching.
    Returns the detected provider name (empty string if undetected).
    """
    from ..providers import detect_provider_from_pane

    detected = (
        await detect_provider_from_pane(
            pane_current_command, pane_tty=pane_tty, window_id=window_id
        )
        if pane_current_command
        else ""
    )
    if detected:
        session_manager.set_window_provider(window_id, detected)
        if detected == "shell":
            from ..providers.shell import setup_shell_prompt

            await setup_shell_prompt(window_id, clear=False)
    return detected


async def _forward_pending_text(
    bot: "Bot",
    user_id: int,
    thread_id: int,
    window_id: str,
    text: str,
    provider_name: str,
    *,
    is_existing_window: bool = False,
) -> None:
    """Forward pending text to a newly bound window, routing shell via LLM.

    Args:
        is_existing_window: True when binding an existing window (not a fresh
            one from directory browser).  For shell, skips handle_shell_message
            to avoid _ensure_prompt_marker racing with the offer keyboard.
    """
    if provider_name == "shell" and not is_existing_window:
        from .shell_commands import handle_shell_message

        await handle_shell_message(bot, user_id, thread_id, window_id, text)
    else:
        # For non-shell providers or existing shell windows, send raw text.
        # Existing shell windows skip handle_shell_message to avoid
        # _ensure_prompt_marker racing with the offer keyboard just shown.
        send_ok, send_msg = await session_manager.send_to_window(window_id, text)
        if not send_ok:
            logger.warning("Failed to forward pending text: %s", send_msg)
            await safe_send(
                bot,
                session_manager.resolve_chat_id(user_id, thread_id),
                f"❌ Failed to send pending message: {send_msg}",
                message_thread_id=thread_id,
            )


async def _handle_bind(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_WIN_BIND: bind existing unbound window to current topic."""
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    if pending_tid is not None and get_thread_id(update) != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    try:
        idx = int(data[len(CB_WIN_BIND) :])
    except ValueError:
        await query.answer("Invalid data")
        return

    cached_windows: list[str] = (
        context.user_data.get(UNBOUND_WINDOWS_KEY, []) if context.user_data else []
    )
    if idx < 0 or idx >= len(cached_windows):
        await query.answer("Window list changed, please retry", show_alert=True)
        return
    selected_wid = cached_windows[idx]

    w = await tmux_manager.find_window_by_id(selected_wid)
    if not w:
        display = session_manager.get_display_name(selected_wid)
        await query.answer(f"Window '{display}' no longer exists", show_alert=True)
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Not in a topic", show_alert=True)
        return

    display = w.window_name
    clear_window_picker_state(context.user_data)
    session_manager.bind_thread(user_id, thread_id, selected_wid, window_name=display)
    _store_group_chat_id(user_id, thread_id, update, query)

    detected = await _detect_and_setup_provider(
        selected_wid,
        w.pane_current_command,
        pane_tty=w.pane_tty,
        bot=context.bot,
        user_id=user_id,
        thread_id=thread_id,
    )

    try:
        await context.bot.edit_forum_topic(
            chat_id=session_manager.resolve_chat_id(user_id, thread_id),
            message_thread_id=thread_id,
            name=format_topic_name_for_mode(
                display, session_manager.get_approval_mode(selected_wid)
            ),
        )
    except TelegramError as e:
        logger.debug("Failed to rename topic: %s", e)

    await safe_edit(
        query,
        f"✅ Bound to window `{display}`",
    )

    pending_text = (
        context.user_data.get(PENDING_THREAD_TEXT) if context.user_data else None
    )
    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_TEXT, None)
        context.user_data.pop(PENDING_THREAD_ID, None)
    if pending_text:
        await _forward_pending_text(
            context.bot,
            user_id,
            thread_id,
            selected_wid,
            pending_text,
            detected,
            is_existing_window=True,
        )
    await query.answer("Bound")


async def _handle_new(
    query: CallbackQuery,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_WIN_NEW: transition from window picker to directory browser."""
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    if pending_tid is not None and get_thread_id(update) != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    clear_window_picker_state(context.user_data)
    start_path = str(Path.cwd())
    msg_text, keyboard, subdirs = build_directory_browser(start_path, user_id=user_id)
    if context.user_data is not None:
        context.user_data[STATE_KEY] = STATE_BROWSING_DIRECTORY
        context.user_data[BROWSE_PATH_KEY] = start_path
        context.user_data[BROWSE_PAGE_KEY] = 0
        context.user_data[BROWSE_DIRS_KEY] = subdirs
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer()


async def _handle_cancel(
    query: CallbackQuery,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_WIN_CANCEL: cancel the window picker."""
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    if pending_tid is not None and get_thread_id(update) != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    clear_window_picker_state(context.user_data)
    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_ID, None)
        context.user_data.pop(PENDING_THREAD_TEXT, None)
    await safe_edit(query, "Cancelled")
    await query.answer("Cancelled")
