"""Telegram bot handlers — the main UI layer of CCGram.

Registers all command/callback/message handlers and manages the bot lifecycle.
Each Telegram topic maps 1:1 to a tmux window (Claude session).

Core responsibilities:
  - Command handlers: /new (+ /start alias), /history, /sessions, /resume,
    /screenshot, /panes, /toolbar, /restore, plus forwarding unknown /commands to Claude Code via tmux.
  - Callback query handler: thin dispatcher routing to dedicated handler modules.
  - Topic-based routing: each named topic binds to one tmux window.
    Unbound topics trigger the directory browser to create a new session.
  - Topic lifecycle: closing a topic unbinds the window (kept alive for
    rebinding). Unbound windows are auto-killed after TTL by status polling.
    Unsupported content (images, stickers, etc.) is rejected with a warning.
  - Bot lifecycle management: post_init, post_shutdown, create_bot.

Key functions: create_bot(), handle_new_message().
"""

import asyncio
import contextlib
import structlog
import os
import re
import signal
from pathlib import Path

from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
)
from telegram.error import BadRequest, Conflict, NetworkError, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from .cc_commands import (
    discover_provider_commands,
    register_commands,
)
from .providers import (
    get_provider,
    get_provider_for_window,
)
from .config import config
from .handlers.topic_orchestration import (
    adopt_unbound_windows as _adopt_unbound_windows,
    handle_new_window as _handle_new_window,
)
from .handlers.command_orchestration import (
    forward_command_handler,
    sync_scoped_menu_for_text_context as _sync_scoped_menu_for_text_context,
    sync_scoped_provider_menu as _sync_scoped_provider_menu,
    setup_menu_refresh_job,
)
from .handlers.callback_data import (
    CB_DIR_CANCEL,
    CB_DIR_CONFIRM,
    CB_DIR_FAV,
    CB_DIR_HOME,
    CB_DIR_PAGE,
    CB_DIR_SELECT,
    CB_DIR_STAR,
    CB_DIR_UP,
    CB_MODE_SELECT,
    CB_PROV_SELECT,
    CB_HISTORY_NEXT,
    CB_HISTORY_PREV,
    CB_KEYS_PREFIX,
    CB_PANE_SCREENSHOT,
    CB_RECOVERY_BACK,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_PICK,
    CB_RECOVERY_RESUME,
    CB_RESUME_CANCEL,
    CB_RESUME_PAGE,
    CB_RESUME_PICK,
    CB_SCREENSHOT_REFRESH,
    CB_SESSIONS_KILL,
    CB_SESSIONS_KILL_CONFIRM,
    CB_SESSIONS_NEW,
    CB_SESSIONS_REFRESH,
    CB_STATUS_ESC,
    CB_STATUS_NOTIFY,
    CB_STATUS_RECALL,
    CB_STATUS_REMOTE,
    CB_STATUS_SCREENSHOT,
    CB_SYNC_DISMISS,
    CB_SYNC_FIX,
    CB_TOOLBAR_CTRLC,
    CB_TOOLBAR_DISMISS,
    CB_VOICE,
    CB_SHELL_CANCEL,
    CB_SHELL_CONFIRM_DANGER,
    CB_SHELL_EDIT,
    CB_SHELL_RUN,
    CB_WIN_BIND,
    CB_WIN_CANCEL,
    CB_WIN_NEW,
)
from .handlers.callback_helpers import get_thread_id as _get_thread_id
from .handlers.callback_helpers import user_owns_window as _user_owns_window
from .handlers.directory_callbacks import handle_directory_callback
from .handlers.history_callbacks import handle_history_callback
from .handlers.interactive_callbacks import (
    handle_interactive_callback,
    match_interactive_prefix as _match_interactive_prefix,
)
from .handlers.recovery_callbacks import handle_recovery_callback
from .handlers.restore_command import restore_command
from .handlers.resume_command import handle_resume_command_callback, resume_command
from .handlers.screenshot_callbacks import handle_screenshot_callback
from .handlers.voice_callbacks import handle_voice_callback
from .handlers.window_callbacks import handle_window_callback
from .handlers.directory_browser import clear_browse_state
from .handlers.cleanup import clear_topic_state
from .handlers.topic_emoji import strip_emoji_prefix, update_stored_topic_name
from .handlers.history import send_history
from .handlers.sessions_dashboard import (
    handle_sessions_kill,
    handle_sessions_kill_confirm,
    handle_sessions_refresh,
    sessions_command,
)
from .handlers.sync_command import (
    handle_sync_dismiss,
    handle_sync_fix,
    sync_command,
)
from .handlers.upgrade import upgrade_command
from .handlers.interactive_ui import (
    INTERACTIVE_TOOL_NAMES,
    clear_interactive_mode,
    clear_interactive_msg,
    get_interactive_msg_id,
    handle_interactive_ui,
    set_interactive_mode,
)
from .handlers.message_queue import (
    enqueue_content_message,
    enqueue_status_update,
    get_message_queue,
    shutdown_workers,
)
from .handlers.message_sender import safe_reply
from .handlers.response_builder import build_response_parts
from .handlers.status_polling import status_poll_loop
from .handlers.file_handler import handle_document_message, handle_photo_message
from .handlers.voice_handler import handle_voice_message
from .handlers.text_handler import handle_text_message
from .session import session_manager
from .session_monitor import NewMessage, NewWindowEvent, SessionMonitor
from .thread_router import thread_router
from .telegram_request import ResilientPollingHTTPXRequest
from .tmux_manager import tmux_manager
from .utils import handle_general_topic_message, is_general_topic, task_done_callback

logger = structlog.get_logger()

# Error keyword pattern for errors_only notification mode (word boundaries)
_ERROR_KEYWORDS_RE = re.compile(
    r"\b(?:error|exception|failed|traceback|stderr|assertion)\b", re.IGNORECASE
)

# Max label length for /recall command buttons (wider than status bar buttons)
_RECALL_LABEL_MAX = 40
# Session monitor instance
session_monitor: SessionMonitor | None = None

# Status polling task
_status_poll_task: asyncio.Task | None = None


def is_user_allowed(user_id: int | None) -> bool:
    return user_id is not None and config.is_user_allowed(user_id)


# Group filter: when CCBOT_GROUP_ID is set, only process updates from that group.
# filters.ALL is a no-op — single-instance backward compat.
_group_filter: filters.BaseFilter = (
    filters.Chat(chat_id=config.group_id) if config.group_id else filters.ALL
)


# --- Command handlers ---


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        if update.message:
            await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    clear_browse_state(context.user_data)

    if update.message:
        await safe_reply(
            update.message,
            "\U0001f916 *Claude Code Monitor*\n\n"
            "Each topic is a session. Create a new topic to start.",
        )


async def history_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show message history for the active session or bound thread."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    window_id = thread_router.resolve_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "\u274c No session bound to this topic.")
        return

    provider = get_provider_for_window(window_id)
    if not provider.capabilities.supports_structured_transcript:
        await safe_reply(update.message, "No transcript available for this provider.")
        return

    await send_history(update.message, window_id)


async def commands_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show provider-specific slash commands for the current topic."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    window_id = thread_router.resolve_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "\u274c No session bound to this topic.")
        return

    provider = get_provider_for_window(window_id)
    await _sync_scoped_provider_menu(update.message, user.id, provider)
    commands = discover_provider_commands(provider)
    if not commands:
        await safe_reply(
            update.message,
            f"Provider: `{provider.capabilities.name}`\nNo discoverable commands.",
        )
        return

    lines = [f"Provider: `{provider.capabilities.name}`", "Supported commands:"]
    for cmd in sorted(commands, key=lambda c: c.telegram_name):
        if not cmd.telegram_name:
            continue
        original = cmd.name if cmd.name.startswith("/") else f"/{cmd.name}"
        lines.append(f"- `/{cmd.telegram_name}` \u2192 `{original}`")
    await safe_reply(update.message, "\n".join(lines))


async def topic_closed_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle topic closure — unbind thread but keep the tmux window alive.

    The window becomes "unbound" and is available for rebinding via the window
    picker when a new topic is created. Unbound windows are auto-killed after
    the configured TTL (autoclose_done_minutes) by the status polling loop.
    """
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if window_id:
        display = thread_router.get_display_name(window_id)
        thread_router.unbind_thread(user.id, thread_id)
        # Clean up all memory state for this topic
        await clear_topic_state(
            user.id, thread_id, context.bot, context.user_data, window_id=window_id
        )
        logger.info(
            "Topic closed: window %s unbound (kept alive for rebinding, user=%d, thread=%d)",
            display,
            user.id,
            thread_id,
        )
    else:
        logger.debug(
            "Topic closed: no binding (user=%d, thread=%d)", user.id, thread_id
        )


async def topic_edited_handler(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle topic rename — sync new name to tmux window and emoji cache.

    Ignores icon-only edits (name is None) and emoji-only changes from the bot
    itself (clean name unchanged after stripping prefixes).
    """
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message or not update.message.forum_topic_edited:
        return

    new_name = update.message.forum_topic_edited.name
    if not new_name:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    window_id = thread_router.get_window_for_chat_thread(chat_id, thread_id)
    if not window_id:
        logger.debug("Topic edited: no binding (thread=%d)", thread_id)
        return

    clean_name = strip_emoji_prefix(new_name)

    # Loop guard: if clean name matches current display name, this was a
    # bot-originated emoji/mode change — skip to prevent rename loops.
    current_display = thread_router.get_display_name(window_id)
    if current_display and strip_emoji_prefix(current_display) == clean_name:
        logger.debug(
            "Topic edited: name unchanged after strip, skipping (thread=%d)", thread_id
        )
        return

    renamed = await tmux_manager.rename_window(window_id, clean_name)
    if renamed:
        thread_router.set_display_name(window_id, clean_name)
        update_stored_topic_name(chat_id, thread_id, clean_name)
        logger.info(
            "Topic renamed: window %s → %r (thread=%d)",
            window_id,
            clean_name,
            thread_id,
        )


async def unbind_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Disconnect a topic from its tmux window without killing the session."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    display = thread_router.get_display_name(window_id)
    # Enqueue a status clear to actually delete the Telegram status message
    # (clear_topic_state only clears the tracking dict, leaving a ghost)
    await enqueue_status_update(context.bot, user.id, window_id, None, thread_id)
    await clear_topic_state(
        user.id, thread_id, context.bot, context.user_data, window_id=window_id
    )
    thread_router.unbind_thread(user.id, thread_id)
    await safe_reply(
        update.message,
        f"\u2702 Unbound from window `{display}`. The session is still running.\n"
        "Send a message in this topic to rebind or create a new session.",
    )


async def screenshot_command(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Capture and send a terminal screenshot for the current topic."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await safe_reply(update.message, "\u274c Window no longer exists.")
        return

    pane_text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not pane_text:
        await safe_reply(update.message, "\u274c Failed to capture terminal.")
        return

    import io

    from .handlers.screenshot_callbacks import build_screenshot_keyboard
    from .screenshot import text_to_image

    png_bytes = await text_to_image(pane_text, with_ansi=True)
    keyboard = build_screenshot_keyboard(window_id)
    chat_id = thread_router.resolve_chat_id(user.id, thread_id)
    try:
        await update.message.get_bot().send_document(
            chat_id=chat_id,
            document=io.BytesIO(png_bytes),
            filename="screenshot.png",
            reply_markup=keyboard,
            message_thread_id=thread_id,
        )
    except TelegramError as e:
        logger.error("Failed to send screenshot: %s", e)
        await safe_reply(update.message, "\u274c Failed to send screenshot.")


async def panes_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all panes in the current topic's window."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    panes = await tmux_manager.list_panes(window_id)
    if len(panes) <= 1:
        await safe_reply(
            update.message,
            "\U0001f4d0 Single pane \u2014 no multi-pane layout detected.",
        )
        return

    from .handlers.status_polling import has_pane_alert

    lines = [f"\U0001f4d0 {len(panes)} panes in window\n"]
    buttons: list[InlineKeyboardButton] = []
    for pane in panes:
        prefix = "\U0001f4cd" if pane.active else "  "
        label = f"Pane {pane.index} ({pane.command})"
        suffix_parts: list[str] = []
        if pane.active:
            suffix_parts.append("active")
        if has_pane_alert(pane.pane_id):
            prefix = "\u26a0\ufe0f"
            suffix_parts.append("blocked")
        elif not pane.active:
            suffix_parts.append("running")
        suffix = f" \u2014 {', '.join(suffix_parts)}" if suffix_parts else ""
        lines.append(f"{prefix} {label}{suffix}")
        buttons.append(
            InlineKeyboardButton(
                f"\U0001f4f7 {pane.index}",
                callback_data=f"{CB_PANE_SCREENSHOT}{window_id}:{pane.pane_id}"[:64],
            )
        )

    keyboard = InlineKeyboardMarkup([buttons]) if buttons else None
    await safe_reply(update.message, "\n".join(lines), reply_markup=keyboard)


async def recall_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent command history for the current topic."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    from .handlers.command_history import (
        INLINE_QUERY_MAX,
        get_history,
        truncate_for_display,
    )

    history = get_history(user.id, thread_id, limit=10)
    if not history:
        await safe_reply(update.message, "\U0001f4cb No command history yet.")
        return

    rows = []
    for cmd in history:
        label = truncate_for_display(cmd, _RECALL_LABEL_MAX)
        query = cmd[:INLINE_QUERY_MAX]
        rows.append(
            [InlineKeyboardButton(label, switch_inline_query_current_chat=query)]
        )
    keyboard = InlineKeyboardMarkup(rows)
    await safe_reply(
        update.message, "\U0001f4cb Recent commands:", reply_markup=keyboard
    )


async def toolbar_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show persistent action toolbar with inline keyboard buttons."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    from .handlers.screenshot_callbacks import build_toolbar_keyboard

    keyboard = build_toolbar_keyboard(window_id)
    display = thread_router.get_display_name(window_id)
    await safe_reply(
        update.message,
        f"\U0001f39b `{display}` toolbar",
        reply_markup=keyboard,
    )


async def verbose_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle tool call batching for this topic."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    new_mode = session_manager.cycle_batch_mode(window_id)
    if new_mode == "batched":
        await safe_reply(
            update.message,
            "\u26a1 Tool calls will be *batched* into a single message.",
        )
    else:
        await safe_reply(
            update.message,
            "\U0001f4ac Tool calls will be sent *individually* (verbose mode).",
        )


async def inline_query_handler(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Echo query text as a sendable inline result."""
    if not update.inline_query:
        return
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    text = update.inline_query.query.strip()
    if not text:
        await update.inline_query.answer([])
        return

    result = InlineQueryResultArticle(
        id="cmd",
        title=text,
        description="Tap to send",
        input_message_content=InputTextMessageContent(message_text=text),
    )
    await update.inline_query.answer([result], cache_time=0, is_personal=True)


async def unsupported_content_handler(
    update: Update,
    _context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Reply to non-text messages (images, stickers, voice, etc.)."""
    if not update.message:
        return
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    logger.debug("Unsupported content from user %d", user.id)
    # Omit "voice" from the list when whisper is configured (has its own handler)
    media_list = (
        "Stickers, voice, video" if not config.whisper_provider else "Stickers, video"
    )
    await safe_reply(
        update.message,
        f"\u26a0 {media_list}, and similar media are not supported. Use text, photos, or documents.",
    )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        if update.message:
            await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    if not update.message or not update.message.text:
        return

    await _sync_scoped_menu_for_text_context(update, user.id)
    await handle_text_message(update, context)


# --- Callback query handler (thin dispatcher) ---

# Callback prefixes that route to dedicated handler modules.
# Order matters: prefixes checked via startswith must be longest-first
# to avoid false matches (e.g. CB_SESSIONS_KILL_CONFIRM before CB_SESSIONS_KILL).
_CB_HISTORY = (CB_HISTORY_PREV, CB_HISTORY_NEXT)
_CB_DIRECTORY = (
    CB_DIR_FAV,
    CB_DIR_STAR,
    CB_DIR_SELECT,
    CB_DIR_UP,
    CB_DIR_HOME,
    CB_DIR_PAGE,
    CB_DIR_CONFIRM,
    CB_PROV_SELECT,
    CB_MODE_SELECT,
    CB_DIR_CANCEL,
)
_CB_WINDOW = (CB_WIN_BIND, CB_WIN_NEW, CB_WIN_CANCEL)
_CB_SCREENSHOT = (
    CB_SCREENSHOT_REFRESH,
    CB_STATUS_RECALL,
    CB_STATUS_ESC,
    CB_STATUS_NOTIFY,
    CB_STATUS_SCREENSHOT,
    CB_KEYS_PREFIX,
    CB_PANE_SCREENSHOT,
    CB_STATUS_REMOTE,
    CB_TOOLBAR_CTRLC,
    CB_TOOLBAR_DISMISS,
)
_CB_RECOVERY = (
    CB_RECOVERY_BACK,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_RESUME,
    CB_RECOVERY_PICK,
    CB_RECOVERY_CANCEL,
)
_CB_VOICE = (CB_VOICE,)
_CB_RESUME = (CB_RESUME_PICK, CB_RESUME_PAGE, CB_RESUME_CANCEL)
_CB_SHELL = (
    CB_SHELL_RUN,
    CB_SHELL_EDIT,
    CB_SHELL_CANCEL,
    CB_SHELL_CONFIRM_DANGER,
)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch callback queries to dedicated handler modules."""
    # CallbackQueryHandler doesn't support filters= param, so check inline.
    if config.group_id:
        chat = update.effective_chat
        if not chat or chat.id != config.group_id:
            return

    query = update.callback_query
    if not query or not query.data:
        return

    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        await query.answer("Not authorized")
        return

    # Store group chat_id for forum topic message routing
    if query.message and query.message.chat.type in ("group", "supergroup"):
        cb_thread_id = _get_thread_id(update)
        if cb_thread_id is not None:
            thread_router.set_group_chat_id(
                user.id, cb_thread_id, query.message.chat.id
            )

    data = query.data

    # History pagination
    if data.startswith(_CB_HISTORY):
        await handle_history_callback(query, user.id, data, update, context)

    # Directory browser
    elif data.startswith(_CB_DIRECTORY):
        await handle_directory_callback(query, user.id, data, update, context)

    # Window picker
    elif data.startswith(_CB_WINDOW):
        await handle_window_callback(query, user.id, data, update, context)

    # Screenshot / status buttons / quick keys
    elif data.startswith(_CB_SCREENSHOT):
        await handle_screenshot_callback(query, user.id, data, update, context)

    # No-op
    elif data == "noop":
        await query.answer()

    # Interactive UI (AskUserQuestion / ExitPlanMode navigation)
    elif _match_interactive_prefix(data):
        await handle_interactive_callback(query, user.id, data, update, context)

    # Recovery UI
    elif data.startswith(_CB_RECOVERY):
        await handle_recovery_callback(query, user.id, data, update, context)

    # Resume command UI
    elif data.startswith(_CB_RESUME):
        await handle_resume_command_callback(query, user.id, data, update, context)

    # Sessions dashboard
    elif data == CB_SESSIONS_REFRESH:
        await handle_sessions_refresh(query, user.id)
        await query.answer("Refreshed")
    elif data == CB_SESSIONS_NEW:
        await query.answer("Create a new topic to start a session.")
    elif data.startswith(CB_SESSIONS_KILL_CONFIRM):
        window_id = data[len(CB_SESSIONS_KILL_CONFIRM) :]
        if not _user_owns_window(user.id, window_id):
            await query.answer("Not your session", show_alert=True)
            return
        await handle_sessions_kill_confirm(query, user.id, window_id, context.bot)
        await query.answer("Killed")
    elif data.startswith(CB_SESSIONS_KILL):
        window_id = data[len(CB_SESSIONS_KILL) :]
        if not _user_owns_window(user.id, window_id):
            await query.answer("Not your session", show_alert=True)
            return
        await handle_sessions_kill(query, user.id, window_id)
        await query.answer()

    # Voice message callbacks
    elif data.startswith(_CB_VOICE):
        await handle_voice_callback(update, context)

    # Shell command approval
    elif data.startswith(_CB_SHELL):
        from .handlers.shell_commands import handle_shell_callback

        thread_id = _get_thread_id(update)
        await handle_shell_callback(query, user.id, data, context.bot, thread_id)

    # Sync command
    elif data == CB_SYNC_FIX:
        await handle_sync_fix(query)
        await query.answer("Fixed")
    elif data == CB_SYNC_DISMISS:
        await handle_sync_dismiss(query)
        await query.answer("Dismissed")


# --- Streaming response / notifications ---


async def handle_new_message(msg: NewMessage, bot: Bot) -> None:
    """Handle a new assistant message — enqueue for sequential processing.

    Messages are queued per-user to ensure status messages always appear last.
    Routes via thread_bindings to deliver to the correct topic.
    """
    status = "complete" if msg.is_complete else "streaming"
    logger.info(
        "handle_new_message [%s]: session=%s, text_len=%d",
        status,
        msg.session_id,
        len(msg.text),
    )

    # Find users whose thread-bound window matches this session
    active_users = session_manager.find_users_for_session(msg.session_id)

    if not active_users:
        logger.info("No active users for session %s", msg.session_id)
        return

    for user_id, window_id, thread_id in active_users:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            window_id=window_id, session_id=msg.session_id
        )
        # Check notification mode — skip suppressed messages.
        # All tool_use/tool_result MUST pass through regardless of mode: the message
        # queue edits tool_use messages in-place when tool_result arrives, so filtering
        # one half would break pairing and leave orphaned messages. This means muted/
        # errors_only sessions still deliver tool flow — an accepted trade-off.
        notif_mode = session_manager.get_notification_mode(window_id)
        is_tool_flow = msg.tool_name in INTERACTIVE_TOOL_NAMES or msg.content_type in (
            "tool_use",
            "tool_result",
        )
        if not is_tool_flow:
            if notif_mode == "muted":
                continue
            if notif_mode == "errors_only" and not _ERROR_KEYWORDS_RE.search(
                msg.text or ""
            ):
                continue

        # Handle interactive tools specially - capture terminal and send UI
        if msg.tool_name in INTERACTIVE_TOOL_NAMES and msg.content_type == "tool_use":
            # Mark interactive mode BEFORE sleeping so polling skips this window
            set_interactive_mode(user_id, window_id, thread_id)
            # Flush pending messages (e.g. plan content) before sending interactive UI
            queue = get_message_queue(user_id)
            if queue:
                await queue.join()
            # Wait briefly for Claude Code to render the question UI
            await asyncio.sleep(0.3)
            handled = await handle_interactive_ui(bot, user_id, window_id, thread_id)
            if handled:
                # Update user's read offset
                session = await session_manager.resolve_session_for_window(window_id)
                if session and session.file_path:
                    try:
                        file_size = Path(session.file_path).stat().st_size
                        session_manager.update_user_window_offset(
                            user_id, window_id, file_size
                        )
                    except OSError:
                        pass
                continue  # Don't send the normal tool_use message
            else:
                # UI not rendered — clear the early-set mode
                clear_interactive_mode(user_id, thread_id)

        # Any non-interactive message means the interaction is complete — delete the UI message
        if get_interactive_msg_id(user_id, thread_id):
            await clear_interactive_msg(user_id, bot, thread_id)

        parts = build_response_parts(
            msg.text,
            msg.is_complete,
            msg.content_type,
            msg.role,
        )

        if msg.is_complete:
            # Enqueue content message task
            # Note: tool_result editing is handled inside _process_content_task
            # to ensure sequential processing with tool_use message sending
            await enqueue_content_message(
                bot=bot,
                user_id=user_id,
                window_id=window_id,
                parts=parts,
                tool_use_id=msg.tool_use_id,
                content_type=msg.content_type,
                text=msg.text,
                thread_id=thread_id,
            )

            # Update user's read offset to current file position
            # This marks these messages as "read" for this user
            session = await session_manager.resolve_session_for_window(window_id)
            if session and session.file_path:
                try:
                    file_size = Path(session.file_path).stat().st_size
                    session_manager.update_user_window_offset(
                        user_id, window_id, file_size
                    )
                except OSError:
                    pass


# --- App lifecycle ---


def _global_exception_handler(
    _loop: asyncio.AbstractEventLoop, context: dict[str, object]
) -> None:
    """Last-resort handler for uncaught exceptions in asyncio tasks."""
    exc = context.get("exception")
    msg = context.get("message", "Unhandled exception in event loop")
    if isinstance(exc, BaseException):
        logger.error(
            "asyncio exception handler: %s",
            msg,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logger.error("asyncio exception handler: %s", msg)


async def post_init(application: Application) -> None:
    global session_monitor, _status_poll_task

    # Install global asyncio exception handler as safety net
    asyncio.get_running_loop().set_exception_handler(_global_exception_handler)

    default_provider = get_provider()
    try:
        await register_commands(application.bot, provider=default_provider)
    except TelegramError:
        logger.warning("Failed to register bot commands at startup, will retry later")
    setup_menu_refresh_job(application)

    # Re-resolve stale window IDs from persisted state against live tmux windows
    await session_manager.resolve_stale_ids()

    await _adopt_unbound_windows(application.bot)

    # Warn if Claude Code hooks are not installed (provider-aware, non-blocking)
    provider = get_provider()
    if provider.capabilities.supports_hook:
        from .hook import _claude_settings_file, get_installed_events

        settings_file = _claude_settings_file()
        import json

        if settings_file.exists():
            try:
                settings = json.loads(settings_file.read_text())
                events = get_installed_events(settings)
                missing = [e for e, ok in events.items() if not ok]
                if missing:
                    logger.warning(
                        "Claude Code hooks incomplete — %d missing: %s. "
                        "Run: ccgram hook --install",
                        len(missing),
                        ", ".join(missing),
                    )
            except (json.JSONDecodeError, OSError):  # fmt: skip
                logger.warning(
                    "Claude Code hooks not installed. Run: ccgram hook --install"
                )
        else:
            logger.warning(
                "Claude Code hooks not installed (%s missing). "
                "Run: ccgram hook --install",
                settings_file,
            )

    monitor = SessionMonitor()
    # Expose to other modules (status_polling activity heuristic)
    from ccgram.session_monitor import set_active_monitor

    set_active_monitor(monitor)

    async def message_callback(msg: NewMessage) -> None:
        await handle_new_message(msg, application.bot)

    monitor.set_message_callback(message_callback)

    async def new_window_callback(event: NewWindowEvent) -> None:
        await _handle_new_window(event, application.bot)

    monitor.set_new_window_callback(new_window_callback)

    # Wire hook event dispatcher for structured Claude Code events
    from ccgram.handlers.hook_events import HookEvent, dispatch_hook_event

    async def hook_event_callback(event: HookEvent) -> None:
        await dispatch_hook_event(event, application.bot)

    monitor.set_hook_event_callback(hook_event_callback)
    monitor.start()
    session_monitor = monitor
    logger.info("Session monitor started")

    # Start status polling task (routed through PTB error handler)
    _status_poll_task = asyncio.create_task(status_poll_loop(application.bot))
    _status_poll_task.add_done_callback(task_done_callback)
    logger.info("Status polling task started")


async def _send_shutdown_notification(application: Application) -> None:
    """Send a shutdown notification to the General topic if a group is configured."""
    from .main import _shutdown_signal

    if not config.group_id:
        return

    sig = _shutdown_signal
    reason = f"Received {signal.Signals(sig).name}" if sig else "Clean exit"

    from . import __version__

    text = f"🔌 ccgram stopped — {reason} (v{__version__})"
    try:
        await application.bot.send_message(
            chat_id=config.group_id,
            text=text,
            message_thread_id=1,  # General topic
        )
    except (TelegramError, RuntimeError) as exc:
        logger.debug("Shutdown notification skipped: %s", exc)


async def post_stop(application: Application) -> None:
    """Send shutdown notification while HTTP transport is still alive."""
    await _send_shutdown_notification(application)


async def post_shutdown(_application: Application) -> None:
    global _status_poll_task

    # Stop status polling
    if _status_poll_task:
        _status_poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _status_poll_task
        _status_poll_task = None
        logger.info("Status polling stopped")

    # Stop all queue workers
    await shutdown_workers()

    if session_monitor:
        session_monitor.stop()
        logger.info("Session monitor stopped")

    # Flush debounced state to disk AFTER workers/monitor stop (captures final mutations)
    session_manager.flush_state()


async def _error_handler(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle bot-level errors from updater and handlers."""
    if isinstance(context.error, Conflict):
        logger.critical(
            "Another bot instance is polling with the same token. "
            "Shutting down to avoid conflicts."
        )
        os.kill(os.getpid(), signal.SIGINT)
        return
    if isinstance(context.error, BadRequest) and "too old" in str(context.error):
        logger.debug("Callback query expired (query too old)")
        return
    if isinstance(context.error, NetworkError) and not isinstance(
        context.error, BadRequest
    ):
        logger.warning("Transient network error (PTB will retry): %s", context.error)
        return
    logger.error("Unhandled bot error", exc_info=context.error)


def create_bot() -> Application:
    # Suppress PTBUserWarning about JobQueue (we intentionally don't use it for core tasks)
    import warnings

    warnings.filterwarnings("ignore", message=".*JobQueue.*", category=UserWarning)
    application = (
        Application.builder()
        .token(config.telegram_bot_token)
        .get_updates_request(ResilientPollingHTTPXRequest(connection_pool_size=1))
        .post_init(post_init)
        .post_stop(post_stop)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_error_handler(_error_handler)
    application.add_handler(CommandHandler("new", new_command, filters=_group_filter))
    application.add_handler(
        CommandHandler("start", new_command, filters=_group_filter)  # compat alias
    )
    application.add_handler(
        CommandHandler("history", history_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("commands", commands_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("sessions", sessions_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("resume", resume_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("unbind", unbind_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("upgrade", upgrade_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("recall", recall_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("screenshot", screenshot_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("panes", panes_command, filters=_group_filter)
    )
    application.add_handler(CommandHandler("sync", sync_command, filters=_group_filter))
    application.add_handler(
        CommandHandler("toolbar", toolbar_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("verbose", verbose_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("restore", restore_command, filters=_group_filter)
    )
    application.add_handler(CallbackQueryHandler(callback_handler))
    # Topic closed event — unbind window (kept alive for rebinding)
    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_CLOSED & _group_filter,
            topic_closed_handler,
        )
    )
    # Topic renamed event — sync name to tmux window
    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_EDITED & _group_filter,
            topic_edited_handler,
        )
    )
    # Forward any other /command to the topic's provider CLI
    application.add_handler(
        MessageHandler(filters.COMMAND & _group_filter, forward_command_handler)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & _group_filter, text_handler)
    )
    # Photos
    application.add_handler(
        MessageHandler(filters.PHOTO & _group_filter, handle_photo_message)
    )
    # Documents
    application.add_handler(
        MessageHandler(filters.Document.ALL & _group_filter, handle_document_message)
    )
    # Voice messages (transcription when configured)
    application.add_handler(
        MessageHandler(filters.VOICE & _group_filter, handle_voice_message)
    )
    # Catch-all: unsupported content (stickers, voice, video, etc.)
    application.add_handler(
        MessageHandler(
            ~filters.COMMAND
            & ~filters.TEXT
            & ~filters.PHOTO
            & ~filters.Document.ALL
            & ~filters.VOICE
            & ~filters.StatusUpdate.ALL
            & _group_filter,
            unsupported_content_handler,
        )
    )
    # Inline query handler (serves switch_inline_query_current_chat from history buttons)
    application.add_handler(InlineQueryHandler(inline_query_handler))

    return application
