"""``/dashboard`` command: deliver the native Mini App button via private chat.

Telegram only allows ``web_app`` inline buttons in private chats, so the
status bubble in forum topics carries a plain-URL button. This command sends
the native WebApp button (initData, theme) for the topic's window to the
user's private chat with the bot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import InlineKeyboardMarkup
from telegram.error import TelegramError

from .callback_helpers import get_thread_id
from ..config import config
from .messaging_pipeline.message_sender import safe_reply
from .status.status_bar_actions import build_dashboard_button
from ..telegram_client import PTBTelegramClient
from ..thread_router import thread_router
from ..utils import handle_general_topic_message, is_general_topic

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = structlog.get_logger()


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """DM the user a native WebApp dashboard button for this topic's window."""
    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        if update.effective_chat and is_general_topic(update.message):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "❌ Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "❌ This topic is not bound to any session.")
        return

    # None exactly when the Mini App is disabled (unset base URL).
    button = build_dashboard_button(window_id, user.id)
    if button is None:
        await safe_reply(update.message, "❌ Mini App is not enabled on this instance.")
        return

    display = thread_router.get_display_name(window_id)
    client = PTBTelegramClient(context.bot)
    try:
        await client.send_message(
            chat_id=user.id,
            text=f"🪟 Dashboard for {display}",
            reply_markup=InlineKeyboardMarkup([[button]]),
        )
    except TelegramError as e:
        logger.warning("dashboard DM failed for user %s: %s", user.id, e)
        await safe_reply(
            update.message,
            "❌ Could not message you privately. Open a private chat with this "
            "bot, send /start once, then retry /dashboard.",
        )
        return
    await safe_reply(
        update.message, "📨 Sent the dashboard button to your private chat."
    )
