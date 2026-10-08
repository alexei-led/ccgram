"""Topic-scoped /dashboard command and private-chat onboarding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from telegram import Chat, InlineKeyboardMarkup, Update
from telegram.error import Forbidden

from .. import session_query
from ..config import config
from ..telegram_client import PTBTelegramClient
from .messaging_pipeline.message_sender import safe_reply

if TYPE_CHECKING:
    from telegram.ext import ContextTypes


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """DM the current bound topic's Mini App button to its authorized user."""
    user = update.effective_user
    message = update.message
    chat = update.effective_chat
    if not user or not config.is_user_allowed(user.id) or message is None:
        return

    thread_id = message.message_thread_id
    if chat is None or thread_id is None:
        await safe_reply(message, "❌ Use /dashboard inside a bound forum topic.")
        return

    window_id = session_query.resolve_window_for_topic(user.id, thread_id, chat.id)
    if window_id is None:
        await safe_reply(message, "❌ This topic is not bound to any session.")
        return

    # Lazy: importing status actions registers callbacks and topic-state cleanups.
    from .status.status_bar_actions import build_dashboard_button

    button = build_dashboard_button(window_id, user.id)
    if button is None:
        await safe_reply(message, "❌ The Mini App dashboard is disabled.")
        return

    markup = InlineKeyboardMarkup([[button]])
    client = PTBTelegramClient(context.bot)
    try:
        await client.send_message(
            chat_id=user.id,
            text="Open the dashboard for this session:",
            reply_markup=markup,
        )
    except Forbidden:
        await safe_reply(
            message,
            "❌ I can't message you privately yet. Start a private chat with me "
            "using /start, then retry /dashboard.",
        )


async def private_start_greeting(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Confirm a permitted user's private chat so dashboard DMs can be sent."""
    user = update.effective_user
    chat = update.effective_chat
    message = update.message
    if (
        not user
        or not config.is_user_allowed(user.id)
        or chat is None
        or chat.type != Chat.PRIVATE
        or message is None
    ):
        return

    await safe_reply(
        message,
        "✅ You're ready to use the dashboard. Run /dashboard in a bound forum "
        "topic and ccgram will send its private dashboard button here.",
    )
