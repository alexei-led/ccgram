"""Integration tests for PTB Application handler dispatch.

Tests that handlers are correctly registered and PTB routes updates
to the right handler functions. Uses real PTB Application with mocked
external dependencies (Bot API, TmuxManager, SessionManager).
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, Message, MessageEntity, Update, User
from telegram.ext import Application

pytestmark = pytest.mark.integration

TEST_USER_ID = 12345
TEST_CHAT_ID = -100999
TEST_THREAD_ID = 42


def _make_update(
    text=None,
    *,
    bot=None,
    thread_id=TEST_THREAD_ID,
    update_id=1,
    user_id=TEST_USER_ID,
    chat_type="supergroup",
):
    user = User(id=user_id, first_name="Test", is_bot=False)
    chat = Chat(id=TEST_CHAT_ID, type=chat_type)
    entities = None
    if text and text.startswith("/"):
        cmd_end = text.index(" ") if " " in text else len(text)
        entities = [
            MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=cmd_end)
        ]
    message = Message(
        message_id=update_id,
        date=None,
        chat=chat,
        from_user=user,
        text=text,
        entities=entities,
        message_thread_id=thread_id,
    )
    update = Update(update_id=update_id, message=message)
    if bot:
        update.set_bot(bot)
        message.set_bot(bot)
        for entity in entities or []:
            entity.set_bot(bot)
    return update


@pytest.fixture
async def app():
    """Real PTB Application with ccgram handlers registered."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    application = Application.builder().token(token).build()

    from ccgram.bot import (
        callback_handler,
        forward_command_handler,
        history_command,
        new_command,
        sessions_command,
        text_handler,
        topic_closed_handler,
    )
    from telegram.ext import (
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    application.add_handler(CommandHandler("new", new_command))
    application.add_handler(CommandHandler("start", new_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("sessions", sessions_command))
    application.add_handler(CallbackQueryHandler(callback_handler))
    application.add_handler(
        MessageHandler(filters.StatusUpdate.FORUM_TOPIC_CLOSED, topic_closed_handler)
    )
    application.add_handler(MessageHandler(filters.COMMAND, forward_command_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
    )

    mock_post = AsyncMock(
        return_value={
            "id": 1,
            "first_name": "Bot",
            "is_bot": True,
            "username": "testbot",
        }
    )
    with patch.object(type(application.bot), "_do_post", mock_post):
        async with application:
            yield application


async def test_text_routed_to_text_handler(app) -> None:
    update = _make_update("hello world", bot=app.bot)

    with (
        patch("ccgram.bot.handle_text_message", new_callable=AsyncMock) as mock_handler,
        patch("ccgram.bot.is_user_allowed", return_value=True),
    ):
        await app.process_update(update)
        mock_handler.assert_awaited_once()


async def test_unauthorized_user_rejected(app) -> None:
    update = _make_update("hello", bot=app.bot, user_id=99999)

    with (
        patch("ccgram.bot.handle_text_message", new_callable=AsyncMock) as mock_handler,
        patch("ccgram.bot.is_user_allowed", return_value=False),
    ):
        await app.process_update(update)
        mock_handler.assert_not_awaited()


async def test_new_command_dispatched(app) -> None:
    update = _make_update("/new", bot=app.bot)

    with (
        patch("ccgram.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        patch("ccgram.bot.is_user_allowed", return_value=True),
    ):
        await app.process_update(update)
        mock_reply.assert_awaited_once()


async def test_history_command_dispatched(app) -> None:
    update = _make_update("/history", bot=app.bot)

    with (
        patch("ccgram.bot.is_user_allowed", return_value=True),
        patch(
            "ccgram.bot.session_manager.resolve_window_for_thread", return_value=None
        ),
        patch("ccgram.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
    ):
        await app.process_update(update)
        mock_reply.assert_awaited_once()


async def test_unknown_command_forwarded(app) -> None:
    update = _make_update("/sometool", bot=app.bot)

    with (
        patch("ccgram.bot.is_user_allowed", return_value=True),
        patch(
            "ccgram.bot.session_manager.resolve_window_for_thread", return_value="@0"
        ),
        patch(
            "ccgram.bot.tmux_manager.find_window_by_id",
            new_callable=AsyncMock,
            return_value=MagicMock(window_id="@0"),
        ),
        patch(
            "ccgram.bot.session_manager.send_to_window",
            new_callable=AsyncMock,
            return_value=(True, "Sent"),
        ),
        patch("ccgram.bot.session_manager.get_display_name", return_value="test-win"),
        patch("ccgram.bot.safe_reply", new_callable=AsyncMock),
        patch.object(Chat, "send_action", new_callable=AsyncMock),
    ):
        await app.process_update(update)


async def test_command_priority_over_text(app) -> None:
    """Commands like /history should be handled by CommandHandler, not text_handler."""
    update = _make_update("/history", bot=app.bot)

    with (
        patch("ccgram.bot.is_user_allowed", return_value=True),
        patch("ccgram.bot.handle_text_message", new_callable=AsyncMock) as mock_text,
        patch(
            "ccgram.bot.session_manager.resolve_window_for_thread", return_value=None
        ),
        patch("ccgram.bot.safe_reply", new_callable=AsyncMock),
    ):
        await app.process_update(update)
        mock_text.assert_not_awaited()
