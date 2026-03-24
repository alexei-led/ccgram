"""Integration tests for PTB dispatch routing to shell handler.

Uses real PTB Application with _do_post patch to verify that text messages
in shell-bound topics route through to handle_shell_message, and that
callback queries with shell prefixes dispatch to handle_shell_callback.
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from telegram import (
    CallbackQuery,
    Chat,
    Message,
    Update,
    User,
)
from telegram.ext import Application

pytestmark = pytest.mark.integration

TEST_USER_ID = 12345
TEST_CHAT_ID = -100999
TEST_THREAD_ID = 42


def _make_text_update(text, *, bot=None, update_id=1):
    user = User(id=TEST_USER_ID, first_name="Test", is_bot=False)
    chat = Chat(id=TEST_CHAT_ID, type="supergroup")
    message = Message(
        message_id=update_id,
        date=datetime.now(),
        chat=chat,
        from_user=user,
        text=text,
        message_thread_id=TEST_THREAD_ID,
    )
    update = Update(update_id=update_id, message=message)
    if bot:
        update.set_bot(bot)
        message.set_bot(bot)
    return update


def _make_callback_update(data, *, bot=None, update_id=1):
    user = User(id=TEST_USER_ID, first_name="Test", is_bot=False)
    query = CallbackQuery(
        id="1",
        from_user=user,
        chat_instance="test",
        data=data,
    )
    update = Update(update_id=update_id, callback_query=query)
    if bot:
        update.set_bot(bot)
        query.set_bot(bot)
    return update


@pytest.fixture
async def app():
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


@pytest.mark.asyncio()
async def test_text_in_shell_topic_reaches_text_handler(app) -> None:
    update = _make_text_update("list files", bot=app.bot)

    with (
        patch("ccgram.bot.handle_text_message", new_callable=AsyncMock) as mock_handler,
        patch("ccgram.bot.is_user_allowed", return_value=True),
    ):
        await app.process_update(update)
        mock_handler.assert_awaited_once()

        call_args = mock_handler.call_args[0]
        assert call_args[0].message.text == "list files"
        assert call_args[0].message.message_thread_id == TEST_THREAD_ID


@pytest.mark.asyncio()
async def test_bang_prefix_reaches_text_handler(app) -> None:
    update = _make_text_update("!ls -la", bot=app.bot)

    with (
        patch("ccgram.bot.handle_text_message", new_callable=AsyncMock) as mock_handler,
        patch("ccgram.bot.is_user_allowed", return_value=True),
    ):
        await app.process_update(update)
        mock_handler.assert_awaited_once()

        call_args = mock_handler.call_args[0]
        assert call_args[0].message.text == "!ls -la"


@pytest.mark.asyncio()
async def test_shell_callback_dispatches_to_shell_handler(app) -> None:
    from ccgram.handlers.callback_data import CB_SHELL_RUN

    update = _make_callback_update(f"{CB_SHELL_RUN}@0", bot=app.bot)

    with (
        patch(
            "ccgram.handlers.shell_commands.handle_shell_callback",
            new_callable=AsyncMock,
        ) as mock_shell_cb,
        patch("ccgram.bot.is_user_allowed", return_value=True),
    ):
        await app.process_update(update)
        mock_shell_cb.assert_awaited_once()

        call_args = mock_shell_cb.call_args
        assert call_args[0][2].startswith(CB_SHELL_RUN)
