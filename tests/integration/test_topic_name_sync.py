import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, ForumTopicEdited, Message, MessageEntity, Update, User
from telegram.ext import Application

from ccgram.session import AuditResult

pytestmark = pytest.mark.integration

TEST_USER_ID = 12345
TEST_CHAT_ID = -100999
TEST_THREAD_ID = 42


def _make_command_update(text: str, *, bot=None, update_id: int = 1) -> Update:
    user = User(id=TEST_USER_ID, first_name="Test", is_bot=False)
    chat = Chat(id=TEST_CHAT_ID, type="supergroup")
    command_end = text.index(" ") if " " in text else len(text)
    entities = [
        MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=command_end)
    ]
    message = Message(
        message_id=update_id,
        date=datetime.now(),
        chat=chat,
        from_user=user,
        text=text,
        entities=entities,
        message_thread_id=TEST_THREAD_ID,
    )
    update = Update(update_id=update_id, message=message)
    if bot:
        update.set_bot(bot)
        message.set_bot(bot)
        for entity in entities:
            entity.set_bot(bot)
    return update


def _make_topic_edited_update(
    name: str, *, bot=None, update_id: int = 2, thread_id: int = TEST_THREAD_ID
) -> Update:
    user = User(id=TEST_USER_ID, first_name="Test", is_bot=False)
    chat = Chat(id=TEST_CHAT_ID, type="supergroup")
    message = Message(
        message_id=update_id,
        date=datetime.now(),
        chat=chat,
        from_user=user,
        forum_topic_edited=ForumTopicEdited(name=name, icon_custom_emoji_id=None),
        message_thread_id=thread_id,
    )
    update = Update(update_id=update_id, message=message)
    if bot:
        update.set_bot(bot)
        message.set_bot(bot)
    return update


@pytest.fixture
async def app():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    application = Application.builder().token(token).build()

    from ccgram.handlers.topics.topic_lifecycle import topic_edited_handler
    from ccgram.handlers.sync_command import sync_command
    from telegram.ext import CommandHandler, MessageHandler, filters

    application.add_handler(CommandHandler("sync", sync_command))
    application.add_handler(
        MessageHandler(filters.StatusUpdate.FORUM_TOPIC_EDITED, topic_edited_handler)
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


async def test_sync_dispatches_live_topic_name_reconciliation(app) -> None:
    update = _make_command_update("/sync", bot=app.bot)

    with (
        patch("ccgram.handlers.sync_command.config.is_user_allowed", return_value=True),
        patch(
            "ccgram.multiplexer.tmux.tmux_manager.list_windows",
            new_callable=AsyncMock,
            return_value=[MagicMock(window_id="@0", window_name="ccgram-codex")],
        ),
        patch(
            "ccgram.handlers.sync_command.thread_router.iter_thread_bindings",
            return_value=[(TEST_USER_ID, TEST_THREAD_ID, "@0")],
        ),
        patch(
            "ccgram.handlers.sync_command.thread_router.resolve_chat_id",
            return_value=TEST_CHAT_ID,
        ),
        patch(
            "ccgram.handlers.sync_command.thread_router.get_display_name",
            return_value="ccgram-codex",
        ),
        patch(
            "ccgram.handlers.sync_command._run_audit",
            new_callable=AsyncMock,
            return_value=AuditResult(
                issues=[],
                total_bindings=1,
                live_binding_count=1,
            ),
        ),
        patch(
            "ccgram.handlers.sync_command._probe_dead_topics",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "ccgram.handlers.sync_command.sync_topic_name",
            new_callable=AsyncMock,
        ) as mock_sync_topic_name,
        patch("ccgram.handlers.sync_command.safe_reply", new_callable=AsyncMock),
    ):
        await app.process_update(update)
        mock_sync_topic_name.assert_awaited_once()
        args = mock_sync_topic_name.call_args.args
        from ccgram.telegram_client import PTBTelegramClient

        assert isinstance(args[0], PTBTelegramClient)
        assert args[0].bot is app.bot
        assert args[1:] == (TEST_CHAT_ID, TEST_THREAD_ID, "ccgram-codex")


async def test_topic_edited_dispatches_rename_to_tmux(app) -> None:
    update = _make_topic_edited_update("bun", bot=app.bot)

    with (
        patch("ccgram.bot.is_user_allowed", return_value=True),
        patch(
            "ccgram.bot.thread_router.get_window_for_chat_thread",
            return_value="@0",
        ),
        patch("ccgram.bot.thread_router.get_display_name", return_value="fish"),
        patch(
            "ccgram.multiplexer.tmux.tmux_manager.rename_window",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_rename_window,
        patch("ccgram.bot.session_manager.set_display_name"),
    ):
        await app.process_update(update)
        mock_rename_window.assert_awaited_once_with("@0", "bun")


async def test_topic_edited_ignores_bot_generated_name_update(app) -> None:
    update = _make_topic_edited_update("\U0001f7e1 ccgram-codex", bot=app.bot)

    with (
        patch("ccgram.bot.is_user_allowed", return_value=True),
        patch(
            "ccgram.bot.thread_router.get_window_for_chat_thread",
            return_value="@0",
        ),
        patch(
            "ccgram.bot.thread_router.get_display_name",
            return_value="ccgram-codex",
        ),
        patch(
            "ccgram.multiplexer.tmux.tmux_manager.rename_window",
            new_callable=AsyncMock,
        ) as mock_rename_window,
    ):
        await app.process_update(update)
        mock_rename_window.assert_not_awaited()
