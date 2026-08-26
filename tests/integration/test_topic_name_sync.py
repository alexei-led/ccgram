"""Topic-name sync: a user rename reaches the multiplexer, ours does not.

``/sync`` audits bindings without touching live topic names; a
``forum_topic_edited`` update renames the window only when the new name did not
come from the bot itself (the emoji-prefixed status label).
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, ForumTopicEdited, Message, Update, User
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from ccgram.session import AuditResult

pytestmark = pytest.mark.integration

TEST_USER_ID = 12345
TEST_CHAT_ID = -100999
TEST_THREAD_ID = 42

_MOD_SYNC = "ccgram.handlers.sync_command"


def _register(application: Application) -> None:
    from ccgram.handlers.sync_command import sync_command
    from ccgram.handlers.topics.topic_lifecycle import topic_edited_handler

    application.add_handler(CommandHandler("sync", sync_command))
    application.add_handler(
        MessageHandler(filters.StatusUpdate.FORUM_TOPIC_EDITED, topic_edited_handler)
    )


@pytest.fixture
async def app(make_ptb_app) -> Application:
    return await make_ptb_app(_register)


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


async def test_sync_audit_does_not_mutate_live_topic_names(
    app, make_text_update
) -> None:
    update = make_text_update("/sync", bot=app.bot)

    with (
        patch(f"{_MOD_SYNC}.config.is_user_allowed", return_value=True),
        patch(
            "ccgram.multiplexer.tmux.tmux_manager.list_windows",
            new_callable=AsyncMock,
            return_value=[MagicMock(window_id="@0", window_name="ccgram-codex")],
        ),
        patch(
            f"{_MOD_SYNC}.thread_router.iter_thread_bindings",
            return_value=[(TEST_USER_ID, TEST_THREAD_ID, "@0")],
        ),
        patch(
            f"{_MOD_SYNC}.thread_router.resolve_chat_id",
            return_value=TEST_CHAT_ID,
        ),
        patch(
            f"{_MOD_SYNC}.thread_router.get_display_name",
            return_value="ccgram-codex",
        ),
        patch(
            f"{_MOD_SYNC}._run_audit",
            new_callable=AsyncMock,
            return_value=AuditResult(
                issues=[],
                total_bindings=1,
                live_binding_count=1,
            ),
        ),
        patch(
            f"{_MOD_SYNC}._probe_dead_topics",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(f"{_MOD_SYNC}.sync_topic_name", new_callable=AsyncMock) as mock_sync,
        patch(f"{_MOD_SYNC}.safe_reply", new_callable=AsyncMock),
    ):
        await app.process_update(update)

    mock_sync.assert_not_awaited()


@pytest.mark.parametrize(
    ("new_name", "current_display_name", "expected_rename"),
    [
        pytest.param("bun", "fish", "bun", id="user-rename-forwarded"),
        pytest.param(
            "\U0001f7e1 ccgram-codex",
            "ccgram-codex",
            None,
            id="bot-status-label-ignored",
        ),
    ],
)
async def test_topic_edited_renames_window_only_for_user_edits(
    app, new_name: str, current_display_name: str, expected_rename: str | None
) -> None:
    update = _make_topic_edited_update(new_name, bot=app.bot)

    with (
        patch("ccgram.bot.is_user_allowed", return_value=True),
        patch("ccgram.bot.thread_router.get_window_for_chat_thread", return_value="@0"),
        patch(
            "ccgram.bot.thread_router.get_display_name",
            return_value=current_display_name,
        ),
        patch(
            "ccgram.multiplexer.tmux.tmux_manager.rename_window",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_rename,
        patch("ccgram.bot.session_manager.set_display_name"),
    ):
        await app.process_update(update)

    if expected_rename is None:
        mock_rename.assert_not_awaited()
    else:
        mock_rename.assert_awaited_once_with("@0", expected_rename)
