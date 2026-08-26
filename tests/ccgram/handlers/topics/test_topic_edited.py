"""Tests for FORUM_TOPIC_EDITED handler (bidirectional name sync)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.status.topic_emoji import _topic_names, reset_all_state
from ccgram.handlers.topics.topic_lifecycle import topic_edited_handler

CHAT_ID = -100
THREAD_ID = 42


@pytest.fixture(autouse=True)
def _reset():
    reset_all_state()
    yield
    reset_all_state()


@pytest.fixture(autouse=True)
def allowed_user():
    with patch("ccgram.config.Config.is_user_allowed", return_value=True):
        yield


@pytest.fixture
def mux():
    with patch("ccgram.handlers.topics.topic_lifecycle.tmux_manager") as mock_mux:
        mock_mux.rename_window = AsyncMock(return_value=True)
        yield mock_mux


@pytest.fixture
def router():
    with patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_router:
        yield mock_router


@pytest.fixture
def session():
    with patch("ccgram.handlers.topics.topic_lifecycle.session_manager") as mock_sm:
        yield mock_sm


def _make_update(
    new_name: str | None,
    thread_id: int = THREAD_ID,
    chat_id: int = CHAT_ID,
    user_id: int = 1,
) -> MagicMock:
    """Create a mock Update for FORUM_TOPIC_EDITED."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.message.forum_topic_edited.name = new_name
    update.message.forum_topic_edited.icon_custom_emoji_id = None
    update.message.message_thread_id = thread_id
    return update


class TestTopicEditedRenamesWindow:
    @pytest.mark.parametrize(
        ("window_id", "old_display"),
        [("@0", "old-name"), ("w1:t1", "workspace ▸ old-agent")],
        ids=["tmux_window", "herdr_tab"],
    )
    async def test_rename_reaches_the_multiplexer_proxy(
        self,
        mux: MagicMock,
        router: MagicMock,
        session: MagicMock,
        window_id: str,
        old_display: str,
    ) -> None:
        router.get_window_for_chat_thread.return_value = window_id
        router.get_display_name.return_value = old_display

        await topic_edited_handler(_make_update("new-name"), MagicMock())

        mux.rename_window.assert_called_once_with(window_id, "new-name")
        session.set_display_name.assert_called_once_with(window_id, "new-name")

    async def test_updates_emoji_cache(
        self, mux: MagicMock, router: MagicMock, session: MagicMock
    ) -> None:
        _topic_names[(CHAT_ID, THREAD_ID)] = "old-name"
        router.get_window_for_chat_thread.return_value = "@0"
        router.get_display_name.return_value = "old-name"

        await topic_edited_handler(_make_update("new-name"), MagicMock())

        assert _topic_names[(CHAT_ID, THREAD_ID)] == "new-name"


class TestTopicEditedIgnoredEdits:
    async def test_ignores_emoji_only_change(
        self, mux: MagicMock, router: MagicMock
    ) -> None:
        """The bot itself wrote "🟢 myproject"; the clean name is unchanged."""
        router.get_window_for_chat_thread.return_value = "@0"
        router.get_display_name.return_value = "myproject"

        await topic_edited_handler(_make_update("\U0001f7e2 myproject"), MagicMock())

        mux.rename_window.assert_not_called()

    async def test_ignores_icon_only_edit(
        self, mux: MagicMock, router: MagicMock
    ) -> None:
        await topic_edited_handler(_make_update(None), MagicMock())

        router.get_window_for_chat_thread.assert_not_called()
        mux.rename_window.assert_not_called()

    async def test_ignores_unbound_topic(
        self, mux: MagicMock, router: MagicMock
    ) -> None:
        router.get_window_for_chat_thread.return_value = None

        await topic_edited_handler(_make_update("new-name"), MagicMock())

        mux.rename_window.assert_not_called()

    async def test_caches_unchanged_when_rename_fails(
        self, mux: MagicMock, router: MagicMock
    ) -> None:
        _topic_names[(CHAT_ID, THREAD_ID)] = "old-name"
        router.get_window_for_chat_thread.return_value = "@0"
        router.get_display_name.return_value = "old-name"
        mux.rename_window = AsyncMock(return_value=False)

        await topic_edited_handler(_make_update("new-name"), MagicMock())

        assert _topic_names[(CHAT_ID, THREAD_ID)] == "old-name"
        router.set_display_name.assert_not_called()
