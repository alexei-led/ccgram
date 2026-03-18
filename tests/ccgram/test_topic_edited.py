"""Tests for FORUM_TOPIC_EDITED handler (bidirectional name sync)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.topic_emoji import reset_all_state


@pytest.fixture(autouse=True)
def _reset():
    reset_all_state()
    yield
    reset_all_state()


def _make_update(
    new_name: str | None, thread_id: int = 42, chat_id: int = -100, user_id: int = 1
) -> MagicMock:
    """Create a mock Update for FORUM_TOPIC_EDITED."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.message.forum_topic_edited.name = new_name
    update.message.forum_topic_edited.icon_custom_emoji_id = None
    update.message.message_thread_id = thread_id
    return update


_PATCH_ALLOWED = patch("ccgram.bot.is_user_allowed", return_value=True)


class TestTopicEditedHandler:
    @_PATCH_ALLOWED
    @patch("ccgram.bot.tmux_manager")
    @patch("ccgram.bot.session_manager")
    async def test_renames_tmux_window(
        self, mock_sm: MagicMock, mock_tm: MagicMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_edited_handler

        mock_sm.get_window_for_chat_thread.return_value = "@0"
        mock_sm.get_display_name.return_value = "old-name"
        mock_tm.rename_window = AsyncMock(return_value=True)

        update = _make_update("new-name")
        await topic_edited_handler(update, MagicMock())

        mock_tm.rename_window.assert_called_once_with("@0", "new-name")
        mock_sm.set_display_name.assert_called_once_with("@0", "new-name")

    @_PATCH_ALLOWED
    @patch("ccgram.bot.tmux_manager")
    @patch("ccgram.bot.session_manager")
    async def test_ignores_emoji_only_change(
        self, mock_sm: MagicMock, mock_tm: MagicMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_edited_handler

        mock_sm.get_window_for_chat_thread.return_value = "@0"
        mock_sm.get_display_name.return_value = "myproject"
        mock_tm.rename_window = AsyncMock()

        # Bot set "🟢 myproject" — clean name matches current display
        update = _make_update("\U0001f7e2 myproject")
        await topic_edited_handler(update, MagicMock())

        mock_tm.rename_window.assert_not_called()

    @patch("ccgram.bot.tmux_manager")
    @patch("ccgram.bot.session_manager")
    async def test_ignores_icon_only_edit(
        self, mock_sm: MagicMock, mock_tm: MagicMock
    ) -> None:
        from ccgram.bot import topic_edited_handler

        mock_tm.rename_window = AsyncMock()

        update = _make_update(None)
        await topic_edited_handler(update, MagicMock())

        mock_sm.get_window_for_chat_thread.assert_not_called()
        mock_tm.rename_window.assert_not_called()

    @_PATCH_ALLOWED
    @patch("ccgram.bot.tmux_manager")
    @patch("ccgram.bot.session_manager")
    async def test_ignores_unbound_topic(
        self, mock_sm: MagicMock, mock_tm: MagicMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_edited_handler

        mock_sm.get_window_for_chat_thread.return_value = None
        mock_tm.rename_window = AsyncMock()

        update = _make_update("new-name")
        await topic_edited_handler(update, MagicMock())

        mock_tm.rename_window.assert_not_called()

    @_PATCH_ALLOWED
    @patch("ccgram.bot.tmux_manager")
    @patch("ccgram.bot.session_manager")
    async def test_updates_emoji_cache(
        self, mock_sm: MagicMock, mock_tm: MagicMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_edited_handler
        from ccgram.handlers.topic_emoji import _topic_names

        _topic_names[(-100, 42)] = "old-name"
        mock_sm.get_window_for_chat_thread.return_value = "@0"
        mock_sm.get_display_name.return_value = "old-name"
        mock_tm.rename_window = AsyncMock(return_value=True)

        update = _make_update("new-name")
        await topic_edited_handler(update, MagicMock())

        assert _topic_names[(-100, 42)] == "new-name"

    @_PATCH_ALLOWED
    @patch("ccgram.bot.tmux_manager")
    @patch("ccgram.bot.session_manager")
    async def test_caches_unchanged_when_rename_fails(
        self, mock_sm: MagicMock, mock_tm: MagicMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_edited_handler
        from ccgram.handlers.topic_emoji import _topic_names

        _topic_names[(-100, 42)] = "old-name"
        mock_sm.get_window_for_chat_thread.return_value = "@0"
        mock_sm.get_display_name.return_value = "old-name"
        mock_tm.rename_window = AsyncMock(return_value=False)

        update = _make_update("new-name")
        await topic_edited_handler(update, MagicMock())

        assert _topic_names[(-100, 42)] == "old-name"
        mock_sm.set_display_name.assert_not_called()
