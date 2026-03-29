"""Tests for FORUM_TOPIC_CLOSED handler (unbind thread, keep window)."""

from unittest.mock import AsyncMock, MagicMock, patch


def _make_update(thread_id: int = 42, user_id: int = 1) -> MagicMock:
    """Create a mock Update for FORUM_TOPIC_CLOSED."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.message_thread_id = thread_id
    return update


_PATCH_ALLOWED = patch("ccgram.bot.is_user_allowed", return_value=True)


class TestTopicClosedHandler:
    @_PATCH_ALLOWED
    @patch("ccgram.bot.clear_topic_state", new_callable=AsyncMock)
    @patch("ccgram.bot.thread_router")
    async def test_unbinds_bound_topic(
        self, mock_tr: MagicMock, mock_clear: AsyncMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_closed_handler

        mock_tr.get_window_for_thread.return_value = "@0"
        mock_tr.get_display_name.return_value = "my-project"

        update = _make_update()
        ctx = MagicMock()
        await topic_closed_handler(update, ctx)

        mock_tr.get_window_for_thread.assert_called_once_with(1, 42)
        mock_tr.unbind_thread.assert_called_once_with(1, 42)
        mock_clear.assert_called_once_with(
            1, 42, ctx.bot, ctx.user_data, window_id="@0"
        )

    @_PATCH_ALLOWED
    @patch("ccgram.bot.clear_topic_state", new_callable=AsyncMock)
    @patch("ccgram.bot.thread_router")
    async def test_skips_unbound_topic(
        self, mock_tr: MagicMock, mock_clear: AsyncMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_closed_handler

        mock_tr.get_window_for_thread.return_value = None

        update = _make_update()
        await topic_closed_handler(update, MagicMock())

        mock_tr.unbind_thread.assert_not_called()
        mock_clear.assert_not_called()

    @patch("ccgram.bot.clear_topic_state", new_callable=AsyncMock)
    @patch("ccgram.bot.thread_router")
    async def test_skips_disallowed_user(
        self, mock_tr: MagicMock, mock_clear: AsyncMock
    ) -> None:
        from ccgram.bot import topic_closed_handler

        update = _make_update()
        with patch("ccgram.bot.is_user_allowed", return_value=False):
            await topic_closed_handler(update, MagicMock())

        mock_tr.get_window_for_thread.assert_not_called()
        mock_clear.assert_not_called()

    @_PATCH_ALLOWED
    @patch("ccgram.bot.clear_topic_state", new_callable=AsyncMock)
    @patch("ccgram.bot.thread_router")
    async def test_skips_general_topic(
        self, mock_tr: MagicMock, mock_clear: AsyncMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_closed_handler

        update = MagicMock()
        update.effective_user.id = 1
        update.message.message_thread_id = 1

        await topic_closed_handler(update, MagicMock())

        mock_tr.get_window_for_thread.assert_not_called()
        mock_clear.assert_not_called()

    @_PATCH_ALLOWED
    @patch("ccgram.bot.clear_topic_state", new_callable=AsyncMock)
    @patch("ccgram.bot.thread_router")
    async def test_skips_no_thread_id(
        self, mock_tr: MagicMock, mock_clear: AsyncMock, _allowed: MagicMock
    ) -> None:
        from ccgram.bot import topic_closed_handler

        update = MagicMock()
        update.effective_user.id = 1
        update.message.message_thread_id = None

        await topic_closed_handler(update, MagicMock())

        mock_tr.get_window_for_thread.assert_not_called()
        mock_clear.assert_not_called()
