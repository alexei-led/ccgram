"""Tests for topic state cleanup."""

from unittest.mock import AsyncMock, patch

from ccgram.handlers.cleanup import clear_topic_state


class TestClearTopicState:
    async def test_enqueues_status_clear_when_bot_available(self) -> None:
        bot = AsyncMock()
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update") as mock_enqueue,
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.handlers.cleanup.clear_topic_emoji_state"),
            patch("ccgram.handlers.cleanup.clear_tool_msg_ids_for_topic"),
            patch("ccgram.handlers.cleanup.clear_status_msg_info") as mock_clear_info,
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, bot=bot, window_id="@0")

        mock_enqueue.assert_called_once()
        args = mock_enqueue.call_args
        assert args[0][1] == 1  # user_id
        assert args[0][2] == "@0"  # window_id passed through
        assert args[0][3] is None  # status_text (clear)
        assert args[1]["thread_id"] == 42
        mock_clear_info.assert_not_called()

    async def test_clears_tracking_only_when_no_bot(self) -> None:
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update") as mock_enqueue,
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.handlers.cleanup.clear_topic_emoji_state"),
            patch("ccgram.handlers.cleanup.clear_tool_msg_ids_for_topic"),
            patch("ccgram.handlers.cleanup.clear_status_msg_info") as mock_clear_info,
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, bot=None, window_id="@0")

        mock_enqueue.assert_not_called()
        mock_clear_info.assert_called_once_with(1, 42)

    async def test_enqueues_empty_window_id_when_none(self) -> None:
        bot = AsyncMock()
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update") as mock_enqueue,
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.handlers.cleanup.clear_topic_emoji_state"),
            patch("ccgram.handlers.cleanup.clear_tool_msg_ids_for_topic"),
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, bot=bot, window_id=None)

        mock_enqueue.assert_called_once()
        assert mock_enqueue.call_args[0][2] == ""
