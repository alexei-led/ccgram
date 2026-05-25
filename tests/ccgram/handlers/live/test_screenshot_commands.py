"""Tests for /live and related slash commands in screenshot_callbacks."""

from unittest.mock import AsyncMock, MagicMock, patch

from telegram.error import TelegramError

from ccgram.handlers.live.screenshot_callbacks import (
    live_command,
    screenshot_command,
)
from ccgram.terminal_backends import (
    TerminalBackendUnavailableError,
    TerminalNotFoundError,
)

_SC = "ccgram.handlers.live.screenshot_callbacks"
_LV = "ccgram.handlers.live.live_view"


def _make_update(
    user_id: int = 100,
    thread_id: int | None = 42,
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.message = MagicMock()
    update.message.message_thread_id = thread_id
    update.message.get_bot = MagicMock(return_value=MagicMock(send_photo=AsyncMock()))
    update.message.reply_text = AsyncMock()
    return update


class TestLiveCommand:
    @patch(f"{_LV}._active_views", new_callable=dict)
    @patch(f"{_SC}.text_to_image", new_callable=AsyncMock, return_value=b"png")
    @patch(f"{_SC}.tmux_manager")
    @patch(f"{_SC}.thread_router")
    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_starts_live_view(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        _mock_render: AsyncMock,
        active_views: dict,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_tr.get_window_for_thread.return_value = "@0"
        mock_tr.resolve_chat_id.return_value = -100
        mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@0"))
        mock_tm.capture_pane = AsyncMock(return_value="some terminal text")

        update = _make_update()
        sent_message = MagicMock()
        sent_message.message_id = 555
        update.message.get_bot.return_value.send_photo = AsyncMock(
            return_value=sent_message
        )

        with patch(f"{_SC}.get_thread_id", return_value=42):
            await live_command(update, MagicMock())

        update.message.get_bot.return_value.send_photo.assert_awaited_once()
        kwargs = update.message.get_bot.return_value.send_photo.call_args.kwargs
        assert kwargs["chat_id"] == -100
        assert kwargs["message_thread_id"] == 42
        assert "Live" in kwargs["caption"]
        assert (100, 42) in active_views
        mock_reply.assert_not_awaited()

    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_unauthorized_silent(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = False
        await live_command(_make_update(), MagicMock())
        mock_reply.assert_not_awaited()

    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_no_thread_replies_error(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        update = _make_update(thread_id=None)
        update.effective_chat = None
        with patch(f"{_SC}.get_thread_id", return_value=None):
            await live_command(update, MagicMock())
        mock_reply.assert_awaited_once()
        assert "topic" in mock_reply.call_args.args[1].lower()

    @patch(f"{_LV}._active_views", new_callable=dict)
    @patch(f"{_SC}.thread_router")
    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_already_live_returns_message(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
        _mock_tr: MagicMock,
        active_views: dict,
    ) -> None:
        mock_config.is_user_allowed.return_value = True

        from ccgram.handlers.live.live_view import LiveViewState

        active_views[(100, 42)] = LiveViewState(
            chat_id=-100,
            message_id=1,
            thread_id=42,
            user_id=100,
            window_id="@0",
        )

        with patch(f"{_SC}.get_thread_id", return_value=42):
            await live_command(_make_update(), MagicMock())

        mock_reply.assert_awaited_once()
        assert "already" in mock_reply.call_args.args[1].lower()

    @patch(f"{_SC}.thread_router")
    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_unbound_topic_replies(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
        mock_tr: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_tr.get_window_for_thread.return_value = None

        with patch(f"{_SC}.get_thread_id", return_value=42):
            await live_command(_make_update(), MagicMock())

        mock_reply.assert_awaited_once()
        assert "not bound" in mock_reply.call_args.args[1]

    @patch(f"{_SC}.tmux_manager")
    @patch(f"{_SC}.thread_router")
    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_dead_window_replies(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_tr.get_window_for_thread.return_value = "@0"
        mock_tm.find_window_by_id = AsyncMock(return_value=None)

        with patch(f"{_SC}.get_thread_id", return_value=42):
            await live_command(_make_update(), MagicMock())

        mock_reply.assert_awaited_once()
        assert "no longer exists" in mock_reply.call_args.args[1]

    @patch(f"{_LV}._active_views", new_callable=dict)
    @patch(f"{_SC}.text_to_image", new_callable=AsyncMock, return_value=b"png")
    @patch(f"{_SC}.tmux_manager")
    @patch(f"{_SC}.thread_router")
    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_send_photo_failure_replies(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        _mock_render: AsyncMock,
        active_views: dict,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_tr.get_window_for_thread.return_value = "@0"
        mock_tr.resolve_chat_id.return_value = -100
        mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@0"))
        mock_tm.capture_pane = AsyncMock(return_value="x")

        update = _make_update()
        update.message.get_bot.return_value.send_photo = AsyncMock(
            side_effect=TelegramError("denied")
        )

        with patch(f"{_SC}.get_thread_id", return_value=42):
            await live_command(update, MagicMock())

        mock_reply.assert_awaited_once()
        assert "Failed to start" in mock_reply.call_args.args[1]
        assert (100, 42) not in active_views


class TestScreenshotCommand:
    @patch(f"{_SC}.terminal_capture", new_callable=AsyncMock)
    @patch(f"{_SC}.thread_router")
    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_window_not_found_replies(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
        mock_tr: MagicMock,
        mock_capture: AsyncMock,
    ) -> None:
        mock_config.is_user_allowed.return_value = True
        mock_tr.get_window_for_thread.return_value = "@0"
        mock_capture.side_effect = TerminalNotFoundError("gone")

        with patch(f"{_SC}.get_thread_id", return_value=42):
            await screenshot_command(_make_update(), MagicMock())

        mock_reply.assert_awaited_once()
        assert "no longer exists" in mock_reply.call_args.args[1]

    @patch(f"{_SC}.terminal_capture", new_callable=AsyncMock)
    @patch(f"{_SC}.thread_router")
    @patch("ccgram.config.config")
    @patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    )
    async def test_backend_unavailable_replies_friendly_error(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
        mock_tr: MagicMock,
        mock_capture: AsyncMock,
    ) -> None:
        """cmux sidecar down / backend unregistered must not bubble up."""
        mock_config.is_user_allowed.return_value = True
        mock_tr.get_window_for_thread.return_value = "cmux:abc"
        mock_capture.side_effect = TerminalBackendUnavailableError(
            "cmux backend not registered"
        )

        with patch(f"{_SC}.get_thread_id", return_value=42):
            await screenshot_command(_make_update(), MagicMock())

        mock_reply.assert_awaited_once()
        body = mock_reply.call_args.args[1]
        assert "Terminal backend error" in body
        assert "unavailable" in body
