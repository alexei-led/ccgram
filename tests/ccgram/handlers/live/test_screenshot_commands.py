"""Tests for /live and related slash commands in screenshot_callbacks."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import TelegramError

from ccgram.handlers.live.live_view import LiveViewState
from ccgram.handlers.live.screenshot_callbacks import live_command

_SC = "ccgram.handlers.live.screenshot_callbacks"
_LV = "ccgram.handlers.live.live_view"

USER_ID = 100
THREAD_ID = 42
CHAT_ID = -100


def _make_update(thread_id: int | None = THREAD_ID) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = USER_ID
    update.message = MagicMock()
    update.message.message_thread_id = thread_id
    sent = MagicMock()
    sent.message_id = 555
    update.message.get_bot = MagicMock(
        return_value=MagicMock(send_photo=AsyncMock(return_value=sent))
    )
    update.message.reply_text = AsyncMock()
    return update


def _send_photo(update: MagicMock) -> AsyncMock:
    return update.message.get_bot.return_value.send_photo


@pytest.fixture
def active_views() -> Iterator[dict]:
    with patch(f"{_LV}._active_views", new_callable=dict) as views:
        yield views


@pytest.fixture
def reply() -> Iterator[AsyncMock]:
    with patch(
        "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
        new_callable=AsyncMock,
    ) as mock_reply:
        yield mock_reply


@contextmanager
def _live_env(
    *,
    allowed: bool = True,
    thread_id: int | None = THREAD_ID,
    window_id: str | None = "@0",
    window_alive: bool = True,
    capture: str = "some terminal text",
):
    """Patch the /live collaborators: auth, thread routing and the multiplexer."""
    with (
        patch("ccgram.config.config") as mock_config,
        patch(f"{_SC}.get_thread_id", return_value=thread_id),
        patch(f"{_SC}.thread_router") as mock_tr,
        patch(f"{_SC}.tmux_manager") as mock_tm,
        patch(f"{_SC}.text_to_image", new_callable=AsyncMock, return_value=b"png"),
    ):
        mock_config.is_user_allowed.return_value = allowed
        mock_tr.get_window_for_thread.return_value = window_id
        mock_tr.resolve_chat_id.return_value = CHAT_ID
        mock_tm.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id=window_id) if window_alive else None
        )
        mock_tm.capture_pane = AsyncMock(return_value=capture)
        yield


def _reply_text(reply: AsyncMock) -> str:
    return reply.call_args.args[1]


class TestLiveCommand:
    async def test_starts_live_view(self, active_views: dict, reply: AsyncMock) -> None:
        update = _make_update()
        with _live_env():
            await live_command(update, MagicMock())

        _send_photo(update).assert_awaited_once()
        kwargs = _send_photo(update).call_args.kwargs
        assert kwargs["chat_id"] == CHAT_ID
        assert kwargs["message_thread_id"] == THREAD_ID
        assert "Live" in kwargs["caption"]
        assert (USER_ID, THREAD_ID) in active_views
        reply.assert_not_awaited()

    async def test_unauthorized_silent(self, reply: AsyncMock) -> None:
        with _live_env(allowed=False):
            await live_command(_make_update(), MagicMock())
        reply.assert_not_awaited()

    async def test_no_thread_replies_error(self, reply: AsyncMock) -> None:
        update = _make_update(thread_id=None)
        update.effective_chat = None
        with _live_env(thread_id=None):
            await live_command(update, MagicMock())
        reply.assert_awaited_once()
        assert "topic" in _reply_text(reply).lower()

    async def test_already_live_returns_message(
        self, active_views: dict, reply: AsyncMock
    ) -> None:
        active_views[(USER_ID, THREAD_ID)] = LiveViewState(
            chat_id=CHAT_ID,
            message_id=1,
            thread_id=THREAD_ID,
            user_id=USER_ID,
            window_id="@0",
        )
        with _live_env():
            await live_command(_make_update(), MagicMock())
        reply.assert_awaited_once()
        assert "already" in _reply_text(reply).lower()

    async def test_unbound_topic_replies(self, reply: AsyncMock) -> None:
        with _live_env(window_id=None):
            await live_command(_make_update(), MagicMock())
        reply.assert_awaited_once()
        assert "not bound" in _reply_text(reply)

    async def test_dead_window_replies(self, reply: AsyncMock) -> None:
        with _live_env(window_alive=False):
            await live_command(_make_update(), MagicMock())
        reply.assert_awaited_once()
        assert "no longer exists" in _reply_text(reply)

    async def test_send_photo_failure_replies(
        self, active_views: dict, reply: AsyncMock
    ) -> None:
        update = _make_update()
        _send_photo(update).side_effect = TelegramError("denied")
        with _live_env(capture="x"):
            await live_command(update, MagicMock())

        reply.assert_awaited_once()
        assert "Failed to start" in _reply_text(reply)
        assert (USER_ID, THREAD_ID) not in active_views
