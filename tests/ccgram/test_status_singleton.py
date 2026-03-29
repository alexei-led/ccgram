"""Tests for status message singleton behavior (no pile-up).

Verifies three invariants:
1. Edit failure does NOT send a new status message (clears tracking only).
2. Content delivery does NOT eagerly recreate status (poll loop handles it).
3. _do_send_status_message edits existing status instead of sending new.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.message_queue import (
    MessageTask,
    _do_send_status_message,
    _process_status_update_task,
    _status_msg_info,
)

USER_ID = 1
THREAD_ID = 10
WINDOW_ID = "@0"
SKEY = (USER_ID, THREAD_ID)


@pytest.fixture(autouse=True)
def _clear_status_tracking():
    _status_msg_info.pop(SKEY, None)
    yield
    _status_msg_info.pop(SKEY, None)


def _status_task(text: str = "running...", window_id: str = WINDOW_ID) -> MessageTask:
    return MessageTask(
        task_type="status_update",
        text=text,
        window_id=window_id,
        thread_id=THREAD_ID,
    )


class TestEditFailureNoNewMessage:
    """Change 1: edit failure clears tracking, does NOT send a new message."""

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.edit_with_fallback", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue.rate_limit_send_message", new_callable=AsyncMock
    )
    async def test_edit_failure_clears_tracking_no_send(
        self, mock_send, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        mock_edit.return_value = False  # edit fails

        # Pre-populate: existing status message tracked
        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text")

        bot = AsyncMock()
        await _process_status_update_task(bot, USER_ID, _status_task("new text"))

        # Tracking should be cleared
        assert SKEY not in _status_msg_info

        # No new message should be sent
        mock_send.assert_not_called()

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.edit_with_fallback", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue.rate_limit_send_message", new_callable=AsyncMock
    )
    async def test_edit_success_updates_tracking(
        self, mock_send, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        mock_edit.return_value = True  # edit succeeds

        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text")

        bot = AsyncMock()
        await _process_status_update_task(bot, USER_ID, _status_task("new text"))

        # Tracking should be updated with new text, same message id
        assert _status_msg_info[SKEY] == (100, WINDOW_ID, "new text")
        mock_send.assert_not_called()


class TestDoSendGuard:
    """Change 3: _do_send_status_message edits existing instead of sending new."""

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.edit_with_fallback", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue.rate_limit_send_message", new_callable=AsyncMock
    )
    async def test_existing_status_same_window_edits_in_place(
        self, mock_send, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        mock_edit.return_value = True

        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text")

        bot = AsyncMock()
        await _do_send_status_message(bot, USER_ID, THREAD_ID, WINDOW_ID, "new text")

        # Should edit, not send new
        mock_edit.assert_called_once()
        mock_send.assert_not_called()
        assert _status_msg_info[SKEY] == (100, WINDOW_ID, "new text")

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.edit_with_fallback", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue.rate_limit_send_message", new_callable=AsyncMock
    )
    async def test_existing_status_identical_text_skips(
        self, mock_send, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42

        _status_msg_info[SKEY] = (100, WINDOW_ID, "running...")

        bot = AsyncMock()
        await _do_send_status_message(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        # Should do nothing — identical text
        mock_edit.assert_not_called()
        mock_send.assert_not_called()

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.edit_with_fallback", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue.rate_limit_send_message", new_callable=AsyncMock
    )
    async def test_no_existing_status_sends_new(
        self, mock_send, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 200
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _do_send_status_message(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        mock_edit.assert_not_called()
        mock_send.assert_called_once()
        assert _status_msg_info[SKEY] == (200, WINDOW_ID, "running...")

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.edit_with_fallback", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue.rate_limit_send_message", new_callable=AsyncMock
    )
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_existing_status_different_window_clears_and_sends(
        self, mock_clear, mock_send, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 300
        mock_send.return_value = sent_msg

        _status_msg_info[SKEY] = (100, "@1", "running...")  # different window

        bot = AsyncMock()
        await _do_send_status_message(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        mock_clear.assert_called_once_with(bot, USER_ID, THREAD_ID)
        mock_send.assert_called_once()
        assert _status_msg_info[SKEY] == (300, WINDOW_ID, "running...")

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.edit_with_fallback", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue.rate_limit_send_message", new_callable=AsyncMock
    )
    async def test_existing_status_edit_fails_falls_through_to_send(
        self, mock_send, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        mock_edit.return_value = False  # edit fails
        sent_msg = MagicMock()
        sent_msg.message_id = 400
        mock_send.return_value = sent_msg

        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text")

        bot = AsyncMock()
        await _do_send_status_message(bot, USER_ID, THREAD_ID, WINDOW_ID, "new text")

        # Edit attempted first, then falls through to send
        mock_edit.assert_called_once()
        mock_send.assert_called_once()
        assert _status_msg_info[SKEY] == (400, WINDOW_ID, "new text")
