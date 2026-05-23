"""Tests for status message singleton behavior (no pile-up).

Verifies three invariants:
1. Edit failure recovers by sending a new status message (no ghost messages).
2. Content delivery does NOT eagerly recreate status (poll loop handles it).
3. send_status_text edits existing status instead of sending new.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.claude_task_state import claude_task_state
from ccgram.handlers.messaging_pipeline.message_task import (
    StatusClearTask,
    StatusUpdateTask,
)
from ccgram.handlers.status.status_bubble import (
    _status_msg_info,
    process_status_clear,
    process_status_update,
    send_status_text,
)

USER_ID = 1
THREAD_ID = 10
WINDOW_ID = "@0"
CHAT_ID = 42
SKEY = (USER_ID, THREAD_ID)


@pytest.fixture(autouse=True)
def _clear_status_tracking():
    _status_msg_info.pop(SKEY, None)
    yield
    _status_msg_info.pop(SKEY, None)


def _make_bot(send_id: int = 200) -> AsyncMock:
    bot = AsyncMock()
    sent = MagicMock()
    sent.message_id = send_id
    bot.send_message.return_value = sent
    return bot


def _status_task(
    text: str = "running...", window_id: str = WINDOW_ID
) -> StatusUpdateTask:
    return StatusUpdateTask(
        text=text,
        window_id=window_id,
        thread_id=THREAD_ID,
    )


class TestEditFailureNoNewMessage:
    """Edit failure recovers by sending a new message (no ghost messages)."""

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_edit_failure_recovers_with_new_send(
        self, mock_edit, mock_tr
    ) -> None:
        # No active DraftStream → falls back to edit_with_fallback,
        # which fails, triggering a new send via DraftStream.start.
        mock_tr.resolve_chat_id.return_value = 42
        mock_edit.return_value = False  # edit fails

        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text", CHAT_ID)

        bot = _make_bot(send_id=200)
        await process_status_update(bot, USER_ID, _status_task("new text"))

        bot.send_message.assert_awaited_once()
        assert _status_msg_info[SKEY][2] == "new text"

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_edit_success_updates_tracking(self, mock_edit, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        mock_edit.return_value = True  # edit succeeds

        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text", CHAT_ID)

        bot = _make_bot()
        await process_status_update(bot, USER_ID, _status_task("new text"))

        # Tracking updated with new text, same message id; no fresh send.
        assert _status_msg_info[SKEY] == (100, WINDOW_ID, "new text", CHAT_ID)
        bot.send_message.assert_not_called()

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    async def test_status_update_appends_claude_tasks(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = CHAT_ID
        claude_task_state.apply_entries(
            WINDOW_ID,
            "session-1",
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "TodoWrite",
                                "input": {
                                    "todos": [
                                        {
                                            "content": "Review changes",
                                            "status": "completed",
                                        },
                                        {
                                            "content": "Write tests",
                                            "status": "in_progress",
                                            "activeForm": "Writing tests",
                                        },
                                    ]
                                },
                            }
                        ]
                    },
                }
            ],
        )

        bot = _make_bot(send_id=500)
        await process_status_update(bot, USER_ID, _status_task("Working"))

        sent_text = bot.send_message.call_args.kwargs["text"]
        assert sent_text.startswith("Working")
        assert "2 tasks (1 done, 1 open)" in sent_text
        assert "✔ #1 Review changes" in sent_text
        assert "◔ #2 Writing tests" in sent_text

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_status_clear_renders_task_only_when_snapshot_exists(
        self, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = CHAT_ID
        mock_edit.return_value = True
        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text", CHAT_ID)
        claude_task_state.apply_entries(
            WINDOW_ID,
            "session-1",
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "TodoWrite",
                                "input": {
                                    "todos": [
                                        {
                                            "content": "Review changes",
                                            "status": "completed",
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            ],
        )

        bot = AsyncMock()
        await process_status_clear(
            bot,
            USER_ID,
            StatusClearTask(thread_id=THREAD_ID, window_id=WINDOW_ID),
        )

        sent_text = mock_edit.call_args[0][3]
        assert sent_text.startswith("1 tasks (1 done, 0 open)")
        assert "✔ #1 Review changes" in sent_text


class TestDoSendGuard:
    """Change 3: send_status_text edits existing instead of sending new."""

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_existing_status_same_window_edits_in_place(
        self, mock_edit, mock_tr
    ) -> None:
        # Pre-existing tracking but no DraftStream → falls back to edit_with_fallback.
        mock_tr.resolve_chat_id.return_value = 42
        mock_edit.return_value = True

        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text", CHAT_ID)

        bot = _make_bot()
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "new text")

        mock_edit.assert_awaited_once()
        bot.send_message.assert_not_called()
        assert _status_msg_info[SKEY] == (100, WINDOW_ID, "new text", CHAT_ID)

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_existing_status_identical_text_skips(
        self, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42

        _status_msg_info[SKEY] = (100, WINDOW_ID, "running...", CHAT_ID)

        bot = _make_bot()
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        # Identical text → no API calls.
        mock_edit.assert_not_called()
        bot.send_message.assert_not_called()
        bot.edit_message_text.assert_not_called()

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_no_existing_status_sends_new(self, mock_edit, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42

        bot = _make_bot(send_id=200)
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        mock_edit.assert_not_called()
        bot.send_message.assert_awaited_once()
        assert _status_msg_info[SKEY] == (200, WINDOW_ID, "running...", CHAT_ID)

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.status.status_bubble.clear_status_message",
        new_callable=AsyncMock,
    )
    async def test_existing_status_different_window_clears_and_sends(
        self, mock_clear, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42

        _status_msg_info[SKEY] = (100, "@1", "running...", CHAT_ID)  # different window

        bot = _make_bot(send_id=300)
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        mock_clear.assert_called_once_with(bot, USER_ID, THREAD_ID)
        bot.send_message.assert_awaited_once()
        assert _status_msg_info[SKEY] == (300, WINDOW_ID, "running...", CHAT_ID)

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_existing_status_edit_fails_falls_through_to_send(
        self, mock_edit, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        mock_edit.return_value = False  # edit fails

        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text", CHAT_ID)

        bot = _make_bot(send_id=400)
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "new text")

        # Edit attempted first, then falls through to send.
        mock_edit.assert_awaited_once()
        bot.send_message.assert_awaited_once()
        assert _status_msg_info[SKEY] == (400, WINDOW_ID, "new text", CHAT_ID)
