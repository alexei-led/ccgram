"""Status bubble rendering at the queue-worker entry points.

``send_status_text`` mechanics (edit-in-place, dedup, edit-failure recovery)
live in ``test_status_bubble.py``.  This file covers what
``process_status_update`` / ``process_status_clear`` put *into* the bubble on
top of them: the Claude task list, driven by real ``claude_task_state`` entries
rather than a stubbed snapshot.
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


def _seed_todos(*todos: dict) -> None:
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
                            "input": {"todos": list(todos)},
                        }
                    ]
                },
            }
        ],
    )


class TestProcessStatusUpdate:
    @patch("ccgram.handlers.status.status_bubble.thread_router")
    async def test_appends_claude_task_list_to_status_text(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = CHAT_ID
        _seed_todos(
            {"content": "Review changes", "status": "completed"},
            {
                "content": "Write tests",
                "status": "in_progress",
                "activeForm": "Writing tests",
            },
        )

        bot = _make_bot(send_id=500)
        await process_status_update(
            bot,
            USER_ID,
            StatusUpdateTask(text="Working", window_id=WINDOW_ID, thread_id=THREAD_ID),
        )

        sent_text = bot.send_message.call_args.kwargs["text"]
        assert sent_text.startswith("Working")
        assert "2 tasks (1 done, 1 open)" in sent_text
        assert "✔ #1 Review changes" in sent_text
        assert "◔ #2 Writing tests" in sent_text


class TestProcessStatusClear:
    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_keeps_bubble_showing_the_task_list(self, mock_edit, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = CHAT_ID
        mock_edit.return_value = True
        _status_msg_info[SKEY] = (100, WINDOW_ID, "old text", CHAT_ID)
        _seed_todos({"content": "Review changes", "status": "completed"})

        await process_status_clear(
            AsyncMock(),
            USER_ID,
            StatusClearTask(thread_id=THREAD_ID, window_id=WINDOW_ID),
        )

        sent_text = mock_edit.call_args[0][3]
        assert sent_text.startswith("1 tasks (1 done, 0 open)")
        assert "✔ #1 Review changes" in sent_text
