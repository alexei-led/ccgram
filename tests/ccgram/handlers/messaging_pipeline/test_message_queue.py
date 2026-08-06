import ast
import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.handlers.messaging_pipeline.message_queue import (
    MERGE_MAX_LENGTH,
    _can_merge_tasks,
    _coalesce_status_updates,
    _dispatch,
    _handle_content_task,
    _merge_content_tasks,
    _process_content_task,
    get_or_create_queue,
    shutdown_workers,
)
from ccgram.handlers.messaging_pipeline.message_task import (
    ContentTask,
    ContentType,
    StatusClearTask,
    StatusUpdateTask,
)
from ccgram.telegram_client import FakeTelegramClient


@pytest.fixture
def bot() -> FakeTelegramClient:
    return FakeTelegramClient()


@pytest.fixture
def queue():
    return asyncio.Queue()


@pytest.fixture
def lock():
    return asyncio.Lock()


def _content_task(
    text: str = "hello",
    window_id: str = "@0",
    content_type: ContentType = "text",
    thread_id: int | None = 42,
    tool_use_id: str | None = None,
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=(text,),
        content_type=content_type,
        thread_id=thread_id,
        tool_use_id=tool_use_id,
    )


def _status_task(
    text: str = "Thinking...",
    window_id: str = "@0",
    thread_id: int | None = 42,
) -> StatusUpdateTask:
    return StatusUpdateTask(
        text=text,
        window_id=window_id,
        thread_id=thread_id,
    )


def _clear_task(
    window_id: str = "@0",
    thread_id: int | None = 42,
) -> StatusClearTask:
    return StatusClearTask(
        window_id=window_id,
        thread_id=thread_id,
    )


class TestGetOrCreateQueue:
    async def test_creates_queue_and_worker(self, bot):
        user_id = 99990
        from ccgram.handlers.messaging_pipeline.message_queue import (
            _message_queues,
            _queue_workers,
        )

        _message_queues.pop(user_id, None)
        _queue_workers.pop(user_id, None)

        try:
            q = get_or_create_queue(bot, user_id)
            assert q is not None
            assert user_id in _queue_workers
        finally:
            await shutdown_workers()

    async def test_reuses_existing_queue(self, bot):
        user_id = 99991
        from ccgram.handlers.messaging_pipeline.message_queue import (
            _message_queues,
            _queue_workers,
        )

        _message_queues.pop(user_id, None)
        _queue_workers.pop(user_id, None)

        try:
            q1 = get_or_create_queue(bot, user_id)
            q2 = get_or_create_queue(bot, user_id)
            assert q1 is q2
        finally:
            await shutdown_workers()


class TestCanMergeTasks:
    def test_same_window_text_tasks_merge(self):
        a = _content_task("hello")
        b = _content_task("world")
        assert _can_merge_tasks(a, b)

    def test_different_window_blocks_merge(self):
        a = _content_task("hello", window_id="@0")
        b = _content_task("world", window_id="@1")
        assert not _can_merge_tasks(a, b)

    def test_tool_use_base_blocks_merge(self):
        a = _content_task("hello", content_type="tool_use")
        b = _content_task("world")
        assert not _can_merge_tasks(a, b)

    def test_tool_result_candidate_blocks_merge(self):
        a = _content_task("hello")
        b = _content_task("world", content_type="tool_result")
        assert not _can_merge_tasks(a, b)

    def test_non_content_candidate_blocks_merge(self):
        a = _content_task("hello")
        b = _status_task()
        assert not _can_merge_tasks(a, b)


class TestMergeContentTasks:
    async def test_merges_consecutive_text_tasks(self, queue, lock):
        queue.put_nowait(_content_task("second"))
        queue.put_nowait(_content_task("third"))
        first = _content_task("first")

        merged, count = await _merge_content_tasks(queue, first, lock)

        assert count == 2
        assert merged.parts == ("first", "second", "third")

    async def test_stops_on_tool_use(self, queue, lock):
        queue.put_nowait(_content_task("second"))
        queue.put_nowait(_content_task("tool", content_type="tool_use"))
        queue.put_nowait(_content_task("after"))
        first = _content_task("first")

        merged, count = await _merge_content_tasks(queue, first, lock)

        assert count == 1
        assert merged.parts == ("first", "second")
        assert queue.qsize() == 2

    async def test_stops_at_length_limit(self, queue, lock):
        big_text = "x" * MERGE_MAX_LENGTH
        queue.put_nowait(_content_task("overflow"))
        first = _content_task(big_text)

        merged, count = await _merge_content_tasks(queue, first, lock)

        assert count == 0
        assert merged.parts == (big_text,)
        assert queue.qsize() == 1

    async def test_no_merge_returns_zero(self, queue, lock):
        first = _content_task("solo")

        merged, count = await _merge_content_tasks(queue, first, lock)

        assert count == 0
        assert merged is first


class TestCoalesceStatusUpdates:
    async def test_keeps_latest_status(self, queue, lock):
        queue.put_nowait(_status_task("Thinking..."))
        queue.put_nowait(_status_task("Writing..."))
        first = _status_task("Reading...")

        selected, dropped = await _coalesce_status_updates(queue, first, lock)

        assert selected.text == "Writing..."
        assert dropped == 2

    async def test_preserves_non_status_tasks(self, queue, lock):
        queue.put_nowait(_content_task("hello"))
        queue.put_nowait(_status_task("Writing..."))
        first = _status_task("Reading...")

        selected, dropped = await _coalesce_status_updates(queue, first, lock)

        assert selected.text == "Writing..."
        assert dropped == 1
        assert queue.qsize() == 1


class TestDispatch:
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.flush_if_active",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=False,
    )
    async def test_content_task_dispatch(
        self, mock_eligible, mock_flush, mock_process, bot, queue, lock
    ):
        ct = _content_task("hello")
        extra = await _dispatch(bot, 1, ct, queue, lock)
        assert extra == 0
        mock_flush.assert_awaited_once_with(bot, 1, ct)
        mock_process.assert_awaited_once()

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_tool_event",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=True,
    )
    async def test_content_task_batch_eligible(
        self, mock_eligible, mock_tool_event, mock_process, bot, queue, lock
    ):
        ct = _content_task("tool", content_type="tool_use")
        mock_tool_event.return_value = None
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
            return_value=False,
        ):
            extra = await _dispatch(bot, 1, ct, queue, lock)
        assert extra == 0
        mock_tool_event.assert_awaited_once_with(bot, 1, ct)
        mock_process.assert_not_awaited()

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_tool_event",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=True,
    )
    async def test_content_task_batch_with_followup(
        self, mock_eligible, mock_tool_event, mock_process, bot, queue, lock
    ):
        ct = _content_task("tool", content_type="tool_use")
        followup = _content_task("overflow")
        mock_tool_event.return_value = followup
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
            return_value=False,
        ):
            await _dispatch(bot, 1, ct, queue, lock)
        mock_process.assert_awaited_once_with(bot, 1, followup)

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_status_update",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._flush_batch_for_task",
        new_callable=AsyncMock,
    )
    async def test_status_update_dispatch(
        self, mock_flush, mock_status, bot, queue, lock
    ):
        st = _status_task("Working...")
        extra = await _dispatch(bot, 1, st, queue, lock)
        assert extra == 0
        mock_flush.assert_awaited_once_with(1, st, bot)
        mock_status.assert_awaited_once()

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_status_clear",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._flush_batch_for_task",
        new_callable=AsyncMock,
    )
    async def test_status_clear_dispatch(
        self, mock_flush, mock_clear, bot, queue, lock
    ):
        cl = _clear_task()
        extra = await _dispatch(bot, 1, cl, queue, lock)
        assert extra == 0
        mock_flush.assert_awaited_once_with(1, cl, bot)
        mock_clear.assert_awaited_once_with(bot, 1, cl)


class TestChatScopedContentDelivery:
    async def test_content_task_uses_explicit_chat_id(self) -> None:
        client = FakeTelegramClient()
        task = ContentTask(
            window_id="@0",
            parts=("hello",),
            thread_id=42,
            chat_id=-1002,
        )

        with (
            patch(
                "ccgram.handlers.messaging_pipeline.message_queue.convert_status_to_content",
                new_callable=AsyncMock,
            ) as mock_convert,
            patch(
                "ccgram.handlers.messaging_pipeline.message_queue.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_send,
        ):
            await _process_content_task(client, 100, task)

        mock_convert.assert_not_awaited()
        mock_send.assert_awaited_once()
        assert mock_send.await_args_list[0].args[:2] == (client, -1002)


class TestNoBackEdgeImports:
    def _get_imports(self, filepath: Path) -> set[str]:
        tree = ast.parse(filepath.read_text())
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        return modules

    def test_tool_batch_does_not_import_message_queue(self):
        src = Path("src/ccgram/handlers/messaging_pipeline/tool_batch.py")
        imports = self._get_imports(src)
        assert not any("message_queue" in m for m in imports), (
            f"tool_batch.py must not import from message_queue: {imports}"
        )

    def test_status_bubble_does_not_import_message_queue(self):
        src = Path("src/ccgram/handlers/status/status_bubble.py")
        imports = self._get_imports(src)
        assert not any("message_queue" in m for m in imports), (
            f"status_bubble.py must not import from message_queue: {imports}"
        )


class TestMessageQueueWorker:
    async def test_telegram_error_calls_task_done(self, bot):
        from ccgram.handlers.messaging_pipeline.message_queue import (
            _message_queue_worker,
            _message_queues,
            _queue_locks,
        )
        from telegram.error import TelegramError

        user_id = 88001
        _message_queues[user_id] = asyncio.Queue()
        _queue_locks[user_id] = asyncio.Lock()
        q = _message_queues[user_id]
        q.put_nowait(_content_task("hello"))
        worker = asyncio.create_task(_message_queue_worker(bot, user_id))
        try:
            with patch(
                "ccgram.handlers.messaging_pipeline.message_queue._dispatch",
                new_callable=AsyncMock,
                side_effect=TelegramError("fail"),
            ):
                await asyncio.wait_for(q.join(), timeout=1.0)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            _message_queues.pop(user_id, None)
            _queue_locks.pop(user_id, None)

    async def test_oserror_calls_task_done(self, bot):
        from ccgram.handlers.messaging_pipeline.message_queue import (
            _message_queue_worker,
            _message_queues,
            _queue_locks,
        )

        user_id = 88002
        _message_queues[user_id] = asyncio.Queue()
        _queue_locks[user_id] = asyncio.Lock()
        q = _message_queues[user_id]
        q.put_nowait(_content_task("hello"))
        worker = asyncio.create_task(_message_queue_worker(bot, user_id))
        try:
            with patch(
                "ccgram.handlers.messaging_pipeline.message_queue._dispatch",
                new_callable=AsyncMock,
                side_effect=OSError("disk error"),
            ):
                await asyncio.wait_for(q.join(), timeout=1.0)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            _message_queues.pop(user_id, None)
            _queue_locks.pop(user_id, None)

    async def test_cancelled_error_exits_cleanly(self, bot):
        from ccgram.handlers.messaging_pipeline.message_queue import (
            _message_queue_worker,
            _message_queues,
            _queue_locks,
        )

        user_id = 88003
        _message_queues[user_id] = asyncio.Queue()
        _queue_locks[user_id] = asyncio.Lock()
        worker = asyncio.create_task(_message_queue_worker(bot, user_id))
        try:
            await asyncio.sleep(0)
            worker.cancel()
            await asyncio.wait_for(worker, timeout=1.0)
        except asyncio.CancelledError:
            pass
        finally:
            _message_queues.pop(user_id, None)
            _queue_locks.pop(user_id, None)

        assert worker.done()
        assert not worker.exception() if not worker.cancelled() else True


class TestThinkingGate:
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.flush_if_active",
        new_callable=AsyncMock,
    )
    async def test_hidden_suppresses_thinking(
        self, mock_flush, mock_process, bot, queue, lock
    ):
        ct = _content_task("private reasoning", content_type="thinking")
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.config.hide_thinking",
            True,
        ):
            extra = await _handle_content_task(bot, 1, ct, queue, lock)

        assert extra == 0
        mock_flush.assert_not_awaited()
        mock_process.assert_not_awaited()

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.flush_if_active",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=False,
    )
    async def test_shown_processes_thinking(
        self, mock_eligible, mock_flush, mock_process, bot, queue, lock
    ):
        ct = _content_task("public reasoning", content_type="thinking")
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.config.hide_thinking",
            False,
        ):
            extra = await _handle_content_task(bot, 1, ct, queue, lock)

        assert extra == 0
        mock_flush.assert_awaited_once_with(bot, 1, ct)
        mock_process.assert_awaited_once_with(bot, 1, ct)


class TestToolCallsGate:
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_tool_event",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.flush_if_active",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=True,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
        return_value=True,
    )
    async def test_hidden_suppresses_tool_use(
        self,
        mock_hidden,
        mock_eligible,
        mock_flush,
        mock_tool_event,
        mock_process,
        bot,
        queue,
        lock,
    ):
        ct = _content_task("tool", content_type="tool_use")
        extra = await _handle_content_task(bot, 1, ct, queue, lock)
        assert extra == 0
        mock_tool_event.assert_not_awaited()
        mock_process.assert_not_awaited()
        mock_flush.assert_not_awaited()

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_tool_event",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.flush_if_active",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=True,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
        return_value=True,
    )
    async def test_hidden_suppresses_tool_result(
        self,
        mock_hidden,
        mock_eligible,
        mock_flush,
        mock_tool_event,
        mock_process,
        bot,
        queue,
        lock,
    ):
        ct = _content_task("res", content_type="tool_result", tool_use_id="t1")
        extra = await _handle_content_task(bot, 1, ct, queue, lock)
        assert extra == 0
        mock_tool_event.assert_not_awaited()
        mock_process.assert_not_awaited()

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_tool_event",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=True,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
        return_value=False,
    )
    async def test_shown_allows_tool_use(
        self,
        mock_hidden,
        mock_eligible,
        mock_tool_event,
        mock_process,
        bot,
        queue,
        lock,
    ):
        ct = _content_task("tool", content_type="tool_use")
        mock_tool_event.return_value = None
        extra = await _handle_content_task(bot, 1, ct, queue, lock)
        assert extra == 0
        mock_tool_event.assert_awaited_once_with(bot, 1, ct)

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.flush_if_active",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=False,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
        return_value=True,
    )
    async def test_text_unaffected_when_hidden(
        self,
        mock_hidden,
        mock_eligible,
        mock_flush,
        mock_process,
        bot,
        queue,
        lock,
    ):
        ct = _content_task("hello", content_type="text")
        extra = await _handle_content_task(bot, 1, ct, queue, lock)
        assert extra == 0
        mock_flush.assert_awaited_once_with(bot, 1, ct)
        mock_process.assert_awaited_once()

    @patch("ccgram.handlers.messaging_pipeline.message_queue._tool_msg_ids", {})
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_tool_event",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_batch_eligible",
        return_value=True,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
        return_value=True,
    )
    async def test_hidden_does_not_register_tool_msg_ids(
        self,
        mock_hidden,
        mock_eligible,
        mock_tool_event,
        mock_process,
        bot,
        queue,
        lock,
    ):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        mq._tool_msg_ids.clear()
        ct = _content_task("tool", content_type="tool_use", tool_use_id="t-xyz")
        await _handle_content_task(bot, 1, ct, queue, lock)
        assert not mq._tool_msg_ids


class TestTruncateCaption:
    def test_short_text_unchanged(self):
        from ccgram.handlers.messaging_pipeline.message_queue import _truncate_caption

        assert _truncate_caption("hello world") == "hello world"

    def test_truncates_at_whitespace(self):
        from ccgram.handlers.messaging_pipeline.message_queue import _truncate_caption

        text = ("word " * 220).rstrip()  # > 1024 chars
        result = _truncate_caption(text)
        assert len(result) <= 1024
        assert result.endswith("…")
        assert not result[:-1].endswith(" ")

    def test_no_whitespace_still_within_limit(self):
        from ccgram.handlers.messaging_pipeline.message_queue import _truncate_caption

        text = "x" * 1100  # no spaces
        result = _truncate_caption(text)
        assert len(result) <= 1024  # must not exceed limit

    def test_exact_limit_unchanged(self):
        from ccgram.handlers.messaging_pipeline.message_queue import _truncate_caption

        text = "a" * 1024
        assert _truncate_caption(text) == text
