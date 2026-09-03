import ast
import asyncio
import contextlib
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import RetryAfter, TelegramError
from telegramify_markdown import utf16_len

from ccgram.delivery_contract import DeliveryOutcome
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
    chat_id: int | None = -1001,
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=(text,),
        content_type=content_type,
        thread_id=thread_id,
        tool_use_id=tool_use_id,
        chat_id=chat_id,
    )


def _status_task(
    text: str = "Thinking...",
    window_id: str = "@0",
    thread_id: int | None = 42,
    *,
    transient: bool = False,
) -> StatusUpdateTask:
    return StatusUpdateTask(
        text=text,
        window_id=window_id,
        thread_id=thread_id,
        transient=transient,
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
    async def test_coalesces_queued_status_operations(self, bot, monkeypatch):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 99989
        queue = asyncio.Queue()
        monkeypatch.setattr(mq, "get_or_create_queue", lambda *_: queue)
        monkeypatch.setattr(mq, "is_window_live", lambda _: True)

        await mq.enqueue_status_update(bot, user_id, "@0", "first", thread_id=42)
        await mq.enqueue_status_update(bot, user_id, "@0", "latest", thread_id=42)

        assert queue.qsize() == 1
        assert mq._pending_status_updates == {(user_id, "@0", 42, False)}
        queue.get_nowait()
        queue.task_done()
        mq._pending_status_updates.clear()

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
        assert _can_merge_tasks(_content_task("hello"), _content_task("world"))

    @pytest.mark.parametrize("base_type", ["tool_use", "tool_result"])
    @pytest.mark.parametrize("candidate_type", ["text", "tool_use", "tool_result"])
    def test_tool_base_blocks_merge(
        self, base_type: ContentType, candidate_type: ContentType
    ):
        base = _content_task("hello", content_type=base_type)
        candidate = _content_task("world", content_type=candidate_type)
        assert not _can_merge_tasks(base, candidate)

    @pytest.mark.parametrize("candidate_type", ["tool_use", "tool_result"])
    def test_tool_candidate_blocks_merge(self, candidate_type: ContentType):
        candidate = _content_task("world", content_type=candidate_type)
        assert not _can_merge_tasks(_content_task("hello"), candidate)

    def test_different_window_blocks_merge(self):
        a = _content_task("hello", window_id="@0")
        b = _content_task("world", window_id="@1")
        assert not _can_merge_tasks(a, b)

    @pytest.mark.parametrize("content_type", ["thinking", "tool_use", "tool_result"])
    def test_non_plain_text_blocks_merge(self, content_type: ContentType):
        candidate = _content_task("world", content_type=content_type)
        assert not _can_merge_tasks(_content_task("hello"), candidate)

    def test_chat_and_thread_boundaries_block_merge(self):
        base = ContentTask(
            window_id="@0", parts=("hello",), thread_id=42, chat_id=-1001
        )
        assert not _can_merge_tasks(
            base,
            ContentTask(window_id="@0", parts=("world",), thread_id=43, chat_id=-1001),
        )
        assert not _can_merge_tasks(
            base,
            ContentTask(window_id="@0", parts=("world",), thread_id=42, chat_id=-1002),
        )
        assert not _can_merge_tasks(
            ContentTask(window_id="@0", parts=("hello",), thread_id=42),
            ContentTask(window_id="@0", parts=("world",), thread_id=42),
        )

    def test_tool_metadata_and_paginated_parts_block_merge(self):
        base = _content_task("hello")
        assert not _can_merge_tasks(base, _content_task("world", tool_use_id="tool-1"))
        assert not _can_merge_tasks(
            base,
            ContentTask(window_id="@0", parts=("page one", "page two"), thread_id=42),
        )

    def test_non_content_candidate_blocks_merge(self):
        assert not _can_merge_tasks(_content_task("hello"), _status_task())


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

    async def test_merges_up_to_the_exact_length_limit(self, queue, lock):
        half = "x" * ((MERGE_MAX_LENGTH - 2) // 2)
        queue.put_nowait(_content_task(half))
        queue.put_nowait(_content_task("overflow"))
        first = _content_task(half)

        merged, count = await _merge_content_tasks(queue, first, lock)

        assert count == 1
        assert sum(len(p) for p in merged.parts) + 2 <= MERGE_MAX_LENGTH
        assert merged.is_text_batch is True
        assert queue.qsize() == 1

    async def test_separator_counts_toward_length_limit(self, queue, lock):
        first = _content_task("x" * (MERGE_MAX_LENGTH - 1))
        queue.put_nowait(_content_task("x"))

        merged, count = await _merge_content_tasks(queue, first, lock)

        assert merged is first
        assert count == 0
        assert queue.qsize() == 1

    async def test_utf16_units_count_toward_length_limit(self, queue, lock):
        first = _content_task("😀" * (MERGE_MAX_LENGTH // 2))
        queue.put_nowait(_content_task("x"))

        merged, count = await _merge_content_tasks(queue, first, lock)

        assert utf16_len(first.parts[0]) == MERGE_MAX_LENGTH
        assert merged is first
        assert count == 0
        assert queue.qsize() == 1

    async def test_no_merge_returns_zero(self, queue, lock):
        first = _content_task("solo")

        merged, count = await _merge_content_tasks(queue, first, lock)

        assert count == 0
        assert merged is first


class TestActualTextBatching:
    async def test_consecutive_text_tasks_use_one_formatted_api_call(
        self, bot, queue, lock
    ):
        first = ContentTask(
            window_id="@0", parts=("**first**",), thread_id=42, chat_id=-1001
        )
        queue.put_nowait(
            ContentTask(
                window_id="@0", parts=("_second_",), thread_id=42, chat_id=-1001
            )
        )

        result = await _handle_content_task(bot, 1, first, queue, lock)

        assert result == 1
        assert bot.call_count("send_message") == 1
        call = bot.last_call("send_message")
        assert call is not None
        assert call.kwargs["text"] == "first\n\nsecond"
        assert {entity.type for entity in call.kwargs["entities"]} == {"bold", "italic"}
        assert [entity.offset for entity in call.kwargs["entities"]] == [0, 7]

    async def test_tts_media_tasks_do_not_merge(self, monkeypatch):
        first = _content_task("first")
        candidate = _content_task("second")
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.message_queue.config.tts_provider",
            "edge",
        )

        assert not _can_merge_tasks(first, candidate)


class TestCoalesceStatusUpdates:
    async def test_keeps_latest_status(self, queue, lock):
        queue.put_nowait(_status_task("Thinking..."))
        queue.put_nowait(_status_task("Writing..."))
        first = _status_task("Reading...")

        selected, dropped = await _coalesce_status_updates(queue, first, lock)

        assert selected.text == "Writing..."
        assert dropped == 2

    async def test_stops_at_status_for_a_different_window(self, queue, lock):
        queue.put_nowait(_status_task("Writing...", window_id="@0"))
        queue.put_nowait(_status_task("other window", window_id="@1"))
        first = _status_task("Reading...", window_id="@0")

        selected, dropped = await _coalesce_status_updates(queue, first, lock)

        assert selected.text == "Writing..."
        assert dropped == 1
        remaining = queue.get_nowait()
        assert isinstance(remaining, StatusUpdateTask)
        assert remaining.text == "other window"

    async def test_does_not_collapse_durable_notice_with_transient_refresh(
        self, queue, lock
    ):
        transient = _status_task("Working...", transient=True)
        queue.put_nowait(transient)
        durable = _status_task("Done")

        selected, dropped = await _coalesce_status_updates(queue, durable, lock)

        assert selected is durable
        assert dropped == 0
        assert queue.get_nowait() is transient

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
        from ccgram.handlers.messaging_pipeline.tool_batch import ToolEventResult

        ct = _content_task("tool", content_type="tool_use")
        mock_tool_event.return_value = ToolEventResult()
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
            return_value=False,
        ):
            result = await _dispatch(bot, 1, ct, queue, lock)
        assert result == 0
        assert result.outcome.value == "delivered"
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
    async def test_content_task_batch_failure_is_not_delivered(
        self, mock_eligible, mock_tool_event, mock_process, bot, queue, lock
    ):
        from ccgram.handlers.messaging_pipeline.tool_batch import (
            ToolEventOutcome,
            ToolEventResult,
        )

        ct = _content_task("tool", content_type="tool_use")
        mock_tool_event.return_value = ToolEventResult(outcome=ToolEventOutcome.FAILED)
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
            return_value=False,
        ):
            result = await _dispatch(bot, 1, ct, queue, lock)

        assert result.outcome.value == "failed"
        mock_process.assert_not_awaited()

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
    async def test_stale_backlog_notice_is_not_sent(self) -> None:
        client = FakeTelegramClient()
        task = ContentTask(
            window_id="@0",
            parts=("skipped",),
            thread_id=42,
            chat_id=-1002,
            is_backlog_notice=True,
        )

        with (
            patch(
                "ccgram.handlers.messaging_pipeline.message_queue.thread_router.resolve_window_for_thread",
                return_value="@other",
            ),
            patch(
                "ccgram.handlers.messaging_pipeline.message_queue.rate_limit_send_message",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            outcome = await _process_content_task(client, 100, task)

        assert outcome is DeliveryOutcome.FAILED
        mock_send.assert_not_awaited()

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
    async def test_confirmed_dead_window_drops_queued_task_without_dispatch(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq
        from ccgram.multiplexer.base import WindowRef
        from ccgram.multiplexer.window_liveness import note_live_windows

        user_id = 88006
        await mq.shutdown_workers()
        note_live_windows(
            [WindowRef(window_id="@1", window_name="live", cwd="/tmp")],
            tracked_window_ids=["@0"],
        )
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipt = mq.DeliveryReceipt()
        receipt.track()
        receipt.close()
        q = mq._message_queues[user_id]
        q.put_nowait(
            ContentTask(
                window_id="@0",
                parts=("stale",),
                thread_id=42,
                delivery_receipts=(receipt,),
            )
        )
        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with patch.object(mq, "_dispatch", new_callable=AsyncMock) as dispatch:
                await asyncio.wait_for(q.join(), timeout=1.0)
            dispatch.assert_not_awaited()
            assert receipt.commit_ready
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            await mq.shutdown_workers()

    async def test_retry_flood_stops_when_target_session_closes(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq
        from ccgram.multiplexer.base import WindowRef
        from ccgram.multiplexer.window_liveness import note_live_windows

        user_id = 88009
        await mq.shutdown_workers()
        note_live_windows(
            [WindowRef(window_id="@0", window_name="live", cwd="/tmp")],
            tracked_window_ids=["@0"],
        )
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipt = mq.DeliveryReceipt()
        receipt.track()
        receipt.close()
        q = mq._message_queues[user_id]
        q.put_nowait(
            ContentTask(
                window_id="@0",
                parts=("flooded",),
                thread_id=42,
                delivery_receipts=(receipt,),
            )
        )

        async def close_session_during_backoff(_delay: float) -> None:
            note_live_windows([], tracked_window_ids=["@0"])

        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with (
                patch.object(
                    mq, "_dispatch", new_callable=AsyncMock, side_effect=RetryAfter(30)
                ) as dispatch,
                patch.object(
                    mq.asyncio, "sleep", new_callable=AsyncMock
                ) as retry_sleep,
            ):
                retry_sleep.side_effect = close_session_during_backoff
                await asyncio.wait_for(q.join(), timeout=1.0)
            dispatch.assert_awaited_once()
            retry_sleep.assert_awaited_once()
            assert receipt.commit_ready
            assert not receipt.failed
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            await mq.shutdown_workers()

    async def test_large_stale_backlog_drains_without_telegram_flood(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq
        from ccgram.multiplexer.base import WindowRef
        from ccgram.multiplexer.window_liveness import note_live_windows

        user_id = 88010
        stale_count = 1_000
        await mq.shutdown_workers()
        note_live_windows(
            [WindowRef(window_id="live", window_name="live", cwd="/tmp")],
            tracked_window_ids=["closed", "live"],
        )
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipts: list[mq.DeliveryReceipt] = []
        q = mq._message_queues[user_id]
        for index in range(stale_count):
            receipt = mq.DeliveryReceipt()
            receipt.track()
            receipt.close()
            receipts.append(receipt)
            q.put_nowait(
                ContentTask(
                    window_id="closed",
                    parts=(f"stale-{index}",),
                    thread_id=42,
                    delivery_receipts=(receipt,),
                )
            )
        q.put_nowait(ContentTask(window_id="live", parts=("current",), thread_id=43))

        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with (
                patch.object(
                    mq,
                    "_dispatch",
                    new_callable=AsyncMock,
                    return_value=mq.DispatchResult(0, mq.DeliveryOutcome.DELIVERED),
                ) as dispatch,
                patch.object(mq.logger, "info"),
            ):
                await asyncio.wait_for(q.join(), timeout=2.0)
            dispatch.assert_awaited_once()
            assert all(receipt.commit_ready for receipt in receipts)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            await mq.shutdown_workers()

    async def test_status_rate_limit_drops_without_blocking_content(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88013
        queue: asyncio.Queue = asyncio.Queue()
        lock = asyncio.Lock()
        status = StatusUpdateTask(
            window_id="@0", text="Working", thread_id=42, transient=True
        )
        with (
            patch.object(
                mq, "_dispatch", new_callable=AsyncMock, side_effect=RetryAfter(30)
            ) as dispatch,
            patch.object(mq.asyncio, "sleep", new_callable=AsyncMock) as sleep,
        ):
            outcome = await mq._dispatch_with_retry(
                bot, user_id, mq.RetryDispatchState(status), queue, lock
            )

        assert outcome is mq.DeliveryOutcome.INTENTIONALLY_DROPPED
        dispatch.assert_awaited_once()
        sleep.assert_not_awaited()
        assert mq._status_suppressed_until[user_id] > mq.time.monotonic()
        mq._status_suppressed_until.clear()

    async def test_status_clear_rate_limit_drops_without_sleeping(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88015
        queue: asyncio.Queue = asyncio.Queue()
        lock = asyncio.Lock()
        clear = StatusClearTask(window_id="@0", thread_id=42)
        with (
            patch.object(
                mq, "_dispatch", new_callable=AsyncMock, side_effect=RetryAfter(30)
            ) as dispatch,
            patch.object(mq.asyncio, "sleep", new_callable=AsyncMock) as sleep,
        ):
            outcome = await mq._dispatch_with_retry(
                bot, user_id, mq.RetryDispatchState(clear), queue, lock
            )

        assert outcome is mq.DeliveryOutcome.INTENTIONALLY_DROPPED
        dispatch.assert_awaited_once()
        sleep.assert_not_awaited()
        mq._status_suppressed_until.clear()

    async def test_durable_status_retries_after_flood_control(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88016
        queue: asyncio.Queue = asyncio.Queue()
        lock = asyncio.Lock()
        status = StatusUpdateTask(window_id="@0", text="Done", thread_id=42)
        with (
            patch.object(
                mq,
                "_dispatch",
                new_callable=AsyncMock,
                side_effect=[
                    RetryAfter(1),
                    mq.DispatchResult(0, mq.DeliveryOutcome.DELIVERED),
                ],
            ) as dispatch,
            patch.object(mq.asyncio, "sleep", new_callable=AsyncMock) as sleep,
            patch.object(mq.random, "uniform", return_value=0),
        ):
            outcome = await mq._dispatch_with_retry(
                bot, user_id, mq.RetryDispatchState(status), queue, lock
            )

        assert outcome is mq.DeliveryOutcome.DELIVERED
        assert dispatch.await_count == 2
        sleep.assert_awaited_once()
        assert user_id not in mq._status_suppressed_until

    async def test_suppressed_status_update_is_not_enqueued(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88014
        queue: asyncio.Queue = asyncio.Queue()
        mq._status_suppressed_until[user_id] = mq.time.monotonic() + 30
        with patch.object(mq, "get_or_create_queue", return_value=queue) as get_queue:
            await mq.enqueue_status_update(
                bot, user_id, "@0", "Working", 42, transient=True
            )

        get_queue.assert_not_called()
        assert queue.empty()
        mq._status_suppressed_until.clear()

    async def test_suppression_keeps_durable_status_notice(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88017
        queue: asyncio.Queue = asyncio.Queue()
        mq._status_suppressed_until[user_id] = mq.time.monotonic() + 30
        with patch.object(mq, "get_or_create_queue", return_value=queue):
            await mq.enqueue_status_update(bot, user_id, "@0", "Done", 42)

        task = queue.get_nowait()
        assert isinstance(task, StatusUpdateTask)
        assert task.text == "Done"
        assert not task.transient
        queue.task_done()
        mq._pending_status_updates.clear()
        mq._status_suppressed_until.clear()

    async def test_retry_budget_fails_bound_task_after_budget(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88007
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipt = mq.DeliveryReceipt()
        receipt.track()
        receipt.close()
        q = mq._message_queues[user_id]
        q.put_nowait(
            ContentTask(window_id="@0", parts=("hello",), delivery_receipts=(receipt,))
        )
        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with (
                patch.object(mq, "_QUEUE_RETRY_BUDGET_SECONDS", 0),
                patch.object(
                    mq, "_dispatch", new_callable=AsyncMock, side_effect=RetryAfter(1)
                ) as dispatch,
            ):
                await asyncio.wait_for(q.join(), timeout=1.0)
            dispatch.assert_awaited_once()
            assert receipt.failed
            assert not receipt.commit_ready
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_retry_after_tool_edit_keeps_original_message_id(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        key = ("tool-1", 88008, 42)
        mq._tool_msg_ids.clear()
        mq._tool_msg_ids[key] = 123
        task = ContentTask(
            window_id="@0",
            parts=("result",),
            content_type="tool_result",
            tool_use_id="tool-1",
            thread_id=42,
            chat_id=-1001,
        )
        bot.set_side_effect("edit_message_text", [RetryAfter(1), True])
        with patch.object(mq, "clear_status_message", new_callable=AsyncMock):
            with pytest.raises(RetryAfter):
                await mq._try_edit_tool_result(bot, 88008, -1001, 42, task)
            assert mq._tool_msg_ids[key] == 123
            assert await mq._try_edit_tool_result(bot, 88008, -1001, 42, task)
        assert key not in mq._tool_msg_ids
        assert bot.call_count("edit_message_text") == 2

    def test_closed_session_drop_log_is_throttled(self):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88015
        task = ContentTask(window_id="closed", parts=("stale",), thread_id=42)
        mq._message_queues[user_id] = asyncio.Queue()
        mq._message_queues[user_id].put_nowait(task)
        mq._stale_drop_log_last_at.clear()
        try:
            with (
                patch.object(mq, "is_window_live", return_value=False),
                patch.object(mq.time, "monotonic", side_effect=(100.0, 110.0, 131.0)),
                patch.object(mq.logger, "info") as info,
            ):
                assert mq._is_stale_task(user_id, task)
                assert mq._is_stale_task(user_id, task)
                assert mq._is_stale_task(user_id, task)

            assert info.call_count == 2
            assert info.call_args_list[0].args == (
                "Dropping queued messages for closed multiplexer session",
            )
            assert info.call_args_list[0].kwargs["queued_tasks"] == 1
        finally:
            mq._message_queues.pop(user_id, None)
            mq._stale_drop_log_last_at.clear()

    def test_rate_limit_debug_log_is_throttled_and_reports_queue_state(self):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88012
        task = ContentTask(window_id="@0", parts=("blocked",), thread_id=42)
        mq._message_queues[user_id] = asyncio.Queue()
        mq._message_queues[user_id].put_nowait(
            ContentTask(window_id="@0", parts=("waiting",), thread_id=42)
        )
        mq._rate_limit_log_last_at.clear()
        try:
            with (
                patch.object(mq.time, "monotonic", side_effect=(100.0, 110.0, 131.0)),
                patch.object(mq.logger, "debug") as debug,
            ):
                mq._log_rate_limit_queue_state(user_id, task, 1, 5.0, 5.5, 0.2)
                mq._log_rate_limit_queue_state(user_id, task, 2, 5.0, 5.5, 10.2)
                mq._log_rate_limit_queue_state(user_id, task, 3, 5.0, 5.5, 31.2)

            assert debug.call_count == 2
            assert debug.call_args_list[0].args == (
                "Telegram rate limit is blocking message queue",
            )
            assert debug.call_args_list[0].kwargs["queued_tasks"] == 1
            assert debug.call_args_list[0].kwargs["current_task_in_flight"] is True
            assert debug.call_args_list[1].kwargs["retry"] == 3
        finally:
            mq._message_queues.pop(user_id, None)
            mq._rate_limit_log_last_at.clear()

    async def test_retry_after_backoff_keeps_receipt_pending_until_delivery(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88003
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipt = mq.DeliveryReceipt()
        receipt.track()
        receipt.close()
        q = mq._message_queues[user_id]
        q.put_nowait(
            ContentTask(
                window_id="@0",
                parts=("hello",),
                thread_id=42,
                delivery_receipts=(receipt,),
            )
        )
        dispatch = AsyncMock(
            side_effect=[
                RetryAfter(timedelta(seconds=3)),
                RetryAfter(timedelta(seconds=3)),
                mq.DispatchResult(0, mq.DeliveryOutcome.DELIVERED),
            ]
        )
        pending_during_sleep: list[bool] = []
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)
            pending_during_sleep.append(not receipt.failed and not receipt.commit_ready)

        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with (
                patch.object(mq, "_dispatch", dispatch),
                patch.object(mq.asyncio, "sleep", side_effect=record_sleep),
                patch("random.uniform", return_value=0.25),
            ):
                await asyncio.wait_for(q.join(), timeout=1.0)

            assert delays == [3.25, 4.25]
            assert pending_during_sleep == [True, True]
            assert dispatch.await_count == 3
            assert receipt.commit_ready is True
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_terminal_send_failure_blocks_delivery_receipt(self, bot):
        """A drained queue is not a delivery acknowledgement after a send error."""
        from ccgram.handlers.messaging_pipeline import message_queue as mq
        from telegram.error import TelegramError

        user_id = 88000
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipt = mq.DeliveryReceipt()
        receipt.track()
        q = mq._message_queues[user_id]
        q.put_nowait(
            ContentTask(
                window_id="@0",
                parts=("hello",),
                thread_id=42,
                delivery_receipts=(receipt,),
            )
        )
        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with patch.object(
                mq,
                "_dispatch",
                new_callable=AsyncMock,
                side_effect=TelegramError("fail"),
            ):
                await asyncio.wait_for(q.join(), timeout=1.0)
            assert receipt.commit_ready is False
            assert receipt.failed is True
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_intentional_drop_acknowledges_delivery_receipt(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        receipt = mq.DeliveryReceipt()
        receipt.track()
        receipt.close()
        task = ContentTask(
            window_id="@0",
            parts=("private",),
            content_type="thinking",
            delivery_receipts=(receipt,),
        )
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.config.hide_thinking",
            True,
        ):
            result = await mq._dispatch(bot, 1, task, asyncio.Queue(), asyncio.Lock())
        receipt.settle(result.outcome)

        assert result.outcome is mq.DeliveryOutcome.INTENTIONALLY_DROPPED
        assert receipt.commit_ready is True

    async def test_terminal_sender_none_is_a_failed_outcome(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        task = ContentTask(
            window_id="@0", parts=("hello",), thread_id=42, chat_id=-1001
        )
        with patch.object(
            mq, "rate_limit_send_message", new_callable=AsyncMock, return_value=None
        ):
            outcome = await mq._process_content_task(bot, 1, task)

        assert outcome is mq.DeliveryOutcome.FAILED

    async def test_queue_join_waits_for_inflight_send(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 87999
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        q = mq._message_queues[user_id]
        q.put_nowait(_content_task("hello"))
        started = asyncio.Event()
        release = asyncio.Event()

        async def block_dispatch(*_args):
            started.set()
            await release.wait()
            return 0

        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with patch.object(mq, "_dispatch", side_effect=block_dispatch):
                await started.wait()
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(q.join(), timeout=0.01)
                release.set()
                await asyncio.wait_for(q.join(), timeout=1.0)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_cancelled_inflight_send_does_not_acknowledge_receipt(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 87998
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipt = mq.DeliveryReceipt()
        receipt.track()
        receipt.close()
        mq._message_queues[user_id].put_nowait(
            ContentTask(window_id="@0", parts=("hello",), delivery_receipts=(receipt,))
        )
        started = asyncio.Event()

        async def block_dispatch(*_args):
            started.set()
            await asyncio.Event().wait()
            return 0

        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with patch.object(mq, "_dispatch", side_effect=block_dispatch):
                await started.wait()
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
            assert receipt.failed is True
            assert receipt.commit_ready is False
        finally:
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_merged_content_settles_every_delivery_receipt(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 87997
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipts = [mq.DeliveryReceipt(), mq.DeliveryReceipt()]
        tasks = []
        for text, receipt in zip(("first", "second"), receipts, strict=True):
            receipt.track()
            receipt.close()
            tasks.append(
                ContentTask(
                    window_id="@0",
                    parts=(text,),
                    role="assistant",
                    delivery_receipts=(receipt,),
                )
            )
            mq._message_queues[user_id].put_nowait(tasks[-1])

        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with patch.object(
                mq,
                "_process_content_task",
                new_callable=AsyncMock,
                return_value=mq.DeliveryOutcome.DELIVERED,
            ):
                await asyncio.wait_for(mq._message_queues[user_id].join(), timeout=1)

            assert all(receipt.commit_ready for receipt in receipts)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_failed_merged_content_settles_every_receipt(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 87998
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipts = [mq.DeliveryReceipt(), mq.DeliveryReceipt()]
        for text, receipt in zip(("first", "second"), receipts, strict=True):
            receipt.track()
            receipt.close()
            mq._message_queues[user_id].put_nowait(
                ContentTask(
                    window_id="@0",
                    parts=(text,),
                    role="assistant",
                    thread_id=42,
                    chat_id=-1001,
                    delivery_receipts=(receipt,),
                )
            )

        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with patch.object(
                mq,
                "_process_content_task",
                new_callable=AsyncMock,
                side_effect=TelegramError("send failed"),
            ) as process_content:
                await asyncio.wait_for(mq._message_queues[user_id].join(), timeout=1)

            process_content.assert_awaited_once()
            assert process_content.await_args_list[0].args[2].parts == (
                "first",
                "second",
            )
            assert all(receipt.failed for receipt in receipts)
            assert all(not receipt.commit_ready for receipt in receipts)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_cancelled_merged_retry_fails_every_receipt(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88011
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipts = [mq.DeliveryReceipt(), mq.DeliveryReceipt()]
        for text, receipt in zip(("first", "second"), receipts, strict=True):
            receipt.track()
            receipt.close()
            mq._message_queues[user_id].put_nowait(
                ContentTask(
                    window_id="@0",
                    parts=(text,),
                    role="assistant",
                    thread_id=42,
                    chat_id=-1001,
                    delivery_receipts=(receipt,),
                )
            )
        retry_sleep_started = asyncio.Event()

        async def block_retry_sleep(_delay: float) -> None:
            retry_sleep_started.set()
            await asyncio.Event().wait()

        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with (
                patch.object(
                    mq,
                    "_process_content_task",
                    new_callable=AsyncMock,
                    side_effect=RetryAfter(1),
                ) as process_content,
                patch.object(mq.asyncio, "sleep", side_effect=block_retry_sleep),
                patch("random.uniform", return_value=0),
            ):
                await retry_sleep_started.wait()
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
            await asyncio.wait_for(mq._message_queues[user_id].join(), timeout=1)
            process_content.assert_awaited_once()
            assert process_content.await_args_list[0].args[2].parts == (
                "first",
                "second",
            )
            assert all(receipt.failed for receipt in receipts)
            assert all(not receipt.commit_ready for receipt in receipts)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_batched_retry_after_retries_one_combined_payload_and_receipts(
        self, bot
    ):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88004
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipts = [mq.DeliveryReceipt(), mq.DeliveryReceipt()]
        for receipt in receipts:
            receipt.track()
            receipt.close()
        mq._message_queues[user_id].put_nowait(
            ContentTask(
                window_id="@0",
                parts=("first",),
                thread_id=42,
                chat_id=-1001,
                delivery_receipts=(receipts[0],),
            )
        )
        mq._message_queues[user_id].put_nowait(
            ContentTask(
                window_id="@0",
                parts=("second",),
                thread_id=42,
                chat_id=-1001,
                delivery_receipts=(receipts[1],),
            )
        )
        bot.set_side_effect("send_message", [RetryAfter(timedelta(seconds=0)), True])
        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            with (
                patch.object(mq.asyncio, "sleep", new_callable=AsyncMock),
                patch("random.uniform", return_value=0),
            ):
                await asyncio.wait_for(mq._message_queues[user_id].join(), timeout=1)

            assert bot.call_count("send_message") == 2
            assert [call.kwargs["text"] for call in bot.calls] == [
                "first\n\nsecond",
                "first\n\nsecond",
            ]
            assert all(receipt.commit_ready for receipt in receipts)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

    async def test_batched_terminal_failure_fails_every_receipt(self, bot):
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        user_id = 88005
        mq._message_queues[user_id] = asyncio.Queue()
        mq._queue_locks[user_id] = asyncio.Lock()
        receipts = [mq.DeliveryReceipt(), mq.DeliveryReceipt()]
        for receipt in receipts:
            receipt.track()
            receipt.close()
        for text, receipt in zip(("first", "second"), receipts, strict=True):
            mq._message_queues[user_id].put_nowait(
                ContentTask(
                    window_id="@0",
                    parts=(text,),
                    thread_id=42,
                    chat_id=-1001,
                    delivery_receipts=(receipt,),
                )
            )
        # Entity and plain fallback both fail, so the batch reports a terminal
        # delivery failure rather than advancing either transcript receipt.
        bot.set_side_effect(
            "send_message", [TelegramError("entity"), TelegramError("plain")]
        )
        worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
        try:
            await asyncio.wait_for(mq._message_queues[user_id].join(), timeout=1)

            assert bot.call_count("send_message") == 2
            assert all(receipt.failed for receipt in receipts)
            assert all(not receipt.commit_ready for receipt in receipts)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            mq._message_queues.pop(user_id, None)
            mq._queue_locks.pop(user_id, None)

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


class TestShutdownDrain:
    """Issue #179: graceful shutdown must drain queued tasks before cancelling."""

    async def test_drain_delivers_pending_task_before_cancel(self) -> None:
        from ccgram.handlers.messaging_pipeline.message_queue import (
            get_or_create_queue,
            shutdown_workers,
        )
        from ccgram.handlers.messaging_pipeline.message_task import ContentTask

        await shutdown_workers()  # clean state
        delivered: list[str] = []

        async def fake_handle(*args, **kwargs):
            delivered.append("delivered")

        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue._handle_content_task",
            new=AsyncMock(side_effect=fake_handle),
        ):
            bot = AsyncMock()
            queue = get_or_create_queue(bot, 99)
            queue.put_nowait(
                ContentTask(window_id="@0", parts=("last words",), content_type="text")
            )
            await shutdown_workers(drain_timeout=2.0)

        assert delivered == ["delivered"]

    async def test_drain_timeout_does_not_hang(self) -> None:
        from ccgram.handlers.messaging_pipeline.message_queue import (
            get_or_create_queue,
            shutdown_workers,
        )
        from ccgram.handlers.messaging_pipeline.message_task import ContentTask

        async def block_dispatch(*_args, **_kwargs) -> int:
            await asyncio.sleep(10)
            return 0

        await shutdown_workers()  # clean state
        bot = AsyncMock()
        queue = get_or_create_queue(bot, 98)
        queue.put_nowait(ContentTask(window_id="@0", parts=("x",), content_type="text"))
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue._handle_content_task",
            new=AsyncMock(side_effect=block_dispatch),
        ):
            await asyncio.wait_for(shutdown_workers(drain_timeout=0.3), timeout=5.0)
