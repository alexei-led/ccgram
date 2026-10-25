import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import RetryAfter, TelegramError

from ccgram.delivery_contract import (
    DeliveryReceipt,
    activate_delivery_receipt,
    deactivate_delivery_receipt,
)
from ccgram.handlers.messaging_pipeline import message_queue as mq
from ccgram.handlers.messaging_pipeline import message_sender
from ccgram.handlers.messaging_pipeline.message_task import ContentTask
from ccgram.telegram_client import FakeTelegramClient


@pytest.fixture
def bot() -> FakeTelegramClient:
    return FakeTelegramClient()


def _receipt() -> DeliveryReceipt:
    receipt = DeliveryReceipt()
    receipt.track()
    receipt.close()
    return receipt


async def test_revoked_active_receipt_is_rejected_before_enqueue(bot):
    receipt = DeliveryReceipt()
    receipt.revoke()
    queue: asyncio.Queue = asyncio.Queue()
    token = activate_delivery_receipt(receipt)
    try:
        with (
            patch.object(mq, "_is_ghost_window_task_at_enqueue", return_value=False),
            patch.object(mq, "is_window_live", return_value=True),
            patch.object(mq, "get_or_create_queue", return_value=queue) as get_queue,
        ):
            enqueued = await mq.enqueue_content_message(
                bot, 1, "@0", ["stale"], thread_id=42, chat_id=-1001
            )
    finally:
        deactivate_delivery_receipt(token)

    assert enqueued is False
    assert receipt._pending == 0
    assert queue.empty()
    get_queue.assert_not_called()


async def test_mixed_revoked_and_current_tasks_keep_only_current_payload(bot):
    user_id = 82450
    await mq.shutdown_workers()
    queue: asyncio.Queue = asyncio.Queue()
    lock = asyncio.Lock()
    mq._message_queues[user_id] = queue
    mq._queue_locks[user_id] = lock
    revoked, current = _receipt(), _receipt()
    revoked.revoke()
    queue.put_nowait(
        ContentTask(
            window_id="@0",
            parts=("stale",),
            thread_id=42,
            chat_id=-1001,
            delivery_receipts=(revoked,),
        )
    )
    queue.put_nowait(
        ContentTask(
            window_id="@0",
            parts=("current",),
            thread_id=42,
            chat_id=-1001,
            delivery_receipts=(current,),
        )
    )
    sent: list[str] = []

    async def record_send(_client, _chat_id, text, **_kwargs):
        sent.append(text)
        return SimpleNamespace(message_id=len(sent))

    worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
    try:
        with (
            patch.object(mq, "is_window_live", return_value=True),
            patch.object(mq, "rate_limit_send_message", side_effect=record_send),
        ):
            await asyncio.wait_for(queue.join(), timeout=1)
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        mq._message_queues.pop(user_id, None)
        mq._queue_locks.pop(user_id, None)

    assert sent == ["current"]
    assert current.commit_ready
    assert revoked._pending == 0
    assert not revoked.failed
    assert not revoked.commit_ready


async def test_revocation_after_first_part_stops_parts_and_tts(bot, monkeypatch):
    receipt = _receipt()
    task = ContentTask(
        window_id="@0",
        parts=("first", "second"),
        thread_id=42,
        chat_id=-1001,
        delivery_receipts=(receipt,),
    )
    sent: list[str] = []

    async def send_part(_client, _chat_id, text, **_kwargs):
        sent.append(text)
        receipt.revoke()
        return SimpleNamespace(message_id=len(sent))

    monkeypatch.setattr(mq.config, "tts_provider", "edge")
    with (
        patch.object(mq, "rate_limit_send_message", side_effect=send_part),
        patch.object(mq, "prepare_tts_text", return_value="spoken text"),
        patch.object(mq, "get_synthesizer") as get_synthesizer,
    ):
        outcome = await mq._process_content_task(bot, 1, task)

    assert outcome is mq.DeliveryOutcome.INTENTIONALLY_DROPPED
    assert sent == ["first"]
    get_synthesizer.assert_not_called()


async def test_revocation_during_tts_synthesis_prevents_voice_rpc(bot, monkeypatch):
    receipt = _receipt()
    task = ContentTask(
        window_id="@0",
        parts=("text",),
        thread_id=42,
        chat_id=-1001,
        delivery_receipts=(receipt,),
    )
    synthesis_started = asyncio.Event()
    finish_synthesis = asyncio.Event()

    async def synthesize(_text):
        synthesis_started.set()
        await finish_synthesis.wait()
        return SimpleNamespace(data=b"audio", filename="reply.ogg")

    synthesizer = SimpleNamespace(synthesize=AsyncMock(side_effect=synthesize))
    monkeypatch.setattr(mq.config, "tts_provider", "edge")
    with (
        patch.object(
            mq,
            "rate_limit_send_message",
            new=AsyncMock(return_value=SimpleNamespace(message_id=1)),
        ),
        patch.object(mq, "prepare_tts_text", return_value="spoken text"),
        patch.object(mq, "get_synthesizer", return_value=synthesizer),
    ):
        processing = asyncio.create_task(mq._process_content_task(bot, 1, task))
        await asyncio.wait_for(synthesis_started.wait(), timeout=1)
        receipt.revoke()
        finish_synthesis.set()
        outcome = await asyncio.wait_for(processing, timeout=1)

    assert outcome is mq.DeliveryOutcome.INTENTIONALLY_DROPPED
    synthesizer.synthesize.assert_awaited_once_with("spoken text")
    assert bot.call_count("send_voice") == 0


async def test_retry_after_revocation_during_backoff_does_not_retry(bot):
    user_id = 82451
    receipt = _receipt()
    task = ContentTask(window_id="@0", parts=("flooded",), delivery_receipts=(receipt,))
    state = mq.RetryDispatchState(task)
    queue: asyncio.Queue = asyncio.Queue()
    dispatch = AsyncMock(
        side_effect=[
            RetryAfter(1),
            mq.DispatchResult(0, mq.DeliveryOutcome.DELIVERED),
        ]
    )

    async def revoke_during_backoff(_delay):
        receipt.revoke()

    with (
        patch.object(mq, "_is_stale_task", return_value=False),
        patch.object(mq, "_dispatch", dispatch),
        patch.object(mq.asyncio, "sleep", side_effect=revoke_during_backoff),
        patch.object(mq.random, "uniform", return_value=0),
    ):
        outcome = await mq._dispatch_with_retry(
            bot, user_id, state, queue, asyncio.Lock()
        )

    assert outcome is mq.DeliveryOutcome.INTENTIONALLY_DROPPED
    dispatch.assert_awaited_once()


async def test_rate_limited_send_rechecks_admission_before_first_rpc(bot):
    receipt = DeliveryReceipt()

    async def revoke_while_waiting(_chat_id):
        receipt.revoke()

    with (
        patch.object(
            message_sender, "rate_limit_send", side_effect=revoke_while_waiting
        ),
    ):
        sent = await message_sender.rate_limit_send_message(
            bot, 123, "stale", admit=lambda: not receipt.revoked
        )

    assert sent is None
    assert bot.call_count("send_message") == 0


async def test_revocation_after_entity_error_suppresses_plain_fallback(bot):
    receipt = DeliveryReceipt()

    def revoke_then_fail(**_kwargs):
        receipt.revoke()
        raise TelegramError("entity send failed")

    bot.returns["send_message"] = revoke_then_fail
    with patch.object(message_sender, "rate_limit_send", new_callable=AsyncMock):
        sent = await message_sender.rate_limit_send_message(
            bot, 123, "hello", admit=lambda: not receipt.revoked
        )

    assert sent is None
    assert bot.call_count("send_message") == 1


async def test_revocation_after_entity_edit_failure_suppresses_plain_fallback(bot):
    receipt = DeliveryReceipt()

    def revoke_then_fail(**_kwargs):
        receipt.revoke()
        raise TelegramError("entity edit failed")

    bot.returns["edit_message_text"] = revoke_then_fail
    result = await message_sender.edit_with_fallback(
        bot, 123, 7, "hello", admit=lambda: not receipt.revoked
    )

    assert result is False
    assert bot.call_count("edit_message_text") == 1


async def test_safe_send_rechecks_admission_before_entity_fallback(bot):
    receipt = DeliveryReceipt()

    def revoke_then_fail(**_kwargs):
        receipt.revoke()
        raise TelegramError("first attempt failed")

    bot.returns["send_message"] = revoke_then_fail
    result = await message_sender.safe_send(
        bot, 123, "hello", admit=lambda: not receipt.revoked
    )
    assert result is None
    assert bot.call_count("send_message") == 1


async def test_failed_but_current_receipt_keeps_normal_send_behavior(bot):
    receipt = _receipt()
    receipt.fail()
    task = ContentTask(
        window_id="@0",
        parts=("still current",),
        thread_id=42,
        chat_id=-1001,
        delivery_receipts=(receipt,),
    )
    with patch.object(
        mq,
        "rate_limit_send_message",
        new=AsyncMock(return_value=SimpleNamespace(message_id=1)),
    ) as send:
        outcome = await mq._process_content_task(bot, 1, task)
    assert outcome is mq.DeliveryOutcome.DELIVERED
    send.assert_awaited_once()
    assert receipt.failed
    assert not receipt.revoked


async def test_tool_batch_rechecks_cached_ownership_after_rate_limit(bot, monkeypatch):
    from ccgram.handlers.messaging_pipeline import tool_batch

    old = DeliveryReceipt()
    current = DeliveryReceipt()
    batch = tool_batch.ToolBatch(window_id="@0", thread_id=42)
    batch.entries = [
        tool_batch.ToolBatchEntry("old", "stale tool", delivery_receipts=(old,)),
        tool_batch.ToolBatchEntry("new", "current tool", delivery_receipts=(current,)),
    ]

    async def revoke_old(_chat_id):
        old.revoke()

    monkeypatch.setattr(tool_batch, "is_ephemeral_tools", lambda _window: True)
    monkeypatch.setattr(tool_batch, "_rate_limit_chat", revoke_old)
    with patch(
        "ccgram.handlers.status.status_bubble.clear_status_message",
        new_callable=AsyncMock,
    ):
        delivered = await tool_batch._send_or_edit_batch(
            bot, 1, batch, -1001, 42, 42, admit=lambda: not current.revoked
        )
    assert delivered is False
    assert current.revoked is False
    assert bot.call_count("send_message") == 0


@pytest.mark.parametrize("retry", ["none", "once", "exhausted"])
async def test_merged_tasks_settle_each_shared_receipt_track(bot, retry):
    user_id = 82453
    receipt = DeliveryReceipt(checkpoint=90)
    receipt.track()
    receipt.track()
    receipt.close()
    queue: asyncio.Queue = asyncio.Queue()
    mq._message_queues[user_id] = queue
    mq._queue_locks[user_id] = asyncio.Lock()
    for text in ("first", "second"):
        queue.put_nowait(
            ContentTask(
                window_id="@0",
                parts=(text,),
                thread_id=42,
                chat_id=-1001,
                source_session_id="s1",
                delivery_receipts=(receipt,),
            )
        )
    outcomes = (
        [RetryAfter(0), SimpleNamespace(message_id=1)]
        if retry != "none"
        else [SimpleNamespace(message_id=1)]
    )
    worker = asyncio.create_task(mq._message_queue_worker(bot, user_id))
    try:
        with (
            patch.object(mq, "is_window_live", return_value=True),
            patch.object(mq, "rate_limit_send_formatted_message", side_effect=outcomes),
            patch.object(mq.asyncio, "sleep", new_callable=AsyncMock),
            patch.object(mq.random, "uniform", return_value=0),
            patch.object(
                mq, "_QUEUE_RETRY_BUDGET_SECONDS", 0 if retry == "exhausted" else 300
            ),
        ):
            await asyncio.wait_for(queue.join(), timeout=2)
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        mq._message_queues.pop(user_id, None)
        mq._queue_locks.pop(user_id, None)
    assert receipt._pending == 0
    assert receipt.commit_ready is (retry != "exhausted")
    assert receipt.failed is (retry == "exhausted")
