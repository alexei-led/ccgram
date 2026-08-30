"""Safety coverage for source-scoped backlog progress and live jumps."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.delivery_contract import DeliveryOutcome, get_active_delivery_receipt
from ccgram.handlers.callback_data import (
    CB_STATUS_BACKLOG_CANCEL,
    CB_STATUS_BACKLOG_CONFIRM,
    CB_STATUS_BACKLOG_JUMP,
)
from ccgram.handlers.messaging_pipeline import message_queue as mq
from ccgram.handlers.messaging_pipeline.backlog import BacklogSnapshot
from ccgram.handlers.messaging_pipeline.message_task import (
    ContentTask,
    StatusUpdateTask,
)
from ccgram.handlers.status.status_bar_actions import _handle_status_bar_action
from ccgram.handlers.status.status_bubble import (
    build_status_keyboard,
    format_backlog_status,
)
from ccgram.monitor_state import TrackedSession
from ccgram.session_monitor import SessionMonitor


def _callbacks(kb) -> list[str]:
    return [
        button.callback_data
        for row in kb.inline_keyboard
        for button in row
        if isinstance(button.callback_data, str)
    ]


def test_jump_button_is_only_rendered_for_severe_backlog() -> None:
    assert not any(
        data.startswith(CB_STATUS_BACKLOG_JUMP)
        for data in _callbacks(build_status_keyboard("@0", backlog_severe=False))
    )
    assert any(
        data.startswith(CB_STATUS_BACKLOG_JUMP)
        for data in _callbacks(build_status_keyboard("@0", backlog_severe=True))
    )


async def test_status_update_includes_backlog_progress() -> None:
    with (
        patch(
            "ccgram.handlers.status.status_bubble.format_claude_task_status",
            return_value="Working",
        ),
        patch(
            "ccgram.handlers.status.status_bubble.format_backlog_status",
            return_value=("Queue: 2 pending · age 4s · delivery lag 1.0s", False),
        ),
        patch(
            "ccgram.handlers.status.status_bubble.send_status_text",
            new_callable=AsyncMock,
        ) as send,
    ):
        from ccgram.handlers.status.status_bubble import process_status_update

        await process_status_update(
            MagicMock(),
            1,
            StatusUpdateTask(window_id="@0", text="Working", thread_id=4),
        )
    call = send.await_args
    assert call is not None
    assert call.args[4] == "Working\nQueue: 2 pending · age 4s · delivery lag 1.0s"
    assert call.kwargs["backlog_severe"] is False


def test_backlog_status_reports_count_age_lag_and_throttles() -> None:
    from ccgram.handlers.status import status_bubble

    status_bubble._backlog_status_cache.clear()
    snapshot = BacklogSnapshot(100, 301.0, 2.5)
    with patch(
        "ccgram.handlers.messaging_pipeline.backlog.get_backlog_snapshot",
        return_value=snapshot,
    ) as get_snapshot:
        line, severe = format_backlog_status(1, 2, "@0")
        second, second_severe = format_backlog_status(1, 2, "@0")
    assert "100 pending" in line
    assert "age 301s" in line
    assert "delivery lag 2.5s" in line
    assert severe is True and second_severe is True
    assert second == line
    assert get_snapshot.call_count == 2


async def test_purge_is_source_scoped_and_settles_only_retired_receipts() -> None:
    user_id = 701
    queue: asyncio.Queue = asyncio.Queue()
    mq._message_queues[user_id] = queue
    mq._queue_locks[user_id] = asyncio.Lock()
    retired = mq.DeliveryReceipt(checkpoint=50)
    retained = mq.DeliveryReceipt(checkpoint=60)
    for receipt in (retired, retained):
        receipt.track()
        receipt.close()
    queue.put_nowait(
        ContentTask(
            window_id="@0",
            parts=("old",),
            thread_id=4,
            source_session_id="s1",
            source_checkpoint=50,
            delivery_receipts=(retired,),
        )
    )
    queue.put_nowait(
        ContentTask(
            window_id="@0",
            parts=("new",),
            thread_id=4,
            source_session_id="s1",
            source_checkpoint=60,
            delivery_receipts=(retained,),
        )
    )
    queue.put_nowait(
        ContentTask(
            window_id="@1",
            parts=("other",),
            thread_id=4,
            source_session_id="s1",
            source_checkpoint=50,
        )
    )
    try:
        assert await mq.purge_source_tasks(user_id, "@0", 4, "s1", 50) == 1
        assert retired.commit_ready is True
        assert retained.commit_ready is False
        assert queue.qsize() == 2
    finally:
        mq._message_queues.clear()
        mq._queue_locks.clear()


async def test_notice_receipt_precedes_watermark_and_resumes_after_restart(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b"x" * 50)
    state_file = tmp_path / "monitor.json"
    monitor = SessionMonitor(projects_path=tmp_path, state_file=state_file)
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=50)
    )
    monitor._last_session_map = {"@0": {"session_id": "s1"}}
    purged: list[int] = []

    async def purge(intent) -> int:
        purged.append(intent.snapshot_offset)
        return 3

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(purge=purge, notice=notice)
    intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
    assert intent is not None
    assert intent.snapshot_offset == 50
    assert intent.range_start == 10
    assert intent.skipped_count == 3
    assert purged == [50]
    monitor.commit_delivered_watermarks()
    assert monitor.state.get_session("s1").last_byte_offset == 10  # type: ignore[union-attr]

    first_notice = monitor._skip_notice_receipts["s1"]
    first_notice.settle(DeliveryOutcome.FAILED)
    monitor.commit_delivered_watermarks()
    assert monitor.state.get_session("s1").last_byte_offset == 10  # type: ignore[union-attr]
    assert "s1" in monitor.state.pending_skips

    restarted = SessionMonitor(projects_path=tmp_path, state_file=state_file)
    resumed: list[str] = []

    async def resumed_notice(intent) -> None:
        resumed.append(intent.session_id)
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    restarted.set_skip_callbacks(purge=purge, notice=resumed_notice)
    await restarted._resume_pending_skip_notices()
    assert resumed == ["s1"]
    restarted._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
    restarted.commit_delivered_watermarks()
    assert restarted.state.get_session("s1").last_byte_offset == 50  # type: ignore[union-attr]
    assert "s1" not in restarted.state.pending_skips
    assert transcript.read_bytes() == b"x" * 50


async def test_confirmation_and_cancellation_do_not_skip_without_confirmation() -> None:
    query = AsyncMock()
    query.message.chat.id = 99
    query.get_bot.return_value = AsyncMock()
    update = MagicMock()
    monitor = MagicMock()
    monitor.request_backlog_skip = AsyncMock(return_value=MagicMock(skipped_count=2))
    with (
        patch(
            "ccgram.handlers.status.status_bar_actions.user_owns_window",
            return_value=True,
        ),
        patch(
            "ccgram.handlers.status.status_bar_actions.get_thread_id", return_value=4
        ),
        patch("ccgram.session_monitor.get_active_monitor", return_value=monitor),
    ):
        await _handle_status_bar_action(
            query, 1, f"{CB_STATUS_BACKLOG_CANCEL}@0", update, MagicMock()
        )
        monitor.request_backlog_skip.assert_not_awaited()
        await _handle_status_bar_action(
            query, 1, f"{CB_STATUS_BACKLOG_CONFIRM}@0", update, MagicMock()
        )
    monitor.request_backlog_skip.assert_awaited_once_with(1, "@0", 4, 99)
