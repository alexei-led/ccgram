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
from ccgram.monitor_state import BacklogSkipIntent, TrackedSession
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


async def test_purge_refuses_rebound_topic_before_draining() -> None:
    user_id = 702
    queue: asyncio.Queue = asyncio.Queue()
    mq._message_queues[user_id] = queue
    mq._queue_locks[user_id] = asyncio.Lock()
    queue.put_nowait(
        ContentTask(
            window_id="@0",
            parts=("old",),
            thread_id=4,
            chat_id=99,
            source_session_id="s1",
            source_checkpoint=50,
        )
    )
    try:
        with patch.object(
            mq.thread_router, "resolve_window_for_thread", return_value="@1"
        ):
            assert await mq.purge_source_tasks(user_id, "@0", 4, "s1", 50, 99) is None
        assert queue.qsize() == 1
    finally:
        mq._message_queues.clear()
        mq._queue_locks.clear()


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


async def test_failed_notice_retries_in_process_before_watermark_advances(
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

    notice_attempts: list[str] = []

    async def notice(intent) -> None:
        notice_attempts.append(intent.session_id)
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
    assert intent is not None
    assert intent.snapshot_offset == 50
    assert intent.range_start == 10
    assert intent.skipped_count == 3
    assert purged == [50]
    assert notice_attempts == ["s1"]
    monitor.commit_delivered_watermarks()
    assert monitor.state.get_session("s1").last_byte_offset == 10  # type: ignore[union-attr]

    first_notice = monitor._skip_notice_receipts["s1"]
    first_notice.settle(DeliveryOutcome.FAILED)
    monitor.commit_delivered_watermarks()
    assert monitor.state.get_session("s1").last_byte_offset == 10  # type: ignore[union-attr]
    assert "s1" in monitor.state.pending_skips
    assert "s1" not in monitor._skip_notice_receipts

    # Controlled backoff suppresses an immediate hot-loop retry.
    await monitor._resume_pending_skip_notices()
    assert notice_attempts == ["s1"]

    monitor._skip_retry_at["s1"] = 0.0
    await monitor._resume_pending_skip_notices()
    assert notice_attempts == ["s1", "s1"]
    monitor._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
    monitor.commit_delivered_watermarks()
    assert monitor.state.get_session("s1").last_byte_offset == 50  # type: ignore[union-attr]
    assert "s1" not in monitor.state.pending_skips
    assert transcript.read_bytes() == b"x" * 50


async def test_notice_enqueue_failure_retries_in_process(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b"x" * 50)
    monitor = SessionMonitor(
        projects_path=tmp_path, state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=50)
    )
    monitor._last_session_map = {"@0": {"session_id": "s1"}}

    async def purge(_intent) -> int:
        return 3

    attempts = 0

    async def notice(_intent) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("queue unavailable")
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
    assert intent is None
    assert attempts == 1
    assert "s1" not in monitor._skip_notice_receipts
    assert "s1" in monitor.state.pending_skips

    monitor._skip_retry_at["s1"] = 0.0
    await monitor._resume_pending_skip_notices()
    assert attempts == 2
    monitor._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
    monitor.commit_delivered_watermarks()
    assert monitor.state.get_session("s1").last_byte_offset == 50  # type: ignore[union-attr]
    assert "s1" not in monitor.state.pending_skips


async def test_failed_purge_resumes_after_restart_before_notice(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b"x" * 50)
    state_file = tmp_path / "monitor.json"
    monitor = SessionMonitor(projects_path=tmp_path, state_file=state_file)
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=50)
    )
    monitor._last_session_map = {"@0": {"session_id": "s1"}}

    async def failed_purge(_intent) -> int:
        raise RuntimeError("queue unavailable")

    notice = AsyncMock()
    monitor.set_skip_callbacks(
        purge=failed_purge, notice=notice, validate=lambda _intent: True
    )
    intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
    assert intent is None
    assert monitor.state.pending_skips["s1"].purge_complete is False
    notice.assert_not_awaited()
    assert "s1" in monitor.state.pending_skips

    restarted = SessionMonitor(projects_path=tmp_path, state_file=state_file)

    async def purge(_intent) -> int:
        return 3

    async def resumed_notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    restarted.set_skip_callbacks(
        purge=purge, notice=resumed_notice, validate=lambda _intent: True
    )
    await restarted._resume_pending_skip_notices()
    assert restarted.state.pending_skips["s1"].purge_complete is True
    assert restarted.state.pending_skips["s1"].skipped_count == 3
    restarted._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
    restarted.commit_delivered_watermarks()
    assert restarted.state.get_session("s1").last_byte_offset == 50  # type: ignore[union-attr]
    assert "s1" not in restarted.state.pending_skips


async def test_rebound_after_notice_delivery_does_not_commit_skip(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b"x" * 50)
    monitor = SessionMonitor(
        projects_path=tmp_path, state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=50)
    )
    monitor._last_session_map = {"@0": {"session_id": "s1"}}
    binding_current = True

    async def purge(_intent) -> int:
        return 3

    def validate(_intent) -> bool:
        return binding_current

    async def notice(_intent) -> None:
        nonlocal binding_current
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()
        binding_current = False

    monitor.set_skip_callbacks(purge=purge, notice=notice, validate=validate)
    intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
    assert intent is not None
    monitor._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
    monitor.commit_delivered_watermarks()

    assert monitor.state.get_session("s1").last_byte_offset == 10  # type: ignore[union-attr]
    assert "s1" not in monitor.state.pending_skips


async def test_purge_retry_preserves_retired_count_after_state_save_failure(
    tmp_path: Path, monkeypatch
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b"x" * 50)
    monitor = SessionMonitor(
        projects_path=tmp_path, state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=50)
    )
    monitor._last_session_map = {"@0": {"session_id": "s1"}}
    purge_calls = 0

    async def purge(_intent) -> int:
        nonlocal purge_calls
        purge_calls += 1
        return 3 if purge_calls == 1 else 0

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    real_save = monitor.state.save_if_dirty
    save_calls = 0

    def save_once_as_failed() -> bool:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            return False
        return real_save()

    monkeypatch.setattr(monitor.state, "save_if_dirty", save_once_as_failed)
    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    assert await monitor.request_backlog_skip(1, "@0", 4, 99) is None
    assert monitor.state.pending_skips["s1"].skipped_count == 3

    monitor._skip_retry_at["s1"] = 0.0
    await monitor._resume_pending_skip_notices()

    assert monitor.state.pending_skips["s1"].skipped_count == 3
    assert monitor.state.pending_skips["s1"].purge_complete is True


async def test_purge_complete_skip_is_cancelled_after_topic_rebound(
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
    intent = BacklogSkipIntent("s1", "@0", 1, 4, 99, 50, 10)
    monitor.state.begin_skip(intent)
    monitor.state.update_skip_count("s1", 3)
    assert monitor.state.save_if_dirty()

    restarted = SessionMonitor(projects_path=tmp_path, state_file=state_file)
    notice = AsyncMock()
    restarted.set_skip_callbacks(
        purge=AsyncMock(return_value=0),
        notice=notice,
        validate=lambda _intent: False,
    )
    await restarted._resume_pending_skip_notices()

    notice.assert_not_awaited()
    assert "s1" not in restarted.state.pending_skips


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
        patch(
            "ccgram.handlers.messaging_pipeline.backlog.get_backlog_snapshot",
            return_value=BacklogSnapshot(100, 0.0, None),
        ),
        patch("ccgram.handlers.status.status_bar_actions.thread_router") as mock_router,
    ):
        mock_router.resolve_window_for_thread.return_value = "@0"
        await _handle_status_bar_action(
            query, 1, f"{CB_STATUS_BACKLOG_CANCEL}@0", update, MagicMock()
        )
        monitor.request_backlog_skip.assert_not_awaited()
        await _handle_status_bar_action(
            query, 1, f"{CB_STATUS_BACKLOG_CONFIRM}@0", update, MagicMock()
        )
    monitor.request_backlog_skip.assert_awaited_once_with(1, "@0", 4, 99)


async def test_stale_topic_confirmation_cannot_skip_source() -> None:
    query = AsyncMock()
    query.message.chat.id = 99
    update = MagicMock()
    monitor = MagicMock()
    monitor.request_backlog_skip = AsyncMock()
    with (
        patch(
            "ccgram.handlers.status.status_bar_actions.user_owns_window",
            return_value=True,
        ),
        patch(
            "ccgram.handlers.status.status_bar_actions.get_thread_id", return_value=4
        ),
        patch("ccgram.handlers.status.status_bar_actions.thread_router") as mock_router,
        patch("ccgram.session_monitor.get_active_monitor", return_value=monitor),
    ):
        mock_router.resolve_window_for_thread.return_value = "@other"
        await _handle_status_bar_action(
            query, 1, f"{CB_STATUS_BACKLOG_CONFIRM}@0", update, MagicMock()
        )

    monitor.request_backlog_skip.assert_not_awaited()
    query.answer.assert_awaited_once_with("Stale status button", show_alert=True)


async def test_confirmation_rechecks_severe_threshold() -> None:
    query = AsyncMock()
    query.message.chat.id = 99
    update = MagicMock()
    monitor = MagicMock()
    monitor.request_backlog_skip = AsyncMock()
    with (
        patch(
            "ccgram.handlers.status.status_bar_actions.user_owns_window",
            return_value=True,
        ),
        patch(
            "ccgram.handlers.status.status_bar_actions.get_thread_id", return_value=4
        ),
        patch("ccgram.handlers.status.status_bar_actions.thread_router") as mock_router,
        patch(
            "ccgram.handlers.messaging_pipeline.backlog.get_backlog_snapshot",
            return_value=BacklogSnapshot(0, 0.0, None),
        ),
        patch("ccgram.session_monitor.get_active_monitor", return_value=monitor),
    ):
        mock_router.resolve_window_for_thread.return_value = "@0"
        await _handle_status_bar_action(
            query, 1, f"{CB_STATUS_BACKLOG_CONFIRM}@0", update, MagicMock()
        )

    monitor.request_backlog_skip.assert_not_awaited()
    query.answer.assert_awaited_once_with(
        "Backlog is no longer severe", show_alert=True
    )
