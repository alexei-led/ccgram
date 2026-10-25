"""Tests for SessionMonitor."""

import asyncio
import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import structlog

from ccgram.delivery_contract import (
    DeliveryOutcome,
    get_active_delivery_receipt,
    new_delivery_receipt,
)
from ccgram.monitor_state import BacklogSkipIntent, TrackedSession
from ccgram.multiplexer.base import MultiplexerCapabilities, WindowRef
from ccgram.providers.claude import ClaudeProvider
from ccgram.providers.codex import CodexProvider
from ccgram.session import SessionManager
from ccgram.session_monitor import (
    NewMessage,
    NewWindowEvent,
    SessionMonitor,
    _source_range_sha256,
)
from ccgram.transcript_reader import _ToolDeliveryFence
from ccgram.session_protocol.messages import (
    SourceEntry,
    TranscriptBatch,
    TranscriptRecord,
    TranscriptParserState,
)
from ccgram.thread_router import thread_router
from ccgram.telegram_client import FakeTelegramClient
from ccgram.window_state_store import window_store


HERDR_TARGETS = {
    name: "herdr-session-v1-" + digest * 64
    for name, digest in {
        "a": "a",
        "b": "b",
        "shell": "c",
        "bound": "d",
        "known": "e",
        "new": "f",
    }.items()
}


def _transcript_batch(
    session_id: str, checkpoint: int, record_count: int = 1
) -> TranscriptBatch:
    records = tuple(
        (
            SourceEntry(index, {}),
            TranscriptRecord(
                f"message-{index}",
                "agent",
                None,
                ({"type": "text", "text": f"delivered {index}"},),
            ),
        )
        for index in range(record_count)
    )
    return TranscriptBatch(session_id, "claude", records, checkpoint)


def _complete_jsonl_record(size: int) -> bytes:
    prefix = b'{"record":"'
    suffix = b'"}\n'
    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


@pytest.fixture
def monitor(tmp_path) -> SessionMonitor:
    return SessionMonitor(
        projects_path=tmp_path / "projects",
        poll_interval=0.1,
        state_file=tmp_path / "monitor_state.json",
    )


class TestMonitorLoop:
    async def test_stop_and_wait_awaits_producer_before_return(
        self, monitor: SessionMonitor
    ) -> None:
        started = asyncio.Event()

        async def producer() -> None:
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise

        monitor._running = True
        monitor._task = asyncio.create_task(producer())
        await started.wait()
        await monitor.stop_and_wait()

        assert monitor._task is None

    async def test_unavailable_listing_skips_pruning(
        self, monitor: SessionMonitor
    ) -> None:
        current_map = {
            HERDR_TARGETS["a"]: {"session_id": "live"},
            HERDR_TARGETS["b"]: {"session_id": "possibly-live"},
        }
        check_for_updates = AsyncMock(return_value=[])

        async def _stop_after_cycle(_delay: float) -> None:
            monitor._running = False

        with (
            patch.object(monitor, "_cleanup_all_stale_sessions", AsyncMock()),
            patch.object(
                monitor, "_load_current_session_map", AsyncMock(return_value={})
            ),
            patch.object(
                monitor,
                "_detect_and_cleanup_changes",
                AsyncMock(return_value=current_map),
            ),
            patch.object(monitor, "check_for_updates", check_for_updates),
            patch(
                "ccgram.session_monitor.read_session_map_raw",
                AsyncMock(return_value={}),
            ),
            patch("ccgram.session_map.session_map_sync") as mock_sync,
            patch(
                "ccgram.session_monitor.list_windows_for_reconciliation",
                AsyncMock(return_value=None),
            ),
            patch("ccgram.session_monitor.asyncio.sleep", _stop_after_cycle),
        ):
            mock_sync.load_session_map = AsyncMock()
            monitor._running = True
            await monitor._monitor_loop()

        mock_sync.prune_session_map.assert_not_called()
        check_for_updates.assert_awaited_once_with(current_map)

    async def test_each_cycle_clears_prior_message_log_context(
        self, monitor: SessionMonitor
    ) -> None:
        current_map = {HERDR_TARGETS["a"]: {"session_id": "session-a"}}
        contexts: list[dict[str, object]] = []
        cycles = 0
        monitor.set_message_callback(AsyncMock())

        async def inspect_hook_context() -> None:
            contexts.append(structlog.contextvars.get_contextvars())

        async def sleep_until_two_cycles(_delay: float) -> None:
            nonlocal cycles
            cycles += 1
            if cycles == 2:
                monitor._running = False

        with (
            patch.object(monitor, "_cleanup_all_stale_sessions", AsyncMock()),
            patch.object(
                monitor, "_load_current_session_map", AsyncMock(return_value={})
            ),
            patch.object(
                monitor,
                "_detect_and_cleanup_changes",
                AsyncMock(return_value=current_map),
            ),
            patch.object(
                monitor,
                "check_for_updates",
                AsyncMock(
                    side_effect=[
                        [NewMessage("session-a", "first", True)],
                        [],
                    ]
                ),
            ),
            patch.object(monitor, "_read_hook_events", inspect_hook_context),
            patch(
                "ccgram.session_monitor.read_session_map_raw",
                AsyncMock(return_value={}),
            ),
            patch("ccgram.session_map.session_map_sync") as mock_sync,
            patch(
                "ccgram.session_monitor.list_windows_for_reconciliation",
                AsyncMock(return_value=None),
            ),
            patch("ccgram.session_monitor.asyncio.sleep", sleep_until_two_cycles),
        ):
            mock_sync.load_session_map = AsyncMock()
            structlog.contextvars.clear_contextvars()
            monitor._running = True
            await monitor._monitor_loop()

        assert contexts == [{}, {}]

    async def test_map_snapshot_is_refreshed_after_awaited_hook_callback(
        self, monitor: SessionMonitor
    ) -> None:
        window_id = HERDR_TARGETS["a"]
        stale = {window_id: {"session_id": "old-agent"}}
        current = {window_id: {"session_id": "shell-session"}}
        contents = stale
        loaded: list[dict | None] = []

        async def read_map():
            return dict(contents)

        async def process_hooks():
            nonlocal contents
            contents = current

        async def load_map(raw=None):
            loaded.append(raw)

        async def stop_after_cycle(_delay: float) -> None:
            monitor._running = False

        with (
            patch.object(monitor, "_cleanup_all_stale_sessions", AsyncMock()),
            patch.object(
                monitor, "_load_current_session_map", AsyncMock(return_value={})
            ),
            patch.object(
                monitor, "_detect_and_cleanup_changes", AsyncMock(return_value=current)
            ),
            patch.object(monitor, "check_for_updates", AsyncMock(return_value=[])),
            patch("ccgram.session_monitor.read_session_map_raw", read_map),
            patch.object(monitor, "_read_hook_events", process_hooks),
            patch("ccgram.session_map.session_map_sync") as sync,
            patch(
                "ccgram.session_monitor.list_windows_for_reconciliation",
                AsyncMock(return_value=None),
            ),
            patch("ccgram.session_monitor.asyncio.sleep", stop_after_cycle),
        ):
            sync.load_session_map = load_map
            monitor._running = True
            await monitor._monitor_loop()

        assert loaded == [current]

    async def test_reliable_listing_monitors_only_live_windows(
        self, monitor: SessionMonitor
    ) -> None:
        live_id = HERDR_TARGETS["a"]
        current_map = {
            live_id: {"session_id": "live"},
            HERDR_TARGETS["b"]: {"session_id": "stale"},
        }
        live = WindowRef(window_id=live_id, window_name="live", cwd="/live")
        check_for_updates = AsyncMock(return_value=[])

        async def _stop_after_cycle(_delay: float) -> None:
            monitor._running = False

        with (
            patch.object(monitor, "_cleanup_all_stale_sessions", AsyncMock()),
            patch.object(
                monitor, "_load_current_session_map", AsyncMock(return_value={})
            ),
            patch.object(
                monitor,
                "_detect_and_cleanup_changes",
                AsyncMock(return_value=current_map),
            ),
            patch.object(monitor, "check_for_updates", check_for_updates),
            patch(
                "ccgram.session_monitor.read_session_map_raw",
                AsyncMock(return_value={}),
            ),
            patch("ccgram.session_map.session_map_sync") as mock_sync,
            patch(
                "ccgram.session_monitor.list_windows_for_reconciliation",
                AsyncMock(return_value=[live]),
            ),
            patch("ccgram.session_monitor.asyncio.sleep", _stop_after_cycle),
        ):
            mock_sync.load_session_map = AsyncMock()
            monitor._running = True
            await monitor._monitor_loop()

        check_for_updates.assert_awaited_once_with({live_id: current_map[live_id]})

    async def test_rekeyed_window_folds_before_its_map_delta_or_hook_events(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        """A re-keyed window converges before its map delta or hooks are read.

        On a backend whose window id derives from the agent session (herdr),
        ``/clear`` mints a brand-new id for a window that already has a topic.
        The live listing says the new id supersedes the bound one; the session
        map only says a key vanished and another appeared. Reading either the
        map delta or an exactly-routed hook event before folding the alias
        turns one agent into a second topic or drops its hook event.
        """
        thread_router.reset()
        window_store.window_states.clear()
        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
        SessionManager()
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_HERDR_CAPS),
        )

        old_id, new_id = HERDR_TARGETS["a"], HERDR_TARGETS["b"]
        thread_router.bind_thread(100, 42, old_id)

        details = {"session_id": "S-new", "cwd": "/proj", "window_name": ""}
        live = WindowRef(
            window_id=new_id,
            window_name="proj ▸ 1",
            cwd="/proj",
            pane_current_command="claude",
            alias_window_ids=(old_id,),
        )

        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        async def _stop_after_cycle(_delay: float) -> None:
            monitor._running = False

        async def _read_after_identity_convergence() -> None:
            # Hook dispatch uses exact topic bindings. The canonical target
            # must own the legacy topic before event reading advances its offset.
            assert thread_router.thread_bindings[100][42] == new_id

        read_hook_events = AsyncMock(side_effect=_read_after_identity_convergence)
        with (
            patch.object(monitor, "_cleanup_all_stale_sessions", AsyncMock()),
            patch.object(monitor, "_read_hook_events", read_hook_events),
            patch.object(monitor, "check_for_updates", AsyncMock(return_value=[])),
            patch.object(
                monitor,
                "_load_current_session_map",
                AsyncMock(
                    side_effect=[
                        {
                            old_id: {
                                "session_id": "S-old",
                                "cwd": "/proj",
                                "window_name": "",
                            }
                        },
                        {new_id: details},
                    ]
                ),
            ),
            patch(
                "ccgram.session_monitor.read_session_map_raw",
                AsyncMock(return_value={}),
            ),
            patch("ccgram.session_map.session_map_sync") as mock_sync,
            patch("ccgram.session.session_map_sync"),
            patch(
                "ccgram.session_monitor.list_windows_for_reconciliation",
                AsyncMock(return_value=[live]),
            ),
            patch("ccgram.session_monitor.asyncio.sleep", _stop_after_cycle),
        ):
            mock_sync.load_session_map = AsyncMock()
            monitor._running = True
            await monitor._monitor_loop()

        assert thread_router.thread_bindings[100][42] == new_id
        read_hook_events.assert_awaited_once()
        surfaced = [c.args[0].window_id for c in cb.call_args_list]
        assert surfaced == []

    async def test_rekey_precedes_hook_dispatch_and_delivered_watermark_commit(
        self, monitor: SessionMonitor, monkeypatch, tmp_path
    ) -> None:
        """A re-keyed hook routes to its migrated topic before receipt commit."""
        from ccgram.handlers.messaging_pipeline import message_queue as mq

        thread_router.reset()
        window_store.window_states.clear()
        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
        SessionManager()
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_HERDR_CAPS),
        )

        old_id, new_id = HERDR_TARGETS["a"], HERDR_TARGETS["b"]
        session_id = "S-new"
        thread_router.bind_thread(100, 42, old_id)
        live = WindowRef(
            window_id=new_id,
            window_name="proj ▸ 1",
            cwd="/proj",
            pane_current_command="claude",
            alias_window_ids=(old_id,),
        )
        current_map = {
            new_id: {"session_id": session_id, "cwd": "/proj", "window_name": ""}
        }
        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            json.dumps(
                {
                    "ts": 1.0,
                    "event": "Stop",
                    "window_key": f"herdr:{new_id}",
                    "session_id": session_id,
                    "data": {},
                }
            )
            + "\n"
        )
        monkeypatch.setattr("ccgram.session_monitor.config.events_file", events_file)

        dispatched = []

        async def hook_callback(event) -> None:
            assert thread_router.thread_bindings[100][42] == new_id
            dispatched.append(event)

        queued: asyncio.Queue = asyncio.Queue()
        delivered_tasks = []

        async def message_callback(msg: NewMessage) -> None:
            await mq.enqueue_content_message(
                FakeTelegramClient(), 100, new_id, [msg.text], thread_id=42
            )
            task = queued.get_nowait()
            delivered_tasks.append(task)
            for receipt in task.delivery_receipts:
                receipt.settle(mq.DeliveryOutcome.DELIVERED)
            queued.task_done()

        async def check_for_updates(_current_map: dict) -> list[NewMessage]:
            monitor.state.update_session(
                TrackedSession(
                    session_id=session_id,
                    file_path="/transcript.jsonl",
                    last_byte_offset=10,
                    parsed_offset=20,
                )
            )
            return [NewMessage(session_id, "delivered", True)]

        async def _stop_after_cycle(_delay: float) -> None:
            monitor._running = False

        monitor.set_hook_event_callback(hook_callback)
        monitor.set_message_callback(message_callback)
        monkeypatch.setattr(mq, "get_or_create_queue", lambda *_args: queued)
        with (
            patch.object(monitor, "_cleanup_all_stale_sessions", AsyncMock()),
            patch.object(
                monitor, "_load_current_session_map", AsyncMock(return_value={})
            ),
            patch.object(
                monitor,
                "_detect_and_cleanup_changes",
                AsyncMock(return_value=current_map),
            ),
            patch.object(monitor, "check_for_updates", side_effect=check_for_updates),
            patch.object(
                monitor._transcript_reader,
                "drain_batches",
                return_value=(_transcript_batch(session_id, 20),),
            ),
            patch(
                "ccgram.session_monitor.read_session_map_raw",
                AsyncMock(return_value={}),
            ),
            patch("ccgram.session_map.session_map_sync") as mock_sync,
            patch("ccgram.session.session_map_sync"),
            patch(
                "ccgram.session_monitor.list_windows_for_reconciliation",
                AsyncMock(return_value=[live]),
            ),
            patch("ccgram.session_monitor.asyncio.sleep", _stop_after_cycle),
        ):
            mock_sync.load_session_map = AsyncMock()
            monitor._running = True
            await monitor._monitor_loop()

        assert [event.window_key for event in dispatched] == [f"herdr:{new_id}"]
        assert len(delivered_tasks) == 1
        assert delivered_tasks[0].delivery_receipts[0].commit_ready is True
        assert monitor.state.tracked_sessions[session_id].last_byte_offset == 20


async def test_cancelled_dispatch_retains_failed_receipt(
    monitor: SessionMonitor,
) -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def callback(_msg: NewMessage) -> None:
        started.set()
        await blocked.wait()

    monitor.set_message_callback(callback)
    pending = monitor._register_delivery_receipts(
        (_transcript_batch("s1", 10), _transcript_batch("s1", 20))
    )
    first_batch, consumer_name, first_receipt = pending[0]
    task = asyncio.create_task(
        monitor._dispatch_message_with_receipt(
            first_batch, first_receipt, consumer_name
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    receipts = monitor._delivery_receipts["s1"]
    assert len(receipts) == 2
    assert receipts[0].failed is True
    assert receipts[0].commit_ready is False
    assert receipts[1].commit_ready is False


async def test_failed_consumer_fences_batch_checkpoint_for_replay(
    monitor: SessionMonitor,
) -> None:
    async def fail(batch: TranscriptBatch) -> None:
        assert len(batch.records) == 2
        raise RuntimeError("delivery failed")

    async def succeed(batch: TranscriptBatch) -> None:
        assert len(batch.records) == 2
        return None

    monitor.register_delivery_consumer("telegram", fail)
    monitor.register_delivery_consumer("archive", succeed)
    monitor.state.update_session(
        TrackedSession(
            session_id="s1",
            file_path="/transcript.jsonl",
            last_byte_offset=0,
            parsed_offset=20,
        )
    )
    batch = _transcript_batch("s1", 20, record_count=2)
    pending = monitor._register_delivery_receipts((batch,))

    assert [name for _, name, _ in pending] == ["telegram", "archive"]
    assert all(receipt.checkpoint == 20 for _, _, receipt in pending)
    for pending_batch, consumer_name, receipt in pending:
        await monitor._dispatch_message_with_receipt(
            pending_batch, receipt, consumer_name
        )
    await monitor.commit_delivered_watermarks()

    failed, delivered = [receipt for _, _, receipt in pending]
    assert failed.failed is True
    assert delivered.commit_ready is True
    assert monitor.state.tracked_sessions["s1"].last_byte_offset == 0
    assert monitor._delivery_receipts["s1"] == [failed, delivered]


class TestSettledPrefixWatermarkCommit:
    """Upstream #205: the watermark must advance over the longest settled
    receipt run, not wait for every receipt of the session to close; under
    sustained output the all-ready policy never commits and every restart
    replays the whole backlog ahead of live traffic."""

    def _ready(self, checkpoint: int):
        from ccgram.delivery_contract import DeliveryOutcome, new_delivery_receipt

        receipt = new_delivery_receipt(checkpoint=checkpoint)
        receipt.track()
        receipt.settle(DeliveryOutcome.DELIVERED)
        receipt.close()
        return receipt

    def _unsettled(self, checkpoint: int):
        from ccgram.delivery_contract import new_delivery_receipt

        return new_delivery_receipt(checkpoint=checkpoint)

    def _offset(self, monitor: SessionMonitor, session_id: str) -> int:
        session = monitor.state.get_session(session_id)
        assert session is not None
        return session.last_byte_offset

    def _track(self, monitor, session_id: str, receipts: list) -> None:
        monitor.state.update_session(
            TrackedSession(
                session_id=session_id,
                file_path="/transcript.jsonl",
                last_byte_offset=0,
            )
        )
        monitor._delivery_receipts[session_id] = receipts

    async def test_prefix_commit_advances_past_settled_run(
        self, monitor: SessionMonitor
    ) -> None:
        pending = self._unsettled(300)
        pending.track()
        self._track(monitor, "s1", [self._ready(100), self._ready(200), pending])

        await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 200
        # The unsettled tail survives as receipts: a restart replays from
        # the committed fence (200), not from zero.
        assert monitor._delivery_receipts["s1"] == [pending]

    async def test_failed_receipt_fences_prefix(self, monitor: SessionMonitor) -> None:
        from ccgram.delivery_contract import DeliveryOutcome

        failed = self._unsettled(200)
        failed.track()
        failed.settle(DeliveryOutcome.FAILED)
        failed.close()
        self._track(monitor, "s1", [self._ready(100), failed, self._ready(300)])

        await monitor.commit_delivered_watermarks()

        # Progress before the failure is durable; the failed receipt and
        # everything after it replay (at-least-once preserved).
        assert self._offset(monitor, "s1") == 100
        assert len(monitor._delivery_receipts["s1"]) == 2

    async def test_all_ready_commits_full_run_and_clears(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [self._ready(100), self._ready(200)])

        await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 200
        assert "s1" not in monitor._delivery_receipts

    async def test_none_checkpoint_fences_prefix(self, monitor: SessionMonitor) -> None:
        unplaced = self._ready(0)
        unplaced.checkpoint = None
        self._track(monitor, "s1", [self._ready(100), unplaced, self._ready(300)])

        await monitor.commit_delivered_watermarks()

        # A receipt without a checkpoint cannot be ordered: it blocks the
        # commit conservatively (its bytes may tie the boundary), though
        # the settled run before it is consumed; replay re-delivers it.
        assert self._offset(monitor, "s1") == 0
        remaining = monitor._delivery_receipts["s1"]
        assert len(remaining) == 2
        assert remaining[0] is unplaced

    async def test_pending_tools_skip_session(self, monitor: SessionMonitor) -> None:
        self._track(monitor, "s1", [self._ready(100)])
        monitor._transcript_reader._pending_tools["s1"] = {}

        await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 0
        assert len(monitor._delivery_receipts["s1"]) == 1

    async def test_tool_delivery_fence_ignores_receipts_after_result(
        self, monitor: SessionMonitor
    ) -> None:
        from ccgram.transcript_reader import _ToolDeliveryFence

        pending = self._unsettled(300)
        pending.track()
        self._track(
            monitor,
            "s1",
            [self._ready(100), self._ready(200), pending],
        )
        monitor._transcript_reader._tool_delivery_fences["s1"] = _ToolDeliveryFence(
            call_checkpoint=100,
            release_checkpoint=200,
        )

        await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 200
        assert monitor._delivery_receipts["s1"] == [pending]
        assert "s1" not in monitor._transcript_reader._tool_delivery_fences

    async def test_parsed_tool_fences_checkpoint_and_rebuilds_after_restart(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "claude-session.jsonl"
        tool_call = {
            "type": "assistant",
            "uuid": "call-message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    }
                ],
            },
        }
        session_file.write_text(json.dumps(tool_call) + "\n")
        state_file = tmp_path / "monitor-state.json"

        first = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        first.state.update_session(
            TrackedSession(session_id="s1", file_path=str(session_file))
        )
        first.state.save_if_dirty()
        first.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await first._process_session_file("s1", session_file, [])
        first_pending = first._register_delivery_receipts(
            first._transcript_reader.drain_batches()
        )
        for batch, consumer_name, receipt in first_pending:
            await first._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await first.commit_delivered_watermarks()

        assert first.state.tracked_sessions["s1"].last_byte_offset == 0
        assert first._pending_tools["s1"] == {"tool-1": "Read"}

        with session_file.open("a") as transcript:
            transcript.write(
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "result-message",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": "tool-1",
                            "content": "read complete",
                        },
                    }
                )
                + "\n"
            )
        restarted = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        restarted_delivery = AsyncMock()
        restarted.set_message_callback(restarted_delivery)
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await restarted._process_session_file("s1", session_file, [])
        pending = restarted._register_delivery_receipts(
            restarted._transcript_reader.drain_batches()
        )
        for batch, consumer_name, receipt in pending:
            await restarted._dispatch_message_with_receipt(
                batch, receipt, consumer_name
            )
        await restarted.commit_delivered_watermarks()

        assert restarted._pending_tools == {}
        assert (
            restarted.state.tracked_sessions["s1"].last_byte_offset
            == session_file.stat().st_size
        )
        tool_results = [
            call.args[0]
            for call in restarted_delivery.await_args_list
            if call.args[0].content_type == "tool_result"
        ]
        assert len(tool_results) == 1
        assert tool_results[0].tool_name == "Read"
        assert "read complete" in tool_results[0].text

    async def test_tool_fence_waits_for_result_delivery_and_failure_replays_context(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "claude-session.jsonl"
        state_file = tmp_path / "monitor-state.json"
        tool_call = {
            "type": "assistant",
            "uuid": "call-message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    }
                ],
            },
        }
        tool_result = {
            "type": "user",
            "uuid": "result-message",
            "message": {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "content": "read complete",
            },
        }
        session_file.write_text(json.dumps(tool_call) + "\n")

        first = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        first.state.update_session(
            TrackedSession(session_id="s1", file_path=str(session_file))
        )
        first.state.save_if_dirty()
        first_delivery = AsyncMock()
        first.set_message_callback(first_delivery)
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await first._process_session_file("s1", session_file, [])
        call_pending = first._register_delivery_receipts(
            first._transcript_reader.drain_batches()
        )
        for batch, consumer_name, receipt in call_pending:
            await first._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await first.commit_delivered_watermarks()
        assert first.state.tracked_sessions["s1"].last_byte_offset == 0

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(tool_result) + "\n")
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await first._process_session_file("s1", session_file, [])
        result_pending = first._register_delivery_receipts(
            first._transcript_reader.drain_batches()
        )
        assert len(result_pending) == 1
        await first.commit_delivered_watermarks()
        assert first.state.tracked_sessions["s1"].last_byte_offset == 0

        failed = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        failed_results: list[NewMessage] = []

        async def fail_result(message: NewMessage) -> None:
            failed_results.append(message)
            if message.content_type == "tool_result":
                raise RuntimeError("result delivery failed")

        failed.set_message_callback(fail_result)
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await failed._process_session_file("s1", session_file, [])
        failed_pending = failed._register_delivery_receipts(
            failed._transcript_reader.drain_batches()
        )
        for batch, consumer_name, receipt in failed_pending:
            await failed._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await failed.commit_delivered_watermarks()

        failed_tool_results = [
            message
            for message in failed_results
            if message.content_type == "tool_result"
        ]
        assert len(failed_tool_results) == 1
        assert failed_tool_results[0].tool_name == "Read"
        assert failed_tool_results[0].text.find("read complete") >= 0
        assert failed.state.tracked_sessions["s1"].last_byte_offset == 0

        recovered = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        recovered_results: list[NewMessage] = []
        recovered.set_message_callback(
            AsyncMock(side_effect=lambda message: recovered_results.append(message))
        )
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await recovered._process_session_file("s1", session_file, [])
        recovered_pending = recovered._register_delivery_receipts(
            recovered._transcript_reader.drain_batches()
        )
        for batch, consumer_name, receipt in recovered_pending:
            await recovered._dispatch_message_with_receipt(
                batch, receipt, consumer_name
            )
        await recovered.commit_delivered_watermarks()

        recovered_tool_results = [
            message
            for message in recovered_results
            if message.content_type == "tool_result"
        ]
        assert len(recovered_tool_results) == 1
        assert recovered_tool_results[0].tool_name == "Read"
        assert "read complete" in recovered_tool_results[0].text
        assert (
            recovered.state.tracked_sessions["s1"].last_byte_offset
            == session_file.stat().st_size
        )

    @pytest.mark.parametrize("original_receipt_settled", [True, False])
    async def test_suppressed_codex_mirror_gets_durable_checkpoint(
        self, tmp_path, original_receipt_settled: bool
    ) -> None:
        session_file = tmp_path / "codex-session.jsonl"
        state_file = tmp_path / "monitor-state.json"
        response = {
            "type": "response_item",
            "payload": {
                "id": "response-1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "finished"}],
            },
        }
        task_complete = {
            "type": "event_msg",
            "payload": {"type": "task_complete", "last_agent_message": "finished"},
        }
        session_file.write_text(json.dumps(response) + "\n")
        response_checkpoint = session_file.stat().st_size

        first = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        first.state.update_session(
            TrackedSession(session_id="s1", file_path=str(session_file))
        )
        first.state.save_if_dirty()
        first_delivery = AsyncMock()
        first.set_message_callback(first_delivery)
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await first._process_session_file("s1", session_file, [])
        original_batches = first._transcript_reader.drain_batches()
        assert len(original_batches) == 1
        original_pending = first._register_delivery_receipts(original_batches)
        if original_receipt_settled:
            for batch, consumer_name, receipt in original_pending:
                await first._dispatch_message_with_receipt(
                    batch, receipt, consumer_name
                )
        await first.commit_delivered_watermarks()
        expected_offset = response_checkpoint if original_receipt_settled else 0
        assert first.state.tracked_sessions["s1"].last_byte_offset == expected_offset

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(task_complete) + "\n")
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await first._process_session_file("s1", session_file, [])
        mirror_batches = first._transcript_reader.drain_batches()
        assert len(mirror_batches) == 1
        assert mirror_batches[0].records == ()
        mirror_pending = first._register_delivery_receipts(mirror_batches)
        for batch, consumer_name, receipt in mirror_pending:
            await first._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await first.commit_delivered_watermarks()
        expected_offset = session_file.stat().st_size if original_receipt_settled else 0
        assert first.state.tracked_sessions["s1"].last_byte_offset == expected_offset

        restarted = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        restarted_delivery = AsyncMock()
        restarted.set_message_callback(restarted_delivery)
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await restarted._process_session_file("s1", session_file, [])
        restarted_batches = restarted._transcript_reader.drain_batches()
        if original_receipt_settled:
            assert restarted_batches == ()
        else:
            assert len(restarted_batches) == 1
            assert len(restarted_batches[0].records) == 1
            replay = restarted._register_delivery_receipts(restarted_batches)
            for batch, consumer_name, receipt in replay:
                await restarted._dispatch_message_with_receipt(
                    batch, receipt, consumer_name
                )
            assert [
                call.args[0].text for call in restarted_delivery.await_args_list
            ] == ["finished"]
        await restarted.commit_delivered_watermarks()
        assert (
            restarted.state.tracked_sessions["s1"].last_byte_offset
            == session_file.stat().st_size
        )

    @staticmethod
    def _begin_skip(monitor: SessionMonitor, snapshot_offset: int = 500) -> None:
        monitor.state.begin_skip(
            BacklogSkipIntent(
                session_id="s1",
                window_id="window-1",
                user_id=1,
                thread_id=2,
                chat_id=-100,
                snapshot_offset=snapshot_offset,
                range_start=0,
            )
        )

    async def test_skip_range_identity_survives_restart_and_rejects_rewrite(
        self, tmp_path
    ) -> None:
        transcript = tmp_path / "session.jsonl"
        transcript.write_bytes(_complete_jsonl_record(80))
        state_file = tmp_path / "monitor.json"
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        tracked = TrackedSession(
            "s1", str(transcript), last_byte_offset=10, parsed_offset=80
        )
        monitor.state.update_session(tracked)
        bindings = [(1, 99, 4, "@0")]
        session_map = {"@0": {"session_id": "s1", "transcript_path": str(transcript)}}

        async def purge(_intent) -> int:
            return 2

        async def notice(_intent) -> None:
            receipt = get_active_delivery_receipt()
            assert receipt is not None
            receipt.track()

        monitor.set_skip_callbacks(
            purge=purge, notice=notice, validate=lambda _intent: True
        )
        with (
            patch(
                "ccgram.session_monitor.session_lifecycle.resolve_session_id",
                side_effect=lambda _window_id: "s1",
            ),
            patch(
                "ccgram.session_monitor.session_lifecycle._last_session_map",
                session_map,
            ),
            patch.object(
                thread_router,
                "iter_thread_bindings_with_chat",
                side_effect=lambda: iter(bindings),
            ),
        ):
            intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
            assert intent is not None
            assert monitor.state.save_if_dirty()

        restarted = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        persisted_intent = restarted.state.pending_skips["s1"]
        with transcript.open("r+b") as source:
            source.write(b"y" * 80)
        restarted.set_skip_callbacks(
            purge=purge, notice=notice, validate=lambda _intent: True
        )
        restarted._skip_notice_receipts["s1"] = self._ready(80)

        with (
            patch(
                "ccgram.session_monitor.session_lifecycle.resolve_session_id",
                side_effect=lambda _window_id: "s1",
            ),
            patch(
                "ccgram.session_monitor.session_lifecycle._last_session_map",
                session_map,
            ),
            patch.object(
                thread_router,
                "iter_thread_bindings_with_chat",
                side_effect=lambda: iter(bindings),
            ),
        ):
            await restarted._commit_pending_skips()

        restarted_tracked = restarted.state.tracked_sessions["s1"]
        assert restarted_tracked.last_byte_offset == 10
        assert restarted_tracked.parsed_offset == 10
        assert "s1" not in restarted.state.pending_skips
        assert persisted_intent.source_range_sha256 == intent.source_range_sha256

    async def test_initial_skip_persistence_failure_preserves_queued_source_work(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "session.jsonl"
        session_file.write_text("queued transcript bytes\n")
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "monitor-state.json",
        )
        tracked = TrackedSession(
            session_id="s1",
            file_path=str(session_file),
            last_byte_offset=5,
            parsed_offset=12,
        )
        monitor.state.update_session(tracked)
        queued_receipt = self._unsettled(12)
        monitor._delivery_receipts["s1"] = [queued_receipt]
        batch = _transcript_batch("s1", 12)
        monitor._transcript_reader._pending_batches.append(batch)
        purge = AsyncMock(return_value=1)
        monitor.set_skip_callbacks(
            purge=purge,
            notice=AsyncMock(),
            validate=Mock(return_value=True),
        )

        with (
            patch.object(monitor.state, "save_if_dirty", side_effect=[False, True]),
            patch(
                "ccgram.session_monitor.session_lifecycle.resolve_session_id",
                return_value="s1",
            ),
        ):
            result = await monitor.request_backlog_skip(1, "window-1", 2, -100)

        assert result is None
        assert "s1" not in monitor.state.pending_skips
        assert tracked.parsed_offset == 12
        assert monitor._delivery_receipts["s1"] == [queued_receipt]
        assert monitor._transcript_reader.drain_batches() == (batch,)
        purge.assert_not_awaited()

    @pytest.mark.parametrize("newly_tracked_baseline", [False, True])
    async def test_restart_rebuilds_codex_mirror_context_before_task_complete(
        self, tmp_path, newly_tracked_baseline: bool
    ) -> None:
        session_file = tmp_path / "codex-session.jsonl"
        state_file = tmp_path / "monitor-state.json"
        response = {
            "type": "response_item",
            "payload": {
                "id": "response-1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "finished"}],
            },
        }
        task_complete = {
            "type": "event_msg",
            "payload": {"type": "task_complete", "last_agent_message": "finished"},
        }
        session_file.write_text(json.dumps(response) + "\n")
        first = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        if not newly_tracked_baseline:
            first.state.update_session(
                TrackedSession(session_id="s1", file_path=str(session_file))
            )
        first.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await first._process_session_file("s1", session_file, [])
        if newly_tracked_baseline:
            assert first._transcript_reader.drain_batches() == ()
            first.state.save_if_dirty()
        else:
            response_batches = first._transcript_reader.drain_batches()
            pending = first._register_delivery_receipts(response_batches)
            for batch, consumer_name, receipt in pending:
                await first._dispatch_message_with_receipt(
                    batch, receipt, consumer_name
                )
            await first.commit_delivered_watermarks()
        response_offset = session_file.stat().st_size
        assert first.state.tracked_sessions["s1"].last_byte_offset == response_offset

        restarted = (
            first
            if newly_tracked_baseline
            else SessionMonitor(
                projects_path=tmp_path / "projects", state_file=state_file
            )
        )
        delivery = AsyncMock()
        restarted.set_message_callback(delivery)
        with session_file.open("a") as transcript:
            transcript.write(json.dumps(task_complete) + "\n")
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await restarted._process_session_file("s1", session_file, [])

        batches = restarted._transcript_reader.drain_batches()
        assert len(batches) == 1
        assert batches[0].records == ()
        pending = restarted._register_delivery_receipts(batches)
        for batch, consumer_name, receipt in pending:
            await restarted._dispatch_message_with_receipt(
                batch, receipt, consumer_name
            )
        assert delivery.await_count == 0
        await restarted.commit_delivered_watermarks()
        assert (
            restarted.state.tracked_sessions["s1"].last_byte_offset
            == session_file.stat().st_size
        )

    @pytest.mark.parametrize("current", [True, False])
    async def test_completed_skip_retires_tool_fence_with_discarded_receipts(
        self, monitor: SessionMonitor, current: bool
    ) -> None:
        self._track(monitor, "s1", [self._ready(100), self._unsettled(200)])
        reader = monitor._transcript_reader
        call = SourceEntry(
            0,
            {
                "type": "assistant",
                "uuid": "call",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
                    ],
                },
            },
        )
        result = SourceEntry(
            100,
            {
                "type": "user",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "t1",
                    "content": "done",
                },
            },
        )
        for entry, checkpoint in [(call, 100), (result, 200)]:
            reader._append_provider_messages(
                "s1",
                [entry],
                ClaudeProvider(),
                {},
                "",
                [],
                byte_checkpoint=checkpoint,
            )
        assert "s1" in reader._tool_delivery_fences
        self._begin_skip(monitor)
        intent = monitor.state.pending_skips["s1"]
        intent.purge_complete = True
        with patch.object(
            monitor,
            "_verify_skip_content",
            new_callable=AsyncMock,
            return_value=current,
        ):
            monitor._skip_notice_receipts["s1"] = self._ready(500)
            await monitor.commit_delivered_watermarks()

        if not current:
            assert self._offset(monitor, "s1") == 0
            assert "s1" not in reader._tool_delivery_fences
            assert "s1" not in reader._pending_tools
            assert monitor.state.tracked_sessions["s1"].parsed_offset == 0
            return
        assert self._offset(monitor, "s1") == 500
        monitor.state.update_session(
            TrackedSession(
                session_id="s1",
                file_path="/transcript.jsonl",
                last_byte_offset=500,
                parsed_offset=600,
            )
        )
        monitor._delivery_receipts["s1"] = [self._ready(600)]
        await monitor.commit_delivered_watermarks()
        assert self._offset(monitor, "s1") == 600
        assert "s1" not in reader._tool_delivery_fences

    async def test_completed_skip_retires_unread_tool_result_context(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "claude-session.jsonl"
        state_file = tmp_path / "monitor-state.json"
        tool_call = {
            "type": "assistant",
            "uuid": "call-message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    }
                ],
            },
        }
        tool_result = {
            "type": "user",
            "uuid": "result-message",
            "message": {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "content": "skipped result",
            },
        }
        fresh_output = {
            "type": "assistant",
            "uuid": "fresh-message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "after skip"}],
            },
        }
        session_file.write_text(json.dumps(tool_call) + "\n")
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        monitor.state.update_session(
            TrackedSession(session_id="s1", file_path=str(session_file))
        )
        monitor.set_message_callback(AsyncMock())

        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file("s1", session_file, [])
        call_batches = monitor._transcript_reader.drain_batches()
        assert call_batches
        assert monitor._pending_tools["s1"] == {"tool-1": "Read"}
        assert monitor._transcript_reader._protocol_parser_states["s1"][
            1
        ].pending_tools == {"tool-1": "Read"}

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(tool_result) + "\n")
        skipped_offset = session_file.stat().st_size
        monitor.state.update_session(
            TrackedSession(
                session_id="s1",
                file_path=str(session_file),
                last_byte_offset=0,
                parsed_offset=call_batches[0].byte_checkpoint,
            )
        )
        self._begin_skip(monitor, skipped_offset)
        monitor.state.pending_skips["s1"].purge_complete = True
        with patch.object(
            monitor, "_verify_skip_content", new_callable=AsyncMock, return_value=True
        ):
            monitor._skip_notice_receipts["s1"] = self._ready(skipped_offset)
            await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == skipped_offset
        assert "s1" not in monitor._pending_tools
        assert "s1" not in monitor._transcript_reader._tool_delivery_fences
        assert "s1" not in monitor._transcript_reader._protocol_parser_states
        assert "s1" in monitor._transcript_reader._parser_rebuild_pending

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(fresh_output) + "\n")
        messages = []
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file("s1", session_file, messages)
        fresh_batches = monitor._transcript_reader.drain_batches()
        assert len(fresh_batches) == 1
        assert [message.text for message in messages] == ["after skip"]
        assert "s1" not in monitor._transcript_reader._parser_rebuild_pending
        assert (
            monitor._transcript_reader._protocol_parser_states["s1"][1].pending_tools
            == {}
        )
        pending = monitor._register_delivery_receipts(fresh_batches)
        for batch, consumer_name, receipt in pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await monitor.commit_delivered_watermarks()
        assert self._offset(monitor, "s1") == session_file.stat().st_size

        restarted = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        restarted.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await restarted._process_session_file("s1", session_file, [])
        assert restarted._transcript_reader.drain_batches() == ()

    @pytest.mark.parametrize("next_output", ["response_item", "agent_message"])
    async def test_completed_skip_rebuilds_codex_turn_context(
        self, tmp_path, next_output: str
    ) -> None:
        session_file = tmp_path / "codex-session.jsonl"
        state_file = tmp_path / "monitor-state.json"
        prior_turn = [
            {
                "type": "response_item",
                "payload": {
                    "id": "prior-answer",
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "Done"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {"type": "task_complete", "last_agent_message": "Done"},
            },
        ]
        skipped_user_turn = {
            "type": "response_item",
            "payload": {
                "id": "skipped-user",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "next task"}],
            },
        }
        next_entry = (
            {
                "type": "response_item",
                "payload": {
                    "id": "next-answer",
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "Done"}],
                },
            }
            if next_output == "response_item"
            else {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "Done"},
            }
        )
        session_file.write_text("".join(json.dumps(item) + "\n" for item in prior_turn))
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        monitor.state.update_session(
            TrackedSession(session_id="s1", file_path=str(session_file))
        )
        delivered: list[NewMessage] = []
        monitor.set_message_callback(
            AsyncMock(side_effect=lambda message: delivered.append(message))
        )

        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await monitor._process_session_file("s1", session_file, [])
        prior_batches = monitor._transcript_reader.drain_batches()
        assert len(prior_batches) == 1
        prior_pending = monitor._register_delivery_receipts(prior_batches)
        for batch, consumer_name, receipt in prior_pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await monitor.commit_delivered_watermarks()
        prior_offset = session_file.stat().st_size
        assert self._offset(monitor, "s1") == prior_offset
        assert [message.text for message in delivered] == ["Done"]

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(skipped_user_turn) + "\n")
        skip_offset = session_file.stat().st_size
        monitor.state.begin_skip(
            BacklogSkipIntent(
                session_id="s1",
                window_id="window-1",
                user_id=1,
                thread_id=2,
                chat_id=-100,
                snapshot_offset=skip_offset,
                range_start=prior_offset,
            )
        )
        monitor.state.pending_skips["s1"].purge_complete = True
        with patch.object(
            monitor, "_verify_skip_content", new_callable=AsyncMock, return_value=True
        ):
            monitor._skip_notice_receipts["s1"] = self._ready(skip_offset)
            await monitor.commit_delivered_watermarks()
        assert self._offset(monitor, "s1") == skip_offset

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(next_entry) + "\n")
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await monitor._process_session_file("s1", session_file, [])
        next_batches = monitor._transcript_reader.drain_batches()
        assert len(next_batches) == 1
        assert len(next_batches[0].records) == 1
        next_pending = monitor._register_delivery_receipts(next_batches)
        for batch, consumer_name, receipt in next_pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        assert [message.text for message in delivered] == ["Done", "Done"]
        await monitor.commit_delivered_watermarks()
        assert self._offset(monitor, "s1") == session_file.stat().st_size

    async def test_cancelled_skip_rewinds_and_rebuilds_tool_context(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "claude-session.jsonl"
        session_file.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "call-message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "Read",
                                "input": {"file_path": "README.md"},
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "monitor-state.json",
        )
        monitor.state.update_session(
            TrackedSession(session_id="s1", file_path=str(session_file))
        )
        monitor.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file("s1", session_file, [])
        call_batches = monitor._transcript_reader.drain_batches()
        monitor._register_delivery_receipts(call_batches)
        assert monitor._pending_tools["s1"] == {"tool-1": "Read"}

        with session_file.open("a") as transcript:
            transcript.write(
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "result-message",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": "tool-1",
                            "content": "replayed result",
                        },
                    }
                )
                + "\n"
            )
        self._begin_skip(monitor, session_file.stat().st_size)
        intent = monitor.state.pending_skips["s1"]
        intent.created_at = time.time() - 10_000.0
        intent.purge_complete = True
        with patch.object(monitor, "_skip_is_current", return_value=False):
            monitor._expire_aged_skip_barriers()

        assert self._offset(monitor, "s1") == 0
        assert "s1" not in monitor.state.pending_skips
        assert "s1" not in monitor._delivery_receipts
        assert monitor.state.tracked_sessions["s1"].parsed_offset == 0
        assert "s1" not in monitor._pending_tools

        replayed: list[NewMessage] = []
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file("s1", session_file, replayed)
        batches = monitor._transcript_reader.drain_batches()
        assert len(batches) == 1
        result = next(
            message for message in replayed if message.content_type == "tool_result"
        )
        assert result.tool_name == "Read"
        assert "replayed result" in result.text

    async def test_unconfirmed_skip_preserves_tool_context(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [])
        call = SourceEntry(
            0,
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
                    ],
                },
            },
        )
        monitor._transcript_reader._append_provider_messages(
            "s1", [call], ClaudeProvider(), {}, "", [], byte_checkpoint=100
        )
        parser_state = monitor._transcript_reader._protocol_parser_states["s1"][1]
        self._begin_skip(monitor)
        monitor.state.pending_skips["s1"].purge_complete = True
        unconfirmed = self._unsettled(500)
        unconfirmed.track()
        monitor._skip_notice_receipts["s1"] = unconfirmed

        await monitor.commit_delivered_watermarks()

        assert "s1" in monitor.state.pending_skips
        assert monitor._pending_tools["s1"] == {"t1": "Read"}
        assert parser_state.pending_tools == {"t1": "Read"}
        assert "s1" in monitor._transcript_reader._tool_delivery_fences

    @pytest.mark.parametrize("append_later_text", [True, False])
    async def test_adoption_rebuilds_unsettled_tool_context_from_watermark(
        self, tmp_path, append_later_text: bool
    ) -> None:
        session_file = tmp_path / "claude-session.jsonl"
        state_file = tmp_path / "monitor-state.json"
        tool_call = {
            "type": "assistant",
            "uuid": "call-message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    }
                ],
            },
        }
        tool_result = {
            "type": "user",
            "uuid": "result-message",
            "message": {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "content": "read complete",
            },
        }
        later_text = {
            "type": "assistant",
            "uuid": "later-message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "later"}],
            },
        }
        session_file.write_text(json.dumps(tool_call) + "\n")
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        monitor.state.update_session(
            TrackedSession(session_id="old-id", file_path=str(session_file))
        )
        monitor.set_message_callback(AsyncMock())

        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file("old-id", session_file, [])
        call_batches = monitor._transcript_reader.drain_batches()
        monitor._register_delivery_receipts(call_batches)
        assert "old-id" in monitor._pending_tools

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(tool_result) + "\n")
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file("old-id", session_file, [])
        result_batches = monitor._transcript_reader.drain_batches()
        monitor._register_delivery_receipts(result_batches)
        fence = monitor._transcript_reader._tool_delivery_fences["old-id"]
        assert fence.release_checkpoint == session_file.stat().st_size
        assert monitor._delivery_receipts["old-id"]

        if append_later_text:
            with session_file.open("a") as transcript:
                transcript.write(json.dumps(later_text) + "\n")
        migrated_messages: list[NewMessage] = []
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file(
                "new-id", session_file, migrated_messages
            )

        assert "old-id" not in monitor._delivery_receipts
        assert "old-id" not in monitor._transcript_reader._protocol_parser_states
        assert "old-id" not in monitor._transcript_reader._tool_delivery_fences
        migrated = monitor._transcript_reader.drain_batches()
        assert len(migrated) == 1
        result = next(
            message
            for message in migrated_messages
            if message.content_type == "tool_result"
        )
        assert result.tool_name == "Read"
        assert "read complete" in result.text
        if append_later_text:
            assert any(message.text == "later" for message in migrated_messages)
        else:
            assert not any(message.text == "later" for message in migrated_messages)
        pending = monitor._register_delivery_receipts(migrated)
        for batch, consumer_name, receipt in pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await monitor.commit_delivered_watermarks()
        final_offset = session_file.stat().st_size
        assert monitor.state.tracked_sessions["new-id"].last_byte_offset == final_offset

        restarted = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        restarted.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await restarted._process_session_file("new-id", session_file, [])
        assert restarted._transcript_reader.drain_batches() == ()

    @pytest.mark.parametrize("old_watermark", [0, 10_000])
    async def test_generation_replacement_retires_old_receipts_and_parser_state(
        self, tmp_path, old_watermark: int
    ) -> None:
        session_file = tmp_path / "claude-session.jsonl"
        session_file.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "old-call",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "old-tool", "name": "Read"}
                        ],
                    },
                }
            )
            + "\n"
        )
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "monitor-state.json",
        )
        monitor.state.update_session(
            TrackedSession(session_id="s1", file_path=str(session_file))
        )
        monitor.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file("s1", session_file, [])
        old_batches = monitor._transcript_reader.drain_batches()
        old_deliveries = monitor._register_delivery_receipts(old_batches)
        for _, _, receipt in old_deliveries:
            receipt.fail()
            receipt.close()
        assert "s1" in monitor._delivery_receipts
        assert "s1" in monitor._pending_tools
        monitor.state.tracked_sessions["s1"].last_byte_offset = old_watermark
        monitor.state._dirty = True
        assert monitor.state.save_if_dirty()

        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "fresh-message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "fresh generation"}],
                    },
                }
            )
            + "\n"
        )
        replacement.replace(session_file)
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await monitor._process_session_file("s1", session_file, [])

        assert "s1" not in monitor._delivery_receipts
        assert "s1" not in monitor._pending_tools
        assert "s1" not in monitor._transcript_reader._tool_delivery_fences
        assert "s1" not in monitor._transcript_reader._protocol_parser_states
        replacement_batches = monitor._transcript_reader.drain_batches()
        assert len(replacement_batches) == 1
        assert replacement_batches[0].records == ()
        assert replacement_batches[0].byte_checkpoint == session_file.stat().st_size
        replacement_pending = monitor._register_delivery_receipts(replacement_batches)
        for batch, consumer_name, receipt in replacement_pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await monitor.commit_delivered_watermarks()
        assert self._offset(monitor, "s1") == session_file.stat().st_size

        restarted = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "monitor-state.json",
        )
        restarted.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await restarted._process_session_file("s1", session_file, [])
        assert restarted._transcript_reader.drain_batches() == ()

        with session_file.open("a") as transcript:
            transcript.write(
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "next-message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "next"}],
                        },
                    }
                )
                + "\n"
            )
        next_messages: list[NewMessage] = []
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=ClaudeProvider(),
        ):
            await restarted._process_session_file("s1", session_file, next_messages)
        batches = restarted._transcript_reader.drain_batches()
        assert len(batches) == 1
        assert [message.text for message in next_messages] == ["next"]
        pending = restarted._register_delivery_receipts(batches)
        for batch, consumer_name, receipt in pending:
            await restarted._dispatch_message_with_receipt(
                batch, receipt, consumer_name
            )
        await restarted.commit_delivered_watermarks()
        assert self._offset(restarted, "s1") == session_file.stat().st_size

    async def test_gemini_jsonl_adoption_replays_only_uncommitted_source_range(
        self, tmp_path
    ) -> None:
        from ccgram.providers.gemini import GeminiProvider

        session_file = tmp_path / "gemini-session.jsonl"
        state_file = tmp_path / "monitor-state.json"
        first_entry = {"type": "gemini", "id": "m1", "content": "committed"}
        next_entry = {"type": "gemini", "id": "m2", "content": "pending"}
        session_file.write_text(json.dumps(first_entry) + "\n")
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        monitor.state.update_session(
            TrackedSession(session_id="old-id", file_path=str(session_file))
        )
        monitor.set_message_callback(AsyncMock())

        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=GeminiProvider(),
        ):
            await monitor._process_session_file("old-id", session_file, [])
        first_batches = monitor._transcript_reader.drain_batches()
        assert len(first_batches) == 1
        first_pending = monitor._register_delivery_receipts(first_batches)
        for batch, consumer_name, receipt in first_pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await monitor.commit_delivered_watermarks()
        first_offset = session_file.stat().st_size
        assert monitor.state.tracked_sessions["old-id"].last_byte_offset == first_offset

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(next_entry) + "\n")
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=GeminiProvider(),
        ):
            await monitor._process_session_file("old-id", session_file, [])
        monitor._register_delivery_receipts(monitor._transcript_reader.drain_batches())
        assert monitor._delivery_receipts["old-id"]

        adopted_messages: list[NewMessage] = []
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=GeminiProvider(),
        ):
            await monitor._process_session_file(
                "new-id", session_file, adopted_messages
            )
        adopted_batches = monitor._transcript_reader.drain_batches()

        assert [message.text for message in adopted_messages] == ["pending"]
        assert len(adopted_batches) == 1
        assert adopted_batches[0].byte_checkpoint == session_file.stat().st_size
        assert "old-id" not in monitor._delivery_receipts
        pending = monitor._register_delivery_receipts(adopted_batches)
        for batch, consumer_name, receipt in pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await monitor.commit_delivered_watermarks()
        assert (
            monitor.state.tracked_sessions["new-id"].last_byte_offset
            == session_file.stat().st_size
        )

        restarted = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        restarted.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=GeminiProvider(),
        ):
            await restarted._process_session_file("new-id", session_file, [])
        assert restarted._transcript_reader.drain_batches() == ()

    async def test_adoption_rebuilds_codex_mirror_context_from_committed_prefix(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "codex-session.jsonl"
        state_file = tmp_path / "monitor-state.json"
        response = {
            "type": "response_item",
            "payload": {
                "id": "response-1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "finished"}],
            },
        }
        task_complete = {
            "type": "event_msg",
            "payload": {"type": "task_complete", "last_agent_message": "finished"},
        }
        session_file.write_text(json.dumps(response) + "\n")
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        monitor.state.update_session(
            TrackedSession(session_id="old-id", file_path=str(session_file))
        )
        monitor.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await monitor._process_session_file("old-id", session_file, [])
        response_batches = monitor._transcript_reader.drain_batches()
        assert len(response_batches) == 1
        response_pending = monitor._register_delivery_receipts(response_batches)
        for batch, consumer_name, receipt in response_pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await monitor.commit_delivered_watermarks()
        response_offset = session_file.stat().st_size
        assert (
            monitor.state.tracked_sessions["old-id"].last_byte_offset == response_offset
        )

        with session_file.open("a") as transcript:
            transcript.write(json.dumps(task_complete) + "\n")
        before_adoption: list[NewMessage] = []
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await monitor._process_session_file("old-id", session_file, before_adoption)
        mirror_batches = monitor._transcript_reader.drain_batches()
        assert len(mirror_batches) == 1
        assert mirror_batches[0].records == ()
        monitor._register_delivery_receipts(mirror_batches)
        assert before_adoption == []

        adopted_messages: list[NewMessage] = []
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await monitor._process_session_file(
                "new-id", session_file, adopted_messages
            )
        adopted_batches = monitor._transcript_reader.drain_batches()

        assert adopted_messages == []
        assert len(adopted_batches) == 1
        assert adopted_batches[0].records == ()
        pending = monitor._register_delivery_receipts(adopted_batches)
        for batch, consumer_name, receipt in pending:
            await monitor._dispatch_message_with_receipt(batch, receipt, consumer_name)
        await monitor.commit_delivered_watermarks()
        assert (
            monitor.state.tracked_sessions["new-id"].last_byte_offset
            == session_file.stat().st_size
        )

        restarted = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        restarted.set_message_callback(AsyncMock())
        with patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=CodexProvider(),
        ):
            await restarted._process_session_file("new-id", session_file, [])
        assert restarted._transcript_reader.drain_batches() == ()

    async def test_pending_skip_fences_settled_prefix(
        self, monitor: SessionMonitor
    ) -> None:
        ready = self._ready(100)
        self._track(monitor, "s1", [ready])
        self._begin_skip(monitor)

        await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 0
        assert monitor._delivery_receipts["s1"] == [ready]

    async def test_legacy_skip_without_source_identity_does_not_commit(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [])
        self._begin_skip(monitor)
        monitor.set_skip_callbacks(
            purge=AsyncMock(), notice=AsyncMock(), validate=lambda _intent: True
        )
        monitor._skip_notice_receipts["s1"] = self._ready(500)

        await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 0
        assert "s1" not in monitor.state.pending_skips

    def test_aged_skip_barrier_cancels_and_replays_without_ack(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [self._ready(100)])
        self._begin_skip(monitor)
        intent = monitor.state.pending_skips["s1"]
        intent.created_at = time.time() - 10_000.0
        intent.purge_complete = True

        with patch.object(monitor, "_skip_is_current", return_value=True):
            monitor._expire_aged_skip_barriers()

        assert self._offset(monitor, "s1") == 0
        assert monitor.state.tracked_sessions["s1"].parsed_offset == 0
        assert "s1" not in monitor.state.pending_skips
        assert "s1" not in monitor._skip_notice_receipts

    def test_aged_skip_barrier_on_rebound_topic_cancels(
        self, monitor: SessionMonitor
    ) -> None:
        # A rebound topic never advances the old watermark; the range
        # stays replayable under the new binding.
        self._track(monitor, "s1", [self._ready(100)])
        self._begin_skip(monitor)
        intent = monitor.state.pending_skips["s1"]
        intent.created_at = time.time() - 10_000.0
        intent.purge_complete = True

        with patch.object(monitor, "_skip_is_current", return_value=False):
            monitor._expire_aged_skip_barriers()

        assert self._offset(monitor, "s1") == 0
        assert "s1" not in monitor.state.pending_skips

    async def test_rebound_after_failed_skip_notice_replays_purged_source(
        self, monitor: SessionMonitor
    ) -> None:
        tracked = TrackedSession(
            session_id="s1",
            file_path="/transcript.jsonl",
            last_byte_offset=100,
            parsed_offset=300,
        )
        monitor.state.update_session(tracked)
        source_receipt = self._ready(300)
        monitor._delivery_receipts["s1"] = [source_receipt]
        reader = monitor._transcript_reader
        reader._pending_batches.append(_transcript_batch("s1", 300))
        reader._pending_tools["s1"] = {"tool-call": object()}
        reader._tool_delivery_fences["s1"] = _ToolDeliveryFence(100)
        self._begin_skip(monitor, snapshot_offset=500)
        intent = monitor.state.pending_skips["s1"]

        validate = Mock(side_effect=[True, True, True, False])
        monitor.set_skip_callbacks(
            purge=AsyncMock(return_value=1),
            notice=AsyncMock(side_effect=RuntimeError("notice failed")),
            validate=validate,
        )

        with (
            patch.object(monitor, "_skip_source_is_current", return_value=True),
            patch(
                "ccgram.session_monitor.session_lifecycle.resolve_session_id",
                return_value="s1",
            ),
            patch.object(
                thread_router,
                "iter_thread_bindings_with_chat",
                return_value=[(1, -100, 2, "window-1")],
            ),
        ):
            assert await monitor._prepare_pending_skip(intent) is True
            assert await monitor._enqueue_pending_skip_notice(intent) is False
            assert intent.purge_complete is True
            assert "s1" in monitor.state.pending_skips

            monitor._clear_skip_retry("s1")
            await monitor._resume_pending_skip_notices()

        assert "s1" not in monitor.state.pending_skips
        assert self._offset(monitor, "s1") == 100
        assert tracked.parsed_offset == 100
        assert "s1" not in monitor._delivery_receipts
        assert "s1" not in reader._pending_tools
        assert "s1" not in reader._tool_delivery_fences
        assert not any(batch.session_id == "s1" for batch in reader._pending_batches)
        assert "s1" not in monitor._skip_notice_receipts

    def test_aged_barrier_with_incomplete_purge_replays(
        self, monitor: SessionMonitor
    ) -> None:
        # A barrier whose queued range was never retired must not skip
        # those bytes silently: cancel so they replay.
        self._track(monitor, "s1", [self._ready(100)])
        self._begin_skip(monitor)
        intent = monitor.state.pending_skips["s1"]
        intent.created_at = time.time() - 10_000.0
        assert intent.purge_complete is False

        with patch.object(monitor, "_skip_is_current", return_value=True):
            monitor._expire_aged_skip_barriers()

        assert self._offset(monitor, "s1") == 0
        assert "s1" not in monitor.state.pending_skips

    def test_aged_barrier_cancels_even_when_validator_fails(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [self._ready(100)])
        self._begin_skip(monitor)
        intent = monitor.state.pending_skips["s1"]
        intent.created_at = time.time() - 10_000.0
        intent.purge_complete = True

        validate = Mock(side_effect=RuntimeError("boom"))
        monitor.set_skip_callbacks(
            purge=AsyncMock(), notice=AsyncMock(), validate=validate
        )

        monitor._expire_aged_skip_barriers()

        validate.assert_not_called()
        assert self._offset(monitor, "s1") == 0
        assert monitor.state.tracked_sessions["s1"].parsed_offset == 0
        assert "s1" not in monitor.state.pending_skips

    async def test_aged_barrier_deadline_cancels_without_acknowledgement(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        self._track(monitor, "s1", [self._ready(100)])
        self._begin_skip(monitor)
        intent = monitor.state.pending_skips["s1"]
        intent.created_at = time.time() - 120.0
        intent.purge_complete = True
        monkeypatch.setattr(
            "ccgram.session_monitor.config.skip_barrier_deadline_s", 60.0
        )

        with patch.object(monitor, "_skip_is_current", return_value=True):
            await monitor._advance_skip_barriers()

        assert self._offset(monitor, "s1") == 0
        assert "s1" not in monitor.state.pending_skips
        assert "s1" not in monitor._skip_notice_receipts

    async def test_acknowledged_skip_commits_once_at_deadline(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        self._track(monitor, "s1", [self._ready(100)])
        self._begin_skip(monitor, snapshot_offset=500)
        intent = monitor.state.pending_skips["s1"]
        intent.created_at = time.time() - 120.0
        intent.purge_complete = True
        monitor._skip_notice_receipts["s1"] = self._ready(500)
        monkeypatch.setattr(
            "ccgram.session_monitor.config.skip_barrier_deadline_s", 60.0
        )

        with (
            patch.object(
                monitor,
                "_verify_skip_content",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                monitor.state, "complete_skip", wraps=monitor.state.complete_skip
            ) as complete_skip,
        ):
            await monitor._advance_skip_barriers()
            await monitor._advance_skip_barriers()

        assert self._offset(monitor, "s1") == 500
        assert "s1" not in monitor.state.pending_skips
        complete_skip.assert_called_once_with("s1")

    def test_legacy_barrier_without_stamp_gets_clock_started(
        self, monitor: SessionMonitor
    ) -> None:
        # Barriers persisted before the stamp are aged from first sight,
        # not force-completed on the first cycle.
        self._begin_skip(monitor)
        assert monitor.state.pending_skips["s1"].created_at == 0.0

        monitor._expire_aged_skip_barriers()

        assert monitor.state.pending_skips["s1"].created_at > 0.0
        assert "s1" in monitor.state.pending_skips

    async def test_delivered_skip_wins_and_discards_ordinary_receipts(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [self._ready(100)])
        self._begin_skip(monitor)
        monitor._skip_notice_receipts["s1"] = self._ready(500)

        with patch.object(
            monitor, "_verify_skip_content", new_callable=AsyncMock, return_value=True
        ):
            await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 500
        assert "s1" not in monitor.state.pending_skips
        assert "s1" not in monitor._delivery_receipts

    async def test_shared_batch_checkpoint_defers_commit(
        self, monitor: SessionMonitor
    ) -> None:
        """Code-review P1 (history/shallow 2026-08-30): receipts of one
        parse cycle share the batch-end checkpoint, so a settled sibling
        must NOT commit it while an unsettled sibling sits below it."""
        sibling = self._unsettled(300)
        sibling.track()
        self._track(monitor, "s1", [self._ready(300), sibling])

        await monitor.commit_delivered_watermarks()

        # No commit past a tie: the unsettled sibling's bytes are below
        # the shared 300. The settled receipt is still consumed; replay
        # re-delivers its message (at-least-once permits the duplicate).
        assert self._offset(monitor, "s1") == 0
        assert monitor._delivery_receipts["s1"] == [sibling]

    async def test_next_batch_checkpoint_unblocks_tied_commit(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [self._ready(300), self._ready(400)])
        late = self._unsettled(400)
        late.track()
        monitor._delivery_receipts["s1"].append(late)

        await monitor.commit_delivered_watermarks()

        # The first batch (300) is fully settled and the fence sits
        # strictly beyond it: 300 commits, the tied 400s do not.
        assert self._offset(monitor, "s1") == 300
        assert monitor._delivery_receipts["s1"] == [late]

    async def test_receiptless_tracked_session_does_not_crash_commit(
        self, monitor: SessionMonitor
    ) -> None:
        """Code-review P1 (multi-agent 2026-08-30): a tracked session
        without receipts (the steady state after its receipts were
        consumed) must not break the batched commit for other sessions."""
        self._track(monitor, "s1", [self._ready(100)])
        self._track(monitor, "s2", [])
        monitor._delivery_receipts.pop("s2")

        await monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 100
        assert self._offset(monitor, "s2") == 0


class TestSessionMapReadFailures:
    async def test_unreadable_map_does_not_reconcile_as_empty(
        self, monitor: SessionMonitor
    ) -> None:
        with (
            patch(
                "ccgram.session_monitor.read_session_map_raw",
                AsyncMock(return_value=None),
            ),
            patch("ccgram.session_monitor.session_lifecycle") as lifecycle,
        ):
            await monitor._detect_and_cleanup_changes(adoptable_window_ids=set())
        lifecycle.reconcile.assert_not_called()


class TestPendingToolsCleanup:
    async def test_cleanup_stale_removes_pending_tools(
        self, monitor: SessionMonitor
    ) -> None:
        monitor._pending_tools["stale-session"] = {"tool_1": {"name": "Read"}}
        monitor.state.update_session(
            TrackedSession(session_id="stale-session", file_path="/fake/path")
        )

        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value={},
        ):
            await monitor._cleanup_all_stale_sessions()

        assert "stale-session" not in monitor._pending_tools

    async def test_detect_changes_removes_pending_tools(
        self, monitor: SessionMonitor
    ) -> None:
        old_sid = "old-session"
        new_sid = "new-session"

        monitor._pending_tools[old_sid] = {"tool_1": {"name": "Write"}}
        monitor._last_session_map = {
            "my-window": {"session_id": old_sid, "cwd": "/a", "window_name": ""}
        }
        monitor.state.update_session(
            TrackedSession(session_id=old_sid, file_path="/fake/path")
        )

        new_map = {"my-window": {"session_id": new_sid, "cwd": "/a", "window_name": ""}}
        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=new_map,
        ):
            await monitor._detect_and_cleanup_changes(adoptable_window_ids=set(new_map))

        assert old_sid not in monitor._pending_tools


class TestNewWindowDetection:
    async def test_callback_fires_for_new_window(self, monitor: SessionMonitor) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monitor._last_session_map = {}
        # Adoption is gated on the backend's verdict from the current listing;
        # the loop sets this before reaching here.

        new_map = {"@5": {"session_id": "s1", "cwd": "/proj", "window_name": "proj"}}
        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=new_map,
        ):
            await monitor._detect_and_cleanup_changes(adoptable_window_ids=set(new_map))

        cb.assert_called_once()
        event = cb.call_args[0][0]
        assert isinstance(event, NewWindowEvent)
        assert event.window_id == "@5"
        assert event.session_id == "s1"
        assert event.window_name == "proj"

    async def test_startup_does_not_trigger_callback(
        self, monitor: SessionMonitor
    ) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        initial_map = {"@0": {"session_id": "s0", "cwd": "/a", "window_name": "a"}}
        monitor._last_session_map = initial_map

        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=initial_map,
        ):
            await monitor._detect_and_cleanup_changes(
                adoptable_window_ids=set(initial_map)
            )

        cb.assert_not_called()

    async def test_callback_error_does_not_crash(self, monitor: SessionMonitor) -> None:
        cb = AsyncMock(side_effect=RuntimeError("boom"))
        monitor.set_new_window_callback(cb)
        monitor._last_session_map = {}

        new_map = {"@1": {"session_id": "s1", "cwd": "/x", "window_name": "x"}}
        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=new_map,
        ):
            await monitor._detect_and_cleanup_changes(adoptable_window_ids=set(new_map))

        cb.assert_called_once()


_TMUX_CAPS = MultiplexerCapabilities(
    name="tmux",
    ids_stable_across_restart=True,
    exposes_pane_tty=True,
    native_agent_status=False,
    read_max_lines=None,
    self_identify_env="TMUX_PANE",
    supports_event_stream=False,
    native_worktrees=False,
)
_HERDR_CAPS = MultiplexerCapabilities(
    name="herdr",
    ids_stable_across_restart=False,
    exposes_pane_tty=False,
    native_agent_status=True,
    read_max_lines=1000,
    self_identify_env="HERDR_PANE_ID",
    supports_event_stream=True,
    native_worktrees=True,
    # Production herdr sets this; without it these tests take the agterm
    # branch and guarded-target validation regresses unnoticed.
    native_topic_targets=True,
)


def _winref(window_id: str, command: str, *, eligible: bool | None = None) -> WindowRef:
    """A window as a backend would hand it over.

    ``topic_eligible`` is the backend's verdict, so these fixtures mirror what
    a real adapter stamps: a record with an agent is adoptable, a bare shell
    pane is not. Passing ``eligible`` overrides that for the cases that are
    about the flag itself.
    """
    return WindowRef(
        window_id=window_id,
        window_name=window_id,
        cwd="/proj",
        pane_current_command=command,
        topic_eligible=bool(command.strip()) if eligible is None else eligible,
    )


class TestEmitUnboundWindowEvents:
    """The unbound-window discovery path is capability-gated (Task 10)."""

    @pytest.fixture
    def wired(self, monkeypatch) -> None:
        # Wire thread_router via a real SessionManager and start empty.
        thread_router.reset()
        window_store.window_states.clear()
        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
        SessionManager()

    async def test_tmux_surfaces_every_unbound_window(
        self, monitor: SessionMonitor, wired, monkeypatch
    ) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_TMUX_CAPS),
        )

        windows = [_winref("@1", "zsh"), _winref("@2", "claude")]
        await monitor._emit_unbound_window_events(windows, known_window_ids=set())

        surfaced = {c.args[0].window_id for c in cb.call_args_list}
        assert surfaced == {"@1", "@2"}

    async def test_herdr_surfaces_only_agent_sessions(
        self, monitor: SessionMonitor, wired, monkeypatch
    ) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_HERDR_CAPS),
        )

        # Each sessionful agent target gets a topic; a bare shell does not.
        windows = [
            _winref(HERDR_TARGETS["a"], "claude"),
            _winref(HERDR_TARGETS["b"], "claude"),
            _winref(HERDR_TARGETS["shell"], ""),
        ]
        await monitor._emit_unbound_window_events(windows, known_window_ids=set())

        surfaced = {c.args[0].window_id for c in cb.call_args_list}
        assert surfaced == {HERDR_TARGETS["a"], HERDR_TARGETS["b"]}

    async def test_skips_known_and_bound_windows(
        self, monitor: SessionMonitor, wired, monkeypatch
    ) -> None:
        thread_router.bind_thread(100, 1, HERDR_TARGETS["bound"])
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_HERDR_CAPS),
        )

        windows = [
            _winref(HERDR_TARGETS["known"], "claude"),  # in session_map
            _winref(HERDR_TARGETS["bound"], "claude"),  # bound to a topic
            _winref(HERDR_TARGETS["new"], "claude"),  # genuinely new
        ]
        await monitor._emit_unbound_window_events(
            windows, known_window_ids={HERDR_TARGETS["known"]}
        )

        surfaced = {c.args[0].window_id for c in cb.call_args_list}
        assert surfaced == {HERDR_TARGETS["new"]}

    async def test_case_variant_bound_window_is_not_rediscovered(
        self, monitor: SessionMonitor, wired, monkeypatch
    ) -> None:
        thread_router.bind_thread(100, 42, "abc-def")
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_HERDR_CAPS),
        )

        await monitor._emit_unbound_window_events(
            [_winref("ABC-DEF", "claude")], known_window_ids=set()
        )

        cb.assert_not_awaited()


class TestEmitKnownUnboundWindowEvents:
    """Steady-state self-heal: session_map windows not bound to a topic retry on each poll."""

    @pytest.fixture
    def wired(self, monkeypatch) -> None:
        thread_router.reset()
        window_store.window_states.clear()
        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
        SessionManager()

    async def test_known_unbound_window_surfaces(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """A tab in session_map but not bound fires NewWindowEvent (self-heal path)."""
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "w1:t1": {
                "session_id": "S1",
                "cwd": "/repo",
                "window_name": "agent",
            }
        }
        live_window_ids = {"w1:t1"}  # tab is live

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        cb.assert_called_once()
        event = cb.call_args[0][0]
        assert isinstance(event, NewWindowEvent)
        assert event.window_id == "w1:t1"
        assert event.session_id == "S1"
        assert event.window_name == "agent"

    async def test_bound_window_not_re_fired(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """A tab already bound to a topic is skipped (no spam)."""
        thread_router.bind_thread(100, 42, "w1:t1")
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "w1:t1": {"session_id": "S1", "cwd": "/repo", "window_name": "agent"}
        }
        live_window_ids = {"w1:t1"}

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        cb.assert_not_called()

    async def test_case_variant_bound_window_not_re_fired(
        self, monitor: SessionMonitor, wired
    ) -> None:
        thread_router.bind_thread(100, 42, "abc-def")
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "ABC-DEF": {
                "session_id": "S1",
                "cwd": "/repo",
                "window_name": "agent",
            }
        }

        await monitor._emit_known_unbound_window_events(current_map, {"abc-def"})

        cb.assert_not_called()

    async def test_dead_window_not_surfaced(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """A session_map entry for a window not in live_window_ids is skipped."""
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "w9:t9": {"session_id": "S9", "cwd": "/gone", "window_name": "dead"}
        }
        live_window_ids: set[str] = set()  # tab not alive / __*__-filtered

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        cb.assert_not_called()

    async def test_star_tab_not_surfaced(self, monitor: SessionMonitor, wired) -> None:
        """__*__ tabs are absent from live_window_ids (filtered by list_windows) — never adopted."""
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        # Simulate a __*__ tab somehow in session_map; list_windows filtered it out
        current_map = {
            "w0:t0": {"session_id": "S0", "cwd": "/self", "window_name": "__main__"}
        }
        live_window_ids: set[str] = set()  # __*__ absent from list_windows output

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        cb.assert_not_called()

    async def test_multiple_windows_mixed(self, monitor: SessionMonitor, wired) -> None:
        """Only unbound live tabs surface; bound and dead are skipped."""
        thread_router.bind_thread(100, 1, "w1:t1")  # already bound
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "w1:t1": {"session_id": "S1", "cwd": "/a", "window_name": "bound"},
            "w2:t2": {"session_id": "S2", "cwd": "/b", "window_name": "unbound"},
            "w3:t3": {"session_id": "S3", "cwd": "/c", "window_name": "dead"},
        }
        live_window_ids = {"w1:t1", "w2:t2"}  # w3:t3 not live

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        surfaced = {c.args[0].window_id for c in cb.call_args_list}
        assert surfaced == {"w2:t2"}

    async def test_no_callback_is_noop(self, monitor: SessionMonitor, wired) -> None:
        """No callback registered — returns without error."""
        current_map = {
            "w1:t1": {"session_id": "S1", "cwd": "/repo", "window_name": "agent"}
        }
        # No callback set — must not raise
        await monitor._emit_known_unbound_window_events(current_map, {"w1:t1"})

    async def test_callback_error_does_not_crash(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """Callback error is caught and logged; loop continues."""
        cb = AsyncMock(side_effect=RuntimeError("boom"))
        monitor.set_new_window_callback(cb)

        current_map = {
            "w1:t1": {"session_id": "S1", "cwd": "/a", "window_name": "a"},
            "w2:t2": {"session_id": "S2", "cwd": "/b", "window_name": "b"},
        }
        live_window_ids = {"w1:t1", "w2:t2"}

        # Should not raise despite the callback error
        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        assert cb.call_count == 2


class TestLoadCurrentSessionMapBackend:
    """The monitor's session_map reader must honor the active backend prefix.

    Regression: under herdr the hook writes ``herdr:<opaque-session-target>``
    keys; a tmux-only ``ccgram:`` prefix silently dropped every herdr session.
    """

    async def test_herdr_rejects_raw_locator_keys(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "multiplexer_name", "herdr")
        raw = {
            "herdr:w2:p1": {
                "session_id": "S1",
                "cwd": "/repo",
                "window_name": "agent",
                "transcript_path": "",
                "provider_name": "claude",
            }
        }
        assert await monitor._load_current_session_map(raw) == {}

    async def test_herdr_guarded_target_key_surfaces(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "multiplexer_name", "herdr")
        target = "herdr-session-v1-" + "a" * 64
        raw = {"herdr:" + target: {"session_id": "S1", "cwd": "/repo"}}
        result = await monitor._load_current_session_map(raw)
        assert result[target]["session_id"] == "S1"

    async def test_tmux_skips_herdr_keys(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "multiplexer_name", "tmux")
        raw = {"herdr:w2:p1": {"session_id": "S1", "cwd": "/repo"}}
        assert await monitor._load_current_session_map(raw) == {}


class TestPerWindowProviderResolution:
    async def test_process_session_file_passes_window_id(self, tmp_path) -> None:
        """_process_session_file uses window_id for per-window provider resolution."""
        session_file = tmp_path / "transcript.jsonl"
        line = '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}\n'
        session_file.write_text(line)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-pw",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        new_messages = []
        await monitor._process_session_file(
            "sess-pw", session_file, new_messages, window_id="@42"
        )
        assert len(new_messages) == 1
        assert "hello" in new_messages[0].text

    async def test_process_session_file_prefers_transcript_provider_when_stale(
        self, tmp_path
    ) -> None:
        """A stale hookful provider should not suppress Codex transcript parsing."""
        session_file = (
            tmp_path / ".codex" / "sessions" / "2026" / "03" / "23" / "transcript.jsonl"
        )
        session_file.parent.mkdir(parents=True)
        session_file.write_text(
            '{"timestamp":"2026-03-23T00:00:00Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello codex"}]}}\n'
        )

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-stale",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        new_messages = []
        with (
            patch(
                "ccgram.transcript_reader.get_provider_for_window",
                return_value=ClaudeProvider(),
            ),
            patch(
                "ccgram.transcript_reader.registry.is_valid",
                return_value=True,
            ),
            patch(
                "ccgram.transcript_reader.registry.get",
                return_value=CodexProvider(),
            ),
        ):
            await monitor._process_session_file(
                "sess-stale", session_file, new_messages, window_id="@42"
            )

        assert len(new_messages) == 1
        assert new_messages[0].text == "hello codex"

    async def test_check_for_updates_maps_session_to_window(self, tmp_path) -> None:
        """check_for_updates passes correct window_id to _process_session_file."""
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )

        captured_window_ids = []
        original = monitor._process_session_file

        async def spy(session_id, file_path, new_messages, window_id=""):
            captured_window_ids.append(window_id)
            return await original(
                session_id, file_path, new_messages, window_id=window_id
            )

        monitor._process_session_file = spy

        current_map = {
            "@7": {
                "session_id": "sess-map",
                "cwd": "/proj",
                "window_name": "proj",
                "transcript_path": str(session_file),
            },
        }
        await monitor.check_for_updates(current_map)
        assert "@7" in captured_window_ids


class TestReadNewLines:
    async def test_shrunken_file_resumes_from_eof_no_replay(self, tmp_path) -> None:
        """Shrunken/replaced transcripts must not be replayed (2026-08-17
        flood incident: replaying history flooded Telegram and starved
        every other topic)."""
        session_file = tmp_path / "test.jsonl"
        session_file.write_text(
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
        )

        monitor = SessionMonitor(
            projects_path=tmp_path,
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="t1",
            file_path=str(session_file),
            last_byte_offset=99999,
        )
        entries = await monitor._read_new_lines(tracked, session_file)
        assert tracked.parsed_offset == session_file.stat().st_size
        assert entries == []

    async def test_incremental_read_from_offset(self, tmp_path) -> None:
        session_file = tmp_path / "test.jsonl"
        line1 = '{"type":"assistant","message":{"content":[{"type":"text","text":"first"}]}}\n'
        line2 = '{"type":"assistant","message":{"content":[{"type":"text","text":"second"}]}}\n'
        session_file.write_text(line1 + line2)

        monitor = SessionMonitor(
            projects_path=tmp_path,
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="t1",
            file_path=str(session_file),
            last_byte_offset=len(line1.encode()),
        )
        entries = await monitor._read_new_lines(tracked, session_file)
        assert len(entries) == 1

    async def test_partial_line_stops_reading(self, tmp_path) -> None:
        session_file = tmp_path / "test.jsonl"
        good_line = (
            '{"type":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}\n'
        )
        session_file.write_text(good_line + '{"type":"ass')

        monitor = SessionMonitor(
            projects_path=tmp_path,
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="t1", file_path=str(session_file), last_byte_offset=0
        )
        entries = await monitor._read_new_lines(tracked, session_file)
        assert len(entries) == 1
        assert tracked.parsed_offset == len(good_line.encode())


class TestCorruptedOffset:
    async def test_corrupted_offset_recovers(self, tmp_path) -> None:
        session_file = tmp_path / "test.jsonl"
        line1 = '{"type":"assistant","message":{"content":[{"type":"text","text":"first"}]}}\n'
        line2 = '{"type":"assistant","message":{"content":[{"type":"text","text":"second"}]}}\n'
        session_file.write_text(line1 + line2)

        monitor = SessionMonitor(
            projects_path=tmp_path,
            state_file=tmp_path / "ms.json",
        )
        # Set offset to mid-line1 (corrupted)
        tracked = TrackedSession(
            session_id="t1",
            file_path=str(session_file),
            last_byte_offset=10,
        )
        entries = await monitor._read_new_lines(tracked, session_file)
        # Should recover: skip rest of line1, read line2
        assert len(entries) == 1
        text = (
            entries[0].data.get("message", {}).get("content", [{}])[0].get("text", "")
        )
        assert text == "second"


class TestCheckForUpdates:
    async def test_new_session_initializes_to_eof_fallback(self, tmp_path) -> None:
        """Fallback path: entries without transcript_path use scan_projects."""
        projects_path = tmp_path / "projects"
        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        resolved = str(work_dir.resolve())

        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-new.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": resolved,
            "entries": [
                {
                    "sessionId": "sess-new",
                    "fullPath": str(session_file),
                    "projectPath": resolved,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {"session_id": "sess-new", "cwd": resolved, "window_name": "proj"},
        }
        with patch.object(
            monitor,
            "_get_active_cwds",
            spec=True,
            new_callable=AsyncMock,
            return_value={resolved},
        ):
            msgs = await monitor.check_for_updates(current_map)

        assert msgs == []
        tracked = monitor.state.get_session("sess-new")
        assert tracked is not None
        # New sessions seed the delivered watermark directly at EOF.
        assert tracked.last_byte_offset == session_file.stat().st_size

    async def test_new_session_initializes_to_eof_direct(self, tmp_path) -> None:
        """Primary path: entries with transcript_path are read directly."""
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {
                "session_id": "sess-direct",
                "cwd": "/proj",
                "window_name": "proj",
                "transcript_path": str(session_file),
            },
        }
        msgs = await monitor.check_for_updates(current_map)

        assert msgs == []
        tracked = monitor.state.get_session("sess-direct")
        assert tracked is not None
        # New sessions seed the delivered watermark directly at EOF.
        assert tracked.last_byte_offset == session_file.stat().st_size

    async def test_pending_direct_session_preserves_first_reply(self, tmp_path) -> None:
        """A hook can publish Pi's exact path before Pi creates the file."""
        session_file = tmp_path / "future-pi.jsonl"
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {
                "session_id": "sess-pi",
                "cwd": "/proj",
                "window_name": "proj",
                "transcript_path": str(session_file),
                "replay_from_start": True,
            },
        }

        with patch(
            "ccgram.session_monitor.acknowledge_replay_from_start",
            return_value=True,
        ) as acknowledge:
            assert await monitor.check_for_updates(current_map) == []
        acknowledge.assert_called_once_with("@0", "sess-pi")
        current_map["@0"].pop("replay_from_start")
        tracked = monitor.state.get_session("sess-pi")
        assert tracked is not None
        assert tracked.last_byte_offset == 0
        assert tracked.file_path == str(session_file)

        session_file.write_text(
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "first reply"}],
                    },
                }
            )
            + "\n"
        )
        from ccgram.providers.pi import PiProvider

        with patch(
            "ccgram.transcript_reader.get_provider_for_window",
            return_value=PiProvider(),
        ):
            messages = await monitor.check_for_updates(current_map)

        assert [message.text for message in messages] == ["first reply"]

    async def test_replay_marker_reads_existing_pi_file_from_start(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "pi.jsonl"
        session_file.write_text(
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "fast first reply"}],
                    },
                }
            )
            + "\n"
        )
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {
                "session_id": "sess-pi",
                "cwd": "/proj",
                "window_name": "proj",
                "transcript_path": str(session_file),
                "replay_from_start": True,
            },
        }
        from ccgram.providers.pi import PiProvider

        with (
            patch(
                "ccgram.transcript_reader.get_provider_for_window",
                return_value=PiProvider(),
            ),
            patch(
                "ccgram.session_monitor.acknowledge_replay_from_start",
                return_value=True,
            ) as acknowledge,
        ):
            messages = await monitor.check_for_updates(current_map)

        acknowledge.assert_called_once_with("@0", "sess-pi")
        assert [message.text for message in messages] == ["fast first reply"]
        saved = json.loads((tmp_path / "ms.json").read_text())
        assert saved["tracked_sessions"]["sess-pi"]["last_byte_offset"] == 0

    async def test_unchanged_mtime_skips_read(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        resolved = str(work_dir.resolve())

        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": resolved,
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(session_file),
                    "projectPath": resolved,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        file_size = session_file.stat().st_size
        tracked = TrackedSession(
            session_id="sess-1",
            file_path=str(session_file),
            last_byte_offset=file_size,
        )
        monitor.state.update_session(tracked)
        monitor._file_mtimes["sess-1"] = session_file.stat().st_mtime
        monitor._transcript_reader._protocol_parser_states["sess-1"] = (
            "claude",
            TranscriptParserState(),
        )

        current_map = {
            "@0": {"session_id": "sess-1", "cwd": resolved, "window_name": "proj"},
        }
        with (
            patch.object(
                monitor,
                "_get_active_cwds",
                spec=True,
                new_callable=AsyncMock,
                return_value={resolved},
            ),
            patch.object(
                monitor._transcript_reader,
                "_read_new_lines",
                spec=True,
                new_callable=AsyncMock,
            ) as mock_read,
        ):
            await monitor.check_for_updates(current_map)

        mock_read.assert_not_called()

    async def test_same_mtime_but_larger_size_triggers_read(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        projects_path.mkdir()

        session_file = tmp_path / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')
        original_mtime = session_file.stat().st_mtime

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-1",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)
        monitor._file_mtimes["sess-1"] = original_mtime

        # Append content without changing mtime (simulate sub-second write)
        with open(session_file, "a") as f:
            f.write(
                '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
            )
        os.utime(session_file, (original_mtime, original_mtime))

        current_map = {
            "@0": {
                "session_id": "sess-1",
                "cwd": str(tmp_path),
                "window_name": "proj",
                "transcript_path": str(session_file),
            },
        }
        with patch.object(
            monitor._transcript_reader,
            "_read_new_lines",
            spec=True,
            new_callable=AsyncMock,
        ) as mock_read:
            await monitor.check_for_updates(current_map)

        mock_read.assert_called_once()

    async def test_direct_path_reads_new_content(self, tmp_path) -> None:
        """Primary path reads new content from transcript_path."""
        session_file = tmp_path / "transcript.jsonl"
        line = '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}\n'
        session_file.write_text(line)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        # Pre-track at offset 0 so it reads the content
        tracked = TrackedSession(
            session_id="sess-d",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        current_map = {
            "@1": {
                "session_id": "sess-d",
                "cwd": "/proj",
                "window_name": "proj",
                "transcript_path": str(session_file),
            },
        }
        msgs = await monitor.check_for_updates(current_map)

        assert len(msgs) == 1
        assert msgs[0].session_id == "sess-d"
        assert "hello" in msgs[0].text


class TestCheckForUpdatesExceptionResilience:
    async def test_error_in_one_session_does_not_block_others(self, tmp_path) -> None:
        good_file = tmp_path / "good.jsonl"
        good_file.write_text(
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
        )
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {
                "session_id": "sess-bad",
                "cwd": "/proj",
                "window_name": "bad",
                "transcript_path": str(bad_file),
            },
            "@1": {
                "session_id": "sess-good",
                "cwd": "/proj2",
                "window_name": "good",
                "transcript_path": str(good_file),
            },
        }

        original = monitor._process_session_file

        async def _blow_up(session_id, *args, **kwargs):
            if session_id == "sess-bad":
                raise TypeError("simulated provider bug")
            return await original(session_id, *args, **kwargs)

        with patch.object(monitor, "_process_session_file", side_effect=_blow_up):
            await monitor.check_for_updates(current_map)

        assert monitor.state.get_session("sess-good") is not None
        assert monitor.state.get_session("sess-bad") is None

    async def test_error_in_direct_session_still_saves_state(self, tmp_path) -> None:
        good_file = tmp_path / "good.jsonl"
        good_file.write_text('{"type":"summary"}\n')
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {
                "session_id": "sess-good",
                "cwd": "/proj",
                "window_name": "good",
                "transcript_path": str(good_file),
            },
            "@1": {
                "session_id": "sess-bad",
                "cwd": "/proj2",
                "window_name": "bad",
                "transcript_path": str(bad_file),
            },
        }

        original = monitor._process_session_file

        async def _blow_up(session_id, *args, **kwargs):
            if session_id == "sess-bad":
                raise ValueError("corrupt transcript")
            return await original(session_id, *args, **kwargs)

        with patch.object(monitor, "_process_session_file", side_effect=_blow_up):
            await monitor.check_for_updates(current_map)

        assert monitor.state.get_session("sess-good") is not None


class TestActivityTracking:
    def test_get_last_activity_returns_none_for_unknown(
        self, monitor: SessionMonitor
    ) -> None:
        assert monitor.get_last_activity("unknown-session") is None

    async def test_get_last_activity_updated_after_new_entries(self, tmp_path) -> None:
        session_file = tmp_path / "transcript.jsonl"
        line = (
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
        )
        session_file.write_text(line)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-act",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        new_messages: list = []
        await monitor._process_session_file(
            "sess-act", session_file, new_messages, window_id="@1"
        )
        last = monitor.get_last_activity("sess-act")
        assert last is not None
        assert last > 0

    async def test_get_last_activity_not_updated_without_entries(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text("")

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-empty",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        new_messages: list = []
        await monitor._process_session_file(
            "sess-empty", session_file, new_messages, window_id="@1"
        )
        assert monitor.get_last_activity("sess-empty") is None


class TestScanProjects:
    def test_scan_projects_sync_reads_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        work_dir = tmp_path / "myproject"
        work_dir.mkdir()
        resolved_cwd = str(work_dir.resolve())

        proj_dir = projects_path / "-tmp-myproject"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-123.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": resolved_cwd,
            "entries": [
                {
                    "sessionId": "sess-123",
                    "fullPath": str(session_file),
                    "projectPath": resolved_cwd,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        active_cwds = {resolved_cwd}
        result = monitor._scan_projects_sync(active_cwds)

        assert len(result) == 1
        assert result[0].session_id == "sess-123"

    def test_scan_projects_sync_picks_up_unindexed_jsonl(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        work_dir = tmp_path / "myproject"
        work_dir.mkdir()
        resolved_cwd = str(work_dir.resolve())

        proj_dir = projects_path / "-tmp-myproject"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "orphan-sess.jsonl"
        jsonl.write_text(json.dumps({"cwd": resolved_cwd}) + "\n")

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        active_cwds = {resolved_cwd}
        result = monitor._scan_projects_sync(active_cwds)

        assert len(result) == 1
        assert result[0].session_id == "orphan-sess"

    def test_scan_projects_sync_filters_by_active_cwds(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        resolved_other = str(other_dir.resolve())

        proj_dir = projects_path / "-tmp-other"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-456.jsonl"
        session_file.write_text('{"type":"summary"}\n')
        index = {
            "originalPath": resolved_other,
            "entries": [
                {
                    "sessionId": "sess-456",
                    "fullPath": str(session_file),
                    "projectPath": resolved_other,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        active_cwds = {str((tmp_path / "myproject").resolve())}
        result = monitor._scan_projects_sync(active_cwds)

        assert len(result) == 0

    def test_scan_projects_sync_skips_unindexed_jsonl_without_cwd(
        self, tmp_path
    ) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-my-project"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "orphan.jsonl"
        jsonl.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        # active_cwds value is irrelevant — the skip happens before cwd matching
        active_cwds = {"anything"}
        result = monitor._scan_projects_sync(active_cwds)
        assert result == []

    def test_scan_projects_sync_skips_missing_dir(self, tmp_path) -> None:
        monitor = SessionMonitor(
            projects_path=tmp_path / "nonexistent",
            state_file=tmp_path / "ms.json",
        )
        result = monitor._scan_projects_sync({"/tmp/something"})
        assert result == []


class TestGeminiTranscriptReading:
    """Read Gemini JSONL into the same raw source-entry contract as other providers."""

    _GEMINI_META = {
        "sessionId": "g1",
        "projectHash": "h1",
    }
    _GEMINI_MESSAGES = [
        {"type": "user", "content": [{"text": "hello"}]},
        {"type": "gemini", "content": [{"text": "hi there"}]},
        {"type": "user", "content": [{"text": "thanks"}]},
    ]

    def _write_jsonl(self, path, meta, messages):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    async def test_gemini_reads_jsonl_incrementally(self, tmp_path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        self._write_jsonl(transcript, self._GEMINI_META, self._GEMINI_MESSAGES)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="g1",
            file_path=str(transcript),
            last_byte_offset=0,
        )

        with patch(
            "ccgram.transcript_reader.get_provider_for_window",
            return_value=_make_gemini_provider(),
        ):
            # First read: gets everything
            entries = await monitor._read_new_lines(tracked, transcript, window_id="@5")
            assert len(entries) == 4  # meta + 3 messages
            assert entries[0].data["sessionId"] == "g1"
            assert entries[1].data["type"] == "user"
            assert [entry.offset for entry in entries] == [
                0,
                len((json.dumps(self._GEMINI_META) + "\n").encode()),
                len((json.dumps(self._GEMINI_META) + "\n").encode())
                + len((json.dumps(self._GEMINI_MESSAGES[0]) + "\n").encode()),
                len((json.dumps(self._GEMINI_META) + "\n").encode())
                + sum(
                    len((json.dumps(message) + "\n").encode())
                    for message in self._GEMINI_MESSAGES[:2]
                ),
            ]

            # Second read: nothing new
            entries = await monitor._read_new_lines(tracked, transcript, window_id="@5")
            assert len(entries) == 0

            # Third read: append a message
            with open(transcript, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"type": "gemini", "content": [{"text": "bye"}]}) + "\n"
                )

            entries = await monitor._read_new_lines(tracked, transcript, window_id="@5")
            assert len(entries) == 1
            assert entries[0].data["type"] == "gemini"

    async def test_gemini_end_to_end_process_session(self, tmp_path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        self._write_jsonl(transcript, self._GEMINI_META, self._GEMINI_MESSAGES)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        # We start tracking with offset pointing at the end of the file
        # (simulating a session that was already active at startup)
        st = transcript.stat()
        tracked = TrackedSession(
            session_id="g1",
            file_path=str(transcript),
            last_byte_offset=st.st_size,
        )
        monitor.state.update_session(tracked)

        # Append new message
        with open(transcript, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"type": "gemini", "content": [{"text": "new!"}]}) + "\n"
            )

        new_messages: list = []
        with patch(
            "ccgram.transcript_reader.get_provider_for_window",
            return_value=_make_gemini_provider(),
        ):
            await monitor._process_session_file(
                "g1", transcript, new_messages, window_id="@5"
            )

        assert len(new_messages) == 1
        assert new_messages[0].text == "new!"
        assert new_messages[0].role == "assistant"


def _make_gemini_provider():
    from ccgram.providers.gemini import GeminiProvider

    return GeminiProvider()


class TestAdoptionRespectsBackendEligibility:
    """Every path that can create a topic honours the backend's verdict.

    Three sites fire ``NewWindowEvent``: discovery, the session-map delta and
    the known-unbound self-heal. Only the first consults
    ``is_agent_topic_window``, so a window the backend excluded could still be
    adopted through the other two the moment a globally installed agent hook
    wrote a session_map entry for it.
    """

    @pytest.fixture
    def wired(self, monkeypatch) -> None:
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
        SessionManager()

    async def test_known_unbound_path_skips_an_ineligible_window(
        self, monitor: SessionMonitor, wired
    ) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        current_map = {
            "OUT-OF-SCOPE": {"session_id": "S1", "cwd": "/repo", "window_name": "x"}
        }

        await monitor._emit_known_unbound_window_events(current_map, set())

        cb.assert_not_called()

    async def test_known_unbound_path_still_surfaces_an_eligible_window(
        self, monitor: SessionMonitor, wired
    ) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        current_map = {
            "IN-SCOPE": {"session_id": "S1", "cwd": "/repo", "window_name": "x"}
        }

        await monitor._emit_known_unbound_window_events(current_map, {"IN-SCOPE"})

        cb.assert_called_once()

    async def test_delta_path_refuses_an_out_of_scope_hook_entry(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """Hooks are installed globally, so an agent started in any workspace
        writes an entry. A hook entry is not permission to adopt."""
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        raw = {
            "agterm:OUT-OF-SCOPE": {
                "session_id": "S1",
                "cwd": "/repo",
                "window_name": "elsewhere",
            }
        }

        await monitor._detect_and_cleanup_changes(raw, adoptable_window_ids=set())

        cb.assert_not_called()

    async def test_delta_path_adopts_nothing_when_no_listing_was_available(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """None is not "no restriction": it means ccgram has no verdict.

        Inferring one from the hook entry is what the fail-open cache did.
        """
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        raw = {"agterm:ANY": {"session_id": "S1", "cwd": "/repo", "window_name": "x"}}

        await monitor._detect_and_cleanup_changes(raw, adoptable_window_ids=None)

        cb.assert_not_called()


class TestTheLoopPassesTheEligibleSubset:
    """Behavioural, not source inspection.

    The unit tests above prove the delta path obeys whatever verdict it is
    given. This proves the loop gives it the *eligible* subset rather than
    every live window, which is the mistake the parameter exists to prevent:
    handing over the complete listing would adopt exactly the windows the
    backend refused.
    """

    async def test_the_delta_path_receives_only_eligible_windows(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        received: list[set[str] | None] = []

        async def spy(raw=None, *, adoptable_window_ids):
            received.append(adoptable_window_ids)
            monitor._running = False  # one cycle is enough
            return {}

        monkeypatch.setattr(
            "ccgram.session_monitor.list_windows_for_reconciliation",
            AsyncMock(
                return_value=[
                    _winref("KEEP", "claude"),
                    _winref("REFUSED", "claude", eligible=False),
                ]
            ),
        )
        monkeypatch.setattr(
            "ccgram.session_monitor.read_session_map_raw", AsyncMock(return_value={})
        )
        monkeypatch.setattr(monitor, "_read_hook_events", AsyncMock())
        monkeypatch.setattr(monitor, "_cleanup_all_stale_sessions", AsyncMock())
        monkeypatch.setattr(monitor, "_detect_and_cleanup_changes", spy)
        monitor._running = True

        await monitor._monitor_loop()

        assert received == [{"KEEP"}], (
            "the loop must hand over the eligible subset, not every live window"
        )


class TestAdoptionIdentityFoldsCaseInTheMonitor:
    """Session-map keys are written by the hook; the adoption set comes from
    the backend's listing. On agterm those can spell one UUID differently, and
    comparing raw means the window is never adopted on any cycle.
    """

    def test_the_lookup_folds_case(self) -> None:
        from ccgram.session_monitor import _adoption_lookup

        assert _adoption_lookup({"9F1C2D3E-4A5B"}) == {"9f1c2d3e-4a5b"}

    def test_a_case_variant_key_is_adoptable(self) -> None:
        from ccgram.multiplexer.base import canonical_window_id
        from ccgram.session_monitor import _adoption_lookup

        lookup = _adoption_lookup({"9F1C2D3E-4A5B"})

        assert canonical_window_id("9f1c2d3e-4a5b") in lookup
        assert canonical_window_id("other-id") not in lookup


class TestActiveCwdsUseTheCompleteListing:
    """Transcript discovery is gated on the live cwd set.

    A window missing from that set has no discoverable history at all, so it
    must be built from every live window, not only the ones a backend would
    auto-adopt. On agterm that means a session outside CCGRAM_AGTERM_WORKSPACES
    keeps its /restore and history pager working.
    """

    @staticmethod
    def _reader():
        from ccgram.transcript_reader import TranscriptReader

        return TranscriptReader.__new__(TranscriptReader)

    async def test_includes_a_window_the_ui_listing_hides(self, monkeypatch) -> None:
        from ccgram.multiplexer.base import WindowRef

        excluded = WindowRef(
            window_id="@4", window_name="_paused", cwd="/repo", topic_eligible=False
        )

        async def _complete():
            return [excluded]

        monkeypatch.setattr(
            "ccgram.multiplexer.reconciliation.list_windows_for_reconciliation",
            _complete,
        )

        assert await self._reader()._get_active_cwds() == {"/repo"}

    async def test_ignores_empty_window_cwd(self, monkeypatch) -> None:
        from ccgram.multiplexer.base import WindowRef

        async def _complete():
            return [WindowRef(window_id="@4", window_name="unknown", cwd="")]

        monkeypatch.setattr(
            "ccgram.multiplexer.reconciliation.list_windows_for_reconciliation",
            _complete,
        )

        assert await self._reader()._get_active_cwds() == set()

    async def test_unconfirmed_listing_yields_no_cwds(self, monkeypatch) -> None:
        async def _unavailable():
            return None

        monkeypatch.setattr(
            "ccgram.multiplexer.reconciliation.list_windows_for_reconciliation",
            _unavailable,
        )

        assert await self._reader()._get_active_cwds() == set()


async def test_replay_cap_uses_one_stable_barrier_and_does_not_loop_on_appends(
    tmp_path, monkeypatch
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_complete_jsonl_record(200))
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=10)
    )
    current_map = {
        "@0": {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "provider_name": "claude",
        }
    }
    monitor._last_session_map = current_map
    monkeypatch.setenv("CCGRAM_REPLAY_CAP_MB", "0.0001")
    purge = AsyncMock(return_value=3)
    notice_attempts = []

    async def notice(_intent) -> None:
        notice_attempts.append("s1")
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    process = AsyncMock()
    with (
        patch(
            "ccgram.session_monitor.identity_state.accepts_provider_observation",
            return_value=True,
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
        patch.object(monitor, "_process_session_file", process),
    ):
        await monitor.check_for_updates(current_map)
        assert set(monitor.state.pending_skips) == {"s1"}
        intent = monitor.state.pending_skips["s1"]
        assert intent.snapshot_offset == 200
        assert purge.await_count == 1
        assert notice_attempts == ["s1"]
        process.assert_not_awaited()

        # Appends after the frozen EOF do not mutate the persisted barrier or
        # create a second notice while the first notice is still unacknowledged.
        with transcript.open("ab") as source:
            source.write(b"y" * 40)
        await monitor.check_for_updates(current_map)
        assert monitor.state.pending_skips["s1"].snapshot_offset == 200
        assert purge.await_count == 1
        assert notice_attempts == ["s1"]
        process.assert_not_awaited()

        monitor._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
        await monitor.commit_delivered_watermarks()
        assert monitor.state.tracked_sessions["s1"].last_byte_offset == 200
        assert transcript.read_bytes() == _complete_jsonl_record(200) + b"y" * 40

        # The post-snapshot append is below the cap and resumes normally; the
        # already acknowledged range cannot trigger another skip.
        await monitor.check_for_updates(current_map)
        assert "s1" not in monitor.state.pending_skips
        assert purge.await_count == 1
        assert notice_attempts == ["s1"]
        process.assert_awaited_once()


async def test_replay_cap_defers_if_eof_changes_during_source_reconciliation(
    tmp_path, monkeypatch
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_complete_jsonl_record(200))
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=10)
    )
    current_map = {
        "@0": {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "provider_name": "claude",
        }
    }
    monitor._last_session_map = current_map
    monkeypatch.setenv("CCGRAM_REPLAY_CAP_MB", "0.0001")
    purge = AsyncMock(return_value=3)
    monitor.set_skip_callbacks(
        purge=purge, notice=AsyncMock(), validate=lambda _intent: True
    )

    async def append_during_reconciliation(*_args, **_kwargs):
        with transcript.open("ab") as source:
            source.write(_complete_jsonl_record(16))
        return False, False

    process = AsyncMock()
    with (
        patch(
            "ccgram.session_monitor.identity_state.accepts_provider_observation",
            return_value=True,
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
        patch.object(
            monitor._transcript_reader,
            "_prepare_observed_generation",
            side_effect=append_during_reconciliation,
        ),
        patch.object(monitor, "_process_session_file", process),
    ):
        await monitor.check_for_updates(current_map)

    assert not monitor.state.pending_skips
    purge.assert_not_awaited()
    process.assert_awaited_once()
    assert transcript.read_bytes() == _complete_jsonl_record(
        200
    ) + _complete_jsonl_record(16)


async def test_replay_cap_and_manual_skip_fail_closed_for_shared_recipients(
    tmp_path, monkeypatch
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_complete_jsonl_record(200))
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=10)
    )
    current_map = {
        window_id: {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "provider_name": "claude",
        }
        for window_id in ("@0", "@1")
    }
    monitor._last_session_map = current_map
    purge = AsyncMock(return_value=3)
    notice = AsyncMock()
    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    monkeypatch.setenv("CCGRAM_REPLAY_CAP_MB", "0.0001")
    process = AsyncMock()
    with (
        patch(
            "ccgram.session_monitor.identity_state.accepts_provider_observation",
            return_value=True,
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0"), (2, 100, 7, "@1")],
        ),
        patch.object(monitor, "_process_session_file", process),
        # Isolate the request's early shared-recipient refusal from later gates.
        patch.object(monitor, "_skip_is_current", return_value=True),
    ):
        assert await monitor.request_backlog_skip(1, "@0", 4, 99) is None
        await monitor.check_for_updates(current_map)
        assert monitor.has_shared_backlog_source("@0")

    assert not monitor.state.pending_skips
    purge.assert_not_awaited()
    notice.assert_not_awaited()
    process.assert_awaited()

    # The automatic recipient guard is distinct from the manual request guard.
    request = AsyncMock()
    monitor.request_backlog_skip = request
    with (
        patch(
            "ccgram.session_monitor.identity_state.accepts_provider_observation",
            return_value=True,
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0"), (2, 100, 7, "@1")],
        ),
        patch.object(monitor, "_process_session_file", process),
    ):
        await monitor._maybe_skip_replay_cap(
            "s1", transcript, "@0", current_map["@0"], 1
        )
    request.assert_not_awaited()


async def test_replay_cap_is_disabled_by_default(tmp_path, monkeypatch) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_complete_jsonl_record(200))
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=10)
    )
    current_map = {
        "@0": {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "provider_name": "claude",
        }
    }
    monitor._last_session_map = current_map
    monkeypatch.delenv("CCGRAM_REPLAY_CAP_MB", raising=False)
    purge = AsyncMock(return_value=3)
    monitor.set_skip_callbacks(
        purge=purge, notice=AsyncMock(), validate=lambda _intent: True
    )
    process = AsyncMock()
    with (
        patch.object(monitor, "_process_session_file", process),
        patch.object(
            monitor, "_maybe_skip_replay_cap", new_callable=AsyncMock
        ) as maybe_skip,
    ):
        await monitor.check_for_updates(current_map)

    assert not monitor.state.pending_skips
    purge.assert_not_awaited()
    process.assert_awaited_once()
    maybe_skip.assert_not_awaited()


async def test_expired_auto_skip_waits_for_new_bytes_across_restart(
    tmp_path, monkeypatch
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_complete_jsonl_record(200))
    state_file = tmp_path / "monitor.json"
    monitor = SessionMonitor(projects_path=tmp_path / "projects", state_file=state_file)
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10, parsed_offset=10)
    )
    current_map = {
        "@0": {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "provider_name": "claude",
        }
    }
    monitor._last_session_map = current_map
    monkeypatch.setenv("CCGRAM_REPLAY_CAP_MB", "0.0001")
    monkeypatch.setattr("ccgram.session_monitor.config.skip_barrier_deadline_s", 60.0)
    purge = AsyncMock(return_value=3)
    notice_attempts = []

    async def notice(_intent) -> None:
        notice_attempts.append("s1")
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    with (
        patch(
            "ccgram.session_monitor.identity_state.accepts_provider_observation",
            return_value=True,
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        await monitor.check_for_updates(current_map)
        intent = monitor.state.pending_skips["s1"]
        assert intent.automatic is True
        intent.created_at = time.time() - 120.0
        monitor._expire_aged_skip_barriers()
        assert monitor.state.tracked_sessions["s1"].replay_cap_retry_after == 200

        # The persisted retry floor prevents an immediate repeat after restart
        # while the same EOF still exceeds the cap from the delivered watermark.
        restarted = SessionMonitor(
            projects_path=tmp_path / "projects", state_file=state_file
        )
        restarted._last_session_map = current_map
        restarted.set_skip_callbacks(
            purge=purge, notice=notice, validate=lambda _intent: True
        )
        process = AsyncMock()
        with patch.object(restarted, "_process_session_file", process):
            await restarted.check_for_updates(current_map)
        assert not restarted.state.pending_skips
        assert restarted.state.tracked_sessions["s1"].replay_cap_retry_after == 200
        assert purge.await_count == 1
        process.assert_awaited_once()

        # A new cap-sized append permits a fresh barrier; repeated polls on
        # the unchanged snapshot do not create a retry loop.
        with transcript.open("ab") as source:
            source.write(_complete_jsonl_record(105))
        await restarted.check_for_updates(current_map)
        assert restarted.state.pending_skips["s1"].snapshot_offset == 305
        assert purge.await_count == 2
        assert notice_attempts == ["s1", "s1"]


@pytest.mark.parametrize(
    "mutation",
    [
        "replacement",
        "truncation",
        "truncate-and-regrow",
        "same-size-overwrite",
        "append",
    ],
)
async def test_acknowledged_notice_does_not_advance_after_source_changes(
    tmp_path, mutation: str
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_complete_jsonl_record(80))
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    tracked = TrackedSession(
        "s1", str(transcript), last_byte_offset=10, parsed_offset=80
    )
    monitor.state.update_session(tracked)
    bindings = [(1, 99, 4, "@0")]

    async def purge(_intent) -> int:
        return 2

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    with (
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            side_effect=lambda _window_id: "s1",
        ),
        patch(
            "ccgram.session_monitor.session_lifecycle._last_session_map",
            {"@0": {"session_id": "s1", "transcript_path": str(transcript)}},
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            side_effect=lambda: iter(bindings),
        ),
    ):
        intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
        assert intent is not None
        if mutation == "replacement":
            replacement = tmp_path / "replacement.jsonl"
            replacement.write_bytes(b"replacement data" * 8)
            os.replace(replacement, transcript)
        elif mutation == "truncation":
            transcript.write_bytes(b"short")
        elif mutation == "truncate-and-regrow":
            with transcript.open("r+b") as source:
                source.truncate(0)
                source.write(b"y" * 80)
        elif mutation == "same-size-overwrite":
            with transcript.open("r+b") as source:
                source.write(b"y" * 80)
        else:
            with transcript.open("ab") as source:
                source.write(b"z" * 20)

        monitor._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
        await monitor.commit_delivered_watermarks()

    if mutation == "append":
        assert tracked.last_byte_offset == 80
        assert tracked.parsed_offset == 80
    else:
        assert tracked.last_byte_offset == 10
        assert tracked.parsed_offset == 10
    assert "s1" not in monitor.state.pending_skips


@pytest.mark.parametrize("bind_during_purge", [True, False])
async def test_acknowledged_skip_cancels_when_second_recipient_binds(
    tmp_path, bind_during_purge: bool
) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(_complete_jsonl_record(80))
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    tracked = TrackedSession(
        "s1", str(transcript), last_byte_offset=10, parsed_offset=80
    )
    monitor.state.update_session(tracked)
    source_receipt = new_delivery_receipt(checkpoint=80)
    source_receipt.track()
    monitor._delivery_receipts["s1"] = [source_receipt]
    monitor._transcript_reader._pending_batches.append(_transcript_batch("s1", 80))
    bindings = [(1, 99, 4, "@0")]

    async def purge(_intent) -> int:
        if bind_during_purge:
            bindings.append((2, 100, 7, "@1"))
        return 2

    notice_attempts = 0

    async def notice(_intent) -> None:
        nonlocal notice_attempts
        notice_attempts += 1
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    with (
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            side_effect=lambda _window_id: "s1",
        ),
        patch(
            "ccgram.session_monitor.session_lifecycle._last_session_map",
            {"@0": {"session_id": "s1", "transcript_path": str(transcript)}},
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            side_effect=lambda: iter(bindings),
        ),
    ):
        intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
        if bind_during_purge:
            assert intent is None
        else:
            assert intent is not None
            bindings.append((2, 100, 7, "@1"))
            monitor._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
            await monitor.commit_delivered_watermarks()

    assert notice_attempts == (0 if bind_during_purge else 1)
    assert tracked.last_byte_offset == 10
    assert tracked.parsed_offset == 10
    assert "s1" not in monitor.state.pending_skips
    assert "s1" not in monitor._delivery_receipts
    assert not any(
        batch.session_id == "s1"
        for batch in monitor._transcript_reader._pending_batches
    )


async def test_skip_does_not_freeze_inside_an_incremental_jsonl_record(
    tmp_path,
) -> None:
    first_record = {
        "type": "assistant",
        "uuid": "first",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "first"}],
        },
    }
    second_record = {
        "type": "assistant",
        "uuid": "second",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "second"}],
        },
    }
    first_line = json.dumps(first_record).encode() + b"\n"
    second_line = json.dumps(second_record).encode() + b"\n"
    partial_second = second_line[: len(second_line) // 2]
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(first_line + partial_second)
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=0, parsed_offset=0)
    )
    monitor._last_session_map = {
        "@0": {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "provider_name": "claude",
        }
    }
    purge = AsyncMock(return_value=1)

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    with (
        patch(
            "ccgram.transcript_reader.get_provider_for_window",
            return_value=ClaudeProvider(),
        ),
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        intent = await monitor.request_backlog_skip(1, "@0", 4, 99)
        if intent is not None:
            monitor._skip_notice_receipts["s1"].settle(DeliveryOutcome.DELIVERED)
            await monitor.commit_delivered_watermarks()
        with transcript.open("ab") as source:
            source.write(second_line[len(partial_second) :])

    messages: list[NewMessage] = []
    with (
        patch(
            "ccgram.transcript_reader.get_provider_for_window",
            return_value=ClaudeProvider(),
        ),
        patch(
            "ccgram.transcript_reader.identity_state.accepts_provider_observation",
            return_value=True,
        ),
    ):
        await monitor._process_session_file("s1", transcript, messages, "@0")

    assert not monitor.state.pending_skips
    assert any(message.text == "second" for message in messages)
    if intent is None:
        purge.assert_not_awaited()
        assert not monitor._skip_notice_receipts


async def test_replay_cap_defers_until_incremental_record_is_complete(
    tmp_path, monkeypatch
) -> None:
    first_record = {
        "type": "assistant",
        "uuid": "first",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "first"}],
        },
    }
    second_record = {
        "type": "assistant",
        "uuid": "second",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "second"}],
        },
    }
    first_line = json.dumps(first_record).encode() + b"\n"
    second_line = json.dumps(second_record).encode() + b"\n"
    partial_second = second_line[: len(second_line) // 2]
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(first_line + partial_second)
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=0, parsed_offset=0)
    )
    current_map = {
        "@0": {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "provider_name": "claude",
        }
    }
    monitor._last_session_map = current_map
    monkeypatch.setenv("CCGRAM_REPLAY_CAP_MB", "0.0001")
    purge = AsyncMock(return_value=1)
    notice = AsyncMock()
    process = AsyncMock()
    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )

    with (
        patch(
            "ccgram.session_monitor.identity_state.accepts_provider_observation",
            return_value=True,
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
        patch.object(monitor, "_process_session_file", process),
    ):
        await monitor.check_for_updates(current_map)

    assert not monitor.state.pending_skips
    purge.assert_not_awaited()
    notice.assert_not_awaited()
    process.assert_awaited_once()


async def test_nonincremental_skip_allows_a_source_without_newline(
    tmp_path,
) -> None:
    transcript = tmp_path / "whole-document.json"
    transcript.write_bytes(b'{"whole":"document"}')
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=0)
    )
    monitor._last_session_map = {
        "@0": {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "provider_name": "whole",
        }
    }
    provider = SimpleNamespace(
        capabilities=SimpleNamespace(
            name="whole", supports_hook=False, supports_incremental_read=False
        )
    )
    purge = AsyncMock(return_value=1)

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    with (
        patch(
            "ccgram.transcript_reader.get_provider_for_window", return_value=provider
        ),
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        intent = await monitor.request_backlog_skip(1, "@0", 4, 99)

    assert intent is not None
    assert purge.await_count == 1


def _ready_skip_fixture(tmp_path):
    transcript = tmp_path / "source.jsonl"
    transcript.write_bytes(_complete_jsonl_record(200))
    source_stat = transcript.stat()
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    tracked = TrackedSession("s1", str(transcript), last_byte_offset=10)
    monitor.state.update_session(tracked)
    digest = _source_range_sha256(
        transcript,
        10,
        source_stat.st_size,
        device=source_stat.st_dev,
        inode=source_stat.st_ino,
    )
    assert digest is not None
    intent = BacklogSkipIntent(
        session_id="s1",
        window_id="@0",
        user_id=1,
        thread_id=4,
        chat_id=99,
        snapshot_offset=source_stat.st_size,
        range_start=10,
        created_at=time.time(),
        source_path=str(transcript),
        source_device=source_stat.st_dev,
        source_inode=source_stat.st_ino,
        source_range_sha256=digest,
    )
    monitor.state.begin_skip(intent)
    receipt = new_delivery_receipt(checkpoint=source_stat.st_size)
    receipt.track()
    receipt.settle(DeliveryOutcome.DELIVERED)
    receipt.close()
    monitor._skip_notice_receipts["s1"] = receipt
    monitor._last_session_map = {
        "@0": {"session_id": "s1", "transcript_path": str(transcript)}
    }
    monitor.set_skip_callbacks(
        purge=AsyncMock(return_value=0),
        notice=AsyncMock(),
        validate=lambda _intent: True,
    )
    return monitor, transcript, tracked, intent, receipt


async def test_backlog_freeze_hash_runs_off_loop_and_keeps_loop_responsive(
    tmp_path,
) -> None:
    transcript = tmp_path / "source.jsonl"
    transcript.write_bytes(_complete_jsonl_record(200))
    source_stat = transcript.stat()
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10)
    )
    monitor._last_session_map = {
        "@0": {"session_id": "s1", "transcript_path": str(transcript)}
    }
    digest = _source_range_sha256(
        transcript,
        10,
        source_stat.st_size,
        device=source_stat.st_dev,
        inode=source_stat.st_ino,
    )
    assert digest is not None
    entered = threading.Event()
    release = threading.Event()
    loop_tick = asyncio.Event()
    main_thread = threading.get_ident()
    original_hash = _source_range_sha256

    def slow_hash(*args, **kwargs):
        assert threading.get_ident() != main_thread
        entered.set()
        release.wait(timeout=0.5)
        return original_hash(*args, **kwargs)

    async def purge(_intent) -> int:
        return 0

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    entered_while_loop_responsive = False
    with (
        patch("ccgram.session_monitor._source_range_sha256", slow_hash),
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        task = asyncio.create_task(monitor.request_backlog_skip(1, "@0", 4, 99))
        loop.call_soon(loop_tick.set)
        await loop_tick.wait()
        elapsed = loop.time() - started
        entered_while_loop_responsive = await asyncio.to_thread(entered.wait, 1)
        release.set()
        result = await asyncio.gather(task, return_exceptions=True)

    assert entered_while_loop_responsive
    assert elapsed < 0.1
    assert isinstance(result[0], BacklogSkipIntent)


async def test_concurrent_freezes_create_only_one_pending_barrier(tmp_path) -> None:
    transcript = tmp_path / "source.jsonl"
    transcript.write_bytes(_complete_jsonl_record(200))
    source_stat = transcript.stat()
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.state.update_session(
        TrackedSession("s1", str(transcript), last_byte_offset=10)
    )
    monitor._last_session_map = {
        "@0": {"session_id": "s1", "transcript_path": str(transcript)}
    }
    rendezvous = threading.Barrier(2)
    original_hash = _source_range_sha256

    def concurrent_hash(*args, **kwargs):
        rendezvous.wait(timeout=0.5)
        return original_hash(*args, **kwargs)

    async def purge(_intent) -> int:
        return 0

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    with (
        patch("ccgram.session_monitor._source_range_sha256", concurrent_hash),
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        results = await asyncio.gather(
            monitor.request_backlog_skip(1, "@0", 4, 99),
            monitor.request_backlog_skip(1, "@0", 4, 99),
            return_exceptions=True,
        )

    assert sum(isinstance(result, BacklogSkipIntent) for result in results) == 1
    assert results.count(None) == 1
    assert set(monitor.state.pending_skips) == {"s1"}
    assert source_stat.st_size == monitor.state.pending_skips["s1"].snapshot_offset


async def test_unknown_commit_digest_defers_then_retries_without_cancelling(
    tmp_path,
) -> None:
    monitor, _transcript, tracked, intent, receipt = _ready_skip_fixture(tmp_path)
    with (
        patch(
            "ccgram.session_monitor._source_range_sha256",
            side_effect=[None, intent.source_range_sha256],
        ),
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        await monitor.commit_delivered_watermarks()
        assert monitor.state.pending_skips["s1"] is intent
        assert monitor._skip_notice_receipts["s1"] is receipt
        assert receipt.commit_ready
        assert tracked.last_byte_offset == 10

        monitor._skip_retry_at["s1"] = 0.0
        await monitor.commit_delivered_watermarks()

    assert "s1" not in monitor.state.pending_skips
    assert tracked.last_byte_offset == intent.snapshot_offset
    assert receipt.failed is False


async def test_old_async_verifier_cannot_complete_replacement_barrier(tmp_path) -> None:
    monitor, _transcript, tracked, old_intent, old_receipt = _ready_skip_fixture(
        tmp_path
    )
    entered = threading.Event()
    release = threading.Event()
    replacement = BacklogSkipIntent(
        session_id=old_intent.session_id,
        window_id=old_intent.window_id,
        user_id=old_intent.user_id,
        thread_id=old_intent.thread_id,
        chat_id=old_intent.chat_id,
        snapshot_offset=old_intent.snapshot_offset + 1,
        range_start=old_intent.range_start,
        created_at=time.time(),
        source_path=old_intent.source_path,
        source_device=old_intent.source_device,
        source_inode=old_intent.source_inode,
        source_range_sha256=old_intent.source_range_sha256,
    )
    newer_receipt = new_delivery_receipt(checkpoint=replacement.snapshot_offset)
    newer_receipt.track()
    newer_receipt.settle(DeliveryOutcome.DELIVERED)
    newer_receipt.close()

    def delayed_hash(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=1)
        return old_intent.source_range_sha256

    with (
        patch("ccgram.session_monitor._source_range_sha256", delayed_hash),
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        task = asyncio.create_task(monitor.commit_delivered_watermarks())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            monitor.state.pending_skips["s1"] = replacement
            monitor._skip_notice_receipts["s1"] = newer_receipt
        finally:
            release.set()
        await task

    assert monitor.state.pending_skips["s1"] is replacement
    assert monitor._skip_notice_receipts["s1"] is newer_receipt
    assert newer_receipt.failed is False
    assert tracked.last_byte_offset == 10
    assert old_receipt.commit_ready


@pytest.mark.parametrize("has_receipt", [False, True], ids=["restart", "queued"])
async def test_expired_skip_does_not_enqueue_notice_and_invalidates_receipt(
    tmp_path, monkeypatch, has_receipt: bool
) -> None:
    monitor, _transcript, tracked, intent, _ready_receipt = _ready_skip_fixture(
        tmp_path
    )
    intent.created_at = time.time() - 61
    monitor._skip_notice_receipts.clear()
    receipt = new_delivery_receipt(checkpoint=intent.snapshot_offset)
    receipt.track()
    receipt.close()
    assert not receipt.commit_ready
    if has_receipt:
        monitor._skip_notice_receipts["s1"] = receipt

    async def queued_notice(_intent) -> None:
        queued_receipt = get_active_delivery_receipt()
        assert queued_receipt is not None
        queued_receipt.track()

    purge = AsyncMock(return_value=0)
    notice = AsyncMock(side_effect=queued_notice)
    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    monkeypatch.setattr("ccgram.session_monitor.config.skip_barrier_deadline_s", 60.0)
    with (
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        assert monitor._skip_is_current(intent, strict=True) is True
        await monitor._advance_skip_barriers()

    assert "s1" not in monitor.state.pending_skips
    assert "s1" not in monitor._skip_notice_receipts
    assert tracked.last_byte_offset == 10
    if has_receipt:
        assert receipt.failed
    purge.assert_not_awaited()
    notice.assert_not_awaited()


def test_source_range_hash_refuses_a_rewrite_during_verification(tmp_path, monkeypatch):
    transcript = tmp_path / "source.jsonl"
    transcript.write_bytes(b"original")
    identity = transcript.stat()
    original_open = type(transcript).open

    class RewritingReader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def fileno(self):
            return self.source.fileno()

        def seek(self, *args):
            return self.source.seek(*args)

        def read(self, size):
            data = self.source.read(size)
            with original_open(transcript, "r+b") as replacement:
                replacement.write(b"changed!")
            return data

    def rewrite_on_read(path, mode="r", *args, **kwargs):
        source = original_open(path, mode, *args, **kwargs)
        return (
            RewritingReader(source) if path == transcript and mode == "rb" else source
        )

    monkeypatch.setattr(type(transcript), "open", rewrite_on_read)

    assert (
        _source_range_sha256(
            transcript,
            0,
            identity.st_size,
            device=identity.st_dev,
            inode=identity.st_ino,
        )
        is None
    )


async def test_unknown_source_after_notice_ack_keeps_the_receipt(
    tmp_path, monkeypatch
) -> None:
    monitor, transcript, tracked, intent, _old_receipt = _ready_skip_fixture(tmp_path)
    monitor._skip_notice_receipts.clear()
    monitor.state.update_skip_count("s1", 0)
    original_stat = type(transcript).stat
    source_unavailable = False
    receipts = []

    def source_stat(path, *args, **kwargs):
        if path == transcript and source_unavailable:
            raise OSError("temporarily unavailable")
        return original_stat(path, *args, **kwargs)

    async def notice(_intent) -> None:
        nonlocal source_unavailable
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()
        receipt.settle(DeliveryOutcome.DELIVERED)
        receipts.append(receipt)
        source_unavailable = True

    monitor.set_skip_callbacks(
        purge=AsyncMock(return_value=0), notice=notice, validate=lambda _intent: True
    )
    monkeypatch.setattr(type(transcript), "stat", source_stat)
    with (
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        await monitor._enqueue_pending_skip_notice(intent)
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt.failed is False
        assert receipt.commit_ready
        assert monitor._skip_notice_receipts["s1"] is receipt
        assert monitor.state.pending_skips["s1"] is intent
        assert tracked.last_byte_offset == 10

        source_unavailable = False
        monitor._skip_retry_at["s1"] = 0.0
        await monitor.commit_delivered_watermarks()

    assert tracked.last_byte_offset == intent.snapshot_offset
    assert "s1" not in monitor.state.pending_skips
    assert len(receipts) == 1


@pytest.mark.parametrize("advanced_offset", [100, 200])
async def test_freeze_pins_range_start_while_delivered_watermark_advances(
    tmp_path, advanced_offset
) -> None:
    transcript = tmp_path / "source.jsonl"
    transcript.write_bytes(_complete_jsonl_record(100) * 3)
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    tracked = TrackedSession("s1", str(transcript), last_byte_offset=100)
    monitor.state.update_session(tracked)
    monitor._last_session_map = {
        "@0": {"session_id": "s1", "transcript_path": str(transcript)}
    }
    entered = threading.Event()
    release = threading.Event()
    original_hash = _source_range_sha256

    def delayed_hash(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original_hash(*args, **kwargs)

    async def purge(_intent) -> int:
        return 0

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()
        receipt.settle(DeliveryOutcome.DELIVERED)

    monitor.set_skip_callbacks(
        purge=purge, notice=notice, validate=lambda _intent: True
    )
    with (
        patch("ccgram.session_monitor._source_range_sha256", delayed_hash),
        patch(
            "ccgram.session_monitor.session_lifecycle.resolve_session_id",
            return_value="s1",
        ),
        patch.object(
            thread_router,
            "iter_thread_bindings_with_chat",
            return_value=[(1, 99, 4, "@0")],
        ),
    ):
        request = asyncio.create_task(monitor.request_backlog_skip(1, "@0", 4, 99))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            tracked.last_byte_offset = advanced_offset
        finally:
            release.set()
        intent = await request
        assert intent is not None
        assert intent.range_start == 100
        assert intent.source_device is not None
        assert intent.source_inode is not None
        assert intent.source_range_sha256 == original_hash(
            transcript,
            intent.range_start,
            intent.snapshot_offset,
            device=intent.source_device,
            inode=intent.source_inode,
        )
        await monitor.commit_delivered_watermarks()

    assert "s1" not in monitor.state.pending_skips
    assert tracked.last_byte_offset == intent.snapshot_offset
