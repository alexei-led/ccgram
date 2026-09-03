"""Tests for SessionMonitor."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import structlog

from ccgram.monitor_state import BacklogSkipIntent, TrackedSession
from ccgram.multiplexer.base import MultiplexerCapabilities, WindowRef
from ccgram.providers.claude import ClaudeProvider
from ccgram.providers.codex import CodexProvider
from ccgram.session import SessionManager
from ccgram.session_monitor import NewMessage, NewWindowEvent, SessionMonitor
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
        [
            NewMessage("s1", "first", True),
            NewMessage("s1", "second", True),
        ]
    )
    first_msg, first_receipt = pending[0]
    task = asyncio.create_task(
        monitor._dispatch_message_with_receipt(first_msg, first_receipt)
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

    def test_prefix_commit_advances_past_settled_run(
        self, monitor: SessionMonitor
    ) -> None:
        pending = self._unsettled(300)
        pending.track()
        self._track(monitor, "s1", [self._ready(100), self._ready(200), pending])

        monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 200
        # The unsettled tail survives as receipts: a restart replays from
        # the committed fence (200), not from zero.
        assert monitor._delivery_receipts["s1"] == [pending]

    def test_failed_receipt_fences_prefix(self, monitor: SessionMonitor) -> None:
        from ccgram.delivery_contract import DeliveryOutcome

        failed = self._unsettled(200)
        failed.track()
        failed.settle(DeliveryOutcome.FAILED)
        failed.close()
        self._track(monitor, "s1", [self._ready(100), failed, self._ready(300)])

        monitor.commit_delivered_watermarks()

        # Progress before the failure is durable; the failed receipt and
        # everything after it replay (at-least-once preserved).
        assert self._offset(monitor, "s1") == 100
        assert len(monitor._delivery_receipts["s1"]) == 2

    def test_all_ready_commits_full_run_and_clears(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [self._ready(100), self._ready(200)])

        monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 200
        assert "s1" not in monitor._delivery_receipts

    def test_none_checkpoint_fences_prefix(self, monitor: SessionMonitor) -> None:
        unplaced = self._ready(0)
        unplaced.checkpoint = None
        self._track(monitor, "s1", [self._ready(100), unplaced, self._ready(300)])

        monitor.commit_delivered_watermarks()

        # A receipt without a checkpoint cannot be ordered: it blocks the
        # commit conservatively (its bytes may tie the boundary), though
        # the settled run before it is consumed; replay re-delivers it.
        assert self._offset(monitor, "s1") == 0
        remaining = monitor._delivery_receipts["s1"]
        assert len(remaining) == 2
        assert remaining[0] is unplaced

    def test_pending_tools_skip_session(self, monitor: SessionMonitor) -> None:
        self._track(monitor, "s1", [self._ready(100)])
        monitor._transcript_reader._pending_tools["s1"] = {}

        monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 0
        assert len(monitor._delivery_receipts["s1"]) == 1

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

    def test_pending_skip_fences_settled_prefix(self, monitor: SessionMonitor) -> None:
        ready = self._ready(100)
        self._track(monitor, "s1", [ready])
        self._begin_skip(monitor)

        monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 0
        assert monitor._delivery_receipts["s1"] == [ready]

    def test_delivered_skip_wins_and_discards_ordinary_receipts(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [self._ready(100)])
        self._begin_skip(monitor)
        monitor._skip_notice_receipts["s1"] = self._ready(500)

        with patch.object(monitor, "_skip_is_current", return_value=True):
            monitor.commit_delivered_watermarks()

        assert self._offset(monitor, "s1") == 500
        assert "s1" not in monitor.state.pending_skips
        assert "s1" not in monitor._delivery_receipts

    def test_shared_batch_checkpoint_defers_commit(
        self, monitor: SessionMonitor
    ) -> None:
        """Code-review P1 (history/shallow 2026-08-30): receipts of one
        parse cycle share the batch-end checkpoint, so a settled sibling
        must NOT commit it while an unsettled sibling sits below it."""
        sibling = self._unsettled(300)
        sibling.track()
        self._track(monitor, "s1", [self._ready(300), sibling])

        monitor.commit_delivered_watermarks()

        # No commit past a tie: the unsettled sibling's bytes are below
        # the shared 300. The settled receipt is still consumed; replay
        # re-delivers its message (at-least-once permits the duplicate).
        assert self._offset(monitor, "s1") == 0
        assert monitor._delivery_receipts["s1"] == [sibling]

    def test_next_batch_checkpoint_unblocks_tied_commit(
        self, monitor: SessionMonitor
    ) -> None:
        self._track(monitor, "s1", [self._ready(300), self._ready(400)])
        late = self._unsettled(400)
        late.track()
        monitor._delivery_receipts["s1"].append(late)

        monitor.commit_delivered_watermarks()

        # The first batch (300) is fully settled and the fence sits
        # strictly beyond it: 300 commits, the tied 400s do not.
        assert self._offset(monitor, "s1") == 300
        assert monitor._delivery_receipts["s1"] == [late]

    def test_receiptless_tracked_session_does_not_crash_commit(
        self, monitor: SessionMonitor
    ) -> None:
        """Code-review P1 (multi-agent 2026-08-30): a tracked session
        without receipts (the steady state after its receipts were
        consumed) must not break the batched commit for other sessions."""
        self._track(monitor, "s1", [self._ready(100)])
        self._track(monitor, "s2", [])
        monitor._delivery_receipts.pop("s2")

        monitor.commit_delivered_watermarks()

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
            await monitor._detect_and_cleanup_changes()
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
            await monitor._detect_and_cleanup_changes()

        assert old_sid not in monitor._pending_tools


class TestNewWindowDetection:
    async def test_callback_fires_for_new_window(self, monitor: SessionMonitor) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monitor._last_session_map = {}

        new_map = {"@5": {"session_id": "s1", "cwd": "/proj", "window_name": "proj"}}
        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=new_map,
        ):
            await monitor._detect_and_cleanup_changes()

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
            await monitor._detect_and_cleanup_changes()

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
            await monitor._detect_and_cleanup_changes()

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
)


def _winref(window_id: str, command: str) -> WindowRef:
    return WindowRef(
        window_id=window_id,
        window_name=window_id,
        cwd="/proj",
        pane_current_command=command,
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
        text = entries[0].get("message", {}).get("content", [{}])[0].get("text", "")
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
            },
        }

        assert await monitor.check_for_updates(current_map) == []
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
    """Test _read_new_lines delegation for Gemini with supports_incremental_read=True."""

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
            assert entries[0]["sessionId"] == "g1"
            assert entries[1]["type"] == "user"

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
            assert entries[0]["type"] == "gemini"

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
