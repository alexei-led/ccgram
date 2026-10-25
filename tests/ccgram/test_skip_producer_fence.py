"""Source skips fence readers and dispatches that predate their purge."""

import asyncio
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import threading
from unittest.mock import patch

import aiofiles.threadpool
import pytest

from ccgram.delivery_contract import (
    DeliveryOutcome,
    DeliveryReceipt,
    get_active_delivery_receipt,
)
from ccgram.monitor_state import TrackedSession
from ccgram.providers.claude import ClaudeProvider
from ccgram.session import SessionManager
from ccgram.session_monitor import SessionMonitor
from ccgram.thread_router import thread_router


def _assistant_line(text: str) -> bytes:
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "id": text,
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            }
        )
        + "\n"
    ).encode()


async def _dispatch_pending(monitor, pending) -> None:
    for batch, consumer, receipt in pending:
        await monitor._dispatch_message_with_receipt(batch, receipt, consumer)


@dataclass
class _SourceCase:
    monitor: SessionMonitor
    transcript: Path
    tracked: TrackedSession
    current_map: dict[str, dict[str, str]]
    projected: list[tuple[str, str]] = field(default_factory=list)
    acknowledge_notice: bool = False
    snapshot_offset: int = 0

    async def consume(self, message) -> None:
        self.projected.append((message.session_id, message.text))

    def texts(self, session_id: str) -> list[str]:
        return [text for sid, text in self.projected if sid == session_id]

    def add_other_source(self) -> Path:
        other = self.transcript.with_name("other.jsonl")
        other.write_bytes(_assistant_line("other source"))
        self.monitor.state.update_session(TrackedSession("s2", str(other), 0))
        self.current_map["@1"] = {
            "session_id": "s2",
            "transcript_path": str(other),
        }
        return other

    async def dispatch_buffered(self) -> None:
        pending = self.monitor._register_delivery_receipts(
            self.monitor._transcript_reader.drain_batches()
        )
        await _dispatch_pending(self.monitor, pending)

    async def begin_skip(self, result: str, append: bool) -> None:
        self.acknowledge_notice = result == "committed"
        intent = await self.monitor.request_backlog_skip(1, "@0", 4, 99)
        assert intent is not None
        assert intent.purge_complete
        self.snapshot_offset = intent.snapshot_offset
        if append:
            with self.transcript.open("ab") as target:
                target.write(_assistant_line("new after snapshot"))
        if result == "committed":
            await self.monitor.commit_delivered_watermarks()
            assert not self.monitor.state.pending_skips
        elif result == "cancelled":
            self.monitor._cancel_pending_skip("s1")
            assert self.tracked.last_byte_offset == 0

    async def finish_and_check(self, result: str, append: bool) -> None:
        assert self.texts("s1") == []
        if result == "pending":
            receipt = self.monitor._skip_notice_receipts["s1"]
            receipt.settle(DeliveryOutcome.DELIVERED)
            await self.monitor.commit_delivered_watermarks()
        await self.monitor.check_for_updates(self.current_map)
        await self.dispatch_buffered()
        expected = ["old one", "old two"] if result == "cancelled" else []
        if append:
            expected.append("new after snapshot")
        assert self.texts("s1") == expected
        assert self.tracked.last_byte_offset == (
            0 if result == "cancelled" else self.snapshot_offset
        )


@pytest.fixture
def source_case(tmp_path):
    SessionManager()
    transcript = tmp_path / "source.jsonl"
    transcript.write_bytes(_assistant_line("old one") + _assistant_line("old two"))
    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    tracked = TrackedSession("s1", str(transcript), 0)
    monitor.state.update_session(tracked)
    current_map = {"@0": {"session_id": "s1", "transcript_path": str(transcript)}}
    monitor._last_session_map = current_map
    case = _SourceCase(monitor, transcript, tracked, current_map)

    async def purge(_intent) -> int:
        previous = len(case.projected)
        case.projected[:] = [item for item in case.projected if item[0] != "s1"]
        return previous - len(case.projected)

    async def notice(_intent) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()
        if case.acknowledge_notice:
            receipt.settle(DeliveryOutcome.DELIVERED)

    monitor.set_message_callback(case.consume)
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
        yield case


@dataclass
class _ReadGate:
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    async def wait(self) -> None:
        assert await asyncio.to_thread(self.entered.wait, 2)


@pytest.fixture
def pause_read(monkeypatch):
    original_open = getattr(aiofiles.threadpool, "sync_open")

    def install(path: Path) -> _ReadGate:
        gate = _ReadGate()

        class DelayedReader(io.BufferedReader):
            def read(self, size: int | None = -1, /) -> bytes:
                data = super().read(size)
                if size in (-1, None):
                    gate.entered.set()
                    assert gate.release.wait(timeout=5)
                return data

        def open_source(file, *args, **kwargs):
            source = original_open(file, *args, **kwargs)
            if Path(file) == path and "b" in source.mode:
                return DelayedReader(source.detach())
            return source

        monkeypatch.setattr(aiofiles.threadpool, "sync_open", open_source)
        return gate

    return install


@pytest.mark.parametrize("barrier_result", ["pending", "committed", "cancelled"])
@pytest.mark.parametrize("append_after_snapshot", [False, True])
async def test_skip_fences_an_inflight_read(
    source_case, pause_read, barrier_result: str, append_after_snapshot: bool
) -> None:
    case = source_case
    gate = pause_read(case.transcript)
    producer = asyncio.create_task(case.monitor.check_for_updates(case.current_map))
    try:
        await gate.wait()
        await case.begin_skip(barrier_result, append_after_snapshot)
    finally:
        gate.release.set()
    await producer
    await case.dispatch_buffered()
    await case.finish_and_check(barrier_result, append_after_snapshot)


@pytest.mark.parametrize("barrier_result", ["pending", "committed", "cancelled"])
@pytest.mark.parametrize("append_after_snapshot", [False, True])
async def test_skip_removes_buffered_source_while_another_source_reads(
    source_case, pause_read, barrier_result: str, append_after_snapshot: bool
) -> None:
    case = source_case
    gate = pause_read(case.add_other_source())
    producer = asyncio.create_task(case.monitor.check_for_updates(case.current_map))
    try:
        await gate.wait()
        await case.begin_skip(barrier_result, append_after_snapshot)
    finally:
        gate.release.set()
    await producer
    await case.dispatch_buffered()
    assert case.texts("s2") == ["other source"]
    await case.finish_and_check(barrier_result, append_after_snapshot)
    assert case.texts("s2") == ["other source"]


@pytest.mark.parametrize("barrier_result", ["pending", "committed", "cancelled"])
@pytest.mark.parametrize("append_after_snapshot", [False, True])
async def test_skip_revokes_registered_dispatch_behind_another_consumer(
    source_case, barrier_result: str, append_after_snapshot: bool
) -> None:
    case = source_case
    entered = asyncio.Event()
    release = asyncio.Event()

    async def prior_consumer(_batch) -> None:
        entered.set()
        await release.wait()

    case.monitor.unregister_delivery_consumer("telegram")
    case.monitor.register_delivery_consumer("prior", prior_consumer)
    case.monitor.set_message_callback(case.consume)
    await case.monitor.check_for_updates(case.current_map)
    pending = case.monitor._register_delivery_receipts(
        case.monitor._transcript_reader.drain_batches()
    )
    dispatcher = asyncio.create_task(_dispatch_pending(case.monitor, pending))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await case.begin_skip(barrier_result, append_after_snapshot)
    finally:
        release.set()
    await dispatcher
    await case.finish_and_check(barrier_result, append_after_snapshot)


async def test_revocation_stops_the_rest_of_an_active_consumer(source_case) -> None:
    case = source_case
    entered = asyncio.Event()
    release = asyncio.Event()
    delivered: list[str] = []

    async def consume(message) -> None:
        delivered.append(message.text)
        entered.set()
        await release.wait()

    case.monitor.set_message_callback(consume)
    await case.monitor.check_for_updates(case.current_map)
    pending = case.monitor._register_delivery_receipts(
        case.monitor._transcript_reader.drain_batches()
    )
    dispatcher = asyncio.create_task(_dispatch_pending(case.monitor, pending))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await case.begin_skip("pending", False)
    finally:
        release.set()
    await dispatcher

    assert delivered == ["old one"]
    assert pending[0][2].revoked
    assert pending[0][2].failed is False
    assert not pending[0][2].commit_ready


@pytest.mark.parametrize("stage", ["buffered", "registered"])
async def test_failed_skip_persistence_does_not_fence_valid_work(
    source_case, stage: str
) -> None:
    case = source_case
    await case.monitor.check_for_updates(case.current_map)
    parsed_before = case.tracked.parsed_offset
    pending = []
    if stage == "registered":
        pending = case.monitor._register_delivery_receipts(
            case.monitor._transcript_reader.drain_batches()
        )
    with patch.object(case.monitor.state, "save", return_value=False):
        assert await case.monitor.request_backlog_skip(1, "@0", 4, 99) is None

    assert not case.monitor.state.pending_skips
    assert case.tracked.parsed_offset == parsed_before
    if stage == "registered":
        assert pending
        assert all(not receipt.revoked for _, _, receipt in pending)
    else:
        batches = case.monitor._transcript_reader.drain_batches()
        assert len(batches) == 1
        assert len(batches[0].records) == 2


async def test_automatic_cap_defers_when_its_generation_probe_is_revoked(
    source_case, monkeypatch
) -> None:
    from ccgram import transcript_reader as reader_module

    case = source_case
    await case.monitor.check_for_updates(case.current_map)
    gate = _ReadGate()
    original_digest = reader_module._prefix_digest

    def paused_digest(path: Path, size: int) -> bytes:
        result = original_digest(path, size)
        if path == case.transcript:
            gate.entered.set()
            assert gate.release.wait(timeout=5)
        return result

    monkeypatch.setattr(reader_module, "_prefix_digest", paused_digest)
    with patch(
        "ccgram.session_monitor.get_provider_for_window",
        return_value=ClaudeProvider(),
    ):
        attempt = asyncio.create_task(
            case.monitor._maybe_skip_replay_cap(
                "s1", case.transcript, "@0", case.current_map["@0"], 1
            )
        )
        try:
            await gate.wait()
            case.monitor._transcript_reader.replay_from_watermark("s1")
        finally:
            gate.release.set()
        await attempt

    assert not case.monitor.state.pending_skips
    assert case.tracked.last_byte_offset == 0


async def test_transcript_replacement_keeps_registered_undelivered_tail(
    source_case,
) -> None:
    case = source_case
    await case.monitor.check_for_updates(case.current_map)
    pending = case.monitor._register_delivery_receipts(
        case.monitor._transcript_reader.drain_batches()
    )
    assert pending
    replacement = case.transcript.with_name("replacement.jsonl")
    replacement.write_bytes(_assistant_line("replacement history"))
    replacement.replace(case.transcript)
    await case.monitor.check_for_updates(case.current_map)
    await _dispatch_pending(case.monitor, pending)
    assert case.texts("s1") == ["old one", "old two"]
    assert all(not receipt.revoked for _, _, receipt in pending)


@pytest.mark.parametrize("replace_source", [False, True])
async def test_refresh_waits_for_old_queued_tail_without_blocking_other_sources(
    source_case, replace_source: bool
) -> None:
    case = source_case
    queued: list[tuple[str, str, DeliveryReceipt]] = []

    async def enqueue(message) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()
        queued.append((message.session_id, message.text, receipt))

    case.monitor.set_message_callback(enqueue)
    await case.monitor.check_for_updates(case.current_map)
    await case.dispatch_buffered()
    assert len(queued) == 2
    old_eof = case.transcript.stat().st_size
    if replace_source:
        replacement = case.transcript.with_name("replacement.jsonl")
        replacement.write_bytes(_assistant_line("replacement history"))
        replacement.replace(case.transcript)
    case.add_other_source()
    refreshed = {
        "@0": {"session_id": "refreshed", "transcript_path": str(case.transcript)},
        "@1": case.current_map["@1"],
    }
    case.monitor._last_session_map = refreshed
    await case.monitor.check_for_updates(refreshed)
    await case.dispatch_buffered()
    assert [(sid, text) for sid, text, _ in queued] == [
        ("s1", "old one"),
        ("s1", "old two"),
        ("s2", "other source"),
    ]
    assert case.monitor.state.get_session("refreshed") is None
    assert case.tracked.last_byte_offset == 0
    for _sid, _text, receipt in queued:
        receipt.settle(DeliveryOutcome.DELIVERED)
    # ACK may arrive after the prior cycle's commit and before adoption.
    await case.monitor.check_for_updates(refreshed)
    assert case.tracked.last_byte_offset == old_eof
    await case.dispatch_buffered()
    await case.monitor.commit_delivered_watermarks()
    adopted = case.monitor.state.get_session("refreshed")
    assert adopted is not None
    assert adopted.last_byte_offset == case.transcript.stat().st_size
    assert len(queued) == 3


@pytest.mark.parametrize("retired", ["failed", "revoked"])
async def test_refresh_does_not_wait_forever_for_unsuccessful_receipts(
    source_case, retired: str
) -> None:
    case = source_case
    old_receipts = []

    async def enqueue(_message) -> None:
        receipt = get_active_delivery_receipt()
        assert receipt is not None
        receipt.track()
        old_receipts.append(receipt)

    case.monitor.set_message_callback(enqueue)
    await case.monitor.check_for_updates(case.current_map)
    await case.dispatch_buffered()
    for receipt in old_receipts:
        if retired == "failed":
            receipt.fail()
        else:
            receipt.revoke()
    refreshed = {
        "@0": {"session_id": "refreshed", "transcript_path": str(case.transcript)}
    }
    case.monitor._last_session_map = refreshed
    await case.monitor.check_for_updates(refreshed)
    assert case.monitor.state.get_session("refreshed") is not None
