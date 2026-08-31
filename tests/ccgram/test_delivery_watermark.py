"""At-least-once delivery: only acknowledged transcript work is persisted."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ccgram.delivery_contract import DeliveryOutcome, DeliveryReceipt
from ccgram.handlers.messaging_pipeline import message_queue as mq
from ccgram.monitor_state import MonitorState, TrackedSession


def test_session_monitor_depends_only_on_neutral_delivery_contract() -> None:
    tree = ast.parse(Path("src/ccgram/session_monitor.py").read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert not any("handlers.messaging_pipeline" in module for module in imports)
    assert "delivery_contract" in imports


def _entry(text: str) -> str:
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
        )
        + "\n"
    )


async def test_commit_advances_watermark_when_idle(tmp_path):
    state = MonitorState(state_file=tmp_path / "ms.json")
    session = TrackedSession(
        session_id="s1", file_path="/x", last_byte_offset=10, parsed_offset=50
    )
    state.tracked_sessions["s1"] = session

    with patch.object(mq, "queues_idle", return_value=True):
        advanced = state.commit_parsed_offsets()

    assert advanced is True
    assert session.last_byte_offset == 50


async def test_monitor_does_not_commit_when_queue_busy(tmp_path, monkeypatch):
    """The monitor gate: with queues busy the parsed offset must NOT fold
    into the persisted watermark; a crash then replays the gap."""
    import json as _json
    import os

    from ccgram.handlers.messaging_pipeline import message_queue as mq
    from ccgram.session_monitor import SessionMonitor

    session_file = tmp_path / "s1.jsonl"
    line = (
        _json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hi"}]},
            }
        )
        + "\n"
    )
    session_file.write_text(line)
    size1 = len(line.encode())

    monitor = SessionMonitor(projects_path=tmp_path, state_file=tmp_path / "ms.json")
    # Seed directly (unit scope: the gate, not session discovery).
    from ccgram.monitor_state import TrackedSession

    monitor.state.tracked_sessions["s1"] = TrackedSession(
        session_id="s1", file_path=str(session_file), last_byte_offset=size1
    )

    current_map = {
        "@0": {
            "session_id": "s1",
            "cwd": str(tmp_path),
            "window_name": "w",
            "transcript_path": str(session_file),
        }
    }
    monkeypatch.setattr(mq, "queues_idle", lambda: False)
    monitor.commit_delivered_watermarks()
    with open(session_file, "a") as f:
        f.write(line)
    os.utime(session_file)
    monitor.set_message_callback(AsyncMock(return_value=None))
    messages = await monitor.check_for_updates(current_map)
    pending = monitor._register_delivery_receipts(messages)
    for message, receipt in pending:
        await monitor._dispatch_message_with_receipt(message, receipt)

    tracked = monitor.state.get_session("s1")
    assert tracked is not None
    assert tracked.parsed_offset == 2 * size1, "parse advanced"
    assert tracked.last_byte_offset == size1, "watermark frozen while busy"

    # Queues go idle (and the cycle's messages are dispatched): the loop's
    # post-dispatch commit folds the watermark.
    monkeypatch.setattr(mq, "queues_idle", lambda: True)
    monitor.commit_delivered_watermarks()
    assert tracked.last_byte_offset == 2 * size1


def test_to_dict_excludes_parsed_offset():
    d = TrackedSession(
        session_id="s1", file_path="/x", last_byte_offset=5, parsed_offset=99
    ).to_dict()
    assert "parsed_offset" not in d
    assert d["last_byte_offset"] == 5


async def test_failed_delivery_receipt_withholds_watermark_until_restart(tmp_path):
    """A terminal send failure must replay the parsed range after recovery."""
    from ccgram.session_monitor import SessionMonitor

    monitor = SessionMonitor(projects_path=tmp_path, state_file=tmp_path / "ms.json")
    tracked = TrackedSession(
        session_id="s1", file_path="/x", last_byte_offset=10, parsed_offset=50
    )
    monitor.state.update_session(tracked)
    receipt = DeliveryReceipt(checkpoint=50)
    receipt.track()
    receipt.close()
    receipt.settle(DeliveryOutcome.FAILED)
    monitor._delivery_receipts["s1"] = [receipt]

    monitor.commit_delivered_watermarks()
    assert tracked.last_byte_offset == 10

    # A process restart discards in-memory parse state; the persisted offset
    # remains 10. A successful replay receipt may then commit the range.
    replay = DeliveryReceipt(checkpoint=50)
    replay.track()
    replay.close()
    replay.settle(DeliveryOutcome.DELIVERED)
    monitor._delivery_receipts["s1"] = [replay]
    monitor.commit_delivered_watermarks()
    assert tracked.last_byte_offset == 50


def test_ready_receipt_commits_only_its_checkpoint(tmp_path):
    from ccgram.session_monitor import SessionMonitor

    monitor = SessionMonitor(projects_path=tmp_path, state_file=tmp_path / "ms.json")
    tracked = TrackedSession(
        session_id="s1", file_path="/x", last_byte_offset=10, parsed_offset=80
    )
    monitor.state.update_session(tracked)
    receipt = DeliveryReceipt(checkpoint=50)
    receipt.track()
    receipt.close()
    receipt.settle(DeliveryOutcome.DELIVERED)
    monitor._delivery_receipts["s1"] = [receipt]

    monitor.commit_delivered_watermarks()

    assert tracked.last_byte_offset == 50


def test_receipt_free_parsed_offset_is_not_committed(tmp_path):
    from ccgram.session_monitor import SessionMonitor

    monitor = SessionMonitor(projects_path=tmp_path, state_file=tmp_path / "ms.json")
    tracked = TrackedSession(
        session_id="s1", file_path="/x", last_byte_offset=10, parsed_offset=50
    )
    monitor.state.update_session(tracked)

    monitor.commit_delivered_watermarks()

    assert tracked.last_byte_offset == 10


async def test_queues_idle_semantics():
    import asyncio

    # No queues at all: idle.
    mq._message_queues.clear()
    mq._inflight_count = 0
    assert mq.queues_idle() is True

    q = mq._message_queues.setdefault(1, asyncio.Queue())
    q.put_nowait(object())  # type: ignore[arg-type]
    assert mq.queues_idle() is False

    q.get_nowait()
    mq._inflight_count = 1
    assert mq.queues_idle() is False  # empty queue but a send in flight
    mq._inflight_count = 0
    assert mq.queues_idle() is True
    mq._message_queues.clear()


def test_settled_prefix_before_failure_commits(tmp_path):
    """The #205 policy change: progress strictly below a failed receipt is
    durable; the failed receipt and everything after it replay."""
    from ccgram.session_monitor import SessionMonitor

    monitor = SessionMonitor(projects_path=tmp_path, state_file=tmp_path / "ms.json")
    tracked = TrackedSession(
        session_id="s1", file_path="/x", last_byte_offset=10, parsed_offset=90
    )
    monitor.state.update_session(tracked)
    ok = DeliveryReceipt(checkpoint=50)
    ok.track()
    ok.close()
    ok.settle(DeliveryOutcome.DELIVERED)
    failed = DeliveryReceipt(checkpoint=90)
    failed.track()
    failed.close()
    failed.settle(DeliveryOutcome.FAILED)
    monitor._delivery_receipts["s1"] = [ok, failed]

    monitor.commit_delivered_watermarks()

    assert tracked.last_byte_offset == 50
    assert monitor._delivery_receipts["s1"] == [failed]
