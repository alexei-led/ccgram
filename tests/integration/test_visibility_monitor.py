import json

import pytest

from ccgram.monitor_state import TrackedSession

pytestmark = pytest.mark.integration


async def test_internal_only_batch_settles_without_public_messages_and_survives_restart(
    state_dir, make_monitor
):
    path = state_dir / ".claude" / "projects" / "project" / "session.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "type": "api-request-blob",
            "message": {"role": "system", "content": "INTERNAL_CONTEXT" * 1000},
        },
        {
            "type": "api-request-blob",
            "message": {"role": "assistant", "content": "INTERNAL_CONTEXT"},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monitor = make_monitor()
    monitor.state.update_session(
        TrackedSession(session_id="session", file_path=str(path))
    )
    delivered = []

    async def receive(message):
        delivered.append(message)

    monitor.set_message_callback(receive)
    await monitor._process_session_file("session", path, [])
    batches = monitor._transcript_reader.drain_batches()
    assert len(batches) == 1
    assert batches[0].records == ()
    for batch, consumer, receipt in monitor._register_delivery_receipts(batches):
        await monitor._dispatch_message_with_receipt(batch, receipt, consumer)
    monitor.commit_delivered_watermarks()

    assert delivered == []
    assert (
        monitor.state.tracked_sessions["session"].last_byte_offset
        == path.stat().st_size
    )
    restarted = make_monitor()
    restarted.set_message_callback(receive)
    assert (
        restarted.state.tracked_sessions["session"].last_byte_offset
        == path.stat().st_size
    )
    await restarted._process_session_file("session", path, [])
    assert restarted._transcript_reader.drain_batches() == ()
    assert delivered == []
