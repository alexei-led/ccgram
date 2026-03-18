"""Tests for session monitor hook event reading."""

import json
from pathlib import Path

import pytest

from ccgram.handlers.hook_events import HookEvent
from ccgram.session_monitor import SessionMonitor


@pytest.fixture
def monitor(tmp_path: Path) -> SessionMonitor:
    return SessionMonitor(
        projects_path=tmp_path / "projects",
        poll_interval=0.1,
        state_file=tmp_path / "monitor_state.json",
    )


@pytest.fixture
def events_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr("ccgram.session_monitor.config.events_file", path)
    return path


class TestReadHookEvents:
    async def test_reads_events_incrementally(
        self, monitor: SessionMonitor, events_file: Path
    ) -> None:
        received: list[HookEvent] = []

        async def cb(event: HookEvent) -> None:
            received.append(event)

        monitor.set_hook_event_callback(cb)

        # Write two events
        events_file.write_text(
            json.dumps(
                {
                    "ts": 1.0,
                    "event": "Stop",
                    "window_key": "ccgram:@0",
                    "session_id": "s1",
                    "data": {},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "ts": 2.0,
                    "event": "Notification",
                    "window_key": "ccgram:@1",
                    "session_id": "s2",
                    "data": {"tool_name": "AskUserQuestion"},
                }
            )
            + "\n"
        )

        await monitor._read_hook_events()
        assert len(received) == 2
        assert received[0].event_type == "Stop"
        assert received[1].event_type == "Notification"

        # Second read should return nothing (offset advanced)
        received.clear()
        await monitor._read_hook_events()
        assert len(received) == 0

    async def test_empty_file(self, monitor: SessionMonitor, events_file: Path) -> None:
        events_file.write_text("")
        received: list[HookEvent] = []
        monitor.set_hook_event_callback(lambda e: received.append(e))  # type: ignore[arg-type, return-value]
        await monitor._read_hook_events()
        assert len(received) == 0

    async def test_missing_file(
        self, monitor: SessionMonitor, events_file: Path
    ) -> None:
        # File doesn't exist — no error
        received: list[HookEvent] = []
        monitor.set_hook_event_callback(lambda e: received.append(e))  # type: ignore[arg-type, return-value]
        await monitor._read_hook_events()
        assert len(received) == 0

    async def test_malformed_line_skipped(
        self, monitor: SessionMonitor, events_file: Path
    ) -> None:
        events_file.write_text(
            "not-json\n"
            + json.dumps(
                {
                    "ts": 1.0,
                    "event": "Stop",
                    "window_key": "ccgram:@0",
                    "session_id": "s1",
                    "data": {},
                }
            )
            + "\n"
        )
        received: list[HookEvent] = []

        async def cb(event: HookEvent) -> None:
            received.append(event)

        monitor.set_hook_event_callback(cb)
        await monitor._read_hook_events()
        assert len(received) == 1
        assert received[0].event_type == "Stop"

    async def test_truncation_detection(
        self, monitor: SessionMonitor, events_file: Path
    ) -> None:
        # Write a long line, read it, then truncate the file
        line = json.dumps(
            {
                "ts": 1.0,
                "event": "Stop",
                "window_key": "ccgram:@0",
                "session_id": "s1",
                "data": {},
            }
        )
        events_file.write_text(line + "\n")

        received: list[HookEvent] = []

        async def cb(event: HookEvent) -> None:
            received.append(event)

        monitor.set_hook_event_callback(cb)
        await monitor._read_hook_events()
        assert len(received) == 1

        # Simulate file truncation (smaller than previous offset)
        events_file.write_text("")
        received.clear()
        await monitor._read_hook_events()
        assert len(received) == 0  # No events in empty file

    async def test_no_callback_set(
        self, monitor: SessionMonitor, events_file: Path
    ) -> None:
        events_file.write_text(
            json.dumps(
                {
                    "ts": 1.0,
                    "event": "Stop",
                    "window_key": "ccgram:@0",
                    "session_id": "s1",
                    "data": {},
                }
            )
            + "\n"
        )
        # No callback set — should not crash
        await monitor._read_hook_events()


class TestRecordHookActivity:
    def test_records_activity(self, monitor: SessionMonitor) -> None:
        monitor._last_session_map = {
            "ccgram:@0": {"session_id": "s1", "cwd": "/tmp"},
        }
        monitor.record_hook_activity("@0")
        assert monitor._last_activity.get("s1") is not None

    def test_no_match(self, monitor: SessionMonitor) -> None:
        monitor._last_session_map = {}
        monitor.record_hook_activity("@99")
        assert len(monitor._last_activity) == 0
