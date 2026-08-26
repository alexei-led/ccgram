"""Tests for event_reader — incremental events.jsonl reading."""

import json
from pathlib import Path

import pytest

from ccgram.event_reader import read_new_events
from ccgram.providers.base import HookEvent


def _write_event(path: Path, event_type: str, window_key: str, session_id: str) -> None:
    with path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "event": event_type,
                    "window_key": window_key,
                    "session_id": session_id,
                    "data": {},
                    "ts": 1234567890.0,
                }
            )
            + "\n"
        )


async def test_returns_empty_when_file_missing(tmp_path: Path) -> None:
    events, offset = await read_new_events(tmp_path / "missing.jsonl", 0)
    assert events == []
    assert offset == 0


async def test_reads_new_events_from_zero(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_event(path, "Stop", "ccgram:@0", "sess-1")
    _write_event(path, "SessionStart", "ccgram:@1", "sess-2")

    events, offset = await read_new_events(path, 0)
    assert len(events) == 2
    assert events[0].event_type == "Stop"
    assert events[0].window_key == "ccgram:@0"
    assert events[1].event_type == "SessionStart"
    assert offset == path.stat().st_size


async def test_reads_only_new_events_after_offset(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_event(path, "Stop", "ccgram:@0", "sess-1")
    _, offset_after_first = await read_new_events(path, 0)

    _write_event(path, "SessionStart", "ccgram:@1", "sess-2")
    events, offset = await read_new_events(path, offset_after_first)
    assert len(events) == 1
    assert events[0].event_type == "SessionStart"
    assert offset > offset_after_first


@pytest.mark.parametrize(
    "unusable_line",
    [
        pytest.param("", id="blank-line"),
        pytest.param("not-json", id="invalid-json"),
        pytest.param("[]", id="json-array"),
        pytest.param(
            '{"event": "Stop", "window_key": "ccgram:@0"}', id="no-session-id"
        ),
        pytest.param(
            json.dumps(
                {
                    "schema_version": 9999,
                    "event": "Stop",
                    "window_key": "ccgram:@0",
                    "session_id": "sess-1",
                }
            ),
            id="future-schema-version",
        ),
    ],
)
async def test_unusable_line_is_skipped_and_the_offset_advances_past_it(
    tmp_path: Path, unusable_line: str
) -> None:
    """A line the reader cannot use must never wedge the poll loop: it is
    dropped and the offset moves past it so the next event is still read.
    """
    path = tmp_path / "events.jsonl"
    path.write_text(unusable_line + "\n")
    offset_after_bad = path.stat().st_size
    _write_event(path, "Stop", "ccgram:@0", "sess-1")

    events, offset = await read_new_events(path, 0)

    assert [event.event_type for event in events] == ["Stop"]
    assert offset > offset_after_bad


async def test_offset_past_eof_rereads_the_file_from_the_start(
    tmp_path: Path,
) -> None:
    """A rotated/truncated events.jsonl leaves the persisted offset beyond EOF;
    the reader must reset instead of silently returning nothing forever."""
    path = tmp_path / "events.jsonl"
    _write_event(path, "Stop", "ccgram:@0", "sess-1")
    file_size = path.stat().st_size

    events, offset = await read_new_events(path, file_size + 9999)

    assert [event.event_type for event in events] == ["Stop"]
    assert offset == file_size


async def test_returns_hook_event_dataclass(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_event(path, "Notification", "ccgram:@5", "abc-123")

    events, _ = await read_new_events(path, 0)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, HookEvent)
    assert ev.event_type == "Notification"
    assert ev.window_key == "ccgram:@5"
    assert ev.session_id == "abc-123"
    assert ev.timestamp == pytest.approx(1234567890.0)
