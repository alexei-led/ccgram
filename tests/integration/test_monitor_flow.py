"""Integration tests for SessionMonitor polling against real JSONL files.

Tests the full check_for_updates pipeline: session initialization,
incremental message delivery, file truncation recovery, and session
change cleanup. Mtime caching and MonitorState round-trips are covered
by unit tests.
"""

import json
import os
import time

import pytest

from ccgram.session_monitor import SessionMonitor

pytestmark = pytest.mark.integration

TEST_SESSION_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _make_assistant_entry(text, *, session_id=TEST_SESSION_ID, cwd="/tmp/test"):
    return {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
        "sessionId": session_id,
        "cwd": cwd,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }


def _write_jsonl(path, entries):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _append_jsonl(path, entries):
    with open(path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _bump_mtime(path):
    """Advance file mtime by 2 seconds to ensure change detection."""
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + 2))


@pytest.fixture
def session_map_with_transcript(state_dir):
    """Write a session_map.json pointing to a transcript file."""

    def _setup(transcript_path, session_id=TEST_SESSION_ID):
        session_map = {
            "ccgram:@0": {
                "session_id": session_id,
                "cwd": "/tmp/test",
                "window_name": "test",
                "transcript_path": str(transcript_path),
                "provider_name": "claude",
            }
        }
        (state_dir / "session_map.json").write_text(json.dumps(session_map))
        return session_map

    return _setup


def _make_monitor(state_dir):
    return SessionMonitor(
        projects_path=state_dir / "projects",
        poll_interval=0.1,
        state_file=state_dir / "monitor_state.json",
    )


async def test_new_session_initializes_offset(
    state_dir, session_map_with_transcript
) -> None:
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(transcript, [_make_assistant_entry("old message")])
    session_map = session_map_with_transcript(transcript)

    monitor = _make_monitor(state_dir)
    initial = await monitor.check_for_updates({"@0": session_map["ccgram:@0"]})
    assert len(initial) == 0

    tracked = monitor.state.get_session(TEST_SESSION_ID)
    assert tracked is not None
    assert tracked.last_byte_offset > 0


async def test_incremental_read_picks_up_new_messages(
    state_dir, session_map_with_transcript
) -> None:
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(transcript, [_make_assistant_entry("old")])
    session_map = session_map_with_transcript(transcript)

    monitor = _make_monitor(state_dir)
    current = {"@0": session_map["ccgram:@0"]}
    assert await monitor.check_for_updates(current) == []

    _append_jsonl(transcript, [_make_assistant_entry("new message")])
    _bump_mtime(transcript)
    new_messages = await monitor.check_for_updates(current)
    assert len(new_messages) == 1
    assert new_messages[0].text == "new message"
    assert new_messages[0].session_id == TEST_SESSION_ID


async def test_file_truncation_resets_offset(
    state_dir, session_map_with_transcript
) -> None:
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(
        transcript,
        [_make_assistant_entry("msg1"), _make_assistant_entry("msg2")],
    )
    session_map = session_map_with_transcript(transcript)

    monitor = _make_monitor(state_dir)
    current = {"@0": session_map["ccgram:@0"]}
    assert await monitor.check_for_updates(current) == []

    transcript.write_text("")
    _write_jsonl(transcript, [_make_assistant_entry("after truncation")])
    _bump_mtime(transcript)
    new_messages = await monitor.check_for_updates(current)
    assert len(new_messages) == 1
    assert new_messages[0].text == "after truncation"


async def test_session_change_cleanup(state_dir, session_map_with_transcript) -> None:
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(transcript, [_make_assistant_entry("init")])
    session_map_with_transcript(transcript)

    monitor = _make_monitor(state_dir)
    old_map = {
        "@0": {
            "session_id": TEST_SESSION_ID,
            "cwd": "/tmp/test",
            "transcript_path": str(transcript),
        }
    }
    await monitor.check_for_updates(old_map)
    assert monitor.state.get_session(TEST_SESSION_ID) is not None

    monitor._last_session_map = old_map

    new_sid = "11111111-2222-3333-4444-555555555555"
    new_map = {
        "@0": {
            "session_id": new_sid,
            "cwd": "/tmp/test",
            "transcript_path": str(transcript),
        }
    }
    (state_dir / "session_map.json").write_text(
        json.dumps({"ccgram:@0": new_map["@0"]})
    )
    await monitor._detect_and_cleanup_changes()

    assert monitor.state.get_session(TEST_SESSION_ID) is None
