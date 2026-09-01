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

from ccgram.claude_task_state import get_claude_task_snapshot
from ccgram.window_state_store import WindowState, window_store

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


def _make_task_create_entry(
    task_id: str,
    subject: str,
    *,
    tool_id: str = "tool-1",
    session_id=TEST_SESSION_ID,
    cwd="/tmp/test",
):
    return [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "TaskCreate",
                        "input": {
                            "subject": subject,
                            "description": "",
                            "activeForm": "",
                        },
                    }
                ]
            },
            "sessionId": session_id,
            "cwd": cwd,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Task #{task_id} created successfully",
                    }
                ]
            },
            "toolUseResult": {"task": {"id": task_id, "subject": subject}},
            "sessionId": session_id,
            "cwd": cwd,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        },
    ]


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


async def test_new_session_initializes_offset(
    state_dir, session_map_with_transcript, make_monitor
) -> None:
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(transcript, [_make_assistant_entry("old message")])
    session_map = session_map_with_transcript(transcript)

    monitor = make_monitor()
    initial = await monitor.check_for_updates({"@0": session_map["ccgram:@0"]})
    assert len(initial) == 0

    tracked = monitor.state.get_session(TEST_SESSION_ID)
    assert tracked is not None
    assert tracked.last_byte_offset > 0


async def test_new_session_seeds_claude_task_snapshot(
    state_dir, session_map_with_transcript, make_monitor
) -> None:
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(transcript, _make_task_create_entry("1", "Review architecture"))
    session_map = session_map_with_transcript(transcript)

    monitor = make_monitor()
    current = {"@0": session_map["ccgram:@0"]}
    assert await monitor.check_for_updates(current) == []

    snapshot = get_claude_task_snapshot("@0")
    assert snapshot is not None
    assert snapshot.total_count == 1
    assert snapshot.items[0].subject == "Review architecture"


async def test_incremental_read_picks_up_new_messages(
    state_dir, session_map_with_transcript, make_monitor
) -> None:
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(transcript, [_make_assistant_entry("old")])
    session_map = session_map_with_transcript(transcript)

    monitor = make_monitor()
    current = {"@0": session_map["ccgram:@0"]}
    assert await monitor.check_for_updates(current) == []

    _append_jsonl(transcript, [_make_assistant_entry("new message")])
    _bump_mtime(transcript)
    new_messages = await monitor.check_for_updates(current)
    assert len(new_messages) == 1
    assert new_messages[0].text == "new message"
    assert new_messages[0].session_id == TEST_SESSION_ID


async def test_post_start_truncation_does_not_replay_snapshot(
    state_dir, session_map_with_transcript, make_monitor
) -> None:
    """A same-inode truncate/rewrite is a replacement snapshot, not new output."""
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(
        transcript,
        [_make_assistant_entry("msg1"), _make_assistant_entry("msg2")],
    )
    session_map = session_map_with_transcript(transcript)

    monitor = make_monitor()
    current = {"@0": session_map["ccgram:@0"]}
    assert await monitor.check_for_updates(current) == []

    transcript.write_text("")
    _write_jsonl(transcript, [_make_assistant_entry("after truncation")])
    _bump_mtime(transcript)
    new_messages = await monitor.check_for_updates(current)
    assert new_messages == []


async def test_nested_session_start_does_not_steal_forwarding(
    state_dir, make_monitor
) -> None:
    parent_id = TEST_SESSION_ID
    child_id = "11111111-2222-3333-4444-555555555555"
    parent_transcript = state_dir / "parent.jsonl"
    child_transcript = state_dir / "child.jsonl"
    _write_jsonl(parent_transcript, [_make_assistant_entry("parent old")])
    _write_jsonl(
        child_transcript, [_make_assistant_entry("child done", session_id=child_id)]
    )
    window_store.window_states["@0"] = WindowState(
        session_id=parent_id,
        cwd="/tmp/test",
        window_name="test",
        transcript_path=str(parent_transcript),
        provider_name="claude",
    )
    (state_dir / "session_map.json").write_text(
        json.dumps(
            {
                "ccgram:@0": {
                    "session_id": child_id,
                    "cwd": "/tmp/test",
                    "window_name": "test",
                    "transcript_path": str(child_transcript),
                    "provider_name": "claude",
                }
            }
        )
    )

    monitor = make_monitor()
    current = await monitor._load_current_session_map()
    assert current["@0"]["session_id"] == parent_id
    assert await monitor.check_for_updates(current) == []

    _append_jsonl(parent_transcript, [_make_assistant_entry("parent new")])
    _bump_mtime(parent_transcript)
    # tmux: every live window is adoptable, which the loop derives per cycle.
    current = await monitor._detect_and_cleanup_changes(adoptable_window_ids={"@0"})
    new_messages = await monitor.check_for_updates(current)

    assert current["@0"]["session_id"] == parent_id
    assert len(new_messages) == 1
    assert new_messages[0].session_id == parent_id
    assert new_messages[0].text == "parent new"
    assert monitor.state.get_session(child_id) is None


async def test_session_change_cleanup(
    state_dir, session_map_with_transcript, make_monitor
) -> None:
    transcript = state_dir / "transcript.jsonl"
    _write_jsonl(transcript, _make_task_create_entry("1", "Old task"))
    session_map_with_transcript(transcript)

    monitor = make_monitor()
    old_map = {
        "@0": {
            "session_id": TEST_SESSION_ID,
            "cwd": "/tmp/test",
            "transcript_path": str(transcript),
        }
    }
    await monitor.check_for_updates(old_map)
    assert monitor.state.get_session(TEST_SESSION_ID) is not None
    assert get_claude_task_snapshot("@0") is not None

    monitor._last_session_map = old_map

    new_sid = "11111111-2222-3333-4444-555555555555"
    new_transcript = (
        state_dir / "transcript_new.jsonl"
    )  # real session change → different file
    _write_jsonl(
        new_transcript, _make_task_create_entry("2", "New task", session_id=new_sid)
    )
    new_map = {
        "@0": {
            "session_id": new_sid,
            "cwd": "/tmp/test",
            "transcript_path": str(new_transcript),
        }
    }
    (state_dir / "session_map.json").write_text(
        json.dumps({"ccgram:@0": new_map["@0"]})
    )
    await monitor._detect_and_cleanup_changes(adoptable_window_ids={"@0"})

    assert monitor.state.get_session(TEST_SESSION_ID) is None
    assert get_claude_task_snapshot("@0") is None
