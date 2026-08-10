"""Tests for transcript reader offset handling."""

from ccgram.idle_tracker import IdleTracker
from ccgram.monitor_state import MonitorState, TrackedSession
from ccgram.transcript_reader import TranscriptReader


async def test_same_transcript_reuses_offset_after_session_map_refresh(
    tmp_path,
) -> None:
    """A tmux rename/session-map refresh must not replay an existing transcript."""
    first = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"old"}]}}\n'
    )
    second = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"new"}]}}\n'
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(first + second, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess-before-rename",
            file_path=str(session_file),
            last_byte_offset=len(first.encode()),
        )
    )
    reader = TranscriptReader(state, IdleTracker())

    messages = []
    await reader._process_session_file(
        "sess-after-rename",
        session_file,
        messages,
        window_id="@1",
    )

    assert [msg.text for msg in messages] == ["new"]
    tracked = state.get_session("sess-after-rename")
    assert tracked is not None
    assert tracked.last_byte_offset == session_file.stat().st_size


async def test_catch_up_read_after_restart_is_not_activity(tmp_path) -> None:
    """Bytes already on disk at startup are history, not a busy agent.

    The byte offset survives a restart but the mtime cache does not, so the
    first poll replays whatever the previous run did not consume. Stamping
    that as activity leaves the topic active with a typing indicator for the
    whole idle window after every restart.
    """
    old = '{"type":"assistant","message":{"content":[{"type":"text","text":"a"}]}}\n'
    unread = '{"type":"assistant","message":{"content":[{"type":"text","text":"b"}]}}\n'
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(old + unread, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=len(old.encode()),
        )
    )
    idle = IdleTracker()
    reader = TranscriptReader(state, idle)

    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    # The missed message is still delivered — only the clock is left alone.
    assert [msg.text for msg in messages] == ["b"]
    assert idle.get_last_activity("sess") is None


async def test_writes_after_the_catch_up_do_count_as_activity(tmp_path) -> None:
    old = '{"type":"assistant","message":{"content":[{"type":"text","text":"a"}]}}\n'
    fresh = '{"type":"assistant","message":{"content":[{"type":"text","text":"b"}]}}\n'
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(old, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=0,
        )
    )
    idle = IdleTracker()
    reader = TranscriptReader(state, idle)

    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")
    assert idle.get_last_activity("sess") is None

    session_file.write_text(old + fresh, newline="\n")
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert idle.get_last_activity("sess") is not None
