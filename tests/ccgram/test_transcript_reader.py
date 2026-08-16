"""Tests for transcript reader offset handling."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ccgram.idle_tracker import IdleTracker
from ccgram.monitor_state import MonitorState, TrackedSession
from ccgram.transcript_reader import TranscriptReader, _StableRead


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

    assert [msg.text for msg in messages] == ["b"]
    assert idle.get_last_activity("sess") is None


async def test_first_poll_counts_only_bytes_written_after_startup(tmp_path) -> None:
    old = '{"type":"assistant","message":{"content":[{"type":"text","text":"a"}]}}\n'
    unread = '{"type":"assistant","message":{"content":[{"type":"text","text":"b"}]}}\n'
    fresh = '{"type":"assistant","message":{"content":[{"type":"text","text":"c"}]}}\n'
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
    with session_file.open("a") as transcript:
        transcript.write(fresh)

    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert [msg.text for msg in messages] == ["b", "c"]
    assert idle.get_last_activity("sess") is not None


async def test_post_start_truncation_begins_a_live_file_generation(tmp_path) -> None:
    old = '{"type":"assistant","message":{"content":[{"type":"text","text":"old"}]}}\n'
    history = old * 10
    fresh = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"new"}]}}\n'
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=len(old.encode()) * 5,
        )
    )
    idle = IdleTracker()
    reader = TranscriptReader(state, idle)
    session_file.write_text(fresh, newline="\n")

    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert [msg.text for msg in messages] == ["new"]
    assert idle.get_last_activity("sess") is not None


async def test_atomic_replacement_after_start_is_read_from_zero(tmp_path) -> None:
    old = '{"type":"assistant","message":{"content":[{"type":"text","text":"old"}]}}\n'
    fresh = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"fresh"}]}}\n'
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(old, newline="\n")
    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess", file_path=str(session_file), last_byte_offset=0
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    await reader._process_session_file("sess", session_file, [], window_id="@1")

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(fresh, newline="\n")
    replacement.replace(session_file)
    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert [msg.text for msg in messages] == ["fresh"]


async def test_same_inode_rewrite_with_preserved_mtime_resets_offset(tmp_path) -> None:
    old = '{"type":"assistant","message":{"content":[{"type":"text","text":"old"}]}}\n'
    new = old.replace("old", "new")
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(old)
    initial_stat = session_file.stat()
    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess", file_path=str(session_file), last_byte_offset=0
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    await reader._process_session_file("sess", session_file, [], window_id="@1")

    session_file.write_text(new)
    os.utime(
        session_file,
        ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
    )
    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert [message.text for message in messages] == ["new"]


async def test_replacement_between_stat_and_open_retries_from_zero(tmp_path) -> None:
    old = '{"type":"assistant","message":{"content":[{"type":"text","text":"old"}]}}\n'
    new = old.replace("old", "new")
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(old)
    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess", file_path=str(session_file), last_byte_offset=0
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    await reader._process_session_file("sess", session_file, [], window_id="@1")
    original_read = reader._read_new_lines
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(new)
    session_file.write_text(old)  # Trigger the read before replacing it in-flight.
    replaced = False

    async def replace_then_read(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            replacement.replace(session_file)
            replaced = True
        return await original_read(*args, **kwargs)

    messages = []
    with patch.object(reader, "_read_new_lines", side_effect=replace_then_read):
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

    assert [message.text for message in messages] == ["new"]


async def test_whole_file_rewrite_bypasses_unchanged_mtime(tmp_path) -> None:
    session_file = tmp_path / "transcript.json"
    session_file.write_text("old")
    old_stat = session_file.stat()
    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess", file_path=str(session_file), last_byte_offset=0
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    provider = SimpleNamespace(
        capabilities=SimpleNamespace(
            supports_incremental_read=False,
            supports_task_tracking=False,
        )
    )
    with (
        patch(
            "ccgram.transcript_reader._resolve_provider_for_file", return_value=provider
        ),
        patch.object(
            reader,
            "_read_session_entries",
            new=AsyncMock(
                side_effect=lambda *_args, **_kwargs: _StableRead(
                    [], session_file.stat(), False
                )
            ),
        ) as read,
        patch.object(reader, "_append_provider_messages"),
    ):
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        session_file.write_text("new-long")
        os.utime(session_file, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        await reader._process_session_file("sess", session_file, [], window_id="@1")
    assert read.await_count == 2


async def test_whole_file_replacement_bypasses_unchanged_mtime(tmp_path) -> None:
    session_file = tmp_path / "transcript.json"
    session_file.write_text("old")
    old_stat = session_file.stat()
    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess", file_path=str(session_file), last_byte_offset=0
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    provider = SimpleNamespace(
        capabilities=SimpleNamespace(
            supports_incremental_read=False,
            supports_task_tracking=False,
        )
    )

    with (
        patch(
            "ccgram.transcript_reader._resolve_provider_for_file", return_value=provider
        ),
        patch.object(
            reader,
            "_read_session_entries",
            new=AsyncMock(
                side_effect=lambda *_args, **_kwargs: _StableRead(
                    [],
                    session_file.stat(),
                    False,
                )
            ),
        ) as read,
        patch.object(reader, "_append_provider_messages"),
    ):
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        replacement = tmp_path / "replacement.json"
        replacement.write_text("new")
        replacement.replace(session_file)
        os.utime(session_file, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

        await reader._process_session_file("sess", session_file, [], window_id="@1")

    assert read.await_count == 2


async def test_failed_startup_read_preserves_catch_up_boundary(tmp_path) -> None:
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

    with patch.object(
        reader, "_read_new_lines", new=AsyncMock(side_effect=OSError("busy"))
    ):
        await reader._process_session_file("sess", session_file, [], window_id="@1")

    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert [msg.text for msg in messages] == ["b"]
    assert idle.get_last_activity("sess") is None
