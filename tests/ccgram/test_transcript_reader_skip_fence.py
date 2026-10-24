"""Regression tests for stale transcript reads crossing skip/reset fences."""

import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from ccgram.idle_tracker import IdleTracker
from ccgram.monitor_state import BacklogSkipIntent, MonitorState, TrackedSession
from ccgram.session_protocol.messages import TranscriptBatch, TranscriptParserState
from ccgram import transcript_reader as transcript_reader_module
from ccgram.transcript_reader import TranscriptReader


class _FilesystemGate:
    """Pause a worker after it has completed real filesystem I/O."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def pause(self) -> None:
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("test did not release the filesystem gate")

    async def wait_until_entered(self) -> None:
        assert await asyncio.to_thread(self.entered.wait, 5)


def _make_reader(
    tmp_path: Path,
    *,
    file_name: str = "transcript.jsonl",
    initial_contents: bytes = b"",
):
    transcript = tmp_path / file_name
    transcript.write_bytes(initial_contents)
    state = MonitorState(state_file=tmp_path / "monitor.json")
    tracked = TrackedSession("sess", str(transcript))
    state.update_session(tracked)
    reader = TranscriptReader(state, IdleTracker())
    return transcript, state, tracked, reader


def _assistant_line(text: str) -> bytes:
    return (
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
        ).encode()
        + b"\n"
    )


def _gemini_provider():
    return SimpleNamespace(
        capabilities=SimpleNamespace(
            name="gemini",
            supports_incremental_read=False,
            supports_task_tracking=False,
        )
    )


async def test_replay_cancels_stale_generation_probe_before_publication(
    tmp_path, monkeypatch
) -> None:
    transcript, state, tracked, reader = _make_reader(
        tmp_path, initial_contents=_assistant_line("old")
    )
    transcript.write_bytes(_assistant_line("replacement"))
    new_messages = []
    gate = _FilesystemGate()
    original_prefix_digest = transcript_reader_module._prefix_digest

    def gated_prefix_digest(path: Path, size: int) -> bytes:
        digest = original_prefix_digest(path, size)
        if path == transcript:
            gate.pause()
        return digest

    monkeypatch.setattr(transcript_reader_module, "_prefix_digest", gated_prefix_digest)
    task = asyncio.create_task(
        reader._process_session_file("sess", transcript, new_messages, window_id="@1")
    )
    try:
        await gate.wait_until_entered()
        state.begin_skip(BacklogSkipIntent("sess", "@1", 1, 4, 99, 100, 0))
        reader.replay_from_watermark("sess")
        state.cancel_skip("sess")

        tracked.last_byte_offset = 3
        tracked.parsed_offset = 3
        parser_state = TranscriptParserState(tool_names={"new": "parser"})
        reader._protocol_parser_states["sess"] = ("claude", parser_state)
        reader._pending_tools["sess"] = {"new": "pending"}
        reader._parser_rebuild_pending.add("sess")
        batch = TranscriptBatch(
            session_id="sess",
            provider_name="claude",
            records=(),
            byte_checkpoint=3,
        )
        reader._pending_batches.append(batch)
        reader._file_mtimes["sess"] = 42.0
        caches = (
            dict(reader._file_generations),
            dict(reader._file_ctimes),
            dict(reader._file_sizes),
            dict(reader._file_prefixes),
            dict(reader._file_markers),
        )
    finally:
        gate.release.set()

    await task

    assert tracked.parsed_offset == 3
    assert reader._file_mtimes["sess"] == 42.0
    assert caches == (
        reader._file_generations,
        reader._file_ctimes,
        reader._file_sizes,
        reader._file_prefixes,
        reader._file_markers,
    )
    assert reader._protocol_parser_states["sess"] == ("claude", parser_state)
    assert reader._pending_tools["sess"] == {"new": "pending"}
    assert "sess" in reader._parser_rebuild_pending
    assert reader.drain_batches() == (batch,)
    assert new_messages == []


async def test_replay_during_incremental_cache_read_drops_stale_batch(
    tmp_path, monkeypatch
) -> None:
    transcript, state, tracked, reader = _make_reader(tmp_path)
    new_line = _assistant_line("late producer")
    transcript.write_bytes(new_line)
    new_messages = []
    gate = _FilesystemGate()
    original_tail_marker = transcript_reader_module._tail_marker

    def gated_tail_marker(path: Path, offset: int) -> bytes:
        marker = original_tail_marker(path, offset)
        if path == transcript and offset == len(new_line):
            gate.pause()
        return marker

    monkeypatch.setattr(transcript_reader_module, "_tail_marker", gated_tail_marker)
    task = asyncio.create_task(
        reader._process_session_file("sess", transcript, new_messages, window_id="@1")
    )
    try:
        await gate.wait_until_entered()
        state.begin_skip(BacklogSkipIntent("sess", "@1", 1, 4, 99, len(new_line), 0))
        reader.replay_from_watermark("sess")
        state.cancel_skip("sess")

        tracked.parsed_offset = 2
        parser_state = TranscriptParserState(tool_names={"new": "state"})
        reader._protocol_parser_states["sess"] = ("claude", parser_state)
        batch = TranscriptBatch(
            session_id="sess",
            provider_name="claude",
            records=(),
            byte_checkpoint=2,
        )
        reader._pending_batches.append(batch)
        reader._file_mtimes["sess"] = 42.0
        reader._file_generations["sess"] = (101, 102)
        reader._file_ctimes["sess"] = 103
        reader._file_sizes["sess"] = 104
        reader._file_prefixes["sess"] = (105, b"new prefix")
        reader._file_markers["sess"] = (106, b"new marker")
        caches = (
            dict(reader._file_mtimes),
            dict(reader._file_generations),
            dict(reader._file_ctimes),
            dict(reader._file_sizes),
            dict(reader._file_prefixes),
            dict(reader._file_markers),
        )
    finally:
        gate.release.set()

    await task

    assert tracked.parsed_offset == 2
    assert caches == (
        reader._file_mtimes,
        reader._file_generations,
        reader._file_ctimes,
        reader._file_sizes,
        reader._file_prefixes,
        reader._file_markers,
    )
    assert reader._protocol_parser_states["sess"] == ("claude", parser_state)
    assert reader.drain_batches() == (batch,)
    assert new_messages == []


async def test_replay_during_retry_reset_probe_does_not_rewind_newer_offset(
    tmp_path, monkeypatch
) -> None:
    first_line = _assistant_line("first")
    second_line = _assistant_line("second")
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_bytes(first_line + second_line)
    state = MonitorState(state_file=tmp_path / "monitor.json")
    tracked = TrackedSession(
        "sess",
        str(transcript),
        last_byte_offset=len(first_line),
        parsed_offset=len(first_line),
    )
    state.update_session(tracked)
    reader = TranscriptReader(state, IdleTracker())
    new_messages = []
    gate = _FilesystemGate()
    original_prefix_digest = transcript_reader_module._prefix_digest
    original_tail_marker = transcript_reader_module._tail_marker
    prefix_reads = 0
    tail_reads = 0

    def rewrite_on_second_marker(path: Path, offset: int) -> bytes:
        nonlocal tail_reads
        if path == transcript:
            tail_reads += 1
            if tail_reads == 2:
                transcript.write_bytes(
                    first_line.replace(b"first", b"other") + second_line
                )
        return original_tail_marker(path, offset)

    def gate_second_prefix_read(path: Path, size: int) -> bytes:
        nonlocal prefix_reads
        digest = original_prefix_digest(path, size)
        if path == transcript:
            prefix_reads += 1
            if prefix_reads == 2:
                gate.pause()
        return digest

    monkeypatch.setattr(
        transcript_reader_module, "_tail_marker", rewrite_on_second_marker
    )
    monkeypatch.setattr(
        transcript_reader_module, "_prefix_digest", gate_second_prefix_read
    )
    task = asyncio.create_task(
        reader._process_session_file("sess", transcript, new_messages, window_id="@1")
    )
    try:
        await gate.wait_until_entered()
        reader.replay_from_watermark("sess")
        tracked.parsed_offset = len(first_line) + 1
        parser_state = TranscriptParserState(tool_names={"new": "retry state"})
        reader._protocol_parser_states["sess"] = ("claude", parser_state)
        reader._pending_tools["sess"] = {"new": "retry tool"}
        reader._parser_rebuild_pending.add("sess")
        caches = (
            dict(reader._file_generations),
            dict(reader._file_ctimes),
            dict(reader._file_sizes),
            dict(reader._file_prefixes),
            dict(reader._file_markers),
        )
    finally:
        gate.release.set()

    await task

    assert tracked.parsed_offset == len(first_line) + 1
    assert caches == (
        reader._file_generations,
        reader._file_ctimes,
        reader._file_sizes,
        reader._file_prefixes,
        reader._file_markers,
    )
    assert reader._protocol_parser_states["sess"] == ("claude", parser_state)
    assert reader._pending_tools["sess"] == {"new": "retry tool"}
    assert "sess" in reader._parser_rebuild_pending
    assert reader.drain_batches() == ()
    assert new_messages == []


async def test_replay_during_parser_prefix_rebuild_preserves_new_context(
    tmp_path, monkeypatch
) -> None:
    prefix = _assistant_line("history")
    suffix = _assistant_line("new output")
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_bytes(prefix)
    state = MonitorState(state_file=tmp_path / "monitor.json")
    tracked = TrackedSession(
        "sess", str(transcript), last_byte_offset=len(prefix), parsed_offset=len(prefix)
    )
    state.update_session(tracked)
    reader = TranscriptReader(state, IdleTracker())
    with transcript.open("ab") as source:
        source.write(suffix)
    new_messages = []
    gate = _FilesystemGate()
    original_read_bytes = Path.read_bytes

    def gated_read_bytes(path: Path) -> bytes:
        contents = original_read_bytes(path)
        if path == transcript:
            gate.pause()
        return contents

    monkeypatch.setattr(Path, "read_bytes", gated_read_bytes)
    task = asyncio.create_task(
        reader._process_session_file("sess", transcript, new_messages, window_id="@1")
    )
    try:
        await gate.wait_until_entered()
        reader.replay_from_watermark("sess")
        tracked.parsed_offset = len(prefix) + 1
        parser_state = TranscriptParserState(tool_names={"new": "rebuilt later"})
        reader._protocol_parser_states["sess"] = ("claude", parser_state)
        reader._pending_tools["sess"] = {"new": "parser tool"}
        reader._parser_rebuild_pending.add("sess")
        batch = TranscriptBatch(
            session_id="sess",
            provider_name="claude",
            records=(),
            byte_checkpoint=len(prefix) + 1,
        )
        reader._pending_batches.append(batch)
        reader._file_mtimes["sess"] = 42.0
        caches = (
            dict(reader._file_generations),
            dict(reader._file_ctimes),
            dict(reader._file_sizes),
            dict(reader._file_prefixes),
            dict(reader._file_markers),
        )
    finally:
        gate.release.set()

    await task

    assert tracked.parsed_offset == len(prefix) + 1
    assert reader._protocol_parser_states["sess"] == ("claude", parser_state)
    assert reader._pending_tools["sess"] == {"new": "parser tool"}
    assert "sess" in reader._parser_rebuild_pending
    assert reader._pending_batches == [batch]
    assert reader._file_mtimes["sess"] == 42.0
    assert caches == (
        reader._file_generations,
        reader._file_ctimes,
        reader._file_sizes,
        reader._file_prefixes,
        reader._file_markers,
    )
    assert new_messages == []


async def test_replay_during_whole_document_filesystem_read_discards_result(
    tmp_path, monkeypatch
) -> None:
    transcript, state, tracked, reader = _make_reader(
        tmp_path, file_name="transcript.json"
    )
    transcript.write_text(
        json.dumps({"messages": [{"type": "gemini", "id": "late", "content": "late"}]})
    )
    new_messages = []
    gate = _FilesystemGate()
    original_read_bytes = Path.read_bytes

    def gated_read_bytes(path: Path) -> bytes:
        contents = original_read_bytes(path)
        if path == transcript:
            gate.pause()
        return contents

    monkeypatch.setattr(Path, "read_bytes", gated_read_bytes)
    monkeypatch.setattr(
        transcript_reader_module,
        "_resolve_provider_for_file",
        lambda *_args: _gemini_provider(),
    )
    task = asyncio.create_task(
        reader._process_session_file("sess", transcript, new_messages, window_id="@1")
    )
    try:
        await gate.wait_until_entered()
        reader.replay_from_watermark("sess")
        tracked.parsed_offset = 5
    finally:
        gate.release.set()

    await task

    assert tracked.parsed_offset == 5
    assert new_messages == []
    assert reader.drain_batches() == ()
    assert "sess" not in reader._file_mtimes


async def test_retirement_and_recreation_reject_obsolete_tracked_session(
    tmp_path, monkeypatch
) -> None:
    transcript, state, old_tracked, reader = _make_reader(tmp_path)
    old_revision = reader._source_revision_for("sess")
    new_line = _assistant_line("old source")
    transcript.write_bytes(new_line)
    gate = _FilesystemGate()
    original_tail_marker = transcript_reader_module._tail_marker

    def gated_tail_marker(path: Path, offset: int) -> bytes:
        marker = original_tail_marker(path, offset)
        if path == transcript and offset == len(new_line):
            gate.pause()
        return marker

    monkeypatch.setattr(transcript_reader_module, "_tail_marker", gated_tail_marker)
    task = asyncio.create_task(
        reader._process_session_file("sess", transcript, [], window_id="@1")
    )
    try:
        await gate.wait_until_entered()
        reader.clear_session("sess")
        new_tracked = TrackedSession(
            "sess",
            str(transcript),
            last_byte_offset=len(new_line),
            parsed_offset=7,
        )
        state.update_session(new_tracked)
        parser_state = TranscriptParserState(tool_names={"new": "source"})
        reader._protocol_parser_states["sess"] = ("claude", parser_state)
        revision_after_retirement = reader._source_revisions["sess"]
    finally:
        gate.release.set()

    await task

    assert old_tracked.parsed_offset == len(new_line)
    assert state.get_session("sess") is new_tracked
    assert new_tracked.parsed_offset == 7
    assert reader._source_revisions["sess"] > old_revision
    assert reader._protocol_parser_states["sess"] == ("claude", parser_state)
    assert reader._file_mtimes == {}
    assert reader._file_generations == {}

    new_tracked.last_byte_offset = len(new_line)
    await reader._process_session_file("sess", transcript, [], window_id="@1")

    assert state.get_session("sess") is new_tracked
    assert reader._source_revisions["sess"] == revision_after_retirement
    assert new_tracked.parsed_offset == len(new_line)
