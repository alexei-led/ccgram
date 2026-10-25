"""Tests for transcript reader offset handling."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ccgram import transcript_reader
from ccgram.idle_tracker import IdleTracker
from ccgram.monitor_state import BacklogSkipIntent, MonitorState, TrackedSession
from ccgram.transcript_reader import TranscriptReader, _StableRead


def test_clear_session_cancels_pending_skip_barrier(tmp_path) -> None:
    state = MonitorState(state_file=tmp_path / "monitor.json")
    state.update_session(TrackedSession("sess", str(tmp_path / "session.jsonl")))
    state.begin_skip(BacklogSkipIntent("sess", "@1", 1, 4, 99, 10, 0))
    reader = TranscriptReader(state, IdleTracker())

    reader.clear_session("sess")

    assert "sess" not in state.pending_skips
    assert state.get_session("sess") is None


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
    state.begin_skip(BacklogSkipIntent("sess-before-rename", "@1", 1, 4, 99, 50, 0))
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
    assert tracked.parsed_offset == session_file.stat().st_size
    assert "sess-before-rename" not in state.pending_skips


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


async def test_post_start_truncation_clamps_to_eof_without_replay(tmp_path) -> None:
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

    assert messages == []


async def test_atomic_replacement_after_start_clamps_to_eof(tmp_path) -> None:
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

    assert messages == []


async def test_same_inode_rewrite_with_preserved_mtime_does_not_replay(
    tmp_path,
) -> None:
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

    assert messages == []


async def test_replacement_between_stat_and_open_clamps_to_eof(tmp_path) -> None:
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
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert messages == []


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
            name="gemini",
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
            name="gemini",
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


async def test_metadata_only_ctime_bump_does_not_replay_transcript(tmp_path) -> None:
    """A ctime bump that leaves consumed bytes intact is not a replacement."""
    history = "".join(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h%d"}]}}\n'
        % index
        for index in range(5)
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=session_file.stat().st_size,
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    os.chmod(session_file, 0o600)

    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert messages == []
    tracked = state.get_session("sess")
    assert tracked is not None
    assert tracked.parsed_offset == session_file.stat().st_size


async def test_append_during_read_does_not_replay_transcript(tmp_path) -> None:
    """A concurrent append mid-read must deliver the new entry, not the file."""
    history = "".join(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h%d"}]}}\n'
        % index
        for index in range(5)
    )
    fresh = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"fresh"}]}}\n'
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=session_file.stat().st_size,
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    original_read = reader._read_new_lines
    appended = False

    async def append_then_read(*args, **kwargs):
        nonlocal appended
        if not appended:
            with session_file.open("a") as transcript:
                transcript.write(fresh)
            appended = True
        return await original_read(*args, **kwargs)

    messages = []
    with patch.object(reader, "_read_new_lines", side_effect=append_then_read):
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

    assert [msg.text for msg in messages] == ["fresh"]


async def test_rewrite_before_tail_marker_does_not_replay(tmp_path) -> None:
    """A rewrite outside the tail marker must not be skipped as an append."""
    history = "".join(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h%d"}]}}\n'
        % index
        for index in range(5)
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=session_file.stat().st_size,
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    await reader._process_session_file("sess", session_file, [], window_id="@1")
    reader._file_mtimes["sess"] = 0.0

    original_read = reader._read_new_lines
    rewritten = False

    async def rewrite_then_read(*args, **kwargs):
        nonlocal rewritten
        if not rewritten:
            session_file.write_text(
                session_file.read_text().replace('"h0"', '"changed"', 1),
                newline="\n",
            )
            rewritten = True
        return await original_read(*args, **kwargs)

    messages = []
    await reader._process_session_file("sess", session_file, messages, window_id="@1")

    assert messages == []


async def test_replaced_smaller_transcript_resumes_from_eof_no_replay(
    tmp_path,
) -> None:
    """A replaced (smaller) transcript must not be replayed as notifications.

    Regression for the 2026-08-17 flood: an adopted offset larger than the
    new file's size used to reset the offset to 0 and re-emit the whole
    history, flooding Telegram and starving every other topic.
    """
    big = '{"type":"assistant","message":{"content":[{"type":"text","text":"old"}]}}\n'
    small = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"fresh"}]}}\n'
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(small, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    # offset from the PREVIOUS (larger) transcript, carried over by adoption
    state.update_session(
        TrackedSession(
            session_id="sess-replaced",
            file_path=str(session_file),
            last_byte_offset=len(big.encode()) * 100,
        )
    )
    reader = TranscriptReader(state, IdleTracker())

    messages = []
    await reader._process_session_file(
        "sess-replaced", session_file, messages, window_id="@1"
    )

    assert messages == []  # nothing replayed from the replaced file
    tracked = state.get_session("sess-replaced")
    assert tracked is not None
    assert tracked.parsed_offset == session_file.stat().st_size


class TestStartupReplayBound:
    """A restart must not re-send an unbounded undelivered backlog.

    The delivered watermark only advances when the queue fully drains, so a
    wedged or flood-limited queue leaves it far behind. Replaying that range
    re-sends thousands of messages and trips Telegram flood control again.
    """

    @staticmethod
    def _record(text: str) -> str:
        return f'{{"type":"assistant","message":{{"content":[{{"type":"text","text":"{text}"}}]}}}}\n'

    def _reader(self, tmp_path, backlog: int, *, watermark: int = 0):
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(
            "".join(self._record(f"m{i}") for i in range(backlog)), newline="\n"
        )
        state = MonitorState(state_file=tmp_path / "monitor_state.json")
        state.update_session(
            TrackedSession(
                session_id="sess",
                file_path=str(session_file),
                last_byte_offset=watermark,
            )
        )
        return TranscriptReader(state, IdleTracker()), state, session_file

    async def test_backlog_larger_than_the_bound_is_skipped(
        self, tmp_path, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, state, session_file = self._reader(tmp_path, backlog=10)

        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

        assert messages == []
        tracked = state.get_session("sess")
        assert tracked is not None
        assert tracked.parsed_offset == session_file.stat().st_size

    async def test_backlog_within_the_bound_still_replays(
        self, tmp_path, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 1 << 20, raising=False)
        reader, _state, session_file = self._reader(tmp_path, backlog=3)

        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

        assert [msg.text for msg in messages] == ["m0", "m1", "m2"]

    async def test_zero_limit_replays_everything(self, tmp_path, monkeypatch) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 0, raising=False)
        reader, _state, session_file = self._reader(tmp_path, backlog=10)

        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

        assert len(messages) == 10

    async def test_only_the_first_poll_is_bounded(self, tmp_path, monkeypatch) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, _state, session_file = self._reader(tmp_path, backlog=10)

        skipped: list = []
        await reader._process_session_file(
            "sess", session_file, skipped, window_id="@1"
        )
        assert skipped == []

        with session_file.open("a") as transcript:
            transcript.write(self._record("fresh"))

        fresh: list = []
        await reader._process_session_file("sess", session_file, fresh, window_id="@1")

        assert [msg.text for msg in fresh] == ["fresh"]

    async def test_a_session_tracked_after_startup_is_never_clamped(
        self, tmp_path, monkeypatch
    ) -> None:
        """A session first tracked by this process has no pre-restart backlog."""
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        session_file = tmp_path / "fresh.jsonl"
        session_file.write_text(
            "".join(self._record(f"m{i}") for i in range(10)), newline="\n"
        )
        state = MonitorState(state_file=tmp_path / "monitor_state.json")
        reader = TranscriptReader(state, IdleTracker())
        # Tracked at offset 0 by a replay request during this process.
        state.update_session(
            TrackedSession(session_id="fresh", file_path=str(session_file))
        )

        messages: list = []
        await reader._process_session_file(
            "fresh", session_file, messages, window_id="@1"
        )

        assert [msg.text for msg in messages] == [f"m{i}" for i in range(10)]

    async def test_an_explicit_replay_request_is_never_clamped(
        self, tmp_path, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, _state, session_file = self._reader(tmp_path, backlog=10)
        reader.allow_replay("sess")

        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

        assert len(messages) == 10

    async def test_a_rebuild_failure_does_not_replay_a_clamped_backlog(
        self, tmp_path, monkeypatch
    ) -> None:
        """A rollback after clamping must re-clamp, not fall back to the watermark."""
        from unittest.mock import AsyncMock, patch

        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, state, session_file = self._reader(tmp_path, backlog=10)
        skipped: list = []
        await reader._process_session_file(
            "sess", session_file, skipped, window_id="@1"
        )
        assert skipped == []

        with session_file.open("a") as transcript:
            transcript.write(self._record("live"))

        # Force the parser-rebuild path and make it fail.
        reader._parser_rebuild_pending.add("sess")
        with patch.object(
            reader,
            "_rebuild_committed_parser_context",
            AsyncMock(return_value=False),
        ):
            blocked: list = []
            await reader._process_session_file(
                "sess", session_file, blocked, window_id="@1"
            )

        tracked = state.get_session("sess")
        assert tracked is not None
        assert tracked.parsed_offset >= reader._startup_clamped["sess"]

        # The live record written after the clamp is still delivered.
        delivered: list = []
        await reader._process_session_file(
            "sess", session_file, delivered, window_id="@1"
        )
        assert [msg.text for msg in delivered] == ["live"]

    async def test_canceling_a_skip_cannot_undo_the_clamp(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A canceled backlog skip must not reopen the clamped range."""
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, state, session_file = self._reader_with_backlog(tmp_path)
        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

        assert messages == []
        assert "sess" in reader._startup_clamped

        reader.replay_from_watermark("sess")

        tracked = state.get_session("sess")
        assert tracked is not None
        assert tracked.parsed_offset >= reader._startup_clamped["sess"]

    async def test_a_refreshed_session_id_keeps_the_clamp(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The same transcript under a new id must not replay its backlog."""
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, _state, session_file = self._reader_with_backlog(tmp_path)
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        clamped = reader._startup_clamped.get("sess")
        assert clamped is not None

        tracked = reader._adopt_tracking_for_file("refreshed", session_file)

        assert tracked is not None
        assert reader._startup_clamped.get("refreshed") == clamped
        assert "refreshed" in reader._startup_sessions
        # The replacement cursor starts at -1, which falls back to the delivered
        # watermark and would replay the clamped range.
        assert tracked.parsed_offset >= clamped

        replayed: list = []
        await reader._process_session_file(
            "refreshed", session_file, replayed, window_id="@1"
        )
        assert replayed == []

    def _reader_with_backlog(self, tmp_path):
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(
            "".join(self._record(f"m{i}") for i in range(10)), newline="\n"
        )
        state = MonitorState(state_file=tmp_path / "monitor_state.json")
        state.update_session(
            TrackedSession(session_id="sess", file_path=str(session_file))
        )
        return TranscriptReader(state, IdleTracker()), state, session_file

    async def test_the_parser_rebuild_reads_up_to_the_clamp_point(
        self, tmp_path, monkeypatch
    ) -> None:
        """A rebuild bounded by the watermark would miss the skipped turn."""
        from ccgram import transcript_reader as module
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, state, session_file = self._reader_with_backlog(tmp_path)
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        tracked = state.get_session("sess")
        assert tracked is not None
        clamp = reader._startup_clamped["sess"]
        watermark = tracked.last_byte_offset

        seen: list[int] = []
        real_parse = module.parse_records

        def spy(entries, provider_name, parser_state, **kwargs):
            seen.extend(entry.offset for entry in entries)
            return real_parse(entries, provider_name, parser_state, **kwargs)

        reader._parser_rebuild_pending.add("sess")
        monkeypatch.setattr(module, "parse_records", spy)
        await reader._process_session_file("sess", session_file, [], window_id="@1")

        assert seen, "rebuild did not parse anything"
        assert max(seen) <= clamp
        if watermark < clamp:
            # Bounded by the watermark, the rebuild would stop before the clamp.
            assert max(seen) > watermark

    async def test_a_refresh_before_the_first_poll_still_bounds_the_replay(
        self, tmp_path, monkeypatch
    ) -> None:
        """The refresh case that the poll-then-refresh test cannot reach."""
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(
            "".join(self._record(f"m{i}") for i in range(10)), newline="\n"
        )
        state = MonitorState(state_file=tmp_path / "monitor_state.json")
        state.update_session(
            TrackedSession(session_id="old", file_path=str(session_file))
        )
        reader = TranscriptReader(state, IdleTracker())

        # Refresh the id before any poll clamped the old one.
        tracked = reader._adopt_tracking_for_file("new", session_file)
        assert tracked is not None
        assert "new" in reader._startup_sessions

        messages: list = []
        await reader._process_session_file(
            "new", session_file, messages, window_id="@1"
        )

        assert messages == []
        assert "new" in reader._startup_clamped


_CLAUDE_PROVIDER = SimpleNamespace(
    capabilities=SimpleNamespace(
        name="claude",
        supports_incremental_read=True,
        supports_task_tracking=False,
    )
)


async def test_consecutive_appends_skip_prefix_digest(tmp_path) -> None:
    """Growing transcripts must not pay a full-file digest every poll cycle."""
    history = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h0"}]}}\n'
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=len(history.encode()),
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    real_digest = transcript_reader._prefix_digest

    delivered = []
    with (
        patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=_CLAUDE_PROVIDER,
        ),
        patch("ccgram.transcript_reader._prefix_digest", wraps=real_digest) as digest,
    ):
        for index in range(4):
            line = (
                '{"type":"assistant","message":{"content":[{"type":"text","text":"m%d"}]}}\n'
                % index
            )
            with session_file.open("a") as transcript:
                transcript.write(line)
            messages = []
            await reader._process_session_file(
                "sess", session_file, messages, window_id="@1"
            )
            delivered.extend(msg.text for msg in messages)
        assert digest.call_count == 0

    assert delivered == ["m0", "m1", "m2", "m3"]


async def test_clamp_cycle_keeps_verify_schedule_at_shortest_interval(
    tmp_path,
) -> None:
    """The cycle after a clamp must verify again, not inherit a doubled interval."""
    history = "".join(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h%d"}]}}\n'
        % index
        for index in range(5)
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=session_file.stat().st_size,
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    real_digest = transcript_reader._prefix_digest
    with patch(
        "ccgram.transcript_reader._resolve_provider_for_file",
        return_value=_CLAUDE_PROVIDER,
    ):
        await reader._process_session_file("sess", session_file, [], window_id="@1")

        session_file.write_text(history.replace('"h4"', '"h9"'), newline="\n")
        reader._file_mtimes["sess"] = 0.0
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        assert reader._prefix_verify["sess"] == (0, 1)

        # One marker-verified append cycle counts up to (1, 1), so the next
        # quiet cycle is due and must run the full verification. Had the
        # clamp cycle's intact read doubled the interval to (0, 2), the state
        # would sit at (1, 2) here and this cycle's digest would be skipped.
        with session_file.open("a") as transcript:
            transcript.write(
                '{"type":"assistant","message":{"content":[{"type":"text","text":"m0"}]}}\n'
            )
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        assert reader._prefix_verify["sess"] == (1, 1)

        reader._file_mtimes["sess"] = 0.0
        with patch(
            "ccgram.transcript_reader._prefix_digest", wraps=real_digest
        ) as digest:
            await reader._process_session_file("sess", session_file, [], window_id="@1")
        assert digest.call_count > 0
        assert reader._prefix_verify["sess"] == (0, 2)


async def test_rewrite_of_tail_marker_window_clamps_without_replay(
    tmp_path,
) -> None:
    """A rewrite inside the tail-marker window must verify and clamp."""
    history = "".join(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h%d"}]}}\n'
        % index
        for index in range(5)
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=session_file.stat().st_size,
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    with patch(
        "ccgram.transcript_reader._resolve_provider_for_file",
        return_value=_CLAUDE_PROVIDER,
    ):
        await reader._process_session_file("sess", session_file, [], window_id="@1")

        session_file.write_text(history.replace('"h4"', '"h9"'), newline="\n")
        reader._file_mtimes["sess"] = 0.0

        messages = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

    assert messages == []
    tracked = state.get_session("sess")
    assert tracked is not None
    assert tracked.parsed_offset == session_file.stat().st_size
    # The clamp resets the schedule to the shortest interval, and the same
    # cycle's intact read must not double the interval away from it.
    assert reader._prefix_verify["sess"] == (0, 1)


async def test_prefix_verify_backoff_resets_on_restart_and_generation_change(
    tmp_path,
) -> None:
    """The digest backoff re-arms immediately after replacement or restart."""
    history = "".join(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h%d"}]}}\n'
        % index
        for index in range(5)
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=session_file.stat().st_size,
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    with patch(
        "ccgram.transcript_reader._resolve_provider_for_file",
        return_value=_CLAUDE_PROVIDER,
    ):
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        assert reader._prefix_verify["sess"] == (0, 2)

        for index in range(2):
            line = (
                '{"type":"assistant","message":{"content":[{"type":"text","text":"m%d"}]}}\n'
                % index
            )
            with session_file.open("a") as transcript:
                transcript.write(line)
            await reader._process_session_file("sess", session_file, [], window_id="@1")
        assert reader._prefix_verify["sess"] == (2, 2)

        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text(
            '{"type":"assistant","message":{"content":[{"type":"text","text":"x"}]}}\n',
            newline="\n",
        )
        replacement.replace(session_file)
        messages = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

    assert messages == []
    assert reader._prefix_verify["sess"] == (0, 1)

    reader.clear_session("sess")
    assert "sess" not in reader._prefix_verify


async def test_idle_due_session_verifies_once_not_every_cycle(tmp_path) -> None:
    """A due session verifies once per interval while idle, not per tick."""
    history = "".join(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h%d"}]}}\n'
        % index
        for index in range(5)
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=session_file.stat().st_size,
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    real_digest = transcript_reader._prefix_digest

    with (
        patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=_CLAUDE_PROVIDER,
        ),
        patch("ccgram.transcript_reader._prefix_digest", wraps=real_digest) as digest,
    ):
        for index in range(2):
            line = (
                '{"type":"assistant","message":{"content":[{"type":"text","text":"m%d"}]}}\n'
                % index
            )
            with session_file.open("a") as transcript:
                transcript.write(line)
            await reader._process_session_file("sess", session_file, [], window_id="@1")
        assert digest.call_count == 0

        for _ in range(3):
            await reader._process_session_file("sess", session_file, [], window_id="@1")
        # The first idle cycle verifies the frozen prefix and extends the
        # baseline, then doubles the interval; the rest defer.
        assert digest.call_count == 2


async def test_overdue_session_forces_verification_even_while_growing(
    tmp_path,
) -> None:
    """Past the defer cap, growth cycles verify instead of deferring forever."""
    history = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"h0"}]}}\n'
    )
    session_file = tmp_path / "transcript.jsonl"
    session_file.write_text(history, newline="\n")

    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess",
            file_path=str(session_file),
            last_byte_offset=len(history.encode()),
        )
    )
    reader = TranscriptReader(state, IdleTracker())
    reader._prefix_verify["sess"] = (
        transcript_reader._PREFIX_VERIFY_CAP,
        transcript_reader._PREFIX_VERIFY_CAP,
    )
    real_digest = transcript_reader._prefix_digest

    line = '{"type":"assistant","message":{"content":[{"type":"text","text":"m0"}]}}\n'
    with (
        patch(
            "ccgram.transcript_reader._resolve_provider_for_file",
            return_value=_CLAUDE_PROVIDER,
        ),
        patch("ccgram.transcript_reader._prefix_digest", wraps=real_digest) as digest,
    ):
        with session_file.open("a") as transcript:
            transcript.write(line)
        messages = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )
        assert digest.call_count >= 1
        assert reader._prefix_verify["sess"][0] < transcript_reader._PREFIX_VERIFY_CAP

    assert [msg.text for msg in messages] == ["m0"]
