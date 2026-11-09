"""Tests for transcript reader offset handling."""

import os
import threading
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

    async def test_a_cleared_session_id_that_returns_is_not_clamped(
        self, tmp_path, monkeypatch
    ) -> None:
        """A cleared id that comes back is a new session, not the old backlog (#367)."""
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, state, session_file = self._reader(tmp_path, backlog=10)

        reader.clear_session("sess")
        state.update_session(
            TrackedSession(session_id="sess", file_path=str(session_file))
        )

        returned: list = []
        await reader._process_session_file(
            "sess", session_file, returned, window_id="@1"
        )

        assert len(returned) == 10

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
        with session_file.open("a") as transcript:
            transcript.write(self._record("live"))
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

    async def test_a_refresh_moves_startup_state_to_the_new_id(
        self, tmp_path, monkeypatch
    ) -> None:
        """The old id must not leak startup state; the replay exemption follows."""
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, _state, session_file = self._reader_with_backlog(tmp_path)
        reader.allow_replay("sess")
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        assert "sess" in reader._startup_replay_checked

        assert reader._adopt_tracking_for_file("refreshed", session_file) is not None

        assert "sess" not in reader._startup_sessions
        assert "sess" not in reader._startup_replay_checked
        assert "sess" not in reader._replay_allowed
        assert "refreshed" in reader._replay_allowed

    async def test_an_explicit_replay_request_survives_a_refresh(
        self, tmp_path, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "startup_replay_max_bytes", 120, raising=False)
        reader, _state, session_file = self._reader_with_backlog(tmp_path)
        reader.allow_replay("sess")
        assert reader._adopt_tracking_for_file("refreshed", session_file) is not None

        messages: list = []
        await reader._process_session_file(
            "refreshed", session_file, messages, window_id="@1"
        )

        assert len(messages) == 10
        assert "refreshed" not in reader._startup_clamped


def _record(text: str) -> str:
    return (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"%s"}]}}\n'
        % text
    )


def _tracked_reader(tmp_path, session_file, offset):
    state = MonitorState(state_file=tmp_path / "monitor_state.json")
    state.update_session(
        TrackedSession(
            session_id="sess", file_path=str(session_file), last_byte_offset=offset
        )
    )
    return TranscriptReader(state, IdleTracker()), state


class TestSteadyStateCost:
    """Polling a growing transcript must not hash the consumed prefix."""

    async def test_appends_and_idle_polls_never_digest_the_prefix(
        self, tmp_path
    ) -> None:
        history = _record("h0") * 50
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        reader, _ = _tracked_reader(tmp_path, session_file, len(history.encode()))
        hashed = 0
        real_range = transcript_reader._hash_range

        def counting_range(digest, path, start, end):
            nonlocal hashed
            hashed += end - start
            return real_range(digest, path, start, end)

        delivered: list[str] = []
        appended = 0
        with (
            patch(
                "ccgram.transcript_reader._prefix_digest",
                wraps=transcript_reader._prefix_digest,
            ) as full_digest,
            patch("ccgram.transcript_reader._hash_range", counting_range),
        ):
            await reader._process_session_file("sess", session_file, [], window_id="@1")
            hashed = 0
            for index in range(5):
                line = _record(f"m{index}")
                with session_file.open("a") as transcript:
                    transcript.write(line)
                appended += len(line.encode())
                messages: list = []
                await reader._process_session_file(
                    "sess", session_file, messages, window_id="@1"
                )
                delivered.extend(msg.text for msg in messages)
                await reader._process_session_file(
                    "sess", session_file, [], window_id="@1"
                )

        assert delivered == [f"m{index}" for index in range(5)]
        assert full_digest.call_count == 0
        assert hashed == appended

    async def test_metadata_only_change_on_idle_file_is_inspected_once(
        self, tmp_path
    ) -> None:
        history = _record("h0") * 5
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        reader, _ = _tracked_reader(tmp_path, session_file, len(history.encode()))
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        os.chmod(session_file, 0o600)
        os.chmod(session_file, 0o644)

        with (
            patch(
                "ccgram.transcript_reader._prefix_digest",
                wraps=transcript_reader._prefix_digest,
            ) as full_digest,
            patch.object(
                reader, "_marker_changed", wraps=reader._marker_changed
            ) as marker,
        ):
            for _ in range(4):
                await reader._process_session_file(
                    "sess", session_file, [], window_id="@1"
                )

        assert full_digest.call_count == 0
        assert marker.call_count <= 1

    async def test_inode_change_with_intact_bytes_digests_once_and_delivers(
        self, tmp_path
    ) -> None:
        history = _record("h0") * 5
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        reader, _ = _tracked_reader(tmp_path, session_file, len(history.encode()))
        await reader._process_session_file("sess", session_file, [], window_id="@1")

        copy = tmp_path / "copy.jsonl"
        copy.write_text(history + _record("fresh"))
        copy.replace(session_file)

        with patch(
            "ccgram.transcript_reader._prefix_digest",
            wraps=transcript_reader._prefix_digest,
        ) as full_digest:
            messages: list = []
            await reader._process_session_file(
                "sess", session_file, messages, window_id="@1"
            )
            await reader._process_session_file("sess", session_file, [], window_id="@1")

        assert [msg.text for msg in messages] == ["fresh"]
        assert full_digest.call_count == 1

    async def test_rewrite_inside_tail_marker_clamps_without_replay(
        self, tmp_path
    ) -> None:
        history = _record("h0") * 5
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        reader, state = _tracked_reader(tmp_path, session_file, len(history.encode()))
        await reader._process_session_file("sess", session_file, [], window_id="@1")

        session_file.write_text(history.replace("h0", "h1") + _record("after"))

        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

        assert messages == []
        tracked = state.get_session("sess")
        assert tracked is not None
        assert tracked.parsed_offset == session_file.stat().st_size

    async def test_replacement_rebaselines_so_later_appends_deliver(
        self, tmp_path
    ) -> None:
        history = _record("h0") * 5
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        reader, _ = _tracked_reader(tmp_path, session_file, len(history.encode()))
        await reader._process_session_file("sess", session_file, [], window_id="@1")

        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text(_record("other") * 3)
        replacement.replace(session_file)
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        with session_file.open("a") as transcript:
            transcript.write(_record("live"))
        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

        assert [msg.text for msg in messages] == ["live"]

    async def test_rewrite_is_detected_after_the_parse_position_rolled_back(
        self, tmp_path
    ) -> None:
        history = _record("h0") * 5
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        reader, state = _tracked_reader(tmp_path, session_file, len(history.encode()))
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        tracked = state.get_session("sess")
        assert tracked is not None
        tracked.parsed_offset = len(_record("h0").encode()) * 2  # rolled back

        session_file.write_text(history.replace("h0", "h1") + _record("after"))
        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )

        assert messages == []
        assert tracked.parsed_offset == session_file.stat().st_size

    def test_a_short_read_baseline_never_claims_unread_bytes(self, tmp_path) -> None:
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_bytes(b"x" * 10)

        size, _ = transcript_reader._advance_prefix(None, session_file, 25)

        assert size == 10


class TestParserRebuildOnActiveFile:
    """The restart rebuild must finish on a transcript that keeps growing."""

    def _setup(self, tmp_path, count=20):
        history = "".join(_record(f"h{i}") for i in range(count))
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        reader, state = _tracked_reader(tmp_path, session_file, len(history.encode()))
        return reader, state, session_file, history

    async def test_rebuild_survives_a_concurrent_append(self, tmp_path) -> None:
        reader, _, session_file, _ = self._setup(tmp_path)
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        reader._parser_rebuild_pending.add("sess")
        with session_file.open("a") as transcript:
            transcript.write(_record("live"))
        real = transcript_reader._replay_committed_prefix

        def append_then_replay(*args, **kwargs):
            with session_file.open("a") as transcript:
                transcript.write(_record("during"))
            return real(*args, **kwargs)

        with patch(
            "ccgram.transcript_reader._replay_committed_prefix", append_then_replay
        ):
            messages: list = []
            await reader._process_session_file(
                "sess", session_file, messages, window_id="@1"
            )

        assert "sess" not in reader._parser_rebuild_pending
        assert "sess" in reader._protocol_parser_states

    async def test_rebuild_rejects_a_replaced_file(self, tmp_path) -> None:
        reader, _, session_file, history = self._setup(tmp_path)
        await reader._process_session_file("sess", session_file, [], window_id="@1")
        stat = session_file.stat()
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text(history)
        replacement.replace(session_file)

        rebuilt = await reader._rebuild_committed_parser_context(
            "sess", session_file, "claude", stat, supports_incremental_read=True
        )

        assert rebuilt is False

    async def test_rebuild_rejects_a_file_truncated_during_the_read(
        self, tmp_path
    ) -> None:
        _, _, session_file, history = self._setup(tmp_path)
        stat = session_file.stat()
        real_feed_read = transcript_reader.read_entries

        def truncate_then_read(raw, provider):
            session_file.write_text(history[:50])
            return real_feed_read(raw, provider)

        with patch("ccgram.transcript_reader.read_entries", truncate_then_read):
            state = transcript_reader._replay_committed_prefix(
                session_file,
                "claude",
                len(history.encode()),
                (stat.st_dev, stat.st_ino),
            )

        assert state is None

    async def test_chunked_rebuild_matches_a_whole_file_rebuild(
        self, tmp_path, monkeypatch
    ) -> None:
        count = 40
        tool_use = (
            '{"type":"assistant","message":{"content":[{"type":"tool_use",'
            '"id":"t1","name":"Bash","input":{"command":"ls"}}]}}\n'
        )
        history = "".join(_record(f"h{i}") for i in range(count)) + tool_use
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        stat = session_file.stat()
        generation = (stat.st_dev, stat.st_ino)
        floor = len(history.encode())

        whole = transcript_reader._replay_committed_prefix(
            session_file, "claude", floor, generation
        )
        monkeypatch.setattr(transcript_reader, "_REBUILD_CHUNK_BYTES", 37)
        chunked = transcript_reader._replay_committed_prefix(
            session_file, "claude", floor, generation
        )

        assert whole is not None and chunked is not None
        assert set(chunked.pending_tools) == set(whole.pending_tools) == {"t1"}

    async def test_rebuild_stops_at_the_floor(self, tmp_path) -> None:
        history = "".join(_record(f"h{i}") for i in range(10))
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        stat = session_file.stat()
        seen: list[int] = []
        real_parse = transcript_reader.parse_records

        def spy(entries, provider_name, parser_state, **kwargs):
            seen.extend(entry.offset for entry in entries)
            return real_parse(entries, provider_name, parser_state, **kwargs)

        floor = len(_record("h0").encode()) * 4
        with patch("ccgram.transcript_reader.parse_records", spy):
            transcript_reader._replay_committed_prefix(
                session_file, "claude", floor, (stat.st_dev, stat.st_ino)
            )

        assert seen == [len(_record("h0").encode()) * i for i in range(4)]


class TestLazyStartupBaseline:
    """Construction must not hash transcripts; the first poll seeds them."""

    def _tracked(self, tmp_path, count=3):
        history = _record("h0") * 5
        state = MonitorState(state_file=tmp_path / "monitor_state.json")
        files = []
        for index in range(count):
            session_file = tmp_path / f"s{index}.jsonl"
            session_file.write_text(history)
            state.update_session(
                TrackedSession(
                    session_id=f"s{index}",
                    file_path=str(session_file),
                    last_byte_offset=len(history.encode()),
                )
            )
            files.append(session_file)
        return state, files, history

    def test_construction_hashes_nothing(self, tmp_path) -> None:
        state, _, _ = self._tracked(tmp_path)

        with (
            patch("ccgram.transcript_reader._hash_range") as hashed,
            patch("ccgram.transcript_reader._prefix_digest") as digest,
        ):
            TranscriptReader(state, IdleTracker())

        assert hashed.call_count == 0
        assert digest.call_count == 0

    async def test_first_poll_hashes_the_baseline_off_the_event_loop(
        self, tmp_path
    ) -> None:
        state, files, _ = self._tracked(tmp_path, count=1)
        reader = TranscriptReader(state, IdleTracker())
        threads: list[threading.Thread] = []
        real_range = transcript_reader._hash_range

        def record_thread(*args, **kwargs):
            threads.append(threading.current_thread())
            return real_range(*args, **kwargs)

        with patch("ccgram.transcript_reader._hash_range", record_thread):
            await reader._process_session_file("s0", files[0], [], window_id="@1")

        assert threads
        assert all(thread is not threading.main_thread() for thread in threads)

    async def test_replacement_before_the_first_poll_still_clamps(
        self, tmp_path
    ) -> None:
        state, files, history = self._tracked(tmp_path, count=1)
        reader = TranscriptReader(state, IdleTracker())
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text(_record("other") * 8)
        replacement.replace(files[0])

        messages: list = []
        await reader._process_session_file("s0", files[0], messages, window_id="@1")

        assert messages == []
        tracked = state.get_session("s0")
        assert tracked is not None
        assert tracked.parsed_offset == files[0].stat().st_size

    async def test_inode_change_after_seeding_is_still_verified_once(
        self, tmp_path
    ) -> None:
        state, files, history = self._tracked(tmp_path, count=1)
        reader = TranscriptReader(state, IdleTracker())
        await reader._process_session_file("s0", files[0], [], window_id="@1")
        copy = tmp_path / "copy.jsonl"
        copy.write_text(history + _record("fresh"))
        copy.replace(files[0])

        messages: list = []
        await reader._process_session_file("s0", files[0], messages, window_id="@1")

        assert [msg.text for msg in messages] == ["fresh"]


class TestDeferredParserRebuild:
    """A restored session rebuilds parser context only when it has output to parse."""

    _TOOL_USE = (
        '{"type":"assistant","message":{"content":[{"type":"tool_use",'
        '"id":"t1","name":"Bash","input":{"command":"ls"}}]}}\n'
    )
    _TOOL_RESULT = (
        '{"type":"user","message":{"content":[{"type":"tool_result",'
        '"tool_use_id":"t1","content":"a\\nb\\nc\\nd"}]}}\n'
    )

    def _restored(self, tmp_path, history: str):
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text(history)
        return (
            *_tracked_reader(tmp_path, session_file, len(history.encode())),
            session_file,
        )

    async def _poll(self, reader, session_file) -> list:
        messages: list = []
        await reader._process_session_file(
            "sess", session_file, messages, window_id="@1"
        )
        return messages

    async def test_idle_restored_session_is_not_rebuilt(self, tmp_path) -> None:
        reader, _, session_file = self._restored(tmp_path, _record("h0") * 20)

        with patch(
            "ccgram.transcript_reader._replay_committed_prefix",
            wraps=transcript_reader._replay_committed_prefix,
        ) as rebuild:
            for _ in range(3):
                assert await self._poll(reader, session_file) == []

        assert rebuild.call_count == 0

    async def test_first_output_after_restart_rebuilds_once(self, tmp_path) -> None:
        reader, _, session_file = self._restored(tmp_path, _record("h0") * 20)
        await self._poll(reader, session_file)

        with patch(
            "ccgram.transcript_reader._replay_committed_prefix",
            wraps=transcript_reader._replay_committed_prefix,
        ) as rebuild:
            with session_file.open("a") as transcript:
                transcript.write(_record("live"))
            delivered = await self._poll(reader, session_file)
            with session_file.open("a") as transcript:
                transcript.write(_record("later"))
            delivered += await self._poll(reader, session_file)

        assert [msg.text for msg in delivered] == ["live", "later"]
        assert rebuild.call_count == 1

    async def test_deferred_rebuild_keeps_context_from_before_the_watermark(
        self, tmp_path
    ) -> None:
        reader, _, session_file = self._restored(
            tmp_path, _record("h0") * 3 + self._TOOL_USE
        )
        await self._poll(reader, session_file)
        await self._poll(reader, session_file)

        with session_file.open("a") as transcript:
            transcript.write(self._TOOL_RESULT)
        messages = await self._poll(reader, session_file)

        results = [msg for msg in messages if msg.content_type == "tool_result"]
        assert [msg.tool_name for msg in results] == ["Bash"]
        assert results[0].text.startswith(
            transcript_reader.format_tool_line("Bash", "")[:6]
        )
