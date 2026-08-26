import json

import pytest

from ccgram.monitor_state import MonitorState, TrackedSession


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "monitor_state.json"


@pytest.fixture
def state(state_file) -> MonitorState:
    return MonitorState(state_file=state_file)


class TestTrackedSession:
    def test_to_dict_from_dict_roundtrip(self):
        original = TrackedSession(
            session_id="sess-1",
            file_path="/tmp/test.jsonl",
            last_byte_offset=42,
        )
        restored = TrackedSession.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_missing_fields_uses_defaults(self):
        session = TrackedSession.from_dict({})
        assert session == TrackedSession(
            session_id="", file_path="", last_byte_offset=0
        )


class TestPersistence:
    def test_save_then_load_restores_offsets(self, state, state_file):
        state.update_session(
            TrackedSession(session_id="s1", file_path="/a.jsonl", last_byte_offset=10)
        )
        state.events_offset = 512
        state.save()

        restored = MonitorState(state_file=state_file)
        restored.load()

        assert restored.tracked_sessions["s1"].last_byte_offset == 10
        assert restored.tracked_sessions["s1"].file_path == "/a.jsonl"
        # The events offset is what stops the bot replaying hook history on
        # restart, so it has to survive the round trip too.
        assert restored.events_offset == 512

    def test_load_missing_file(self, state):
        state.load()
        assert state.tracked_sessions == {}
        assert state.events_offset == 0

    def test_corrupt_state_loads_as_empty(self, state, state_file):
        state_file.write_text("{invalid json!!!")
        state.load()
        assert state.tracked_sessions == {}

    def test_failed_save_leaves_state_dirty_for_a_later_retry(
        self, state, state_file, monkeypatch
    ):
        state.update_session(TrackedSession(session_id="s1", file_path="/a.jsonl"))

        def explode(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("ccgram.monitor_state.atomic_write_json", explode)
        state.save()  # must not raise

        assert not state_file.exists()
        # Still dirty → the next save_if_dirty retries instead of dropping data.
        monkeypatch.undo()
        state.save_if_dirty()
        assert json.loads(state_file.read_text())["tracked_sessions"]["s1"]

    def test_save_if_dirty_skips_an_unchanged_state(self, state, state_file):
        state.save_if_dirty()
        assert not state_file.exists()


class TestSessionRegistry:
    def test_get_session_returns_the_tracked_instance(self, state):
        session = TrackedSession(session_id="s1", file_path="/a.jsonl")
        state.update_session(session)
        assert state.get_session("s1") is session

    def test_get_session_returns_none_when_unknown(self, state):
        assert state.get_session("nonexistent") is None

    def test_update_session_replaces_the_previous_record(self, state):
        state.update_session(TrackedSession(session_id="s1", file_path="/a.jsonl"))
        state.update_session(
            TrackedSession(session_id="s1", file_path="/b.jsonl", last_byte_offset=7)
        )
        assert state.tracked_sessions["s1"].file_path == "/b.jsonl"
        assert state.tracked_sessions["s1"].last_byte_offset == 7

    def test_remove_session_deletes(self, state):
        state.update_session(TrackedSession(session_id="s1", file_path="/a.jsonl"))
        state.remove_session("s1")
        assert "s1" not in state.tracked_sessions

    def test_remove_session_missing_no_error(self, state, state_file):
        state.remove_session("nonexistent")
        assert state.tracked_sessions == {}
        state.save_if_dirty()
        assert not state_file.exists()  # nothing changed → nothing written
