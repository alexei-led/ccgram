"""Tests for hooks.state_files — versioned state-file contracts."""

import pytest

from ccgram.hooks.state_files import (
    EVENTS_SCHEMA_VERSION,
    SESSION_MAP_SCHEMA_VERSION,
    EventLogRecord,
    SessionMapEntry,
    StateFileValidationError,
    parse_event_record,
    parse_session_map_entry,
    serialize_event_record,
    serialize_session_map_entry,
)


# ---------------------------------------------------------------------------
# parse_event_record
# ---------------------------------------------------------------------------


class TestParseEventRecord:
    def test_valid_v1(self) -> None:
        raw = {
            "schema_version": 1,
            "ts": 1234567890.0,
            "event": "SessionStart",
            "window_key": "ccgram:@0",
            "session_id": "abc-123",
            "data": {"key": "val"},
        }
        rec = parse_event_record(raw)
        assert isinstance(rec, EventLogRecord)
        assert rec.schema_version == 1
        assert rec.event == "SessionStart"
        assert rec.window_key == "ccgram:@0"
        assert rec.session_id == "abc-123"
        assert rec.data == {"key": "val"}
        assert rec.ts == 1234567890.0

    def test_legacy_versionless_accepted_as_v1(self) -> None:
        """Records with no schema_version field parse as v1 (legacy compat)."""
        raw = {
            "ts": 1.0,
            "event": "Stop",
            "window_key": "ccgram:@5",
            "session_id": "def-456",
            "data": {},
        }
        rec = parse_event_record(raw)
        assert rec.schema_version == 1
        assert rec.event == "Stop"

    def test_extra_fields_ignored(self) -> None:
        raw = {
            "ts": 2.0,
            "event": "Notification",
            "window_key": "ccgram:@1",
            "session_id": "ghi-789",
            "data": {},
            "future_field": "ignored",
        }
        rec = parse_event_record(raw)
        assert rec.event == "Notification"

    def test_missing_event_field_raises(self) -> None:
        raw = {
            "ts": 1.0,
            "window_key": "ccgram:@0",
            "session_id": "abc-123",
            "data": {},
        }
        with pytest.raises(StateFileValidationError, match="missing required fields"):
            parse_event_record(raw)

    def test_missing_window_key_raises(self) -> None:
        raw = {
            "ts": 1.0,
            "event": "Stop",
            "session_id": "abc-123",
            "data": {},
        }
        with pytest.raises(StateFileValidationError, match="missing required fields"):
            parse_event_record(raw)

    def test_missing_session_id_raises(self) -> None:
        raw = {
            "ts": 1.0,
            "event": "Stop",
            "window_key": "ccgram:@0",
            "data": {},
        }
        with pytest.raises(StateFileValidationError, match="missing required fields"):
            parse_event_record(raw)

    def test_unknown_future_version_raises(self) -> None:
        raw = {
            "schema_version": EVENTS_SCHEMA_VERSION + 1,
            "ts": 1.0,
            "event": "Stop",
            "window_key": "ccgram:@0",
            "session_id": "abc-123",
            "data": {},
        }
        with pytest.raises(
            StateFileValidationError, match="Unsupported events schema_version"
        ):
            parse_event_record(raw)

    def test_non_integer_version_raises(self) -> None:
        raw = {
            "schema_version": "1",
            "ts": 1.0,
            "event": "Stop",
            "window_key": "ccgram:@0",
            "session_id": "abc-123",
            "data": {},
        }
        with pytest.raises(
            StateFileValidationError, match="Unsupported events schema_version"
        ):
            parse_event_record(raw)

    def test_non_dict_list_raises(self) -> None:
        with pytest.raises(StateFileValidationError, match="JSON object"):
            parse_event_record([])  # type: ignore[arg-type]

    def test_non_dict_scalar_raises(self) -> None:
        with pytest.raises(StateFileValidationError, match="JSON object"):
            parse_event_record(5)  # type: ignore[arg-type]

    def test_missing_ts_defaults_to_zero(self) -> None:
        raw = {
            "event": "Stop",
            "window_key": "ccgram:@0",
            "session_id": "abc-123",
            "data": {},
        }
        rec = parse_event_record(raw)
        assert rec.ts == 0.0

    def test_missing_data_defaults_to_empty_dict(self) -> None:
        raw = {
            "event": "Stop",
            "window_key": "ccgram:@0",
            "session_id": "abc-123",
        }
        rec = parse_event_record(raw)
        assert rec.data == {}


# ---------------------------------------------------------------------------
# serialize_event_record
# ---------------------------------------------------------------------------


class TestSerializeEventRecord:
    def test_includes_schema_version(self) -> None:
        d = serialize_event_record("Stop", "abc-123", "ccgram:@0", {})
        assert d["schema_version"] == EVENTS_SCHEMA_VERSION

    def test_all_fields_present(self) -> None:
        d = serialize_event_record("Stop", "abc-123", "ccgram:@0", {"k": "v"})
        assert d["event"] == "Stop"
        assert d["session_id"] == "abc-123"
        assert d["window_key"] == "ccgram:@0"
        assert d["data"] == {"k": "v"}
        assert isinstance(d["ts"], float)

    def test_explicit_ts_honored(self) -> None:
        d = serialize_event_record("Stop", "abc-123", "ccgram:@0", {}, ts=9999.0)
        assert d["ts"] == 9999.0

    def test_round_trip(self) -> None:
        d = serialize_event_record("SessionStart", "s1", "ccgram:@3", {"x": 1}, ts=1.5)
        rec = parse_event_record(d)
        assert rec.event == "SessionStart"
        assert rec.session_id == "s1"
        assert rec.window_key == "ccgram:@3"
        assert rec.data == {"x": 1}
        assert rec.ts == 1.5
        assert rec.schema_version == 1


# ---------------------------------------------------------------------------
# parse_session_map_entry
# ---------------------------------------------------------------------------


class TestParseSessionMapEntry:
    def test_valid_v1(self) -> None:
        raw = {
            "schema_version": 1,
            "session_id": "sess-1",
            "cwd": "/repo",
            "window_name": "repo",
            "transcript_path": "/path/to.jsonl",
            "provider_name": "claude",
        }
        entry = parse_session_map_entry(raw)
        assert isinstance(entry, SessionMapEntry)
        assert entry.schema_version == 1
        assert entry.session_id == "sess-1"
        assert entry.cwd == "/repo"
        assert entry.provider_name == "claude"

    def test_legacy_versionless_accepted_as_v1(self) -> None:
        raw = {
            "session_id": "sess-2",
            "cwd": "/repo2",
            "window_name": "repo2",
            "transcript_path": "",
            "provider_name": "codex",
        }
        entry = parse_session_map_entry(raw)
        assert entry.schema_version == 1
        assert entry.session_id == "sess-2"

    def test_extra_fields_ignored(self) -> None:
        raw = {
            "session_id": "sess-3",
            "cwd": "/x",
            "window_name": "x",
            "transcript_path": "",
            "provider_name": "claude",
            "future_field": "ignored",
        }
        entry = parse_session_map_entry(raw)
        assert entry.session_id == "sess-3"

    def test_missing_session_id_raises(self) -> None:
        raw = {
            "cwd": "/repo",
            "window_name": "repo",
            "transcript_path": "",
            "provider_name": "claude",
        }
        with pytest.raises(StateFileValidationError, match="missing required fields"):
            parse_session_map_entry(raw)

    def test_empty_session_id_raises(self) -> None:
        raw = {
            "session_id": "",
            "cwd": "/repo",
            "window_name": "repo",
            "transcript_path": "",
            "provider_name": "claude",
        }
        with pytest.raises(StateFileValidationError, match="missing required fields"):
            parse_session_map_entry(raw)

    def test_unknown_future_version_raises(self) -> None:
        raw = {
            "schema_version": SESSION_MAP_SCHEMA_VERSION + 1,
            "session_id": "sess-4",
            "cwd": "/repo",
            "window_name": "repo",
            "transcript_path": "",
            "provider_name": "claude",
        }
        with pytest.raises(
            StateFileValidationError, match="Unsupported session_map schema_version"
        ):
            parse_session_map_entry(raw)

    def test_non_dict_list_raises(self) -> None:
        with pytest.raises(StateFileValidationError, match="JSON object"):
            parse_session_map_entry([])  # type: ignore[arg-type]

    def test_non_dict_scalar_raises(self) -> None:
        with pytest.raises(StateFileValidationError, match="JSON object"):
            parse_session_map_entry("oops")  # type: ignore[arg-type]

    def test_optional_fields_default_to_empty_string(self) -> None:
        raw = {"session_id": "sess-5"}
        entry = parse_session_map_entry(raw)
        assert entry.cwd == ""
        assert entry.window_name == ""
        assert entry.transcript_path == ""
        assert entry.provider_name == ""


# ---------------------------------------------------------------------------
# serialize_session_map_entry
# ---------------------------------------------------------------------------


class TestSerializeSessionMapEntry:
    def test_includes_schema_version(self) -> None:
        d = serialize_session_map_entry("s1", "/repo", "repo", "/t.jsonl", "claude")
        assert d["schema_version"] == SESSION_MAP_SCHEMA_VERSION

    def test_all_fields_present(self) -> None:
        d = serialize_session_map_entry("s1", "/repo", "myrepo", "/t.jsonl", "codex")
        assert d["session_id"] == "s1"
        assert d["cwd"] == "/repo"
        assert d["window_name"] == "myrepo"
        assert d["transcript_path"] == "/t.jsonl"
        assert d["provider_name"] == "codex"

    def test_round_trip(self) -> None:
        d = serialize_session_map_entry("s2", "/x", "x", "/x.jsonl", "gemini")
        entry = parse_session_map_entry(d)
        assert entry.session_id == "s2"
        assert entry.cwd == "/x"
        assert entry.window_name == "x"
        assert entry.transcript_path == "/x.jsonl"
        assert entry.provider_name == "gemini"
        assert entry.schema_version == 1
