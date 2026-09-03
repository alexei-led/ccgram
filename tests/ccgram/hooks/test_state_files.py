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

_VALID_EVENT = {
    "schema_version": 1,
    "ts": 1234567890.0,
    "event": "SessionStart",
    "window_key": "ccgram:@0",
    "session_id": "abc-123",
    "data": {"key": "val"},
}

_VALID_SESSION_MAP = {
    "schema_version": 1,
    "session_id": "sess-1",
    "cwd": "/repo",
    "window_name": "repo",
    "transcript_path": "/path/to.jsonl",
    "provider_name": "claude",
}


def _event(**overrides: object) -> dict:
    return _VALID_EVENT | overrides


def _entry(**overrides: object) -> dict:
    return _VALID_SESSION_MAP | overrides


class TestParseEventRecord:
    def test_valid_v1(self) -> None:
        rec = parse_event_record(_VALID_EVENT)
        assert rec == EventLogRecord(
            schema_version=1,
            ts=1234567890.0,
            event="SessionStart",
            window_key="ccgram:@0",
            session_id="abc-123",
            data={"key": "val"},
        )

    def test_legacy_versionless_accepted_as_v1(self) -> None:
        """Records written before versioning must still parse byte-identically."""
        raw = _event()
        del raw["schema_version"]
        assert parse_event_record(raw).schema_version == 1

    def test_extra_fields_ignored(self) -> None:
        rec = parse_event_record(_event(future_field="ignored"))
        assert rec.event == "SessionStart"

    @pytest.mark.parametrize("field", ["event", "window_key", "session_id"])
    def test_missing_required_field_raises(self, field: str) -> None:
        raw = _event()
        del raw[field]
        with pytest.raises(StateFileValidationError, match="missing required fields"):
            parse_event_record(raw)

    @pytest.mark.parametrize("field", ["event", "window_key", "session_id"])
    def test_empty_required_field_raises(self, field: str) -> None:
        with pytest.raises(StateFileValidationError, match="missing required fields"):
            parse_event_record(_event(**{field: ""}))

    @pytest.mark.parametrize(
        "version",
        [
            pytest.param(EVENTS_SCHEMA_VERSION + 1, id="future_int"),
            pytest.param("1", id="string"),
            pytest.param(1.0, id="float"),
        ],
    )
    def test_unsupported_version_raises(self, version: object) -> None:
        with pytest.raises(
            StateFileValidationError, match="Unsupported events schema_version"
        ):
            parse_event_record(_event(schema_version=version))

    @pytest.mark.parametrize(
        "raw", [pytest.param([], id="list"), pytest.param(5, id="scalar")]
    )
    def test_non_dict_raises(self, raw: object) -> None:
        with pytest.raises(StateFileValidationError, match="JSON object"):
            parse_event_record(raw)  # type: ignore[arg-type]

    def test_optional_fields_get_defaults(self) -> None:
        raw = _event()
        del raw["ts"]
        del raw["data"]
        rec = parse_event_record(raw)
        assert rec.ts == 0.0
        assert rec.data == {}


class TestSerializeEventRecord:
    def test_all_fields_present(self) -> None:
        d = serialize_event_record("Stop", "abc-123", "ccgram:@0", {"k": "v"})
        assert d["schema_version"] == EVENTS_SCHEMA_VERSION
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
        assert parse_event_record(d) == EventLogRecord(
            schema_version=1,
            ts=1.5,
            event="SessionStart",
            window_key="ccgram:@3",
            session_id="s1",
            data={"x": 1},
        )


class TestParseSessionMapEntry:
    def test_valid_v1(self) -> None:
        assert parse_session_map_entry(_VALID_SESSION_MAP) == SessionMapEntry(
            schema_version=1,
            session_id="sess-1",
            cwd="/repo",
            window_name="repo",
            transcript_path="/path/to.jsonl",
            provider_name="claude",
        )

    def test_legacy_versionless_accepted_as_v1(self) -> None:
        raw = _entry()
        del raw["schema_version"]
        assert parse_session_map_entry(raw).schema_version == 1

    def test_extra_fields_ignored(self) -> None:
        entry = parse_session_map_entry(_entry(future_field="ignored"))
        assert entry.session_id == "sess-1"

    @pytest.mark.parametrize(
        "session_id",
        [pytest.param(None, id="missing"), pytest.param("", id="empty")],
    )
    def test_session_id_is_required(self, session_id: str | None) -> None:
        raw = _entry()
        if session_id is None:
            del raw["session_id"]
        else:
            raw["session_id"] = session_id
        with pytest.raises(StateFileValidationError, match="missing required fields"):
            parse_session_map_entry(raw)

    @pytest.mark.parametrize(
        "version",
        [
            pytest.param(SESSION_MAP_SCHEMA_VERSION + 1, id="future_int"),
            pytest.param("1", id="string"),
            pytest.param(True, id="bool_is_not_an_int_version"),
        ],
    )
    def test_unsupported_version_raises(self, version: object) -> None:
        with pytest.raises(
            StateFileValidationError, match="Unsupported session_map schema_version"
        ):
            parse_session_map_entry(_entry(schema_version=version))

    @pytest.mark.parametrize(
        "raw", [pytest.param([], id="list"), pytest.param("oops", id="scalar")]
    )
    def test_non_dict_raises(self, raw: object) -> None:
        with pytest.raises(StateFileValidationError, match="JSON object"):
            parse_session_map_entry(raw)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field",
        ["session_id", "cwd", "window_name", "transcript_path", "provider_name"],
    )
    def test_non_string_field_raises(self, field: str) -> None:
        """A corrupt hook write must not put a non-str into WindowState."""
        with pytest.raises(StateFileValidationError, match="must be strings"):
            parse_session_map_entry(_entry(**{field: ["not", "a", "string"]}))

    def test_optional_fields_default_to_empty_string(self) -> None:
        entry = parse_session_map_entry({"session_id": "sess-5"})
        assert (
            entry.cwd,
            entry.window_name,
            entry.transcript_path,
            entry.provider_name,
        ) == ("", "", "", "")
        assert entry.replay_from_start is False

    def test_replay_from_start_must_be_boolean(self) -> None:
        with pytest.raises(
            StateFileValidationError, match="replay_from_start must be a boolean"
        ):
            parse_session_map_entry(_entry(replay_from_start="true"))


class TestSerializeSessionMapEntry:
    def test_all_fields_present(self) -> None:
        d = serialize_session_map_entry("s1", "/repo", "myrepo", "/t.jsonl", "codex")
        assert d == {
            "schema_version": SESSION_MAP_SCHEMA_VERSION,
            "session_id": "s1",
            "cwd": "/repo",
            "window_name": "myrepo",
            "transcript_path": "/t.jsonl",
            "provider_name": "codex",
        }

    def test_round_trip(self) -> None:
        d = serialize_session_map_entry(
            "s2",
            "/x",
            "x",
            "/x.jsonl",
            "gemini",
            replay_from_start=True,
        )
        assert d["replay_from_start"] is True
        assert parse_session_map_entry(d) == SessionMapEntry(
            schema_version=1,
            session_id="s2",
            cwd="/x",
            window_name="x",
            transcript_path="/x.jsonl",
            provider_name="gemini",
            replay_from_start=True,
        )
