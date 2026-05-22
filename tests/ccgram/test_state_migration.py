"""Tests for state file backward compatibility."""

from ccgram.session import WindowState
from ccgram.session_map import parse_session_map


class TestWindowStateSerialization:
    def test_minimal_state_round_trip(self) -> None:
        data = {"session_id": "abc", "cwd": "/tmp"}
        ws = WindowState.from_dict(data)
        assert ws.session_id == "abc"
        assert ws.cwd == "/tmp"

    def test_sparse_serialization_omits_defaults(self) -> None:
        ws = WindowState(session_id="abc", cwd="/tmp")
        d = ws.to_dict()
        assert "window_name" not in d
        assert "transcript_path" not in d


class TestSessionMapParsing:
    def test_basic_session_map_parsing(self) -> None:
        raw = {
            "ccgram:@0": {
                "session_id": "abc-123",
                "cwd": "/tmp/project",
            }
        }
        result = parse_session_map(raw, "ccgram:")
        assert result["@0"]["session_id"] == "abc-123"
        assert result["@0"]["cwd"] == "/tmp/project"
