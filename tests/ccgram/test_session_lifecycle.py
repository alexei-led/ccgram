"""Tests for session-map reconciliation behavior."""

from ccgram.idle_tracker import IdleTracker
from ccgram.session_lifecycle import SessionLifecycle


def test_case_only_session_map_change_is_not_delete_and_create() -> None:
    lifecycle = SessionLifecycle()
    lifecycle.initialize(
        {
            "abc-def": {
                "session_id": "sess-1",
                "cwd": "/repo",
                "window_name": "before",
                "transcript_path": "/tmp/transcript.jsonl",
            }
        }
    )

    result = lifecycle.reconcile(
        {
            "ABC-DEF": {
                "session_id": "sess-1",
                "cwd": "/repo",
                "window_name": "after",
                "transcript_path": "/tmp/transcript.jsonl",
            }
        },
        IdleTracker(),
    )

    assert result.sessions_to_remove == set()
    assert result.new_windows == {}
    assert set(result.current_map) == {"abc-def"}


def test_same_transcript_session_refresh_preserves_monitor_state() -> None:
    lifecycle = SessionLifecycle()
    lifecycle.initialize(
        {
            "@1": {
                "session_id": "sess-before-rename",
                "cwd": "/repo",
                "window_name": "before",
                "transcript_path": "/tmp/transcript.jsonl",
            }
        }
    )

    result = lifecycle.reconcile(
        {
            "@1": {
                "session_id": "sess-after-rename",
                "cwd": "/repo",
                "window_name": "after",
                "transcript_path": "/tmp/transcript.jsonl",
            }
        },
        IdleTracker(),
    )

    assert result.sessions_to_remove == set()
    assert result.changed_windows == {}


def _entry(session_id: str, transcript: str = "/tmp/transcript.jsonl") -> dict:
    return {
        "session_id": session_id,
        "cwd": "/repo",
        "window_name": "w",
        "transcript_path": transcript,
    }


def test_session_that_moves_to_another_window_is_not_removed() -> None:
    lifecycle = SessionLifecycle()
    lifecycle.initialize({"@1": _entry("sess-1")})
    idle = IdleTracker()
    idle.record_activity("sess-1", 100.0)

    result = lifecycle.reconcile({"@2": _entry("sess-1")}, idle)

    assert result.sessions_to_remove == set()
    assert idle.get_last_activity("sess-1") == 100.0
    assert set(result.new_windows) == {"@2"}


def test_session_replaced_in_one_window_but_live_in_another_is_not_removed() -> None:
    lifecycle = SessionLifecycle()
    lifecycle.initialize(
        {"@1": _entry("sess-1", "/tmp/a.jsonl"), "@2": _entry("sess-2", "/tmp/b.jsonl")}
    )
    idle = IdleTracker()
    idle.record_activity("sess-1", 100.0)

    result = lifecycle.reconcile(
        {
            "@1": _entry("sess-3", "/tmp/c.jsonl"),
            "@2": _entry("sess-1", "/tmp/a.jsonl"),
        },
        idle,
    )

    assert "sess-1" not in result.sessions_to_remove
    assert result.sessions_to_remove == {"sess-2"}
    assert idle.get_last_activity("sess-1") == 100.0


def test_deleted_window_session_that_is_gone_everywhere_is_removed() -> None:
    lifecycle = SessionLifecycle()
    lifecycle.initialize({"@1": _entry("sess-1"), "@2": _entry("sess-2", "/tmp/b")})
    idle = IdleTracker()
    idle.record_activity("sess-1", 100.0)

    result = lifecycle.reconcile({"@2": _entry("sess-2", "/tmp/b")}, idle)

    assert result.sessions_to_remove == {"sess-1"}
    assert idle.get_last_activity("sess-1") is None
