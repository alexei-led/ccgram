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
