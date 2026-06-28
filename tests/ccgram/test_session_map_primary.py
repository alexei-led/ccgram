"""Session map primary election regression tests."""

import json
import os
import time
from pathlib import Path

from ccgram.session_map import parse_session_map, session_map_sync
from ccgram.window_state_store import WindowState, window_store


def _write_transcript(path: Path, age_seconds: float) -> None:
    path.write_text('{"type":"assistant"}\n')
    mtime = time.time() - age_seconds
    os.utime(path, (mtime, mtime))


def _info(session_id: str, transcript: Path) -> dict[str, str]:
    return {
        "session_id": session_id,
        "cwd": "/repo",
        "window_name": "repo",
        "transcript_path": str(transcript),
        "provider_name": "claude",
    }


def test_parse_session_map_preserves_fresh_existing_primary(tmp_path: Path) -> None:
    parent = tmp_path / "parent.jsonl"
    child = tmp_path / "child.jsonl"
    _write_transcript(parent, 2)
    _write_transcript(child, 0)
    window_store.window_states["@7"] = WindowState(
        session_id="parent", cwd="/repo", transcript_path=str(parent)
    )

    parsed = parse_session_map({"ccgram:@7": _info("child", child)}, "ccgram:")

    assert parsed["@7"]["session_id"] == "parent"
    assert parsed["@7"]["transcript_path"] == str(parent)


def test_parse_session_map_preserves_existing_primary_when_newer_than_candidate(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.jsonl"
    child = tmp_path / "child.jsonl"
    _write_transcript(parent, 120)
    _write_transcript(child, 180)
    window_store.window_states["@7"] = WindowState(
        session_id="parent", cwd="/repo", transcript_path=str(parent)
    )

    parsed = parse_session_map({"ccgram:@7": _info("child", child)}, "ccgram:")

    assert parsed["@7"]["session_id"] == "parent"


def test_parse_session_map_adopts_newer_stale_candidate(tmp_path: Path) -> None:
    parent = tmp_path / "parent.jsonl"
    new = tmp_path / "new.jsonl"
    _write_transcript(parent, 180)
    _write_transcript(new, 2)
    window_store.window_states["@7"] = WindowState(
        session_id="parent", cwd="/repo", transcript_path=str(parent)
    )

    parsed = parse_session_map({"ccgram:@7": _info("new", new)}, "ccgram:")

    assert parsed["@7"]["session_id"] == "new"


def test_parse_session_map_adopts_when_existing_state_was_cleared(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "new.jsonl"
    _write_transcript(transcript, 0)
    window_store.window_states["@7"] = WindowState()

    parsed = parse_session_map({"ccgram:@7": _info("new", transcript)}, "ccgram:")

    assert parsed["@7"]["session_id"] == "new"


async def test_load_session_map_preserves_primary_window_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent = tmp_path / "parent.jsonl"
    child = tmp_path / "child.jsonl"
    _write_transcript(parent, 2)
    _write_transcript(child, 0)
    session_map_file = tmp_path / "session_map.json"
    session_map_file.write_text(json.dumps({"ccgram:@7": _info("child", child)}))
    monkeypatch.setattr("ccgram.session_map.config.session_map_file", session_map_file)
    monkeypatch.setattr("ccgram.session_map.config.tmux_session_name", "ccgram")
    monkeypatch.setattr(session_map_sync, "_schedule_save", lambda: None)
    window_store.window_states["@7"] = WindowState(
        session_id="parent",
        cwd="/repo",
        window_name="repo",
        transcript_path=str(parent),
    )

    await session_map_sync.load_session_map()

    state = window_store.window_states["@7"]
    assert state.session_id == "parent"
    assert state.transcript_path == str(parent)


async def test_load_session_map_malformed_entry_does_not_delete_window_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A future-version or otherwise invalid session_map entry must NOT cause
    the in-memory WindowState for that window_id to be deleted (MAJOR-1 regression).

    The malformed entry is present in the file so the window_id must be added to
    valid_wids regardless of schema validation outcome.
    """
    transcript = tmp_path / "valid.jsonl"
    _write_transcript(transcript, 0)
    session_map_file = tmp_path / "session_map.json"
    # valid entry for @5, invalid (future schema_version) entry for @9
    session_map_file.write_text(
        json.dumps(
            {
                "ccgram:@5": _info("sess-5", transcript),
                "ccgram:@9": {
                    "schema_version": 9999,
                    "session_id": "sess-9",
                    "cwd": "/repo",
                    "window_name": "repo",
                    "transcript_path": str(transcript),
                    "provider_name": "claude",
                },
            }
        )
    )
    monkeypatch.setattr("ccgram.session_map.config.session_map_file", session_map_file)
    monkeypatch.setattr("ccgram.session_map.config.tmux_session_name", "ccgram")
    monkeypatch.setattr(session_map_sync, "_schedule_save", lambda: None)
    # Pre-populate in-memory state for @9 (no thread binding, not old-format sid)
    window_store.window_states["@5"] = WindowState(session_id="sess-5", cwd="/repo")
    window_store.window_states["@9"] = WindowState(session_id="sess-9", cwd="/repo")

    await session_map_sync.load_session_map()

    # @9's in-memory state must survive — the malformed file entry must not
    # cause _remove_stale_window_states to delete it
    assert "@9" in window_store.window_states, (
        "WindowState for @9 was deleted despite being present in session_map.json"
    )
    # @5 (valid entry) state is updated normally
    assert "@5" in window_store.window_states


def test_grace_env_allows_adopting_candidate(tmp_path: Path, monkeypatch) -> None:
    parent = tmp_path / "parent.jsonl"
    new = tmp_path / "new.jsonl"
    _write_transcript(parent, 5)
    _write_transcript(new, 1)
    monkeypatch.setenv("CCGRAM_NESTED_SESSION_GRACE_SEC", "1")
    window_store.window_states["@7"] = WindowState(
        session_id="parent", cwd="/repo", transcript_path=str(parent)
    )

    parsed = parse_session_map({"ccgram:@7": _info("new", new)}, "ccgram:")

    assert parsed["@7"]["session_id"] == "new"
