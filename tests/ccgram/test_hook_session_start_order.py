"""SessionStart hooks run asynchronously and may finish out of order.

A quick "start claude, then /resume" fires two SessionStart hooks for one
window. session_map must end up with the session from the later-started hook,
whichever process writes last, and a busy lock must not silently drop it.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ccgram import hook
from ccgram.hook import _update_session_map

KEY = "hc:hc-t16"


def _write(tmp_path: Path, session_id: str, started_at: float | None) -> None:
    with patch("ccgram.utils.ccgram_dir", return_value=tmp_path):
        _update_session_map(
            session_window_key=KEY,
            session_id=session_id,
            cwd="/repo",
            window_name="",
            transcript_path=f"/repo/{session_id}.jsonl",
            tmux_session_name="hc",
            started_at=started_at,
        )


def _entry(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "session_map.json").read_text())[KEY]


def test_later_started_hook_wins_even_when_it_writes_first(tmp_path: Path) -> None:
    _write(tmp_path, "resumed", started_at=200.0)
    _write(tmp_path, "fresh-startup", started_at=100.0)

    entry = _entry(tmp_path)
    assert entry["session_id"] == "resumed"
    assert entry["started_at"] == 200.0


def test_in_order_hooks_overwrite(tmp_path: Path) -> None:
    _write(tmp_path, "fresh-startup", started_at=100.0)
    _write(tmp_path, "resumed", started_at=200.0)

    assert _entry(tmp_path)["session_id"] == "resumed"


def test_entries_without_a_start_time_are_overwritten(tmp_path: Path) -> None:
    _write(tmp_path, "legacy", started_at=None)
    _write(tmp_path, "new", started_at=100.0)
    assert _entry(tmp_path)["session_id"] == "new"

    # And a writer without a start time (hookless registration) never skips.
    _write(tmp_path, "unordered", started_at=None)
    assert _entry(tmp_path)["session_id"] == "unordered"


def test_other_windows_do_not_block_the_write(tmp_path: Path) -> None:
    (tmp_path / "session_map.json").write_text(
        json.dumps({"hc:hc-t6": {"session_id": "x", "started_at": 999.0}})
    )
    _write(tmp_path, "mine", started_at=1.0)
    assert _entry(tmp_path)["session_id"] == "mine"


def test_session_start_waits_longer_for_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CCGRAM_SESSION_START_LOCK_TIMEOUT_SEC", raising=False)
    assert hook._session_start_lock_timeout() == 10.0
    assert hook._session_start_lock_timeout() > hook._HOOK_LOCK_TIMEOUT_SECONDS

    monkeypatch.setenv("CCGRAM_SESSION_START_LOCK_TIMEOUT_SEC", "1.5")
    assert hook._session_start_lock_timeout() == 1.5
    monkeypatch.setenv("CCGRAM_SESSION_START_LOCK_TIMEOUT_SEC", "nonsense")
    assert hook._session_start_lock_timeout() == 10.0
    monkeypatch.setenv("CCGRAM_SESSION_START_LOCK_TIMEOUT_SEC", "-1")
    assert hook._session_start_lock_timeout() == 10.0


def test_update_passes_the_lock_timeout(tmp_path: Path) -> None:
    with (
        patch("ccgram.utils.ccgram_dir", return_value=tmp_path),
        patch.object(hook, "_acquire_hook_lock", return_value=False) as acquire,
    ):
        _update_session_map(
            session_window_key=KEY,
            session_id="s",
            cwd="/repo",
            window_name="",
            transcript_path="/repo/s.jsonl",
            tmux_session_name="hc",
            lock_timeout=7.0,
        )
    assert acquire.call_args.args[1] == 7.0
    assert not (tmp_path / "session_map.json").exists()
