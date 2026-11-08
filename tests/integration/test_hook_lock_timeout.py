import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_HOOK_SCRIPT = """
from unittest.mock import patch
from ccgram.hook import hook_main

with patch("ccgram.hook._locate_primary_window", return_value=("ccgram:@1", "@1", "agent")):
    hook_main(provider_name="claude")
"""


@pytest.mark.parametrize(
    ("lock_name", "event"),
    [
        ("session_map.lock", "SessionStart"),
        ("events.jsonl", "SessionStart"),
        ("events.jsonl", "Stop"),
    ],
)
def test_hook_gives_up_on_contended_lock(
    tmp_path: Path, lock_name: str, event: str
) -> None:
    session_map = tmp_path / "session_map.json"
    session_map.write_text('{"untouched": true}')
    events = tmp_path / "events.jsonl"
    events.write_text("")
    payload = json.dumps(
        {
            "hook_event_name": event,
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "cwd": str(tmp_path),
            "transcript_path": str(tmp_path / "transcript.jsonl"),
        }
    )
    # SessionStart waits longer for session_map.lock by default; keep the
    # give-up path quick here (the longer wait has its own test below).
    env = dict(
        os.environ,
        CCGRAM_DIR=str(tmp_path),
        CCGRAM_SESSION_START_LOCK_TIMEOUT_SEC="2",
    )
    env.pop("PI_SUBAGENT_CHILD", None)
    with (tmp_path / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-c", _HOOK_SCRIPT],
            input=payload,
            text=True,
            capture_output=True,
            env=env,
            timeout=4,
        )
        elapsed = time.monotonic() - start

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert elapsed < 3.5
    assert "Timed out" in result.stderr
    assert lock_name in result.stderr
    if lock_name == "session_map.lock":
        assert json.loads(session_map.read_text()) == {"untouched": True}
        assert json.loads(events.read_text())["event"] == event
    else:
        assert events.read_text() == ""

    retry = subprocess.run(
        [sys.executable, "-c", _HOOK_SCRIPT],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        timeout=4,
    )
    assert retry.returncode == 0, retry.stderr
    recorded = [json.loads(line) for line in events.read_text().splitlines()]
    expected_count = 2 if lock_name == "session_map.lock" else 1
    assert [record["event"] for record in recorded] == [event] * expected_count


def test_session_start_outlasts_a_lock_held_past_two_seconds(tmp_path: Path) -> None:
    """A busy machine holding session_map.lock for 3 s no longer drops the update."""
    (tmp_path / "session_map.json").write_text("{}")
    (tmp_path / "events.jsonl").write_text("")
    payload = json.dumps(
        {
            "hook_event_name": "SessionStart",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "cwd": str(tmp_path),
            "transcript_path": str(tmp_path / "transcript.jsonl"),
        }
    )
    env = dict(os.environ, CCGRAM_DIR=str(tmp_path))
    env.pop("PI_SUBAGENT_CHILD", None)
    env.pop("CCGRAM_SESSION_START_LOCK_TIMEOUT_SEC", None)
    with (tmp_path / "session_map.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proc = subprocess.Popen(
            [sys.executable, "-c", _HOOK_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert proc.stdin is not None
        proc.stdin.write(payload)
        proc.stdin.close()
        time.sleep(3)
        fcntl.flock(lock, fcntl.LOCK_UN)
    _, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0, stderr
    assert "Timed out" not in stderr
    entry = json.loads((tmp_path / "session_map.json").read_text())["ccgram:@1"]
    assert entry["session_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert isinstance(entry["started_at"], float)
