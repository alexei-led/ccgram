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
    env = dict(os.environ, CCGRAM_DIR=str(tmp_path))
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
