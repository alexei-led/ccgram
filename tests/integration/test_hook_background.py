import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_LAUNCHER = """
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from ccgram.cli import cli

real_popen = subprocess.Popen
def spawn(command, **kwargs):
    assert command == [sys.executable, "-m", "ccgram.main", "hook", "--provider", "gemini"]
    process = real_popen([sys.executable, sys.argv[1]], **kwargs)
    (Path(os.environ["CCGRAM_DIR"]) / "worker.pid").write_text(str(process.pid))
    return process

with patch("ccgram.hook.subprocess.Popen", side_effect=spawn):
    cli(["hook", "--provider", "gemini", "--background"])
"""

_WORKER = """
import fcntl
import json
import os
import sys
from pathlib import Path

state = Path(os.environ["CCGRAM_DIR"])
payload = sys.stdin.read()
print("worker diagnostic", file=sys.stderr, flush=True)
receipt = state / "received.tmp"
receipt.write_text(json.dumps({
    "pid": os.getpid(), "session": os.getsid(0), "payload": payload,
}))
receipt.replace(state / "received.json")
with (state / "worker.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
(state / "done").touch()
"""


def test_detached_worker_outlives_launcher_without_holding_provider_pipes(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER)
    payload = json.dumps({"hook_event_name": "AfterAgent", "message": "x" * 200_000})
    env = dict(os.environ, CCGRAM_DIR=str(tmp_path))
    pid = None
    with (tmp_path / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            result = subprocess.run(
                [sys.executable, "-c", _LAUNCHER, str(worker)],
                input=payload,
                text=True,
                capture_output=True,
                env=env,
                timeout=5,
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout == ""
            assert result.stderr == ""
            deadline = time.monotonic() + 5
            received = tmp_path / "received.json"
            while not received.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            record = json.loads(received.read_text())
            pid = record["pid"]
            assert record["payload"] == payload
            assert record["session"] == pid
            assert not (tmp_path / "done").exists()
            assert (tmp_path / "hook.log").read_text() == "worker diagnostic\n"
        finally:
            pid_file = tmp_path / "worker.pid"
            if pid is None and pid_file.exists():
                pid = int(pid_file.read_text())
            if pid is not None:
                os.kill(pid, signal.SIGTERM)
