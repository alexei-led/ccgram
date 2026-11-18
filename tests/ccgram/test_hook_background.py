import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ccgram.cli import cli


def test_background_hook_hands_off_payload_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    payload = json.dumps({"hook_event_name": "AfterAgent", "message": "x" * 200_000})
    handed_off = {}

    def spawn(command, **kwargs):
        handed_off["command"] = command
        handed_off["payload"] = kwargs["stdin"].read()
        handed_off["stdout"] = kwargs["stdout"]
        handed_off["detached"] = kwargs["start_new_session"]
        assert kwargs["stderr"].name == str(tmp_path / "hook.log")
        return object()

    with patch("ccgram.hook.subprocess.Popen", side_effect=spawn):
        result = CliRunner().invoke(
            cli, ["hook", "--provider", "gemini", "--background"], input=payload
        )

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert handed_off == {
        "command": [
            sys.executable,
            "-m",
            "ccgram.main",
            "hook",
            "--provider",
            "gemini",
        ],
        "payload": payload,
        "stdout": subprocess.DEVNULL,
        "detached": True,
    }


def test_background_hook_spawn_failure_does_not_block_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    with patch("ccgram.hook.subprocess.Popen", side_effect=OSError("cannot spawn")):
        result = CliRunner().invoke(
            cli, ["hook", "--provider", "gemini", "--background"], input="{}"
        )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "cannot spawn" in result.stderr


def test_background_auto_detection_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    from ccgram.hook import hook_main

    with patch("ccgram.hook.subprocess.Popen") as spawn:
        hook_main(background=True)

    assert spawn.call_args.args[0] == [sys.executable, "-m", "ccgram.main", "hook"]
