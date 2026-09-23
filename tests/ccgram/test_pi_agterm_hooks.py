"""Pi hook-runner lifecycle over agterm's TTY-less window identity."""

import io
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from ccgram.cli import cli
from ccgram.config import config
from ccgram.hook import _encode_pi_cwd_dirname, hook_main
from ccgram.session_monitor import SessionMonitor

_WINDOW_ID = "965E9FB9-365D-4949-A533-0CB498A33C44"
_SESSION_ID = "01a0cf3e-fdab-71f3-ae2b-8c97eff39382"
_OLD_SESSION_ID = "01a0c8d4-373b-71b4-a3fd-769091c5c90b"
_WINDOW_KEY = f"agterm:{_WINDOW_ID}"


@pytest.fixture
def hook_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in (
        "TMUX_PANE",
        "HERDR_WORKSPACE_ID",
        "HERDR_PANE_ID",
        "PI_SUBAGENT_CHILD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("AGTERM_SESSION_ID", _WINDOW_ID)
    monkeypatch.setenv("PI_CODING_AGENT", "true")
    monkeypatch.setattr(
        "ccgram.hook._agterm_hook_target", lambda target, *_args: (target, "")
    )
    return tmp_path


def _hook(
    monkeypatch: pytest.MonkeyPatch, event: str, cwd: str, **extra: object
) -> None:
    payload = {"session_id": _SESSION_ID, "cwd": cwd, "hook_event_name": event, **extra}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    hook_main()


def _transcript(home: Path, cwd: str, session_id: str) -> Path:
    directory = home / ".pi" / "agent" / "sessions" / _encode_pi_cwd_dirname(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"2026-09-23T17-10-07-020Z_{session_id}.jsonl"
    path.write_text(
        json.dumps({"type": "session", "id": session_id, "cwd": cwd}) + "\n"
    )
    return path


def test_pi_lifecycle_resolves_exact_transcript_after_session_start(
    hook_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = str(hook_home / "project")
    stale = _transcript(hook_home, cwd, _OLD_SESSION_ID)
    _hook(monkeypatch, "SessionStart", cwd)
    map_path = hook_home / "state" / "session_map.json"
    entry = json.loads(map_path.read_text())[_WINDOW_KEY]
    assert entry["provider_name"] == "pi"
    assert entry["session_id"] == _SESSION_ID
    assert entry["transcript_path"] == ""
    assert entry["replay_from_start"] is True

    current = _transcript(hook_home, cwd, _SESSION_ID)
    assert current != stale
    _hook(monkeypatch, "Stop", cwd)
    entry = json.loads(map_path.read_text())[_WINDOW_KEY]
    assert entry["provider_name"] == "pi"
    assert entry["transcript_path"] == str(current)
    assert entry["replay_from_start"] is True
    events = [
        json.loads(line)
        for line in (hook_home / "state" / "events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["SessionStart", "Stop"]
    assert all(event["data"]["provider_name"] == "pi" for event in events)


def test_pi_stop_repairs_misclassified_session_map(
    hook_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = str(hook_home / "project")
    current = _transcript(hook_home, cwd, _SESSION_ID)
    map_path = hook_home / "state" / "session_map.json"
    map_path.parent.mkdir()
    map_path.write_text(
        json.dumps(
            {
                _WINDOW_KEY: {
                    "session_id": _OLD_SESSION_ID,
                    "cwd": cwd,
                    "window_name": "pi",
                    "provider_name": "claude",
                    "transcript_path": "",
                }
            }
        )
    )

    _hook(monkeypatch, "Stop", cwd)

    entry = json.loads(map_path.read_text())[_WINDOW_KEY]
    assert entry["session_id"] == _SESSION_ID
    assert entry["provider_name"] == "pi"
    assert entry["transcript_path"] == str(current)
    assert entry["replay_from_start"] is True


@pytest.mark.parametrize("event", ["SessionStart", "Stop", "SessionEnd"])
@pytest.mark.parametrize("annotated", [False, True])
def test_background_pi_child_cannot_replace_parent_or_emit_lifecycle_events(
    hook_home: Path, monkeypatch: pytest.MonkeyPatch, event: str, annotated: bool
) -> None:
    cwd = str(hook_home / "project")
    _transcript(hook_home, cwd, _SESSION_ID)
    _hook(monkeypatch, "SessionStart", cwd)
    map_path = hook_home / "state" / "session_map.json"
    events_path = hook_home / "state" / "events.jsonl"
    before = (map_path.read_bytes(), events_path.read_bytes())
    monkeypatch.setenv("PI_SUBAGENT_CHILD", "1")
    extra = {"provider_name": "pi"} if annotated else {}

    _hook(
        monkeypatch,
        event,
        str(hook_home / "other-project"),
        session_id=_OLD_SESSION_ID,
        **extra,
    )

    assert (map_path.read_bytes(), events_path.read_bytes()) == before


def test_explicit_claude_payload_wins_over_inherited_pi_environment(
    hook_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = str(hook_home / "project")
    transcript = hook_home / ".claude" / "projects" / "project" / f"{_SESSION_ID}.jsonl"
    _hook(monkeypatch, "SessionStart", cwd, transcript_path=str(transcript))
    entry = json.loads((hook_home / "state" / "session_map.json").read_text())[
        _WINDOW_KEY
    ]
    assert entry["provider_name"] == "claude"
    assert entry["transcript_path"] == str(transcript)


async def test_recovered_pi_hook_delivers_existing_reply_to_monitor(
    hook_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = str(hook_home / "project")
    transcript = _transcript(hook_home, cwd, _SESSION_ID)
    with transcript.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Recovered Pi reply"},
                        ],
                    },
                }
            )
            + "\n"
        )
    _hook(monkeypatch, "Stop", cwd)
    map_path = hook_home / "state" / "session_map.json"
    monkeypatch.setattr(config, "session_map_file", map_path)
    monkeypatch.setattr(config, "multiplexer_name", "agterm")
    entry = json.loads(map_path.read_text())[_WINDOW_KEY]
    monitor = SessionMonitor(
        projects_path=hook_home / "projects",
        state_file=hook_home / "monitor_state.json",
    )

    messages = await monitor.check_for_updates({_WINDOW_ID: entry})

    assert [message.text for message in messages] == ["Recovered Pi reply"]
    assert "replay_from_start" not in json.loads(map_path.read_text())[_WINDOW_KEY]


@pytest.mark.parametrize("provider", [None, "claude", "pi"])
def test_cli_provider_option_overrides_pi_environment(
    hook_home: Path, provider: str | None
) -> None:
    cwd = str(hook_home / "project")
    args = ["hook"]
    if provider:
        args += ["--provider", provider]
    result = CliRunner().invoke(
        cli,
        args,
        input=json.dumps(
            {
                "session_id": _SESSION_ID,
                "cwd": cwd,
                "hook_event_name": "SessionStart",
            }
        ),
    )
    assert result.exit_code == 0, result.output
    entry = json.loads((hook_home / "state" / "session_map.json").read_text())[
        _WINDOW_KEY
    ]
    assert entry["provider_name"] == (provider or "pi")


def test_plain_claude_hook_without_pi_environment_keeps_default(
    hook_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PI_CODING_AGENT")
    _hook(monkeypatch, "SessionStart", str(hook_home / "project"))
    entry = json.loads((hook_home / "state" / "session_map.json").read_text())[
        _WINDOW_KEY
    ]
    assert entry["provider_name"] == "claude"
