import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ccgram.hook import _encode_pi_cwd_dirname, _install_hook, hook_main
from ccgram.hooks.adapters import detect_provider_from_payload, get_hook_adapter
from ccgram.multiplexer.herdr import HerdrManager


def _tmux_result() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ccgram\t@0\tproject\n", stderr=""
    )


def _pi_transcript(
    home: Path, cwd: str, session_id: str, *, stamp: str = "2026-05-13T12-26-23-633Z"
) -> Path:
    """Create the on-disk Pi transcript the hook is expected to rediscover."""
    directory = home / ".pi" / "agent" / "sessions" / _encode_pi_cwd_dirname(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    transcript = directory / f"{stamp}_{session_id}.jsonl"
    transcript.write_text('{"type":"session"}\n')
    return transcript


def _write_session_map(state_dir: Path, entry: dict[str, str]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "session_map.json"
    path.write_text(json.dumps({"ccgram:@0": entry}))
    return path


def _run_hook(monkeypatch, payload: dict[str, object], provider_name: str) -> None:
    monkeypatch.setenv("TMUX_PANE", "%0")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with patch("ccgram.hook.subprocess.run", return_value=_tmux_result()):
        hook_main(provider_name=provider_name)


@pytest.mark.parametrize("event_name", ["SessionStart", "Stop"])
def test_herdr_nested_claude_hook_does_not_overwrite_live_pi_session(
    tmp_path: Path, monkeypatch, event_name: str
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("CCGRAM_DIR", str(state_dir))
    monkeypatch.delenv("TMUX_PANE", raising=False)
    monkeypatch.setenv("HERDR_WORKSPACE_ID", "w2")
    monkeypatch.setenv("HERDR_PANE_ID", "w2:p1")

    live_transcript = str(tmp_path / "root-pi-session.jsonl")
    live_record = {
        "agent": "pi",
        "workspace_id": "w2",
        "pane_id": "w2:p1",
        "tab_id": "w2:t1",
        "terminal_id": "term-1",
        "agent_session": {
            "source": "herdr:pi",
            "agent": "pi",
            "kind": "path",
            "value": live_transcript,
        },
    }
    target_id = HerdrManager().target_id_for_live_record(live_record)
    assert target_id is not None
    session_map_path = state_dir / "session_map.json"
    original = json.dumps(
        {
            f"herdr:{target_id}": {
                "session_id": "019e214d-7011-754d-9efb-60106dfa967c",
                "cwd": str(tmp_path / "proj"),
                "window_name": "reflex ▸ wiki",
                "transcript_path": live_transcript,
                "provider_name": "pi",
            }
        },
        indent=2,
    )
    session_map_path.write_text(original)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "cwd": str(tmp_path / "nested-claude"),
                    "hook_event_name": event_name,
                    "transcript_path": str(tmp_path / ".claude" / "child.jsonl"),
                }
            )
        ),
    )
    agent_list = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"result": {"agents": [live_record]}}),
        stderr="",
    )

    with patch("ccgram.hook.subprocess.run", return_value=agent_list):
        hook_main(provider_name="claude")

    assert session_map_path.read_text() == original
    assert not (state_dir / "events.jsonl").exists()


_PI_SESSION_ID = "019e214d-7011-754d-9efb-60106dfa967c"
_STALE_PI_SESSION_ID = "019e214d-7011-754d-9efb-60106dfa0000"


def test_pi_session_start_writes_provider_and_resolves_transcript(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = str(tmp_path / "proj")
    transcript = _pi_transcript(tmp_path, cwd, _PI_SESSION_ID)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "state"))

    _run_hook(
        monkeypatch,
        {
            "session_id": _PI_SESSION_ID,
            "cwd": cwd,
            "hook_event_name": "SessionStart",
            "source": "startup",
        },
        "pi",
    )

    session_map = json.loads((tmp_path / "state" / "session_map.json").read_text())
    entry = session_map["ccgram:@0"]
    assert entry["provider_name"] == "pi"
    assert entry["transcript_path"] == str(transcript)


def test_pi_session_start_ignores_stale_payload_transcript_path(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = str(tmp_path / "proj")
    stale = _pi_transcript(
        tmp_path, cwd, _STALE_PI_SESSION_ID, stamp="2026-05-13T12-00-00-000Z"
    )
    transcript = _pi_transcript(tmp_path, cwd, _PI_SESSION_ID)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "state"))

    _run_hook(
        monkeypatch,
        {
            "session_id": _PI_SESSION_ID,
            "cwd": cwd,
            "hook_event_name": "SessionStart",
            "transcript_path": str(stale),
        },
        "pi",
    )

    entry = json.loads((tmp_path / "state" / "session_map.json").read_text())[
        "ccgram:@0"
    ]
    assert entry["provider_name"] == "pi"
    assert entry["session_id"] == _PI_SESSION_ID
    assert entry["transcript_path"] == str(transcript)


def test_pi_stop_refreshes_stale_transcript_path_when_session_is_in_sync(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = str(tmp_path / "proj")
    stale = _pi_transcript(
        tmp_path, cwd, _STALE_PI_SESSION_ID, stamp="2026-05-13T12-00-00-000Z"
    )
    transcript = _pi_transcript(tmp_path, cwd, _PI_SESSION_ID)
    monkeypatch.setenv("HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    monkeypatch.setenv("CCGRAM_DIR", str(state_dir))
    session_map_path = _write_session_map(
        state_dir,
        {
            "session_id": _PI_SESSION_ID,
            "cwd": cwd,
            "window_name": "project",
            "transcript_path": str(stale),
            "provider_name": "pi",
        },
    )

    _run_hook(
        monkeypatch,
        {
            "session_id": _PI_SESSION_ID,
            "cwd": cwd,
            "hook_event_name": "Stop",
            "transcript_path": str(stale),
        },
        "pi",
    )

    entry = json.loads(session_map_path.read_text())["ccgram:@0"]
    assert entry["session_id"] == _PI_SESSION_ID
    assert entry["provider_name"] == "pi"
    assert entry["transcript_path"] == str(transcript)


def test_pi_stop_refreshes_stale_claude_entry_in_session_map(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = str(tmp_path / "proj")
    pi_session_id = "019e557d-01b3-7e20-9a83-76ba0fdaae3d"
    transcript = _pi_transcript(
        tmp_path, cwd, pi_session_id, stamp="2026-05-23T15-38-36-340Z"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    monkeypatch.setenv("CCGRAM_DIR", str(state_dir))
    session_map_path = _write_session_map(
        state_dir,
        {
            "session_id": "019e557e-f3cc-70c5-95af-d2ea388ed166",
            "cwd": cwd,
            "window_name": "project",
            "transcript_path": "",
            "provider_name": "claude",
        },
    )

    _run_hook(
        monkeypatch,
        {"session_id": pi_session_id, "cwd": cwd, "hook_event_name": "Stop"},
        "pi",
    )

    entry = json.loads(session_map_path.read_text())["ccgram:@0"]
    assert entry["session_id"] == pi_session_id
    assert entry["provider_name"] == "pi"
    assert entry["transcript_path"] == str(transcript)


_CODEX_SESSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_GEMINI_SESSION_ID = "b2c3d4e5-f678-90ab-cdef-1234567890ab"


def test_codex_stop_does_not_refresh_session_map_when_in_sync(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    (tmp_path / "session_map.json").write_text(
        json.dumps(
            {
                "ccgram:@0": {
                    "session_id": _CODEX_SESSION_ID,
                    "cwd": "/tmp/project",
                    "window_name": "project",
                    "transcript_path": "/tmp/.codex/session.jsonl",
                    "provider_name": "codex",
                }
            }
        )
    )

    _run_hook(
        monkeypatch,
        {
            "session_id": _CODEX_SESSION_ID,
            "cwd": "/tmp/project",
            "transcript_path": "/tmp/.codex/session.jsonl",
            "hook_event_name": "Stop",
            "model": "gpt-5",
            "permission_mode": "default",
            "turn_id": "turn",
            "stop_hook_active": False,
        },
        "codex",
    )

    session_map = json.loads((tmp_path / "session_map.json").read_text())
    entry = session_map["ccgram:@0"]
    assert entry["session_id"] == _CODEX_SESSION_ID
    assert entry["provider_name"] == "codex"
    assert entry["transcript_path"] == "/tmp/.codex/session.jsonl"


def test_codex_stop_redacts_raw_prompt_and_tool_payload(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    _run_hook(
        monkeypatch,
        {
            "session_id": _CODEX_SESSION_ID,
            "cwd": "/tmp/project",
            "transcript_path": "/tmp/.codex/session.jsonl",
            "hook_event_name": "Stop",
            "model": "gpt-5",
            "permission_mode": "default",
            "turn_id": "turn",
            "stop_hook_active": False,
            "last_assistant_message": "secret output",
        },
        "codex",
    )

    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["event"] == "Stop"
    assert event["data"]["provider_name"] == "codex"
    assert "last_assistant_message" not in event["data"]


def test_codex_stop_outputs_valid_stop_hook_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    _run_hook(
        monkeypatch,
        {
            "session_id": _CODEX_SESSION_ID,
            "cwd": "/tmp/project",
            "transcript_path": "/tmp/.codex/session.jsonl",
            "hook_event_name": "Stop",
            "model": "gpt-5",
            "permission_mode": "default",
            "turn_id": "turn",
            "stop_hook_active": False,
            "last_assistant_message": "secret output",
        },
        "codex",
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {}


@pytest.mark.parametrize("provider", ["codex", "pi"])
def test_adapter_rejects_non_uuid_session_id(provider: str) -> None:
    adapter = get_hook_adapter(provider)
    assert adapter is not None
    assert (
        adapter.normalize(
            {
                "session_id": "not-a-uuid",
                "cwd": "/tmp/project",
                "hook_event_name": "Stop",
            }
        )
        is None
    )


def test_gemini_after_agent_maps_to_stop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    _run_hook(
        monkeypatch,
        {
            "session_id": _GEMINI_SESSION_ID,
            "cwd": "/tmp/project",
            "transcript_path": "/tmp/.gemini/session.jsonl",
            "hook_event_name": "AfterAgent",
            "timestamp": "2026-05-13T00:00:00Z",
            "prompt": "do thing",
            "prompt_response": "done",
        },
        "gemini",
    )

    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["event"] == "Stop"
    assert event["data"]["provider_name"] == "gemini"
    assert event["data"]["native_event_name"] == "AfterAgent"
    assert "prompt" not in event["data"]
    assert "prompt_response" not in event["data"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param({}, None, id="empty-payload"),
        pytest.param(
            {"transcript_path": "/home/u/.codex/sess.jsonl"}, "codex", id="codex-path"
        ),
        pytest.param(
            {"transcript_path": "/home/u/.gemini/sess.jsonl"},
            "gemini",
            id="gemini-path",
        ),
        pytest.param(
            {"transcript_path": "/home/u/.pi/agent/s.jsonl"}, "pi", id="pi-path"
        ),
        pytest.param({"provider_name": "codex"}, "codex", id="explicit-field"),
        # AfterAgent is unique to Gemini.
        pytest.param(
            {"hook_event_name": "AfterAgent"}, "gemini", id="gemini-only-event"
        ),
        # A model-bearing payload that is not a Claude transcript infers codex...
        pytest.param({"model": "gpt-5-codex"}, "codex", id="codex-model-field"),
        # ...but Claude Stop/Notification payloads now carry ``model`` too, and a
        # Claude transcript path must not be misdetected as codex.
        pytest.param(
            {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "hook_event_name": "Stop",
                "transcript_path": "/home/u/.claude/projects/proj/sess.jsonl",
                "model": "claude-opus-4-8",
                "permission_mode": "default",
            },
            None,
            id="claude-model-field",
        ),
    ],
)
def test_detect_provider_from_payload(
    payload: dict[str, object], expected: str | None
) -> None:
    assert detect_provider_from_payload(payload) == expected


@pytest.mark.parametrize("event_name", ["Stop", "Notification"])
def test_hook_detects_claude_with_custom_config_dir(
    tmp_path: Path, monkeypatch, event_name: str
) -> None:
    custom_config_dir = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_config_dir))
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TMUX_PANE", "%0")
    payload = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "hook_event_name": event_name,
        "transcript_path": str(custom_config_dir / "projects" / "proj" / "sess.jsonl"),
        "model": "claude-opus-4-8",
        "permission_mode": "default",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    with patch("ccgram.hook.subprocess.run", return_value=_tmux_result()):
        hook_main()

    event = json.loads((tmp_path / "state" / "events.jsonl").read_text())
    assert event["event"] == event_name
    assert event["data"]["provider_name"] == "claude"


def test_gemini_install_adds_provider_specific_hooks(
    tmp_path: Path, monkeypatch
) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr("ccgram.hook._gemini_settings_file", lambda: settings_file)

    assert _install_hook("gemini") == 0
    assert _install_hook("gemini") == 0

    settings = json.loads(settings_file.read_text())
    hooks = settings["hooks"]
    for event_type in ("SessionStart", "AfterAgent", "SessionEnd", "Notification"):
        matches = [
            hook
            for group in hooks[event_type]
            for hook in group["hooks"]
            if "--provider gemini" in hook["command"]
        ]
        assert len(matches) == 1
