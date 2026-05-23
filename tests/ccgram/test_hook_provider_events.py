import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from ccgram.hook import _encode_pi_cwd_dirname, _install_hook, hook_main


def _tmux_result() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ccgram\t@0\tproject\n", stderr=""
    )


def _run_hook(monkeypatch, payload: dict[str, object], provider_name: str) -> None:
    monkeypatch.setenv("TMUX_PANE", "%0")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with patch("ccgram.hook.subprocess.run", return_value=_tmux_result()):
        hook_main(provider_name=provider_name)


def test_pi_session_start_writes_provider_and_resolves_transcript(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = str(tmp_path / "proj")
    session_id = "019e214d-7011-754d-9efb-60106dfa967c"
    transcript_dir = (
        tmp_path / ".pi" / "agent" / "sessions" / _encode_pi_cwd_dirname(cwd)
    )
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / f"2026-05-13T12-26-23-633Z_{session_id}.jsonl"
    transcript.write_text('{"type":"session"}\n')
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "state"))

    _run_hook(
        monkeypatch,
        {
            "session_id": session_id,
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


_CODEX_SESSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_GEMINI_SESSION_ID = "b2c3d4e5-f678-90ab-cdef-1234567890ab"


def test_pi_stop_refreshes_stale_claude_entry_in_session_map(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = str(tmp_path / "proj")
    pi_session_id = "019e557d-01b3-7e20-9a83-76ba0fdaae3d"
    stale_claude_session_id = "019e557e-f3cc-70c5-95af-d2ea388ed166"
    transcript_dir = (
        tmp_path / ".pi" / "agent" / "sessions" / _encode_pi_cwd_dirname(cwd)
    )
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / f"2026-05-23T15-38-36-340Z_{pi_session_id}.jsonl"
    transcript.write_text('{"type":"session"}\n')
    monkeypatch.setenv("HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("CCGRAM_DIR", str(state_dir))
    (state_dir / "session_map.json").write_text(
        json.dumps(
            {
                "ccgram:@0": {
                    "session_id": stale_claude_session_id,
                    "cwd": cwd,
                    "window_name": "project",
                    "transcript_path": "",
                    "provider_name": "claude",
                }
            }
        )
    )

    _run_hook(
        monkeypatch,
        {
            "session_id": pi_session_id,
            "cwd": cwd,
            "hook_event_name": "Stop",
        },
        "pi",
    )

    session_map = json.loads((state_dir / "session_map.json").read_text())
    entry = session_map["ccgram:@0"]
    assert entry["session_id"] == pi_session_id
    assert entry["provider_name"] == "pi"
    assert entry["transcript_path"] == str(transcript)


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


def test_codex_adapter_rejects_non_uuid_session_id() -> None:
    from ccgram.hooks.adapters import get_hook_adapter

    adapter = get_hook_adapter("codex")
    assert adapter is not None
    result = adapter.normalize(
        {
            "session_id": "not-a-uuid",
            "cwd": "/tmp/project",
            "hook_event_name": "Stop",
        }
    )
    assert result is None


def test_pi_adapter_rejects_non_uuid_session_id() -> None:
    from ccgram.hooks.adapters import get_hook_adapter

    adapter = get_hook_adapter("pi")
    assert adapter is not None
    result = adapter.normalize(
        {
            "session_id": "garbage",
            "cwd": "/tmp/project",
            "hook_event_name": "Stop",
        }
    )
    assert result is None


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


def test_detect_provider_from_payload_defaults_to_none() -> None:
    from ccgram.hooks.adapters import detect_provider_from_payload

    assert detect_provider_from_payload({}) is None


def test_detect_provider_from_payload_uses_transcript_path_codex() -> None:
    from ccgram.hooks.adapters import detect_provider_from_payload

    assert (
        detect_provider_from_payload({"transcript_path": "/home/u/.codex/sess.jsonl"})
        == "codex"
    )


def test_detect_provider_from_payload_uses_transcript_path_gemini() -> None:
    from ccgram.hooks.adapters import detect_provider_from_payload

    assert (
        detect_provider_from_payload({"transcript_path": "/home/u/.gemini/sess.jsonl"})
        == "gemini"
    )


def test_detect_provider_from_payload_uses_transcript_path_pi() -> None:
    from ccgram.hooks.adapters import detect_provider_from_payload

    assert (
        detect_provider_from_payload({"transcript_path": "/home/u/.pi/agent/s.jsonl"})
        == "pi"
    )


def test_detect_provider_from_payload_uses_explicit_provider_field() -> None:
    from ccgram.hooks.adapters import detect_provider_from_payload

    assert detect_provider_from_payload({"provider_name": "codex"}) == "codex"


def test_detect_provider_from_payload_uses_gemini_only_event_name() -> None:
    from ccgram.hooks.adapters import detect_provider_from_payload

    # AfterAgent is unique to Gemini
    assert detect_provider_from_payload({"hook_event_name": "AfterAgent"}) == "gemini"


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
