from __future__ import annotations

from pathlib import Path

from ccgram.providers import registry, resolve_launch_command
from ccgram.providers.maki import MakiProvider


def test_maki_registered() -> None:
    from ccgram.providers import _ensure_registered

    _ensure_registered()
    provider = registry.get("maki")
    assert isinstance(provider, MakiProvider)
    assert provider.capabilities.name == "maki"


def test_maki_launch_command_default() -> None:
    assert resolve_launch_command("maki") == "maki"


def test_maki_launch_command_override(monkeypatch) -> None:
    monkeypatch.setenv("CCGRAM_MAKI_COMMAND", "maki-dev --flag")
    assert resolve_launch_command("maki") == "maki-dev --flag"


def test_maki_yolo_appends_flag() -> None:
    assert resolve_launch_command("maki", approval_mode="yolo") == "maki --yolo"


def test_maki_provider_make_launch_args() -> None:
    provider = MakiProvider()
    assert provider.make_launch_args() == ""
    assert provider.make_launch_args(use_continue=True) == "--continue"
    assert provider.make_launch_args(resume_id="abc-123") == "--session abc-123"


def test_maki_detects_transcript_path() -> None:
    from ccgram.providers import detect_provider_from_transcript_path

    assert detect_provider_from_transcript_path(
        "/home/u/.maki/sessions/Cdp4pdge8t4paEqFW2Gh6.jsonl"
    ) == "maki"
    assert detect_provider_from_transcript_path(
        "/home/u/.local/state/maki/sessions/CdoLH7HJ22FfBAdU2pnyY.jsonl"
    ) == "maki"


def test_maki_discover_resumable_sessions_from_state_dir(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / ".maki"
    sessions_dir = state_dir / "sessions"
    sessions_dir.mkdir(parents=True)
    session = sessions_dir / "abc123.jsonl"
    session.write_text(
        '{"t":"header","v":2,"id":"abc123","model":"openai/x","cwd":"/repo","created_at":1}\n'
        '{"t":"msg","d":{"role":"user","content":[{"type":"text","text":"Hello from session title"}]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("ccgram.providers.maki._maki_state_dirs", lambda: [state_dir])
    provider = MakiProvider()
    sessions = provider.discover_resumable_sessions(cwd="/repo")
    assert len(sessions) == 1
    assert sessions[0].session_id == "abc123"
    assert sessions[0].provider_name == "maki"
    assert sessions[0].summary.startswith("Hello from session title")


def test_maki_parse_transcript_entries_tool_flow() -> None:
    provider = MakiProvider()
    entries = [
        {
            "t": "msg",
            "d": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect files."},
                    {
                        "type": "tool_use",
                        "id": "call1",
                        "name": "read",
                        "input": {"path": "/tmp/x.py", "offset": 1, "limit": 10},
                    },
                ],
            },
        },
        {
            "t": "out",
            "id": "call1",
            "d": {"Plain": {"text": "1: print('x')"}},
        },
    ]
    messages, pending = provider.parse_transcript_entries(entries, {})
    assert pending == {}
    assert [m.content_type for m in messages] == ["text", "tool_use", "tool_result"]
    assert messages[1].tool_name == "read"
    assert "x.py" in messages[1].text
    assert "print('x')" in messages[2].text


def test_maki_parse_tool_result_inside_message_content() -> None:
    provider = MakiProvider()
    entries = [
        {
            "t": "msg",
            "d": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call1",
                        "content": "done",
                        "is_error": False,
                    }
                ],
            },
        }
    ]
    messages, pending = provider.parse_transcript_entries(entries, {"call1": "bash"})
    assert pending == {}
    assert len(messages) == 1
    assert messages[0].content_type == "tool_result"
    assert messages[0].tool_name == "bash"
    assert messages[0].text == "done"


def test_maki_parse_diff_output_branch() -> None:
    provider = MakiProvider()
    entries = [
        {
            "t": "out",
            "id": "call2",
            "d": {
                "Diff": {
                    "path": "/tmp/file.txt",
                    "summary": "edited file.txt",
                }
            },
        }
    ]
    messages, pending = provider.parse_transcript_entries(entries, {"call2": "edit"})
    assert pending == {}
    assert len(messages) == 1
    assert messages[0].content_type == "tool_result"
    assert "edited file.txt" in messages[0].text
    assert "/tmp/file.txt" in messages[0].text


def test_maki_parse_history_entry_user_text() -> None:
    provider = MakiProvider()
    message = provider.parse_history_entry(
        {
            "t": "msg",
            "d": {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            },
        }
    )
    assert message is not None
    assert message.role == "user"
    assert message.text == "hello"


def test_maki_is_user_transcript_entry() -> None:
    provider = MakiProvider()
    assert provider.is_user_transcript_entry(
        {"t": "msg", "d": {"role": "user", "content": []}}
    )
    assert not provider.is_user_transcript_entry(
        {"t": "msg", "d": {"role": "assistant", "content": []}}
    )


def test_maki_status_parser_marks_interactive_picker() -> None:
    provider = MakiProvider()
    status = provider.parse_terminal_status("choose item\n/sessions\nfoo")
    assert status is not None
    assert status.is_interactive is True
    assert status.ui_type == "/sessions"


def test_maki_status_parser_marks_needs_input() -> None:
    provider = MakiProvider()
    status = provider.parse_terminal_status("Waiting for your response in Kandev")
    assert status is not None
    assert status.display_label == "needs input"
