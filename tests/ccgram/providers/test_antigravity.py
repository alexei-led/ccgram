import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ccgram.providers.antigravity import (
    AntigravityProvider,
    build_hardened_antigravity_launch_command,
    detect_antigravity_from_runtime,
    needs_pane_title_for_detection,
)
from ccgram.providers.base import AgentMessage, SessionStartEvent, StatusUpdate


def _write_antigravity_session(
    tmp_dir: Path,
    project_dir: str,
    project_key: str,
    session_name: str,
    session_id: str,
) -> Path:
    chats_dir = tmp_dir / ".gemini" / "tmp" / project_key / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)
    fpath = chats_dir / f"{session_name}.jsonl"
    payload = {
        "sessionId": session_id,
        "projectHash": hashlib.sha256(project_dir.encode()).hexdigest(),
        "startTime": "2026-03-01T00:00:00.000Z",
        "lastUpdated": "2026-03-01T00:00:00.000Z",
    }
    fpath.write_text(json.dumps(payload) + "\n")
    return fpath


class TestAntigravityCapabilities:
    def test_capabilities(self) -> None:
        provider = AntigravityProvider()
        caps = provider.capabilities
        assert caps.name == "antigravity"
        assert caps.launch_command == "antigravity chat"
        assert caps.supports_hook is False
        assert caps.supports_resume is True
        assert caps.supports_continue is True
        assert caps.supports_structured_transcript is True
        assert caps.supports_incremental_read is True
        assert caps.supports_status_snapshot is True


class TestAntigravityLaunchArgs:
    def test_resume_valid_uuid(self) -> None:
        provider = AntigravityProvider()
        uuid_str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        result = provider.make_launch_args(resume_id=uuid_str)
        assert result == f"--resume {uuid_str}"

    def test_resume_latest(self) -> None:
        provider = AntigravityProvider()
        result = provider.make_launch_args(resume_id="latest")
        assert result == "--resume latest"

    def test_continue_uses_latest(self) -> None:
        provider = AntigravityProvider()
        result = provider.make_launch_args(use_continue=True)
        assert result == "--resume latest"

    def test_invalid_resume_raises_error(self) -> None:
        provider = AntigravityProvider()
        with pytest.raises(ValueError, match="Invalid resume_id"):
            provider.make_launch_args(resume_id="invalid!!resume")


class TestAntigravityLaunchHardening:
    def test_hardening_wraps_command_with_env(self, tmp_path: Path) -> None:
        with patch("ccgram.providers.antigravity.ccgram_dir", return_value=tmp_path):
            cmd = build_hardened_antigravity_launch_command("antigravity chat")
            assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH=" in cmd
            assert "ANTIGRAVITY_CLI_SYSTEM_SETTINGS_PATH=" in cmd
            assert cmd.endswith("antigravity chat")

            # Check that settings json was written correctly
            settings_file = tmp_path / "antigravity-system-settings.json"
            assert settings_file.exists()
            with open(settings_file, "r") as f:
                data = json.load(f)
            assert data["tools"]["shell"]["enableInteractiveShell"] is False


class TestAntigravityTranscriptParsing:
    def test_parses_messages(self) -> None:
        provider = AntigravityProvider()
        entries = [
            {"type": "user", "content": "hello agent"},
            {"type": "antigravity", "content": "hello user"},
            {"type": "gemini", "content": "secondary assistant response"},
            {"type": "info", "content": "Status info"},
            {"type": "error", "content": "An error occurred"},
        ]
        messages, _ = provider.parse_transcript_entries(entries, {})
        assert len(messages) == 5

        assert messages[0].role == "user"
        assert messages[0].text == "hello agent"

        assert messages[1].role == "assistant"
        assert messages[1].text == "hello user"

        assert messages[2].role == "assistant"
        assert messages[2].text == "secondary assistant response"

        assert messages[3].role == "assistant"
        assert messages[3].text == "Status info"

        assert messages[4].role == "assistant"
        assert messages[4].text == "An error occurred"

    def test_parses_array_content(self) -> None:
        provider = AntigravityProvider()
        entries = [
            {
                "type": "user",
                "content": [{"text": "hello "}, {"content": "from list"}],
            }
        ]
        messages, _ = provider.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "hello from list"

    def test_fallback_to_display_content(self) -> None:
        provider = AntigravityProvider()
        entry = {
            "type": "antigravity",
            "content": [{"meta": "ignored"}],
            "displayContent": [{"text": "display content text"}],
        }
        parsed = provider.parse_history_entry(entry)
        assert parsed is not None
        assert parsed.text == "display content text"
        assert parsed.role == "assistant"

    def test_skips_unknown_types(self) -> None:
        provider = AntigravityProvider()
        entries = [{"type": "unrecognized", "content": "ignored text"}]
        messages, _ = provider.parse_transcript_entries(entries, {})
        assert messages == []

    def test_tracks_and_clears_tool_calls(self) -> None:
        provider = AntigravityProvider()
        entries = [
            {
                "type": "antigravity",
                "content": "Running command",
                "toolCalls": [
                    {
                        "id": "tc-1",
                        "name": "run_command",
                        "displayName": "RunCommand",
                        "args": {"cmd": "ls -l"},
                    }
                ],
            }
        ]
        messages, pending = provider.parse_transcript_entries(entries, {})
        assert "tc-1" in pending
        assert pending["tc-1"] == "RunCommand"
        assert len(messages) == 2
        assert messages[0].content_type == "tool_use"
        assert messages[0].tool_name == "RunCommand"
        assert messages[0].text == "**RunCommand** `ls -l`"
        assert messages[1].content_type == "text"
        assert messages[1].text == "Running command"

        # Now test result emission and pending cleanup
        entries_with_result = [
            {
                "type": "antigravity",
                "content": "Command finished",
                "toolCalls": [
                    {
                        "id": "tc-1",
                        "name": "run_command",
                        "displayName": "RunCommand",
                        "args": {"cmd": "ls -l"},
                        "resultDisplay": "file1.txt\nfile2.txt",
                    }
                ],
            }
        ]
        messages_res, pending_res = provider.parse_transcript_entries(entries_with_result, pending)
        assert "tc-1" not in pending_res
        assert len(messages_res) == 3
        assert messages_res[0].content_type == "tool_use"
        assert messages_res[1].content_type == "tool_result"
        assert messages_res[1].text == "file1.txt\nfile2.txt"
        assert messages_res[2].content_type == "text"

    def test_parses_real_agy_transcript_format(self) -> None:
        provider = AntigravityProvider()
        entries = [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "content": "<USER_REQUEST>\nhello, this is a test\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\ntime: 2026\n</ADDITIONAL_METADATA>",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "content": "Analyzing directory...",
                "tool_calls": [
                    {
                        "name": "list_dir",
                        "args": {"DirectoryPath": "/my/dir"},
                    }
                ],
            },
            {
                "step_index": 2,
                "source": "MODEL",
                "type": "LIST_DIRECTORY",
                "content": "file1.py\nfile2.py",
            },
            {
                "step_index": 3,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "content": "I found two files.",
            }
        ]

        messages, pending = provider.parse_transcript_entries(entries, {})
        assert len(pending) == 0
        assert len(messages) == 5

        # 1. User message (extracted and cleaned)
        assert messages[0].role == "user"
        assert messages[0].text == "hello, this is a test"
        assert messages[0].content_type == "text"

        # 2. Tool use message
        assert messages[1].role == "assistant"
        assert messages[1].content_type == "tool_use"
        assert messages[1].tool_name == "list_dir"
        assert "/my/dir" in messages[1].text

        # 3. Standard text response
        assert messages[2].role == "assistant"
        assert messages[2].text == "Analyzing directory..."
        assert messages[2].content_type == "text"

        # 4. Tool result message
        assert messages[3].role == "assistant"
        assert messages[3].content_type == "tool_result"
        assert messages[3].tool_name == "list_dir"
        assert messages[3].text == "file1.py\nfile2.py"

        # 5. Final text response
        assert messages[4].role == "assistant"
        assert messages[4].text == "I found two files."
        assert messages[4].content_type == "text"


class TestAntigravityTerminalStatus:
    SELECTION_PANE = (
        "Some system output\n"
        "Select something:\n"
        "● 1. Option A\n"
        "  2. Option B\n"
        "(Press Enter to select)\n"
    )

    PERMISSION_PANE = (
        "Action Required\n"
        "? Shell ls -F\n"
        "Allow execution?\n"
        "● 1. Allow once\n"
        "  2. Allow for this session\n"
        "(esc\n"
    )

    def test_parses_selection_prompt(self) -> None:
        provider = AntigravityProvider()
        status = provider.parse_terminal_status(self.SELECTION_PANE)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "SelectionUI"
        assert "Select something" in status.raw_text

    def test_parses_permission_prompt(self) -> None:
        provider = AntigravityProvider()
        status = provider.parse_terminal_status(self.PERMISSION_PANE)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"
        assert "Action Required" in status.raw_text

    def test_pane_title_action_required(self) -> None:
        provider = AntigravityProvider()
        status = provider.parse_terminal_status("generic output", pane_title="Action Required ✋")
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"

    def test_pane_content_action_required_fallback(self) -> None:
        provider = AntigravityProvider()
        status = provider.parse_terminal_status("Some error happened and Action Required message printed", pane_title="")
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"

    def test_normal_output_returns_none(self) -> None:
        provider = AntigravityProvider()
        status = provider.parse_terminal_status("Just normal text\nWaiting for something\n", pane_title="")
        assert status is None


class TestAntigravityRuntimeDetection:
    def test_needs_pane_title(self) -> None:
        assert needs_pane_title_for_detection("bun run index.ts") is True
        assert needs_pane_title_for_detection("node main.js") is True
        assert needs_pane_title_for_detection("npx -y antigravity") is True
        assert needs_pane_title_for_detection("python3 app.py") is False

    def test_detects_from_runtime(self) -> None:
        assert detect_antigravity_from_runtime("bun run", "🛸 workspace") is True
        assert detect_antigravity_from_runtime("node cli.js", "Gemini ✦") is False
        assert detect_antigravity_from_runtime("python3 app.py", "🛸 workspace") is False


class TestAntigravityDiscoverTranscript:
    def test_finds_session_via_project_hash_dir(self, tmp_path: Path) -> None:
        project = "/my/antigravity-project"
        project_hash = hashlib.sha256(project.encode()).hexdigest()
        fpath = _write_antigravity_session(
            tmp_path,
            project,
            project_hash,
            "session-2026-03-02T12-00-00abcd",
            "anti-uuid-1",
        )

        provider = AntigravityProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = provider.discover_transcript(project, "ccgram:@7")
        assert event is not None
        assert event.session_id == "anti-uuid-1"
        assert event.cwd == project
        assert event.transcript_path == str(fpath)
        assert event.window_key == "ccgram:@7"

    def test_finds_session_via_projects_alias(self, tmp_path: Path) -> None:
        project = "/my/antigravity-project"
        fpath = _write_antigravity_session(
            tmp_path,
            project,
            "workspace-alias",
            "session-2026-03-02T12-00-00abcd",
            "anti-uuid-2",
        )
        projects = {"projects": {project: "workspace-alias"}}
        (tmp_path / ".gemini").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".gemini" / "projects.json").write_text(json.dumps(projects))

        provider = AntigravityProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = provider.discover_transcript(project, "ccgram:@8")
        assert event is not None
        assert event.session_id == "anti-uuid-2"
        assert event.transcript_path == str(fpath)

    def test_respects_staleness_by_default(self, tmp_path: Path) -> None:
        project = "/my/antigravity-project"
        project_hash = hashlib.sha256(project.encode()).hexdigest()
        fpath = _write_antigravity_session(
            tmp_path,
            project,
            project_hash,
            "session-2026-03-01T09-00-00abcd",
            "anti-old",
        )
        old_time = fpath.stat().st_mtime - 300
        os.utime(fpath, (old_time, old_time))

        provider = AntigravityProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = provider.discover_transcript(project, "ccgram:@7")
        assert event is None

    def test_max_age_zero_ignores_staleness(self, tmp_path: Path) -> None:
        project = "/my/antigravity-project"
        project_hash = hashlib.sha256(project.encode()).hexdigest()
        fpath = _write_antigravity_session(
            tmp_path,
            project,
            project_hash,
            "session-2026-03-01T09-00-00abcd",
            "anti-old",
        )
        old_time = fpath.stat().st_mtime - 300
        os.utime(fpath, (old_time, old_time))

        provider = AntigravityProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = provider.discover_transcript(project, "ccgram:@7", max_age=0)
        assert event is not None
        assert event.session_id == "anti-old"
        assert event.transcript_path == str(fpath)

    def test_finds_session_via_history_jsonl(self, tmp_path: Path) -> None:
        project = "/my/antigravity-project"
        history_dir = tmp_path / ".gemini" / "antigravity-cli"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / "history.jsonl"
        
        # Write history entries
        history_file.write_text(
            json.dumps({"display": "another query", "workspace": "/some/other", "conversationId": "anti-other-id"}) + "\n" +
            json.dumps({"display": "hello", "workspace": project, "conversationId": "anti-uuid-history"}) + "\n"
        )
        
        # Write matching transcript file
        transcript_dir = history_dir / "brain" / "anti-uuid-history" / ".system_generated" / "logs"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = transcript_dir / "transcript.jsonl"
        transcript_file.write_text("{}\n")
        
        provider = AntigravityProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = provider.discover_transcript(project, "ccgram:@7")
        
        assert event is not None
        assert event.session_id == "anti-uuid-history"
        assert event.cwd == project
        assert event.transcript_path == str(transcript_file)



class TestAntigravityStatusSnapshot:
    def test_build_status_snapshot(self, tmp_path: Path) -> None:
        transcript = tmp_path / "antigravity.jsonl"
        transcript.write_text("dummy jsonl content\n")

        provider = AntigravityProvider()
        result = provider.build_status_snapshot(
            str(transcript),
            display_name="test-repo",
            session_id="anti-sess-123456789",
            cwd="/my/repo",
        )
        assert result is not None
        assert "🛸 [test-repo] Antigravity session active." in result
        assert "📁 `/my/repo`" in result
        assert "📄 `antigravity.jsonl`" in result
        assert "⭐ ID: `anti-ses`" in result

    def test_build_status_snapshot_missing_file(self) -> None:
        provider = AntigravityProvider()
        result = provider.build_status_snapshot(
            "/tmp/nonexistent-transcript.jsonl",
            display_name="test",
        )
        assert result is None

    def test_has_output_since(self, tmp_path: Path) -> None:
        transcript = tmp_path / "antigravity.jsonl"
        transcript.write_text("hello\n")

        provider = AntigravityProvider()
        assert provider.has_output_since(str(transcript), 0) is True
        assert provider.has_output_since(str(transcript), 100) is False
        assert provider.has_output_since("/tmp/nonexistent-transcript.jsonl", 0) is False
