import os
import time
import pytest

from ccgram.providers import (
    detect_provider_from_command,
    detect_provider_from_transcript_path,
    resolve_launch_command,
)
from ccgram.providers.antigravity import (
    AntigravityProvider,
    clean_antigravity_content,
    get_antigravity_brain_dirs,
    resolve_antigravity_executable,
    resolve_antigravity_role,
)


class TestAntigravityCapabilities:
    def test_capabilities(self):
        provider = AntigravityProvider()
        caps = provider.capabilities
        assert caps.name == "antigravity"
        assert caps.launch_command == "agy"
        assert caps.has_yolo_confirmation is False
        assert caps.supports_resume is True
        assert caps.supports_continue is True
        assert caps.supports_user_command_discovery is False
        assert caps.supports_structured_transcript is True
        assert "model" in caps.tui_picker_commands
        assert "settings" in caps.tui_picker_commands


class TestAntigravityContentCleaning:
    def test_clean_user_request_tags(self):
        raw = "<USER_REQUEST>\nexplain the project structure\n</USER_REQUEST>"
        cleaned = clean_antigravity_content(raw)
        assert cleaned == "explain the project structure"

    def test_clean_metadata_blocks(self):
        raw = (
            "<USER_REQUEST>\nhelp me debug\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\ntime: 2026-08-06\n</ADDITIONAL_METADATA>\n"
            "<USER_SETTINGS_CHANGE>\nmodel changed\n</USER_SETTINGS_CHANGE>"
        )
        cleaned = clean_antigravity_content(raw)
        assert cleaned == "help me debug"


class TestAntigravityRoleResolution:
    @pytest.mark.parametrize(
        ("entry_type", "source", "expected"),
        [
            ("USER_INPUT", "USER_EXPLICIT", "user"),
            ("USER_INPUT", "USER_IMPLICIT", "user"),
            ("user", "user", "user"),
            ("PLANNER_RESPONSE", "MODEL", "assistant"),
            ("model", "model", "assistant"),
            ("info", "system", "assistant"),
        ],
    )
    def test_role_resolution(self, entry_type, source, expected):
        entry = {"type": entry_type, "source": source}
        assert resolve_antigravity_role(entry) == expected


class TestAntigravityExecutableAndDataResolution:
    def test_command_override(self, monkeypatch):
        monkeypatch.setenv("CCGRAM_ANTIGRAVITY_COMMAND", "custom-agy --effort high")
        assert resolve_antigravity_executable() == "custom-agy --effort high"

    def test_path_lookup(self, monkeypatch):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr(
            "shutil.which", lambda name: "/usr/local/bin/agy" if name == "agy" else None
        )
        assert resolve_antigravity_executable() == "agy"

    def test_data_dir_override(self, tmp_path, monkeypatch):
        override_dir = tmp_path / "custom_brain"
        override_dir.mkdir()
        monkeypatch.setenv("CCGRAM_ANTIGRAVITY_DATA_DIR", str(override_dir))
        dirs = get_antigravity_brain_dirs()
        assert len(dirs) == 1
        assert dirs[0] == override_dir.resolve()

    @pytest.mark.parametrize(
        ("os_name", "home_subpath"),
        [
            ("linux", ".gemini/antigravity-cli/brain"),
            ("darwin-arm64", ".gemini/antigravity-cli/brain"),
            ("darwin-x86_64", ".antigravity/brain"),
        ],
    )
    def test_platform_data_dir_discovery(
        self, tmp_path, monkeypatch, os_name, home_subpath
    ):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_DATA_DIR", raising=False)
        brain = tmp_path / home_subpath
        brain.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        dirs = get_antigravity_brain_dirs()
        assert len(dirs) >= 1
        assert brain.resolve() in dirs


class TestAntigravityCwdMatching:
    def test_discover_transcript_exact_match(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "test-session-id" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        transcript_file = session_dir / "transcript.jsonl"
        proj_dir = tmp_path / "my_project"
        proj_dir.mkdir()
        transcript_file.write_text(f'{{"content": "file://{proj_dir.resolve()}"}}\n')

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(proj_dir), "shared:@0")
        assert event is not None
        assert event.session_id == "test-session-id"

    def test_discover_transcript_no_match_returns_none(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "other-session" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        (session_dir / "transcript.jsonl").write_text(
            f'{{"content": "file://{tmp_path}/other_proj"}}\n'
        )

        target_dir = tmp_path / "target_proj"
        target_dir.mkdir()

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(target_dir), "shared:@0")
        assert event is None

    def test_discover_transcript_sibling_prefix_isolation(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"

        app_dir = tmp_path / "app"
        app_dir.mkdir()
        app_other_dir = tmp_path / "app-other"
        app_other_dir.mkdir()

        s1 = brain / "sess-app-other" / ".system_generated" / "logs"
        s1.mkdir(parents=True)
        (s1 / "transcript.jsonl").write_text(
            f'{{"content": "file://{app_other_dir.resolve()}"}}\n'
        )

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(app_dir), "shared:@0")
        assert event is None

    def test_discover_transcript_multiple_candidates_picks_newest_matching(
        self, tmp_path, monkeypatch
    ):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        proj = tmp_path / "project"
        proj.mkdir()

        now = time.time()
        s1 = brain / "sess-old" / ".system_generated" / "logs"
        s1.mkdir(parents=True)
        f1 = s1 / "transcript.jsonl"
        f1.write_text(f'{{"content": "file://{proj.resolve()}"}}\n')

        s2 = brain / "sess-new" / ".system_generated" / "logs"
        s2.mkdir(parents=True)
        f2 = s2 / "transcript.jsonl"
        f2.write_text(f'{{"content": "file://{proj.resolve()}"}}\n')

        os.utime(f1, (now - 20, now - 20))
        os.utime(f2, (now - 5, now - 5))

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(proj), "shared:@0")
        assert event is not None
        assert event.session_id == "sess-new"

    def test_discover_transcript_max_age_zero(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "test-session-id" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        (session_dir / "transcript.jsonl").write_text(
            f'{{"content": "file://{tmp_path.resolve()}"}}\n'
        )

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(tmp_path), "shared:@0", max_age=0)
        assert event is not None
        assert event.session_id == "test-session-id"


class TestAntigravityResumeAndContinue:
    def test_resume_command_normal(self, monkeypatch):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: "agy")
        provider = AntigravityProvider()
        cmd = provider.get_resume_command(
            "12345678-1234-1234-1234-1234567890ab", approval_mode="normal"
        )
        assert cmd == "agy --conversation 12345678-1234-1234-1234-1234567890ab"

    def test_resume_command_yolo(self, monkeypatch):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: "agy")
        provider = AntigravityProvider()
        cmd = provider.get_resume_command(
            "12345678-1234-1234-1234-1234567890ab", approval_mode="yolo"
        )
        assert (
            cmd
            == "agy --conversation 12345678-1234-1234-1234-1234567890ab --dangerously-skip-permissions"
        )

    def test_continue_command_normal(self, monkeypatch):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: "agy")
        provider = AntigravityProvider()
        cmd = provider.get_continue_command(approval_mode="normal")
        assert cmd == "agy --continue"

    def test_continue_command_yolo(self, monkeypatch):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: "agy")
        provider = AntigravityProvider()
        cmd = provider.get_continue_command(approval_mode="yolo")
        assert cmd == "agy --continue --dangerously-skip-permissions"

    def test_invalid_resume_id(self):
        provider = AntigravityProvider()
        with pytest.raises(ValueError, match="Invalid session_id"):
            provider.get_resume_command("invalid/id; rm -rf /")


class TestAntigravityToolCallParsing:
    def test_tool_call_and_result_parsing(self):
        provider = AntigravityProvider()
        entries = [
            {
                "step_index": 0,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-08-06T09:50:47Z",
                "tool_calls": [{"id": "call-1", "name": "run_command"}],
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "RUN_COMMAND",
                "tool_call_id": "call-1",
                "created_at": "2026-08-06T09:50:48Z",
                "content": "command output result",
            },
        ]
        messages, pending = provider.parse_transcript_entries(entries, {})
        assert len(messages) == 2

        assert messages[0].content_type == "tool_use"
        assert messages[0].tool_name == "run_command"

        assert messages[1].content_type == "tool_result"
        assert messages[1].tool_name == "run_command"
        assert messages[1].text == "command output result"
        assert "call-1" not in pending

    def test_incremental_read_preserves_pending_tools(self):
        provider = AntigravityProvider()
        batch1 = [
            {
                "step_index": 0,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "tool_calls": [{"id": "call-inc", "name": "list_dir"}],
            }
        ]
        msgs1, pending = provider.parse_transcript_entries(batch1, {})
        assert len(msgs1) == 1
        assert pending.get("call-inc") == "list_dir"

        batch2 = [
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "TOOL_RESULT",
                "tool_call_id": "call-inc",
                "content": "file1.py\nfile2.py",
            }
        ]
        msgs2, pending_after = provider.parse_transcript_entries(batch2, pending)
        assert len(msgs2) == 1
        assert msgs2[0].content_type == "tool_result"
        assert msgs2[0].tool_name == "list_dir"
        assert "call-inc" not in pending_after

    def test_malformed_entries_handled_safely(self):
        provider = AntigravityProvider()
        entries = [
            {"invalid": True},
            {"type": "PLANNER_RESPONSE", "tool_calls": "not-a-list"},
            {"type": "RUN_COMMAND", "content": None},
        ]
        messages, pending = provider.parse_transcript_entries(entries, {})
        assert isinstance(messages, list)
        assert isinstance(pending, dict)


class TestAntigravityDetectionAndYolo:
    def test_process_detection(self):
        assert detect_provider_from_command("agy") == "antigravity"
        assert detect_provider_from_command("/usr/local/bin/agy") == "antigravity"
        assert detect_provider_from_command("antigravity") == "antigravity"

    def test_transcript_path_detection(self):
        path1 = "/Users/user/.gemini/antigravity-cli/brain/123/.system_generated/logs/transcript.jsonl"
        path2 = "/Users/user/.config/antigravity/brain/123/.system_generated/logs/transcript.jsonl"
        path3 = (
            "/Users/user/.antigravity/brain/123/.system_generated/logs/transcript.jsonl"
        )
        assert detect_provider_from_transcript_path(path1) == "antigravity"
        assert detect_provider_from_transcript_path(path2) == "antigravity"
        assert detect_provider_from_transcript_path(path3) == "antigravity"

    def test_yolo_resolution(self):
        cmd = resolve_launch_command("antigravity", approval_mode="yolo")
        assert cmd == "agy --dangerously-skip-permissions"
