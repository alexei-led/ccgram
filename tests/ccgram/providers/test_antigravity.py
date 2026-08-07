import json
import os
import time
import pytest

from ccgram.providers import (
    detect_provider_from_command,
    detect_provider_from_transcript_path,
    resolve_launch_command,
)
from ccgram.providers.process_detection import classify_provider_from_args
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

    def test_executable_precedence_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "CCGRAM_ANTIGRAVITY_COMMAND", "/custom/bin/agy --effort high"
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy")
        assert resolve_antigravity_executable() == "/custom/bin/agy --effort high"

    def test_executable_precedence_path_lookup(self, monkeypatch):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr(
            "shutil.which", lambda name: "agy" if name == "agy" else None
        )
        assert resolve_antigravity_executable() == "agy"

    def test_executable_precedence_fallback_candidate(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)

        bin_dir = tmp_path / ".gemini" / "antigravity-cli" / "bin"
        bin_dir.mkdir(parents=True)
        agy_bin = bin_dir / "agy"
        agy_bin.write_text("#!/bin/sh\necho ok\n")
        agy_bin.chmod(0o755)

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        resolved = resolve_antigravity_executable()
        assert resolved == str(agy_bin.resolve())

    def test_resolve_launch_command_uses_antigravity_resolver(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)

        bin_dir = tmp_path / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        agy_bin = bin_dir / "agy"
        agy_bin.write_text("#!/bin/sh\necho ok\n")
        agy_bin.chmod(0o755)

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        cmd = resolve_launch_command("antigravity", approval_mode="normal")
        assert cmd == str(agy_bin.resolve())

    @pytest.mark.parametrize(
        "candidate_subpath",
        [
            ".local/bin/agy",
            ".gemini/antigravity-cli/bin/agy",
            ".antigravity/bin/agy",
        ],
    )
    def test_known_executable_layouts(self, tmp_path, monkeypatch, candidate_subpath):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)

        cand_file = tmp_path / candidate_subpath
        cand_file.parent.mkdir(parents=True, exist_ok=True)
        cand_file.write_text("#!/bin/sh\necho ok\n")
        cand_file.chmod(0o755)

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        resolved = resolve_antigravity_executable()
        assert resolved == str(cand_file.resolve())

        launch_cmd = resolve_launch_command("antigravity", approval_mode="normal")
        assert launch_cmd == str(cand_file.resolve())

    @pytest.mark.parametrize(
        "home_subpath",
        [
            ".config/antigravity/brain",
            ".gemini/antigravity-cli/brain",
            ".antigravity/brain",
        ],
    )
    def test_known_data_dir_layouts(self, tmp_path, monkeypatch, home_subpath):
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_DATA_DIR", raising=False)

        brain = tmp_path / home_subpath
        brain.mkdir(parents=True, exist_ok=True)
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
        transcript_file.write_text(f'{{"cwd": "file://{proj_dir.resolve()}"}}\n')

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(proj_dir), "shared:@0")
        assert event is not None
        assert event.session_id == "test-session-id"

    def test_discover_transcript_requires_cwd(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "test-session-id" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        (session_dir / "transcript.jsonl").write_text(
            f'{{"cwd": "file://{tmp_path.resolve()}"}}\n'
        )

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert provider.discover_transcript("", "shared:@0") is None

    def test_content_file_uri_is_not_workspace_identity(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "content-only" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        target_dir = tmp_path / "target_proj"
        target_dir.mkdir()
        (session_dir / "transcript.jsonl").write_text(
            json.dumps({"type": "USER_INPUT", "content": f"file://{target_dir}"}) + "\n"
        )

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert provider.discover_transcript(str(target_dir), "shared:@0") is None

    def test_nested_content_workspace_is_not_identity(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "nested-content" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        target_dir = tmp_path / "target_proj"
        target_dir.mkdir()
        (session_dir / "transcript.jsonl").write_text(
            json.dumps(
                {"type": "MODEL", "content": {"directory": f"file://{target_dir}"}}
            )
            + "\n"
        )

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert provider.discover_transcript(str(target_dir), "shared:@0") is None

    def test_discover_transcript_no_match_returns_none(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "other-session" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        (session_dir / "transcript.jsonl").write_text(
            f'{{"cwd": "file://{tmp_path}/other_proj"}}\n'
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
            f'{{"cwd": "file://{app_other_dir.resolve()}"}}\n'
        )

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(app_dir), "shared:@0")
        assert event is None

    def test_discover_transcript_descendant_isolation(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"

        app_dir = tmp_path / "app"
        app_dir.mkdir()
        subdir = app_dir / "subdir"
        subdir.mkdir()

        s1 = brain / "sess-subdir" / ".system_generated" / "logs"
        s1.mkdir(parents=True)
        (s1 / "transcript.jsonl").write_text(
            f'{{"cwd": "file://{subdir.resolve()}"}}\n'
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
        f1.write_text(f'{{"cwd": "file://{proj.resolve()}"}}\n')

        s2 = brain / "sess-new" / ".system_generated" / "logs"
        s2.mkdir(parents=True)
        f2 = s2 / "transcript.jsonl"
        f2.write_text(f'{{"cwd": "file://{proj.resolve()}"}}\n')

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
            f'{{"cwd": "file://{tmp_path.resolve()}"}}\n'
        )

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(tmp_path), "shared:@0", max_age=0)
        assert event is not None
        assert event.session_id == "test-session-id"


class TestAntigravityResumeAndContinue:
    def test_resume_launch_args(self):
        provider = AntigravityProvider()
        args = provider.make_launch_args(
            resume_id="12345678-1234-1234-1234-1234567890ab"
        )
        assert args == "--conversation 12345678-1234-1234-1234-1234567890ab"

    def test_continue_launch_args(self):
        assert AntigravityProvider().make_launch_args(use_continue=True) == "--continue"

    def test_invalid_resume_id(self):
        provider = AntigravityProvider()
        with pytest.raises(ValueError, match="Invalid resume_id"):
            provider.make_launch_args(resume_id="invalid/id; rm -rf /")


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
        assert messages[0].tool_use_id == "call-1"

        assert messages[1].content_type == "tool_result"
        assert messages[1].tool_name == "run_command"
        assert messages[1].tool_use_id == "call-1"
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
            {"type": "USER_INPUT", "content": [{"text": 1}, {"text": None}]},
        ]
        messages, pending = provider.parse_transcript_entries(entries, {})
        assert isinstance(messages, list)
        assert isinstance(pending, dict)

    def test_tool_batch_processor_matching_by_tool_use_id(self):
        from ccgram.handlers.messaging_pipeline.message_task import ContentTask
        from ccgram.handlers.messaging_pipeline.tool_batch import (
            ToolBatch,
            ToolBatchEntry,
        )

        provider = AntigravityProvider()
        entries = [
            {
                "step_index": 0,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "tool_calls": [{"id": "tool-call-abc", "name": "run_command"}],
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "RUN_COMMAND",
                "tool_call_id": "tool-call-abc",
                "content": "output text",
            },
        ]
        messages, _ = provider.parse_transcript_entries(entries, {})

        tool_use_msg = messages[0]
        tool_res_msg = messages[1]

        assert tool_use_msg.tool_use_id == "tool-call-abc"
        assert tool_res_msg.tool_use_id == "tool-call-abc"

        task = ContentTask(
            window_id="@1",
            content_type="tool_result",
            tool_name=tool_res_msg.tool_name,
            tool_use_id=tool_res_msg.tool_use_id,
            parts=(tool_res_msg.text,),
        )

        batch = ToolBatch(
            window_id="@1",
            thread_id=123,
            entries=[
                ToolBatchEntry(
                    tool_name="run_command",
                    tool_use_id="tool-call-abc",
                    tool_use_text="run_command()",
                )
            ],
        )

        matching_entry = next(
            (e for e in batch.entries if e.tool_use_id == task.tool_use_id), None
        )
        assert matching_entry is not None
        assert matching_entry.tool_name == "run_command"


class TestAntigravityDetectionAndYolo:
    def test_process_detection(self):
        assert detect_provider_from_command("agy") == "antigravity"
        assert detect_provider_from_command("/usr/local/bin/agy") == "antigravity"
        assert detect_provider_from_command("antigravity") == "antigravity"
        assert (
            classify_provider_from_args("node /opt/antigravity-cli/bin/cli.js")
            == "antigravity"
        )
        assert classify_provider_from_args("node /work/antigravity/cli.js") == ""

    def test_transcript_path_detection(self):
        path1 = "/Users/user/.gemini/antigravity-cli/brain/123/.system_generated/logs/transcript.jsonl"
        path2 = "/Users/user/.config/antigravity/brain/123/.system_generated/logs/transcript.jsonl"
        path3 = (
            "/Users/user/.antigravity/brain/123/.system_generated/logs/transcript.jsonl"
        )
        assert detect_provider_from_transcript_path(path1) == "antigravity"
        assert detect_provider_from_transcript_path(path2) == "antigravity"
        assert detect_provider_from_transcript_path(path3) == "antigravity"
        assert (
            detect_provider_from_transcript_path(
                "/tmp/.antigravity-backup/brain/123/transcript.jsonl"
            )
            == ""
        )
        assert (
            detect_provider_from_transcript_path(
                "/tmp/antigravity-cli-old/brain/123/transcript.jsonl"
            )
            == ""
        )
        assert (
            detect_provider_from_transcript_path(
                "/tmp/antigravity/brain/123/transcript.jsonl"
            )
            == ""
        )

    def test_yolo_resolution(self):
        cmd = resolve_launch_command("antigravity", approval_mode="yolo")
        assert cmd == "agy --dangerously-skip-permissions"


class TestAntigravityRecoveryIntegration:
    def test_resume_picker_scans_antigravity_sessions(self, tmp_path, monkeypatch):
        from ccgram.handlers.recovery.resume_picker import scan_sessions_for_cwd

        brain_dir = tmp_path / "brain"
        s1 = brain_dir / "sess-ag-1" / ".system_generated" / "logs"
        s1.mkdir(parents=True)
        (s1 / "transcript.jsonl").write_text(
            f'{{"cwd": "file://{tmp_path.resolve()}"}}\n'
        )

        monkeypatch.setenv("CCGRAM_ANTIGRAVITY_DATA_DIR", str(brain_dir))
        sessions = scan_sessions_for_cwd(str(tmp_path), provider_name="antigravity")
        assert len(sessions) == 1
        assert sessions[0].session_id == "sess-ag-1"

    def test_resume_command_scans_antigravity_sessions(self, tmp_path, monkeypatch):
        from ccgram.handlers.recovery.resume_command import scan_all_sessions

        brain_dir = tmp_path / "brain"
        s1 = brain_dir / "sess-ag-2" / ".system_generated" / "logs"
        s1.mkdir(parents=True)
        (s1 / "transcript.jsonl").write_text(
            f'{{"cwd": "file://{tmp_path.resolve()}"}}\n'
        )

        monkeypatch.setenv("CCGRAM_ANTIGRAVITY_DATA_DIR", str(brain_dir))
        sessions = scan_all_sessions("antigravity")
        assert len(sessions) == 1
        assert sessions[0].session_id == "sess-ag-2"
