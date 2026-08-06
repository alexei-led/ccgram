import pytest

from ccgram.providers import (
    detect_provider_from_command,
    detect_provider_from_transcript_path,
    resolve_launch_command,
)
from ccgram.providers.antigravity import (
    AntigravityProvider,
    clean_antigravity_content,
    resolve_antigravity_role,
)


class TestAntigravityCapabilities:
    def test_capabilities(self):
        provider = AntigravityProvider()
        caps = provider.capabilities
        assert caps.name == "antigravity"
        assert caps.launch_command == "agy"
        assert caps.has_yolo_confirmation is True
        assert caps.supports_resume is True
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


class TestAntigravityTranscriptParsing:
    def test_parse_transcript_entries(self):
        provider = AntigravityProvider()
        entries = [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-08-06T09:50:46Z",
                "content": "<USER_REQUEST>\nhello world\n</USER_REQUEST>",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-08-06T09:50:47Z",
                "content": "Hello! How can I help you?",
                "tool_calls": [{"name": "view_file"}],
            },
        ]
        messages, pending = provider.parse_transcript_entries(entries, {})
        assert len(messages) == 3

        # User message
        assert messages[0].role == "user"
        assert messages[0].text == "hello world"
        assert messages[0].timestamp == "2026-08-06T09:50:46Z"

        # Tool call message
        assert messages[1].role == "assistant"
        assert messages[1].content_type == "tool_use"
        assert messages[1].tool_name == "view_file"

        # Assistant response
        assert messages[2].role == "assistant"
        assert messages[2].text == "Hello! How can I help you?"

    def test_parse_history_entry(self):
        provider = AntigravityProvider()
        entry = {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-08-06T09:50:46Z",
            "content": "<USER_REQUEST>\ncheck state\n</USER_REQUEST>",
        }
        msg = provider.parse_history_entry(entry)
        assert msg is not None
        assert msg.role == "user"
        assert msg.text == "check state"


class TestAntigravityTranscriptDiscovery:
    def test_discover_transcript(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "test-session-id" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        transcript_file = session_dir / "transcript.jsonl"
        transcript_file.write_text('{"step_index": 0}\n')

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(tmp_path), "shared:@0")
        assert event is not None
        assert event.session_id == "test-session-id"
        assert event.transcript_path == str(transcript_file)

    def test_discover_transcript_max_age_zero(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        session_dir = brain / "test-session-id" / ".system_generated" / "logs"
        session_dir.mkdir(parents=True)
        transcript_file = session_dir / "transcript.jsonl"
        transcript_file.write_text('{"step_index": 0}\n')

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(str(tmp_path), "shared:@0", max_age=0)
        assert event is not None
        assert event.session_id == "test-session-id"

    def test_discover_transcript_filters_by_cwd(self, tmp_path, monkeypatch):
        provider = AntigravityProvider()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        
        # Directory A
        s1 = brain / "sess-proj-a" / ".system_generated" / "logs"
        s1.mkdir(parents=True)
        (s1 / "transcript.jsonl").write_text(f'{{"content": "{tmp_path}/proj-a"}}\n')

        # Directory B (newer mtime)
        s2 = brain / "sess-proj-b" / ".system_generated" / "logs"
        s2.mkdir(parents=True)
        (s2 / "transcript.jsonl").write_text(f'{{"content": "{tmp_path}/proj-b"}}\n')

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        event = provider.discover_transcript(f"{tmp_path}/proj-a", "shared:@0")
        assert event is not None
        assert event.session_id == "sess-proj-a"


class TestAntigravityDetectionAndYolo:
    def test_process_detection(self):
        assert detect_provider_from_command("agy") == "antigravity"
        assert detect_provider_from_command("/usr/local/bin/agy") == "antigravity"
        assert detect_provider_from_command("antigravity") == "antigravity"

    def test_transcript_path_detection(self):
        path = "/Users/user/.gemini/antigravity-cli/brain/123/.system_generated/logs/transcript.jsonl"
        assert detect_provider_from_transcript_path(path) == "antigravity"

    def test_yolo_resolution(self):
        cmd = resolve_launch_command("antigravity", approval_mode="yolo")
        assert cmd == "agy --dangerously-skip-permissions"
