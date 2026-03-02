"""Provider-specific tests for Codex and Gemini (JsonlProvider subclasses).

Tests behavior that differs from the generic contract tests: resume syntax,
builtin command sets, capability flags, and shared JSONL parsing edge cases.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ccbot.providers._jsonl import extract_content_blocks, parse_jsonl_line
from ccbot.providers.codex import CodexProvider
from ccbot.providers.gemini import GeminiProvider

# ── Shared hookless-provider tests (parametrized) ────────────────────────

HOOKLESS_PROVIDERS = [CodexProvider, GeminiProvider]


@pytest.fixture(params=HOOKLESS_PROVIDERS, ids=lambda cls: cls.__name__)
def hookless(request: pytest.FixtureRequest):
    return request.param()


class TestHooklessCapabilities:
    def test_hookless_flags(self, hookless) -> None:
        caps = hookless.capabilities
        assert caps.supports_hook is False
        assert caps.supports_resume is True
        assert caps.supports_continue is True

    def test_invalid_resume_id_raises(self, hookless) -> None:
        with pytest.raises(ValueError, match="Invalid resume_id"):
            hookless.make_launch_args(resume_id="abc; rm -rf /")

    def test_valid_resume_ids(self, hookless) -> None:
        assert hookless.make_launch_args(resume_id="abc-123")
        assert hookless.make_launch_args(resume_id="session_42")


# ── Codex-specific ───────────────────────────────────────────────────────


class TestCodexLaunchArgs:
    def test_resume_uses_subcommand(self) -> None:
        codex = CodexProvider()
        result = codex.make_launch_args(resume_id="abc-123")
        assert result == "resume abc-123"

    def test_continue_uses_resume_last(self) -> None:
        codex = CodexProvider()
        result = codex.make_launch_args(use_continue=True)
        assert result == "resume --last"


# ── Codex capabilities ───────────────────────────────────────────────────


class TestCodexCapabilities:
    def test_no_terminal_ui_patterns(self) -> None:
        codex = CodexProvider()
        assert codex.capabilities.terminal_ui_patterns == ()


# ── Gemini-specific ──────────────────────────────────────────────────────


class TestGeminiCapabilities:
    def test_declares_permission_prompt(self) -> None:
        gemini = GeminiProvider()
        assert "PermissionPrompt" in gemini.capabilities.terminal_ui_patterns


class TestGeminiLaunchArgs:
    def test_resume_uses_flag(self) -> None:
        gemini = GeminiProvider()
        result = gemini.make_launch_args(resume_id="abc-123")
        assert result == "--resume abc-123"

    def test_resume_latest(self) -> None:
        gemini = GeminiProvider()
        result = gemini.make_launch_args(resume_id="latest")
        assert result == "--resume latest"

    def test_continue_uses_resume_latest(self) -> None:
        gemini = GeminiProvider()
        result = gemini.make_launch_args(use_continue=True)
        assert result == "--resume latest"


# ── Codex transcript parsing ────────────────────────────────────────────


class TestCodexTranscriptParsing:
    def test_parses_assistant_response_item(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "hello"
        assert messages[0].role == "assistant"

    def test_parses_user_input_item(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "input_item",
                "payload": {"role": "user", "content": "what is this?"},
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "what is this?"
        assert messages[0].role == "user"

    def test_tracks_function_call_pending(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "role": "assistant",
                    "content": [
                        {"type": "function_call", "call_id": "fc1", "name": "bash"}
                    ],
                },
            }
        ]
        _, pending = codex.parse_transcript_entries(entries, {})
        assert "fc1" in pending

    def test_function_call_output_clears_pending(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call_output",
                            "call_id": "fc1",
                            "output": "ok",
                        }
                    ],
                },
            }
        ]
        _, pending = codex.parse_transcript_entries(entries, {"fc1": "bash"})
        assert "fc1" not in pending

    def test_skips_developer_role(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "system prompt"}],
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert messages == []

    def test_is_user_entry_detects_input_item(self) -> None:
        codex = CodexProvider()
        assert codex.is_user_transcript_entry(
            {"type": "input_item", "payload": {"role": "user"}}
        )

    def test_is_user_entry_skips_system_preamble(self) -> None:
        codex = CodexProvider()
        entry = {
            "type": "response_item",
            "payload": {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "<permissions>...</permissions>"}
                ],
            },
        }
        assert codex.is_user_transcript_entry(entry) is False


# ── Gemini transcript parsing ───────────────────────────────────────────


class TestGeminiTranscriptParsing:
    def test_parses_gemini_message(self) -> None:
        gemini = GeminiProvider()
        entries = [{"type": "gemini", "content": "here is my answer"}]
        messages, _ = gemini.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "here is my answer"
        assert messages[0].role == "assistant"

    def test_parses_user_message(self) -> None:
        gemini = GeminiProvider()
        entries = [{"type": "user", "content": "hello gemini"}]
        messages, _ = gemini.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "hello gemini"
        assert messages[0].role == "user"

    def test_tracks_tool_calls(self) -> None:
        gemini = GeminiProvider()
        entries = [
            {
                "type": "gemini",
                "content": "using tool",
                "toolCalls": [{"id": "tc1", "name": "shell"}],
            }
        ]
        messages, pending = gemini.parse_transcript_entries(entries, {})
        assert "tc1" in pending
        assert messages[0].content_type == "tool_use"

    def test_skips_unknown_types(self) -> None:
        gemini = GeminiProvider()
        entries = [{"type": "system", "content": "some system info"}]
        messages, _ = gemini.parse_transcript_entries(entries, {})
        assert messages == []

    def test_is_user_entry(self) -> None:
        gemini = GeminiProvider()
        assert gemini.is_user_transcript_entry({"type": "user"}) is True
        assert gemini.is_user_transcript_entry({"type": "gemini"}) is False


class TestGeminiTerminalStatus:
    """Gemini CLI interactive UI detection via parse_terminal_status."""

    SHELL_PERMISSION_PANE = (
        "some previous output\n"
        "\n"
        "Action Required\n"
        "? Shell pwd && git branch --show-current && git status -s && ls -F "
        "[current working directory /Users/alexei/Workspace] "
        "(Check current directory, git branch, status, and list …\n"
        "pwd && git branch --show-current && git status -s && ls -F\n"
        "Allow execution of: 'pwd, git, git, ls'?\n"
        "● 1. Allow once\n"
        "  2. Allow for this session\n"
        "  3. Allow for all future sessions\n"
        "  4. No, suggest changes (esc\n"
    )

    WRITE_PERMISSION_PANE = (
        "✦ I'll create the file now.\n"
        "\n"
        "Action Required\n"
        "? WriteFile /tmp/test.txt (Create test file)\n"
        "Allow write to: '/tmp/test.txt'?\n"
        "● 1. Allow once\n"
        "  2. Allow for this session\n"
        "  3. Allow for all future sessions\n"
        "  4. No, suggest changes (esc)\n"
    )

    def test_detects_shell_permission(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(self.SHELL_PERMISSION_PANE)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"

    def test_detects_write_permission(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(self.WRITE_PERMISSION_PANE)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"

    def test_permission_content_includes_options(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(self.SHELL_PERMISSION_PANE)
        assert status is not None
        assert "Allow once" in status.raw_text
        assert "Allow for this session" in status.raw_text
        assert "Action Required" in status.raw_text

    def test_returns_none_for_non_interactive_pane(self) -> None:
        gemini = GeminiProvider()
        pane = "Working on something...\nProcessing files\n"
        status = gemini.parse_terminal_status(pane)
        assert status is None

    def test_returns_none_for_normal_output(self) -> None:
        gemini = GeminiProvider()
        pane = "\u2726 Here is your answer.\n\nSome normal output text.\n> \n"
        status = gemini.parse_terminal_status(pane)
        assert status is None

    def test_returns_none_for_gemini_chrome(self) -> None:
        gemini = GeminiProvider()
        pane = (
            "✦ Here is your answer.\n"
            "[INSERT] ~/Workspace/ccbot (main)           "
            "no sandbox (see /docs)           "
            "/model Auto (Gemini 3) 100% context left | 375.5 MB\n"
        )
        status = gemini.parse_terminal_status(pane)
        assert status is None

    def test_no_interactive_when_bottom_marker_missing(self) -> None:
        pane = "Action Required\n? Shell ls -la\nAllow execution of: 'ls'?\n"
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(pane)
        assert status is None

    def test_no_false_positive_from_response_text(self) -> None:
        pane = (
            "\u2726 Here's what you need to know:\n"
            "\n"
            "Action Required: You must update the config file.\n"
            "Edit settings.json and set the flag to true.\n"
            "Then restart the service.\n"
            "> \n"
        )
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(pane)
        assert status is None


class TestGeminiPaneTitleStatus:
    """Gemini CLI pane-title-based state detection."""

    def test_working_title_returns_working_status(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status("some output", pane_title="Working: ✦")
        assert status is not None
        assert status.is_interactive is False
        assert status.display_label == "\u2026working"

    def test_action_required_title_with_matching_content(self) -> None:
        gemini = GeminiProvider()
        pane = (
            "Action Required\n"
            "? Shell ls\n"
            "Allow execution of: 'ls'?\n"
            "● 1. Allow once\n"
            "  2. No, suggest changes (esc\n"
        )
        status = gemini.parse_terminal_status(pane, pane_title="Action Required: ✋")
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"

    def test_action_required_title_without_matching_content(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            "some output", pane_title="Action Required: ✋"
        )
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"

    def test_ready_title_returns_none(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status("some output", pane_title="Ready: ◇")
        assert status is None

    def test_empty_pane_title_uses_content_only(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status("normal output\n", pane_title="")
        assert status is None


class TestHooklessCommands:
    def test_returns_exact_builtins(self, hookless) -> None:
        result = hookless.discover_commands("/tmp/nonexistent")
        names = {c.name for c in result}
        assert names == set(hookless.capabilities.builtin_commands)


# ── JSONL parsing edge cases (extract_content_blocks) ────────────────────


class TestParseJsonlLine:
    def test_json_array_returns_none(self) -> None:
        assert parse_jsonl_line("[1, 2, 3]") is None

    def test_json_string_returns_none(self) -> None:
        assert parse_jsonl_line('"just a string"') is None

    def test_json_number_returns_none(self) -> None:
        assert parse_jsonl_line("42") is None


class TestExtractContentBlocks:
    def test_string_content(self) -> None:
        text, ct, pending = extract_content_blocks("hello world", {})
        assert text == "hello world"
        assert ct == "text"

    def test_non_list_non_string_returns_empty(self) -> None:
        text, ct, pending = extract_content_blocks(42, {})
        assert text == ""
        assert ct == "text"

    def test_none_content_returns_empty(self) -> None:
        text, ct, pending = extract_content_blocks(None, {})
        assert text == ""
        assert ct == "text"

    def test_non_dict_blocks_skipped(self) -> None:
        text, ct, pending = extract_content_blocks(["not a dict", 42], {})
        assert text == ""

    def test_tool_use_tracked_in_pending(self) -> None:
        blocks = [{"type": "tool_use", "id": "t1", "name": "Read"}]
        _, ct, pending = extract_content_blocks(blocks, {})
        assert ct == "tool_use"
        assert pending == {"t1": "Read"}

    def test_tool_result_clears_pending(self) -> None:
        blocks = [{"type": "tool_result", "tool_use_id": "t1"}]
        _, ct, pending = extract_content_blocks(blocks, {"t1": "Read"})
        assert ct == "tool_result"
        assert "t1" not in pending

    def test_tool_result_without_id_does_not_pop_empty(self) -> None:
        blocks = [{"type": "tool_result"}]
        pending = {"t1": "Read"}
        _, _, result = extract_content_blocks(blocks, pending)
        assert result == {"t1": "Read"}


# ── Gemini whole-file transcript reading ─────────────────────────────────


_SAMPLE_GEMINI_TRANSCRIPT: dict = {
    "sessionId": "gemini-sess-1",
    "projectHash": "abc123",
    "startTime": "2026-01-01T00:00:00Z",
    "lastUpdated": "2026-01-01T00:05:00Z",
    "messages": [
        {"type": "user", "content": "hello gemini"},
        {"type": "gemini", "content": "hi there!"},
        {"type": "user", "content": "what is 2+2?"},
        {"type": "gemini", "content": "4"},
    ],
}


class TestGeminiReadTranscriptFile:
    def test_reads_all_messages_from_zero(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        f.write_text(json.dumps(_SAMPLE_GEMINI_TRANSCRIPT))
        gemini = GeminiProvider()
        entries, offset = gemini.read_transcript_file(str(f), 0)
        assert len(entries) == 4
        assert offset == 4
        assert entries[0]["content"] == "hello gemini"
        assert entries[3]["content"] == "4"

    def test_returns_only_new_messages(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        f.write_text(json.dumps(_SAMPLE_GEMINI_TRANSCRIPT))
        gemini = GeminiProvider()
        entries, offset = gemini.read_transcript_file(str(f), 2)
        assert len(entries) == 2
        assert offset == 4
        assert entries[0]["content"] == "what is 2+2?"

    def test_no_new_messages_when_offset_at_end(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        f.write_text(json.dumps(_SAMPLE_GEMINI_TRANSCRIPT))
        gemini = GeminiProvider()
        entries, offset = gemini.read_transcript_file(str(f), 4)
        assert entries == []
        assert offset == 4

    def test_detects_new_messages_after_file_update(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        data = dict(_SAMPLE_GEMINI_TRANSCRIPT)
        data["messages"] = list(data["messages"][:2])
        f.write_text(json.dumps(data))
        gemini = GeminiProvider()

        entries, offset = gemini.read_transcript_file(str(f), 0)
        assert len(entries) == 2
        assert offset == 2

        data["messages"] = list(_SAMPLE_GEMINI_TRANSCRIPT["messages"])
        f.write_text(json.dumps(data))

        entries, offset = gemini.read_transcript_file(str(f), 2)
        assert len(entries) == 2
        assert offset == 4

    def test_handles_invalid_json(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        f.write_text("{not valid json")
        gemini = GeminiProvider()
        entries, offset = gemini.read_transcript_file(str(f), 0)
        assert entries == []
        assert offset == 0

    def test_handles_missing_file(self, tmp_path) -> None:
        gemini = GeminiProvider()
        entries, offset = gemini.read_transcript_file(
            str(tmp_path / "nonexistent.json"), 0
        )
        assert entries == []
        assert offset == 0

    def test_handles_no_messages_key(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        f.write_text(json.dumps({"sessionId": "s1"}))
        gemini = GeminiProvider()
        entries, offset = gemini.read_transcript_file(str(f), 0)
        assert entries == []
        assert offset == 0

    def test_handles_non_dict_messages(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        data = dict(_SAMPLE_GEMINI_TRANSCRIPT)
        data["messages"] = [{"type": "user", "content": "ok"}, "not a dict", 42]
        f.write_text(json.dumps(data))
        gemini = GeminiProvider()
        entries, offset = gemini.read_transcript_file(str(f), 0)
        assert len(entries) == 1
        assert offset == 3

    def test_handles_non_dict_root(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        f.write_text(json.dumps([1, 2, 3]))
        gemini = GeminiProvider()
        entries, offset = gemini.read_transcript_file(str(f), 0)
        assert entries == []
        assert offset == 0


class TestGeminiMtimeCache:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        from ccbot.providers.gemini import _transcript_cache

        _transcript_cache.clear()
        yield
        _transcript_cache.clear()

    def test_cache_hit_skips_reparse(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        f.write_text(json.dumps(_SAMPLE_GEMINI_TRANSCRIPT))
        gemini = GeminiProvider()

        # First read populates cache
        entries1, _ = gemini.read_transcript_file(str(f), 0)
        assert len(entries1) == 4

        # Patch json.load to prove second read uses cache, not file
        with patch(
            "ccbot.providers.gemini.json.load",
            side_effect=AssertionError("should not be called"),
        ):
            entries2, offset2 = gemini.read_transcript_file(str(f), 0)
        assert len(entries2) == 4
        assert offset2 == 4

    def test_cache_invalidated_on_file_change(self, tmp_path) -> None:
        f = tmp_path / "transcript.json"
        data = dict(_SAMPLE_GEMINI_TRANSCRIPT)
        data["messages"] = list(data["messages"][:2])
        f.write_text(json.dumps(data))
        gemini = GeminiProvider()

        entries1, offset1 = gemini.read_transcript_file(str(f), 0)
        assert len(entries1) == 2
        assert offset1 == 2

        # Overwrite with more messages — size and mtime both change
        data["messages"] = list(_SAMPLE_GEMINI_TRANSCRIPT["messages"])
        f.write_text(json.dumps(data))

        entries2, offset2 = gemini.read_transcript_file(str(f), 2)
        assert len(entries2) == 2
        assert offset2 == 4


# ── Codex transcript discovery ─────────────────────────────────────────


class TestCodexDiscoverTranscript:
    def _write_session(
        self, sessions_dir: Path, date_parts: str, name: str, session_id: str, cwd: str
    ) -> Path:
        """Write a minimal Codex transcript file and return its path."""
        day_dir = sessions_dir / date_parts
        day_dir.mkdir(parents=True, exist_ok=True)
        fpath = day_dir / f"{name}.jsonl"
        meta = json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": session_id, "cwd": cwd},
            }
        )
        fpath.write_text(meta + "\n")
        return fpath

    def test_finds_matching_transcript(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions"
        fpath = self._write_session(
            sessions_dir, "2026/03/02", "test-session", "uuid-abc", "/my/project"
        )
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccbot:@7")
        assert event is not None
        assert event.session_id == "uuid-abc"
        assert event.cwd == "/my/project"
        assert event.transcript_path == str(fpath)
        assert event.window_key == "ccbot:@7"

    def test_returns_none_when_no_cwd_match(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions"
        self._write_session(
            sessions_dir, "2026/03/02", "test-session", "uuid-abc", "/other/project"
        )
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccbot:@7")
        assert event is None

    def test_returns_none_when_no_sessions_dir(self, tmp_path: Path) -> None:
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccbot:@7")
        assert event is None

    def test_picks_most_recent_by_mtime(self, tmp_path: Path) -> None:
        import os
        import time

        sessions_dir = tmp_path / ".codex" / "sessions"
        old = self._write_session(
            sessions_dir, "2026/03/01", "old", "uuid-old", "/my/project"
        )
        # Ensure mtime ordering
        time.sleep(0.05)
        self._write_session(
            sessions_dir, "2026/03/02", "new", "uuid-new", "/my/project"
        )
        # Make old file explicitly older
        os.utime(old, (old.stat().st_mtime - 100, old.stat().st_mtime - 100))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccbot:@7")
        assert event is not None
        assert event.session_id == "uuid-new"

    def test_skips_non_session_meta_first_line(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "03" / "02"
        sessions_dir.mkdir(parents=True)
        fpath = sessions_dir / "bad.jsonl"
        fpath.write_text(json.dumps({"type": "response_item", "payload": {}}) + "\n")
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/any", "ccbot:@7")
        assert event is None

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "03" / "02"
        sessions_dir.mkdir(parents=True)
        fpath = sessions_dir / "corrupt.jsonl"
        fpath.write_text("{not valid json\n")
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/any", "ccbot:@7")
        assert event is None

    def test_skips_empty_session_id(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions"
        self._write_session(sessions_dir, "2026/03/02", "no-id", "", "/my/project")
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccbot:@7")
        assert event is None

    def test_skips_stale_transcript(self, tmp_path: Path) -> None:
        import os

        sessions_dir = tmp_path / ".codex" / "sessions"
        fpath = self._write_session(
            sessions_dir, "2026/03/01", "old-session", "uuid-old", "/my/project"
        )
        old_time = fpath.stat().st_mtime - 300
        os.utime(fpath, (old_time, old_time))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccbot:@7")
        assert event is None

    def test_matches_fresh_transcript_only(self, tmp_path: Path) -> None:
        import os
        import time

        sessions_dir = tmp_path / ".codex" / "sessions"
        stale = self._write_session(
            sessions_dir, "2026/03/01", "stale", "uuid-stale", "/my/project"
        )
        old_time = stale.stat().st_mtime - 300
        os.utime(stale, (old_time, old_time))

        time.sleep(0.05)
        self._write_session(
            sessions_dir, "2026/03/02", "fresh", "uuid-fresh", "/my/project"
        )

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccbot:@7")
        assert event is not None
        assert event.session_id == "uuid-fresh"


class TestHooklessDiscoverTranscriptDefault:
    def test_gemini_returns_none(self) -> None:
        gemini = GeminiProvider()
        assert gemini.discover_transcript("/any/cwd", "ccbot:@0") is None

    def test_codex_returns_none_when_no_sessions(self, tmp_path: Path) -> None:
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            assert codex.discover_transcript("/any", "ccbot:@0") is None


class TestDiscoverTranscriptContract:
    """Contract test: discover_transcript exists on all providers and returns correctly."""

    @pytest.mark.parametrize(
        "provider_cls",
        [CodexProvider, GeminiProvider],
        ids=["codex", "gemini"],
    )
    def test_hookless_provider_has_discover_transcript(self, provider_cls) -> None:
        provider = provider_cls()
        assert hasattr(provider, "discover_transcript")
        result = provider.discover_transcript("/nonexistent", "ccbot:@0")
        assert result is None


class TestGeminiCapabilityFlag:
    def test_gemini_does_not_support_incremental_read(self) -> None:
        gemini = GeminiProvider()
        assert gemini.capabilities.supports_incremental_read is False

    def test_codex_supports_incremental_read(self) -> None:
        codex = CodexProvider()
        assert codex.capabilities.supports_incremental_read is True

    def test_codex_read_transcript_file_raises(self) -> None:
        codex = CodexProvider()
        with pytest.raises(NotImplementedError):
            codex.read_transcript_file("/tmp/fake.jsonl", 0)
