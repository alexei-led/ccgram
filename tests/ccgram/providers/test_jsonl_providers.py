import json
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ccgram.providers.codex import (
    CodexProvider,
)
from ccgram.providers.gemini import (
    GeminiProvider,
    _box_is_interactive,
    _clean_box_lines,
    _extract_active_box,
)
from ccgram.session_protocol.messages import SourceEntry, build_messages


HOOK_AWARE_JSONL_PROVIDERS = [CodexProvider, GeminiProvider]


@pytest.fixture(params=HOOK_AWARE_JSONL_PROVIDERS, ids=lambda cls: cls.__name__)
def jsonl_provider(request: pytest.FixtureRequest):
    return request.param()


class TestHookAwareCapabilities:
    def test_hook_flags(self, jsonl_provider) -> None:
        caps = jsonl_provider.capabilities
        assert caps.supports_hook is True
        assert caps.supports_resume is True
        assert caps.supports_continue is True

    @pytest.mark.parametrize(
        "resume_id",
        ["abc; rm -rf /", "abc && curl evil.sh", "../../etc/passwd", "$(whoami)"],
    )
    def test_shell_metacharacters_in_resume_id_are_rejected(
        self, jsonl_provider, resume_id: str
    ) -> None:
        with pytest.raises(ValueError, match="Invalid resume_id"):
            jsonl_provider.make_launch_args(resume_id=resume_id)

    @pytest.mark.parametrize("resume_id", ["abc-123", "session_42"])
    def test_valid_resume_ids(self, jsonl_provider, resume_id: str) -> None:
        assert resume_id in jsonl_provider.make_launch_args(resume_id=resume_id)


class TestCodexLaunchArgs:
    def test_resume_uses_subcommand(self) -> None:
        codex = CodexProvider()
        result = codex.make_launch_args(resume_id="abc-123")
        assert result == "resume abc-123"

    def test_continue_uses_resume_last(self) -> None:
        codex = CodexProvider()
        result = codex.make_launch_args(use_continue=True)
        assert result == "resume --last"


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


class TestCanonicalCodexTranscript:
    @staticmethod
    def _messages(entries: list[dict]):
        return build_messages(
            [SourceEntry(offset, entry) for offset, entry in enumerate(entries)],
            "codex-test",
            "codex",
        )

    def test_normalizes_response_phase_and_user_input(self) -> None:
        messages = self._messages(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                },
                {
                    "type": "input_item",
                    "payload": {"role": "user", "content": "question"},
                },
            ]
        )
        assert [message.role for message in messages] == ["agent", "user"]
        assert [part.text for part in messages[0].parts] == ["done"]
        assert messages[0].phase == "final_answer"
        assert messages[1].parts[0].text == "question"

    def test_agent_stream_and_completion_share_one_message(self) -> None:
        messages = self._messages(
            [
                {
                    "type": "event_msg",
                    "payload": {"type": "agent_message", "message": "working"},
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "task_complete",
                        "last_agent_message": "working and done",
                    },
                },
            ]
        )
        assert len(messages) == 1
        assert messages[0].parts[0].text == "working and done"
        assert messages[0].complete is True
        assert messages[0].phase == "final_answer"

    def test_tool_result_completes_normalized_call(self) -> None:
        messages = self._messages(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "exec_command",
                        "arguments": '{"cmd":"ls"}',
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "c1",
                        "output": "ok",
                    },
                },
            ]
        )
        assert len(messages) == 1
        tool = messages[0].parts[0]
        assert tool.type == "tool"
        assert tool.tool_call_id == "c1"
        assert tool.name == "exec_command"
        assert tool.status == "completed"
        assert tool.output == {"text": "ok", "truncated": False}


class TestCodexTerminalStatus:
    def test_preserves_advisory_selection_guess(self) -> None:
        codex = CodexProvider()
        pane = "Pick an action:\n› 1. First\n  2. Second"

        status = codex.parse_terminal_status(pane)

        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "SelectionUI"
        assert status.ui_advisory is True

    def test_detects_selection_ui(self) -> None:
        codex = CodexProvider()
        pane = (
            "  Which option should I use?\n"
            "  › Option A\n"
            "    Option B\n"
            "  Press enter to confirm\n"
        )
        status = codex.parse_terminal_status(pane)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "SelectionUI"

    def test_formats_edit_prompt_for_readability(self) -> None:
        codex = CodexProvider()
        pane = (
            "Do you want to make this edit to src/ccgram/bot.py?\n"
            "947    936 -    await register_commands(application.bot, provider=get_provider())"
            "    948 +    await register_commands(application.bot, providers=_menu_providers())\n"
            "953          try:\n"
            "942 -            await register_commands(context.bot, provider=get_provider())"
            "    954 +            await register_commands(context.bot, providers=_menu_providers())\n"
            "› 1. Yes, proceed (y)  2. Yes, and don't ask again for these files (a)"
            "  3. No, and tell Codex what to do differently (esc)\n"
            "Press enter to confirm or esc to cancel\n"
        )
        status = codex.parse_terminal_status(pane)
        assert status is not None
        assert status.is_interactive is True
        assert "File: src/ccgram/bot.py" in status.raw_text
        assert "Changes: +" in status.raw_text
        assert "› 1. Yes, proceed (y)" in status.raw_text
        assert "  2. Yes, and don't ask again for these files (a)" in status.raw_text
        assert "  3. No, and tell Codex what to do differently (esc)" in status.raw_text
        assert "Press enter to confirm or esc to cancel" in status.raw_text

    def test_returns_none_for_non_interactive(self) -> None:
        codex = CodexProvider()
        status = codex.parse_terminal_status("normal output\n")
        assert status is None


class TestCanonicalGeminiTranscript:
    @staticmethod
    def _messages(entries: list[dict]):
        return build_messages(
            [SourceEntry(offset, entry) for offset, entry in enumerate(entries)],
            "gemini-test",
            "gemini",
        )

    def test_normalizes_user_assistant_and_notice_records(self) -> None:
        messages = self._messages(
            [
                {"type": "user", "content": [{"text": "hello"}]},
                {"type": "gemini", "content": "answer"},
                {"type": "error", "content": "failed"},
            ]
        )
        assert [message.role for message in messages] == ["user", "agent", "agent"]
        assert messages[0].parts[0].text == "hello"
        assert messages[1].parts[0].text == "answer"
        assert messages[2].parts[0].type == "notice"
        assert messages[2].parts[0].level == "error"

    def test_embedded_tool_result_completes_call(self) -> None:
        messages = self._messages(
            [
                {
                    "type": "gemini",
                    "content": "Using tool",
                    "toolCalls": [
                        {
                            "id": "tc1",
                            "name": "ReadFile",
                            "args": {"path": "x.py"},
                            "result": "contents",
                        }
                    ],
                }
            ]
        )
        assert len(messages) == 1
        tool = next(part for part in messages[0].parts if part.type == "tool")
        assert tool.tool_call_id == "tc1"
        assert tool.name == "ReadFile"
        assert tool.status == "completed"
        assert tool.output == {"text": "contents", "truncated": False}


class TestGeminiTerminalStatus:
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
    BOXED_PERMISSION_PANE = (
        "╭──────────────────────╮\n"
        "│ Action Required      │\n"
        "│ ? Shell date         │\n"
        "│ Allow execution of: 'date'?\n"
        "│ ● 1. Allow once      │\n"
        "│   2. Allow for this session\n"
        "│   3. Allow for all future sessions\n"
        "│   4. No, suggest changes (esc)\n"
        "╰──────────────────────╯\n"
    )
    SELECT_MODEL_PANE = (
        "Select Model\n"
        "● 1. Auto (Gemini 3)\n"
        "  2. Auto (Gemini 2.5)\n"
        "  3. Manual\n"
        "(Press Esc to close)\n"
    )

    NON_INTERACTIVE_PANES = {
        "plain_progress": "Working on something...\nProcessing files\n",
        "answer_at_prompt": (
            "\u2726 Here is your answer.\n\nSome normal output text.\n> \n"
        ),
        "gemini_status_chrome": (
            "✦ Here is your answer.\n"
            "[INSERT] ~/Workspace/ccgram (main)           "
            "no sandbox (see /docs)           "
            "/model Auto (Gemini 3) 100% context left | 375.5 MB\n"
        ),
        "prompt_without_bottom_marker": (
            "Action Required\n? Shell ls -la\nAllow execution of: 'ls'?\n"
        ),
        "action_required_inside_prose": (
            "\u2726 Here's what you need to know:\n"
            "\n"
            "Action Required: You must update the config file.\n"
            "Edit settings.json and set the flag to true.\n"
            "Then restart the service.\n"
            "> \n"
        ),
    }

    @pytest.mark.parametrize(
        ("pane_attr", "ui_type", "expected_in_raw_text"),
        [
            pytest.param(
                "SHELL_PERMISSION_PANE",
                "PermissionPrompt",
                ("Action Required", "Allow once", "Allow for this session"),
                id="shell_permission",
            ),
            pytest.param(
                "WRITE_PERMISSION_PANE", "PermissionPrompt", (), id="write_permission"
            ),
            pytest.param(
                "BOXED_PERMISSION_PANE",
                "PermissionPrompt",
                ("Allow for this session",),
                id="boxed_permission",
            ),
            pytest.param(
                "SELECT_MODEL_PANE",
                "SelectionUI",
                ("Auto (Gemini 3)",),
                id="model_picker",
            ),
        ],
    )
    def test_detects_interactive_ui(
        self, pane_attr: str, ui_type: str, expected_in_raw_text: tuple[str, ...]
    ) -> None:
        status = GeminiProvider().parse_terminal_status(getattr(self, pane_attr))
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == ui_type
        for fragment in expected_in_raw_text:
            assert fragment in status.raw_text

    def test_boxed_prompt_keeps_the_body_not_just_the_header(self) -> None:
        status = GeminiProvider().parse_terminal_status(self.BOXED_PERMISSION_PANE)
        assert status is not None
        assert status.raw_text != "Action Required"

    @pytest.mark.parametrize("pane_id", list(NON_INTERACTIVE_PANES))
    def test_returns_none_for_non_interactive_pane(self, pane_id: str) -> None:
        pane = self.NON_INTERACTIVE_PANES[pane_id]
        assert GeminiProvider().parse_terminal_status(pane) is None


class TestGeminiBoxedPrompt:
    REAL_TOOL_PERMISSION_BOX = (
        "> use the shell tool to run: whoami\n"
        "\n"
        "  Responding with gemini-3-flash-preview\n"
        "╭────────────────────────────────────────────────────────────────╮\n"
        "│ Answer Questions                                                 │\n"
        "│                                                                  │\n"
        "│ The 'whoami' command was denied by policy. Is there another way "
        "I can help you, or do you want to try a different approach?       │\n"
        "│                                                                  │\n"
        "│ > Explain why you need this or suggest an alternative.           │\n"
        "│                                                                  │\n"
        "│ Enter to submit · Esc to cancel                                  │\n"
        "╰────────────────────────────────────────────────────────────────╯\n"
    )
    REAL_TRUST_SELECTION_BOX = (
        " ╭───────────────────────────────────────────────────────────────╮\n"
        " │                                                                 │\n"
        " │ Do you trust the files in this folder?                          │\n"
        " │                                                                 │\n"
        " │ Trusting a folder allows Gemini CLI to load its local "
        "configurations, including custom commands, hooks, and settings.  │\n"
        " │ These configurations could execute code on your behalf.         │\n"
        " │                                                                 │\n"
        " │ ● 1. Trust folder (gemini-issue75)                              │\n"
        " │   2. Trust parent folder (tmp)                                  │\n"
        " │   3. Don't trust                                                │\n"
        " │                                                                 │\n"
        " ╰───────────────────────────────────────────────────────────────╯\n"
    )
    BOXED_ALLOW_PROMPT = (
        "╭──────────────────────────────────────────────╮\n"
        "│ Allow execution of: 'rm -rf build'?           │\n"
        "│ ● 1. Allow once                               │\n"
        "│   2. Allow for this session                   │\n"
        "│   3. Allow for all future sessions            │\n"
        "│   4. No, suggest changes (esc)                │\n"
        "╰──────────────────────────────────────────────╯\n"
    )
    AUTH_STARTUP_BOX = (
        "╭────────────────────────────────────────────────────────────╮\n"
        "│                                                              │\n"
        "│ ⠧ Waiting for authentication... (Press Esc or Ctrl+C to "
        "cancel)  │\n"
        "│                                                              │\n"
        "╰────────────────────────────────────────────────────────────╯\n"
    )
    TOOL_EXEC_BOX = (
        "╭──────────────────────────────────────────────╮\n"
        "│ ✓  Shell ls -la                               │\n"
        "│                                               │\n"
        "╰──────────────────────────────────────────────╯\n"
    )

    def test_issue_75_real_tool_permission_shows_question(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            self.REAL_TOOL_PERMISSION_BOX, pane_title="Action Required: ✋"
        )
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"
        assert status.raw_text != "Action Required"
        assert "Answer Questions" in status.raw_text
        assert "denied by policy" in status.raw_text

    def test_real_trust_selection_box_detected(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            self.REAL_TRUST_SELECTION_BOX, pane_title="Ready: ◇"
        )
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "SelectionUI"
        assert "Do you trust the files in this folder?" in status.raw_text
        assert "Don't trust" in status.raw_text

    def test_boxed_allow_prompt_includes_options(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            self.BOXED_ALLOW_PROMPT, pane_title="Action Required: ✋"
        )
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"
        assert status.raw_text != "Action Required"
        assert "Allow once" in status.raw_text
        assert "Allow for this session" in status.raw_text

    def test_box_content_strips_border_glyphs(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            self.REAL_TOOL_PERMISSION_BOX, pane_title="Action Required: ✋"
        )
        assert status is not None
        for glyph in ("│", "╭", "╮", "╰", "╯"):
            assert glyph not in status.raw_text

    def test_auth_startup_box_not_interactive(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            self.AUTH_STARTUP_BOX, pane_title="Ready: ◇"
        )
        assert status is None

    def test_tool_exec_box_not_interactive(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(self.TOOL_EXEC_BOX, pane_title="")
        assert status is None

    def test_extract_active_box_returns_cleaned_inner(self) -> None:
        box = (
            "noise above\n"
            "╭────────────╮\n"
            "│ question?  │\n"
            "│ ● 1. yes   │\n"
            "╰────────────╯\n"
        )
        assert _extract_active_box(box) == "question?\n● 1. yes"

    def test_extract_active_box_none_without_box(self) -> None:
        assert _extract_active_box("just normal output\n> \n") is None

    def test_extract_active_box_picks_last_box(self) -> None:
        two = "╭───╮\n│ old │\n╰───╯\ninterim\n╭───╮\n│ new │\n╰───╯\n"
        assert _extract_active_box(two) == "new"

    def test_clean_box_lines_drops_borders_and_padding(self) -> None:
        assert _clean_box_lines(["│ a │", "│   │", "│ b │", "──────"]) == ["a", "b"]

    def test_box_is_interactive_true_for_options(self) -> None:
        assert _box_is_interactive("header\n● 1. choose me\n  2. other") is True

    def test_box_is_interactive_false_for_plain_text(self) -> None:
        assert _box_is_interactive("just a status line\nno options here") is False

    def test_box_is_interactive_single_numbered_line_is_prose(self) -> None:
        assert _box_is_interactive("Step 1. do the thing first") is False

    def test_box_is_interactive_two_numbered_lines_is_prompt(self) -> None:
        assert _box_is_interactive("Pick:\n1. yes\n2. no") is True

    def test_extract_active_box_none_on_dangling_open(self) -> None:
        torn = (
            "╭───╮\n│ old │\n╰───╯\n"
            "interim output\n"
            "╭───────────────────╮\n│ new question being drawn │\n"
        )
        assert _extract_active_box(torn) is None

    def test_extract_active_box_none_when_crossing_box_boundary(self) -> None:
        torn = "╭──╮\n│ a │\n╰──╯\n╰──╯\n"
        assert _extract_active_box(torn) is None

    def test_dangling_open_falls_back_not_stale_question(self) -> None:
        torn = (
            "╭────────────────────────────╮\n"
            "│ Old answered prompt          │\n"
            "│ ● 1. picked                  │\n"
            "╰────────────────────────────╯\n"
            "scrollback\n"
            "╭────────────────────────────╮\n"
            "│ New prompt still rendering   │\n"
        )
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(torn, pane_title="Ready: ◇")
        assert status is None


class TestGeminiPaneTitleStatus:
    def test_working_title_returns_working_status(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status("some output", pane_title="Working: ✦")
        assert status is not None
        assert status.is_interactive is False
        assert status.display_label == "\u2026working"

    def test_working_title_without_emoji_returns_working_status(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            "some output", pane_title="Working… (ccgram)"
        )
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

    def test_action_required_title_without_emoji_still_interactive(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            "some output", pane_title="Action Required (ccgram)"
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
    def test_returns_exact_builtins(self, jsonl_provider) -> None:
        result = jsonl_provider.discover_commands("/tmp/nonexistent")
        names = {c.name for c in result}
        assert names == set(jsonl_provider.capabilities.builtin_commands)


class TestGeminiCommandDiscovery:
    def test_discovers_gemini_toml_commands_via_claude_base_dir(
        self, tmp_path: Path
    ) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir(parents=True)
        gemini_dir = tmp_path / ".gemini" / "commands" / "code"
        gemini_dir.mkdir(parents=True)
        (gemini_dir / "fix.toml").write_text(
            "description = 'Fix all issues'\nprompt = '...'\n"
        )

        gemini = GeminiProvider()
        commands = gemini.discover_commands(str(claude_dir))
        names = {cmd.name for cmd in commands}
        assert "code:fix" in names
        discovered = next(cmd for cmd in commands if cmd.name == "code:fix")
        assert discovered.description == "Fix all issues"
        assert discovered.source == "command"


def _write_gemini_session(
    tmp_dir: Path,
    project_dir: str,
    project_key: str,
    session_name: str,
    session_id: str,
) -> Path:
    chats_dir = tmp_dir / ".gemini" / "tmp" / project_key / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)
    fpath = chats_dir / f"{session_name}.jsonl"
    header = {
        "sessionId": session_id,
        "projectHash": hashlib.sha256(project_dir.encode()).hexdigest(),
        "startTime": "2026-03-01T00:00:00.000Z",
        "lastUpdated": "2026-03-01T00:00:00.000Z",
    }
    fpath.write_text(json.dumps(header) + "\n")
    return fpath


def _write_codex_session(
    sessions_dir: Path,
    date_parts: str,
    name: str,
    session_id: str,
    cwd: str,
    *,
    source: object = "cli",
    originator: object = "codex_cli_rs",
) -> Path:
    """Write a fake Codex session JSONL.

    ``source`` and ``originator`` mirror the keys real Codex writes into the
    ``session_meta`` payload. Pass ``originator=None`` to omit the field
    entirely (simulating older transcripts that predate the field).
    """
    day_dir = sessions_dir / date_parts
    day_dir.mkdir(parents=True, exist_ok=True)
    fpath = day_dir / f"{name}.jsonl"
    payload: dict[str, object] = {"id": session_id, "cwd": cwd, "source": source}
    if originator is not None:
        payload["originator"] = originator
    meta = {"type": "session_meta", "payload": payload}
    fpath.write_text(json.dumps(meta) + "\n")
    return fpath


class TestCodexDiscoverTranscript:
    def test_finds_matching_transcript(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions"
        fpath = _write_codex_session(
            sessions_dir, "2026/03/02", "test-session", "uuid-abc", "/my/project"
        )
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-abc"
        assert event.cwd == "/my/project"
        assert event.transcript_path == str(fpath)
        assert event.window_key == "ccgram:@7"

    def test_returns_none_when_no_cwd_match(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions"
        _write_codex_session(
            sessions_dir, "2026/03/02", "test-session", "uuid-abc", "/other/project"
        )
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is None

    def test_returns_none_when_no_sessions_dir(self, tmp_path: Path) -> None:
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is None

    def test_picks_most_recent_by_mtime(self, tmp_path: Path) -> None:

        sessions_dir = tmp_path / ".codex" / "sessions"
        old = _write_codex_session(
            sessions_dir, "2026/03/01", "old", "uuid-old", "/my/project"
        )
        _write_codex_session(
            sessions_dir, "2026/03/02", "new", "uuid-new", "/my/project"
        )
        os.utime(old, (old.stat().st_mtime - 100, old.stat().st_mtime - 100))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-new"

    def test_searches_configured_codex_homes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A guest's Codex runs with its own CODEX_HOME; a glob covers them all.
        guest_sessions = tmp_path / "admin" / "codex" / "XY" / ".codex" / "sessions"
        fpath = _write_codex_session(
            guest_sessions, "2026/11/12", "guest", "uuid-guest", "/friends/XY"
        )
        from ccgram.config import config

        monkeypatch.setattr(
            config,
            "codex_session_dirs",
            [str(tmp_path / "admin" / "codex" / "*" / ".codex" / "sessions")],
        )
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path / "home"):
            event = codex.discover_transcript("/friends/XY", "hc:hc-t50")
        assert event is not None
        assert event.session_id == "uuid-guest"
        assert event.transcript_path == str(fpath)

    def test_newest_match_wins_across_codex_homes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        own = _write_codex_session(
            tmp_path / ".codex" / "sessions", "2026/11/12", "a", "uuid-own", "/p"
        )
        _write_codex_session(
            tmp_path / "guest" / "sessions", "2026/11/12", "b", "uuid-guest", "/p"
        )
        os.utime(own, (own.stat().st_mtime - 50, own.stat().st_mtime - 50))
        from ccgram.config import config

        monkeypatch.setattr(
            config, "codex_session_dirs", [str(tmp_path / "guest" / "sessions")]
        )
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/p", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-guest"

    def test_skips_non_session_meta_first_line(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "03" / "02"
        sessions_dir.mkdir(parents=True)
        fpath = sessions_dir / "bad.jsonl"
        fpath.write_text(json.dumps({"type": "response_item", "payload": {}}) + "\n")
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/any", "ccgram:@7")
        assert event is None

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions" / "2026" / "03" / "02"
        sessions_dir.mkdir(parents=True)
        fpath = sessions_dir / "corrupt.jsonl"
        fpath.write_text("{not valid json\n")
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/any", "ccgram:@7")
        assert event is None

    def test_skips_empty_session_id(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions"
        _write_codex_session(sessions_dir, "2026/03/02", "no-id", "", "/my/project")
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is None

    def test_skips_stale_transcript(self, tmp_path: Path) -> None:

        sessions_dir = tmp_path / ".codex" / "sessions"
        fpath = _write_codex_session(
            sessions_dir, "2026/03/01", "old-session", "uuid-old", "/my/project"
        )
        old_time = fpath.stat().st_mtime - 300
        os.utime(fpath, (old_time, old_time))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is None

    def test_matches_fresh_transcript_only(self, tmp_path: Path) -> None:

        sessions_dir = tmp_path / ".codex" / "sessions"
        stale = _write_codex_session(
            sessions_dir, "2026/03/01", "stale", "uuid-stale", "/my/project"
        )
        old_time = stale.stat().st_mtime - 300
        os.utime(stale, (old_time, old_time))

        _write_codex_session(
            sessions_dir, "2026/03/02", "fresh", "uuid-fresh", "/my/project"
        )

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-fresh"

    def test_skips_newer_guardian_subagent_transcript(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions"
        main = _write_codex_session(
            sessions_dir, "2026/03/01", "main", "uuid-main", "/my/project"
        )
        _write_codex_session(
            sessions_dir,
            "2026/03/02",
            "guardian",
            "uuid-guardian",
            "/my/project",
            source={"subagent": {"other": "guardian"}},
        )
        # Ensure guardian is "newer" so the test proves content-based filtering,
        # not just mtime ordering — back-date main explicitly.
        os.utime(main, (main.stat().st_mtime - 1, main.stat().st_mtime - 1))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-main"

    def test_returns_none_when_only_guardian_subagent_matches(
        self, tmp_path: Path
    ) -> None:
        sessions_dir = tmp_path / ".codex" / "sessions"
        _write_codex_session(
            sessions_dir,
            "2026/03/02",
            "guardian",
            "uuid-guardian",
            "/my/project",
            source={"subagent": {"other": "guardian"}},
        )

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is None

    def test_skips_codex_exec_session(self, tmp_path: Path) -> None:
        """``codex exec`` runs (originator=codex_exec) must not be claimed by
        hookless discovery, even if a real interactive session exists in the
        same cwd. The interactive session wins."""
        sessions_dir = tmp_path / ".codex" / "sessions"
        main = _write_codex_session(
            sessions_dir,
            "2026/03/01",
            "main",
            "uuid-main",
            "/my/project",
        )
        _write_codex_session(
            sessions_dir,
            "2026/03/02",
            "exec-run",
            "uuid-exec",
            "/my/project",
            source="exec",
            originator="codex_exec",
        )
        # Ensure exec-run is "newer" so the test proves content-based filtering,
        # not mtime ordering — back-date main explicitly.
        os.utime(main, (main.stat().st_mtime - 1, main.stat().st_mtime - 1))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-main"

    def test_codex_desktop_exec_survives_filter(self, tmp_path: Path) -> None:
        """Codex Desktop legitimately writes ``source = "exec"`` while keeping
        a non-``codex_exec`` originator. Those sessions must survive the
        filter."""
        sessions_dir = tmp_path / ".codex" / "sessions"
        _write_codex_session(
            sessions_dir,
            "2026/03/02",
            "desktop-exec",
            "uuid-desktop",
            "/my/project",
            source="exec",
            originator="Codex Desktop",
        )

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-desktop"

    def test_missing_originator_field_is_kept(self, tmp_path: Path) -> None:
        """Older Codex transcripts predate the ``originator`` field. Absence of
        the field must be treated as fail-safe-open (keep the session)."""
        sessions_dir = tmp_path / ".codex" / "sessions"
        _write_codex_session(
            sessions_dir,
            "2026/03/02",
            "no-originator",
            "uuid-bare",
            "/my/project",
            originator=None,
        )

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-bare"

    def test_skips_codex_exec_when_only_match(self, tmp_path: Path) -> None:
        """Reproduce the production bug: when the only cwd-matching session is
        a ``codex exec`` run, discovery must return ``None`` rather than bind
        the wrong session."""
        sessions_dir = tmp_path / ".codex" / "sessions"
        _write_codex_session(
            sessions_dir,
            "2026/03/02",
            "only-exec",
            "uuid-exec",
            "/Users/yubai/Obsidian/byheaven",
            source="exec",
            originator="codex_exec",
        )

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript(
                "/Users/yubai/Obsidian/byheaven", "ccgram:@14"
            )
        assert event is None


class TestCodexDiscoverTranscriptMaxAge:
    def test_max_age_zero_ignores_staleness(self, tmp_path: Path) -> None:

        sessions_dir = tmp_path / ".codex" / "sessions"
        fpath = _write_codex_session(
            sessions_dir, "2026/03/01", "old-session", "uuid-old", "/my/project"
        )
        old_time = fpath.stat().st_mtime - 300
        os.utime(fpath, (old_time, old_time))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7", max_age=0)
        assert event is not None
        assert event.session_id == "uuid-old"

    def test_max_age_none_uses_default(self, tmp_path: Path) -> None:

        sessions_dir = tmp_path / ".codex" / "sessions"
        fpath = _write_codex_session(
            sessions_dir, "2026/03/01", "old-session", "uuid-old", "/my/project"
        )
        old_time = fpath.stat().st_mtime - 300
        os.utime(fpath, (old_time, old_time))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7", max_age=None)
        assert event is None

    def test_explicit_max_age_respected(self, tmp_path: Path) -> None:

        sessions_dir = tmp_path / ".codex" / "sessions"
        fpath = _write_codex_session(
            sessions_dir, "2026/03/01", "session", "uuid-abc", "/my/project"
        )
        old_time = fpath.stat().st_mtime - 200
        os.utime(fpath, (old_time, old_time))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            assert (
                codex.discover_transcript("/my/project", "ccgram:@7", max_age=100)
                is None
            )
            event = codex.discover_transcript("/my/project", "ccgram:@7", max_age=300)
        assert event is not None
        assert event.session_id == "uuid-abc"


class TestGeminiDiscoverTranscript:
    def test_finds_session_via_project_hash_dir(self, tmp_path: Path) -> None:
        project = "/my/project"
        project_hash = hashlib.sha256(project.encode()).hexdigest()
        fpath = _write_gemini_session(
            tmp_path,
            project,
            project_hash,
            "session-2026-03-02T12-00-00abcd",
            "gemini-uuid-1",
        )

        gemini = GeminiProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = gemini.discover_transcript(project, "ccgram:@7")
        assert event is not None
        assert event.session_id == "gemini-uuid-1"
        assert event.cwd == project
        assert event.transcript_path == str(fpath)
        assert event.window_key == "ccgram:@7"

    def test_finds_session_via_projects_alias(self, tmp_path: Path) -> None:
        project = "/my/project"
        fpath = _write_gemini_session(
            tmp_path,
            project,
            "workspace-alias",
            "session-2026-03-02T12-00-00abcd",
            "gemini-uuid-2",
        )
        projects = {"projects": {project: "workspace-alias"}}
        (tmp_path / ".gemini").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".gemini" / "projects.json").write_text(json.dumps(projects))

        gemini = GeminiProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = gemini.discover_transcript(project, "ccgram:@8")
        assert event is not None
        assert event.session_id == "gemini-uuid-2"
        assert event.transcript_path == str(fpath)

    def test_respects_staleness_by_default(self, tmp_path: Path) -> None:
        project = "/my/project"
        project_hash = hashlib.sha256(project.encode()).hexdigest()
        fpath = _write_gemini_session(
            tmp_path,
            project,
            project_hash,
            "session-2026-03-01T09-00-00abcd",
            "gemini-old",
        )
        old_time = fpath.stat().st_mtime - 300
        os.utime(fpath, (old_time, old_time))

        gemini = GeminiProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = gemini.discover_transcript(project, "ccgram:@7")
        assert event is None

    def test_max_age_zero_ignores_staleness(self, tmp_path: Path) -> None:
        project = "/my/project"
        project_hash = hashlib.sha256(project.encode()).hexdigest()
        fpath = _write_gemini_session(
            tmp_path,
            project,
            project_hash,
            "session-2026-03-01T09-00-00abcd",
            "gemini-old",
        )
        old_time = fpath.stat().st_mtime - 300
        os.utime(fpath, (old_time, old_time))

        gemini = GeminiProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = gemini.discover_transcript(project, "ccgram:@7", max_age=0)
        assert event is not None
        assert event.session_id == "gemini-old"
        assert event.transcript_path == str(fpath)

    def test_does_not_scan_unrelated_project_dirs(self, tmp_path: Path) -> None:
        project = "/my/project"
        _write_gemini_session(
            tmp_path,
            project,
            "unrelated-dir-name",
            "session-2026-03-02T12-00-00abcd",
            "gemini-uuid-unrelated",
        )

        gemini = GeminiProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = gemini.discover_transcript(project, "ccgram:@9")
        assert event is None


class TestHooklessDiscoverTranscriptDefault:
    def test_codex_returns_none_when_no_sessions(self, tmp_path: Path) -> None:
        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            assert codex.discover_transcript("/any", "ccgram:@0") is None


class TestDiscoverTranscriptContract:
    @pytest.mark.parametrize(
        "provider_cls",
        [CodexProvider, GeminiProvider],
        ids=["codex", "gemini"],
    )
    def test_hookless_provider_has_discover_transcript(self, provider_cls) -> None:
        provider = provider_cls()
        assert hasattr(provider, "discover_transcript")
        result = provider.discover_transcript("/nonexistent", "ccgram:@0")
        assert result is None


class TestGeminiCapabilityFlag:
    def test_gemini_supports_incremental_read(self) -> None:
        gemini = GeminiProvider()
        assert gemini.capabilities.supports_incremental_read is True

    def test_codex_supports_incremental_read(self) -> None:
        codex = CodexProvider()
        assert codex.capabilities.supports_incremental_read is True
