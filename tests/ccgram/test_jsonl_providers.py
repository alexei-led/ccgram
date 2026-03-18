"""Provider-specific tests for Codex and Gemini (JsonlProvider subclasses).

Tests behavior that differs from the generic contract tests: resume syntax,
builtin command sets, capability flags, and shared JSONL parsing edge cases.
"""

import json
import hashlib
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ccgram.providers._jsonl import extract_content_blocks, parse_jsonl_line
from ccgram.providers.base import EXPANDABLE_QUOTE_START
from ccgram.providers.codex import (
    CodexProvider,
    _format_codex_tool_result,
    _resolve_pending,
)
from ccgram.providers.gemini import GeminiProvider

# ── Pending resolution helper ────────────────────────────────────────────


class TestResolvePending:
    """Unit tests for the shared _resolve_pending helper."""

    @pytest.mark.parametrize(
        "pending_value, expected",
        [
            pytest.param(("shell", "shell"), ("shell", "shell"), id="tuple"),
            pytest.param("shell", ("shell", "shell"), id="legacy_string"),
        ],
    )
    def test_known_formats(
        self,
        pending_value: object,
        expected: tuple[str | None, str | None],
    ) -> None:
        pending: dict[str, object] = {"fc1": pending_value}
        assert _resolve_pending("fc1", pending) == expected
        assert "fc1" not in pending

    def test_missing_key_returns_nones(self) -> None:
        pending: dict[str, object] = {}
        assert _resolve_pending("missing", pending) == (None, None)

    def test_empty_call_id_returns_nones(self) -> None:
        pending: dict[str, object] = {"": ("a", "b")}
        assert _resolve_pending("", pending) == (None, None)
        assert "" in pending

    def test_non_string_call_id_returns_nones(self) -> None:
        pending: dict[str, object] = {}
        assert _resolve_pending(42, pending) == (None, None)


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


# ── Gemini-specific ──────────────────────────────────────────────────────


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
                    "type": "message",
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

    def test_parses_event_agent_message(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "working on it",
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "working on it"
        assert messages[0].role == "assistant"
        assert messages[0].content_type == "text"

    def test_dedupes_identical_event_and_response_messages(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "same text",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "same text"}],
                },
            },
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "same text"

    def test_tracks_function_call_pending(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "fc1",
                    "arguments": '{"cmd":"ls"}',
                },
            }
        ]
        messages, pending = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].content_type == "tool_use"
        assert messages[0].tool_use_id == "fc1"
        assert messages[0].tool_name == "exec_command"
        assert pending["fc1"] == ("exec_command", "exec_command")

    def test_function_call_output_clears_pending(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "fc1",
                    "output": "Chunk ID: abc\nOutput:\nok\n",
                },
            }
        ]
        messages, pending = codex.parse_transcript_entries(
            entries, {"fc1": ("exec_command", "exec_command")}
        )
        assert len(messages) == 1
        assert messages[0].content_type == "tool_result"
        assert messages[0].tool_use_id == "fc1"
        # exec_command output always gets stats + expandable quote
        assert "1 lines" in messages[0].text
        assert "fc1" not in pending

    def test_function_call_output_legacy_string_pending(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "fc1",
                    "output": "result text",
                },
            }
        ]
        messages, pending = codex.parse_transcript_entries(
            entries, {"fc1": "some_tool"}
        )
        assert len(messages) == 1
        assert messages[0].content_type == "tool_result"
        assert "fc1" not in pending

    def test_request_user_input_maps_to_ask_user_question(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "request_user_input",
                    "call_id": "q1",
                    "arguments": (
                        '{"questions":[{"question":"Pick one?",'
                        '"options":[{"label":"A"},{"label":"B"}]}]}'
                    ),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "q1",
                    "output": '{"answers":{"q":{"answers":["A"]}}}',
                },
            },
        ]
        messages, pending = codex.parse_transcript_entries(entries, {})
        assert pending == {}
        assert len(messages) == 2
        assert messages[0].content_type == "tool_use"
        assert messages[0].tool_name == "AskUserQuestion"
        assert "Pick one?" in messages[0].text
        assert messages[1].content_type == "tool_result"
        assert messages[1].text == "Selected: A"

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


class TestFormatCodexToolResult:
    """Unit tests for _format_codex_tool_result: stats, quotes, and apply_patch."""

    @pytest.mark.parametrize(
        "tool_name, text, expect_quote",
        [
            pytest.param(
                "shell", "line1\nline2\nline3", True, id="shell_always_quoted"
            ),
            pytest.param(
                "exec_command", "a\nb\nc\nd\ne", True, id="exec_command_always_quoted"
            ),
            pytest.param("read_file", "short", False, id="short_inline"),
            pytest.param("read_file", "a\nb\nc", False, id="3_lines_at_threshold"),
            pytest.param("read_file", "a\nb\nc\nd", True, id="4_lines_above_threshold"),
        ],
    )
    def test_quote_threshold(
        self, tool_name: str, text: str, expect_quote: bool
    ) -> None:
        result = _format_codex_tool_result(tool_name, text)
        if expect_quote:
            assert EXPANDABLE_QUOTE_START in result
            line_count = text.count("\n") + 1
            assert f"{line_count} lines" in result
        else:
            assert EXPANDABLE_QUOTE_START not in result

    def test_long_non_shell_gets_stats_and_quote(self) -> None:
        text = "\n".join(f"line {i}" for i in range(10))
        result = _format_codex_tool_result("read_file", text)
        assert "10 lines" in result
        assert EXPANDABLE_QUOTE_START in result

    @pytest.mark.parametrize(
        "output_json, expected",
        [
            pytest.param(
                '{"output": "Patch applied successfully"}',
                "Patch applied successfully",
                id="output_key",
            ),
            pytest.param(
                '{"result": "Changes applied"}',
                "Changes applied",
                id="result_key",
            ),
            pytest.param("not json at all", "not json at all", id="non_json_raw"),
            pytest.param('{"status": "ok"}', '{"status": "ok"}', id="no_output_key"),
        ],
    )
    def test_apply_patch_extraction(self, output_json: str, expected: str) -> None:
        assert _format_codex_tool_result("apply_patch", output_json) == expected

    def test_empty_output_returns_done(self) -> None:
        assert _format_codex_tool_result("shell", "") == "Done"


class TestCodexCustomToolCall:
    """Tests for custom_tool_call and custom_tool_call_output parsing."""

    def test_apply_patch_counts_update_files(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "ct1",
                    "input": (
                        "*** Update File: src/foo.py\n"
                        "--- before\n+++ after\n"
                        "*** Update File: src/bar.py\n"
                        "--- before\n+++ after\n"
                    ),
                },
            }
        ]
        messages, pending = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].content_type == "tool_use"
        assert messages[0].tool_name == "Edit"
        assert "2 file(s)" in messages[0].text
        assert "ct1" in pending

    def test_apply_patch_counts_add_and_delete_files(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "ct1",
                    "input": (
                        "*** Add File: src/new.py\ncontent\n"
                        "*** Delete File: src/old.py\n"
                        "*** Update File: src/mod.py\n--- a\n+++ b\n"
                    ),
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert "3 file(s)" in messages[0].text

    def test_non_apply_patch_uses_input_as_summary(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "some_other_tool",
                    "call_id": "ct1",
                    "input": "do something",
                },
            }
        ]
        messages, pending = codex.parse_transcript_entries(entries, {})
        assert messages[0].tool_name == "some_other_tool"
        assert "do something" in messages[0].text
        assert pending["ct1"] == ("some_other_tool", "some_other_tool")

    def test_empty_call_id_not_stored_in_pending(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "",
                    "input": "*** Update File: f.py\n",
                },
            }
        ]
        messages, pending = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert pending == {}
        assert messages[0].tool_use_id is None

    def test_long_input_truncated_in_summary(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "unknown_tool",
                    "call_id": "ct1",
                    "input": "x" * 300,
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert "..." in messages[0].text
        assert len(messages[0].text) < 300


class TestCustomToolCallOutput:
    """Tests for custom_tool_call_output parsing and formatting."""

    def test_apply_patch_output_extracted(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "ct1",
                    "output": '{"output": "Applied successfully"}',
                },
            }
        ]
        messages, pending = codex.parse_transcript_entries(
            entries, {"ct1": ("apply_patch", "Edit")}
        )
        assert messages[0].content_type == "tool_result"
        assert messages[0].text == "Applied successfully"
        assert "ct1" not in pending

    def test_shell_output_gets_quote(self) -> None:
        codex = CodexProvider()
        long_output = "\n".join(f"line {i}" for i in range(20))
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "ct2",
                    "output": long_output,
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(
            entries, {"ct2": ("shell", "shell")}
        )
        assert EXPANDABLE_QUOTE_START in messages[0].text

    def test_dict_output_with_output_key(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "ct1",
                    "output": {"output": "dict result"},
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(
            entries, {"ct1": ("apply_patch", "Edit")}
        )
        assert messages[0].text == "dict result"

    def test_no_pending_match_returns_raw_output(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "unknown",
                    "output": "some output",
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert messages[0].text == "some output"
        assert messages[0].tool_name is None

    def test_empty_output_returns_done(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "ct1",
                    "output": "",
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(
            entries, {"ct1": ("apply_patch", "Edit")}
        )
        assert messages[0].text == "Done"

    def test_legacy_string_pending_compat(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "ct1",
                    "output": "result",
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {"ct1": "shell"})
        assert messages[0].tool_name == "shell"


class TestCodexToolCallIntegration:
    """Integration tests: full transcript flows through parse_transcript_entries."""

    def test_function_call_then_output_shell_formatted(self) -> None:

        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "fc1",
                    "arguments": '{"cmd": "ls -la"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "fc1",
                    "output": "total 42\ndrwxr-xr-x  5 user group  160 Jan  1 00:00 .\ndrwxr-xr-x 10 user group  320 Jan  1 00:00 ..",
                },
            },
        ]
        messages, pending = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 2
        assert messages[0].content_type == "tool_use"
        assert messages[0].tool_name == "shell"
        assert messages[1].content_type == "tool_result"
        assert "3 lines" in messages[1].text
        assert EXPANDABLE_QUOTE_START in messages[1].text
        assert pending == {}

    def test_custom_tool_call_then_output_roundtrip(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "ct1",
                    "input": "*** Update File: src/main.py\n--- a\n+++ b\n",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "ct1",
                    "output": '{"output": "Patch applied"}',
                },
            },
        ]
        messages, pending = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 2
        assert messages[0].content_type == "tool_use"
        assert messages[0].tool_name == "Edit"
        assert "1 file(s)" in messages[0].text
        assert messages[1].content_type == "tool_result"
        assert messages[1].text == "Patch applied"
        assert messages[1].tool_name == "Edit"
        assert pending == {}

    def test_mixed_text_custom_and_function_calls(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me fix that."}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "ct1",
                    "input": "*** Update File: f.py\n",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "ct1",
                    "output": '{"output": "Done"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "fc1",
                    "arguments": '{"cmd": "make test"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "fc1",
                    "output": "tests passed",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "All done!"}],
                },
            },
        ]
        messages, pending = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 6
        types = [m.content_type for m in messages]
        assert types == [
            "text",
            "tool_use",
            "tool_result",
            "tool_use",
            "tool_result",
            "text",
        ]
        assert messages[0].text == "Let me fix that."
        assert messages[5].text == "All done!"
        assert pending == {}

    def test_function_call_output_without_pending_no_formatting(self) -> None:
        codex = CodexProvider()
        entries = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "orphan",
                    "output": "some text",
                },
            }
        ]
        messages, _ = codex.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "some text"
        assert messages[0].tool_name is None


class TestCodexTerminalStatus:
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

    def test_parses_user_array_content(self) -> None:
        gemini = GeminiProvider()
        entries = [
            {
                "type": "user",
                "content": [{"text": "hello "}, {"text": "from array"}],
            }
        ]
        messages, _ = gemini.parse_transcript_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "hello from array"
        assert messages[0].role == "user"

    def test_falls_back_to_display_content(self) -> None:
        gemini = GeminiProvider()
        entry = {
            "type": "user",
            "content": [{"meta": "ignored"}],
            "displayContent": [{"text": "display text"}],
        }
        parsed = gemini.parse_history_entry(entry)
        assert parsed is not None
        assert parsed.text == "display text"
        assert parsed.role == "user"

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

    def test_emits_tool_result_and_clears_pending_when_result_present(self) -> None:
        gemini = GeminiProvider()
        entries = [
            {
                "type": "gemini",
                "content": "",
                "toolCalls": [
                    {
                        "id": "tc1",
                        "name": "read_file",
                        "displayName": "ReadFile",
                        "args": {"file_path": "/tmp/x"},
                        "resultDisplay": "ok",
                    }
                ],
            }
        ]
        messages, pending = gemini.parse_transcript_entries(entries, {})
        assert len(messages) == 2
        assert messages[0].content_type == "tool_use"
        assert messages[0].tool_name == "ReadFile"
        assert messages[1].content_type == "tool_result"
        assert messages[1].text == "ok"
        assert "tc1" not in pending

    def test_parses_info_and_error_entries_as_assistant(self) -> None:
        gemini = GeminiProvider()
        entries = [
            {"type": "info", "content": "Request cancelled."},
            {"type": "error", "content": "API error"},
        ]
        messages, _ = gemini.parse_transcript_entries(entries, {})
        assert len(messages) == 2
        assert all(m.role == "assistant" for m in messages)
        assert messages[0].text == "Request cancelled."
        assert messages[1].text == "API error"

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

    def test_detects_selection_ui(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(self.SELECT_MODEL_PANE)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "SelectionUI"
        assert "Auto (Gemini 3)" in status.raw_text

    def test_permission_content_includes_options(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(self.SHELL_PERMISSION_PANE)
        assert status is not None
        assert "Allow once" in status.raw_text
        assert "Allow for this session" in status.raw_text
        assert "Action Required" in status.raw_text

    def test_detects_boxed_permission_prompt_content(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(self.BOXED_PERMISSION_PANE)
        assert status is not None
        assert status.is_interactive is True
        assert status.ui_type == "PermissionPrompt"
        assert "Allow for this session" in status.raw_text
        assert status.raw_text != "Action Required"

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
            "[INSERT] ~/Workspace/ccgram (main)           "
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

    def test_working_title_without_emoji_returns_working_status(self) -> None:
        gemini = GeminiProvider()
        status = gemini.parse_terminal_status(
            "some output", pane_title="Working… (ccbot)"
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
            "some output", pane_title="Action Required (ccbot)"
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
        from ccgram.providers.gemini import _transcript_cache

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
            "ccgram.providers.gemini.json.load",
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


def _write_gemini_session(
    tmp_dir: Path,
    project_dir: str,
    project_key: str,
    session_name: str,
    session_id: str,
) -> Path:
    """Write a minimal Gemini transcript and return its path."""
    chats_dir = tmp_dir / ".gemini" / "tmp" / project_key / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)
    fpath = chats_dir / f"{session_name}.json"
    payload = {
        "sessionId": session_id,
        "projectHash": hashlib.sha256(project_dir.encode()).hexdigest(),
        "startTime": "2026-03-01T00:00:00.000Z",
        "lastUpdated": "2026-03-01T00:00:00.000Z",
        "messages": [],
    }
    fpath.write_text(json.dumps(payload))
    return fpath


def _write_codex_session(
    sessions_dir: Path, date_parts: str, name: str, session_id: str, cwd: str
) -> Path:
    """Write a minimal Codex transcript file and return its path."""
    day_dir = sessions_dir / date_parts
    day_dir.mkdir(parents=True, exist_ok=True)
    fpath = day_dir / f"{name}.jsonl"
    meta = {"type": "session_meta", "payload": {"id": session_id, "cwd": cwd}}
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
        # Ensure mtime ordering
        time.sleep(0.05)
        _write_codex_session(
            sessions_dir, "2026/03/02", "new", "uuid-new", "/my/project"
        )
        # Make old file explicitly older
        os.utime(old, (old.stat().st_mtime - 100, old.stat().st_mtime - 100))

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-new"

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

        time.sleep(0.05)
        _write_codex_session(
            sessions_dir, "2026/03/02", "fresh", "uuid-fresh", "/my/project"
        )

        codex = CodexProvider()
        with patch.object(Path, "home", return_value=tmp_path):
            event = codex.discover_transcript("/my/project", "ccgram:@7")
        assert event is not None
        assert event.session_id == "uuid-fresh"


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
    """Contract test: discover_transcript exists on all providers and returns correctly."""

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
