import pytest

from ccgram.expandable_quote import EXPANDABLE_QUOTE_END as EXPQUOTE_END
from ccgram.expandable_quote import EXPANDABLE_QUOTE_START as EXPQUOTE_START
from ccgram.transcript_parser import (
    ParsedMessage,
    TranscriptParser,
)


class TestParseLine:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ('{"type": "user"}', {"type": "user"}),
            ("not-json", None),
            ("", None),
            ("   \t  ", None),
        ],
        ids=["valid_json", "invalid_json", "empty", "whitespace"],
    )
    def test_parse_line(self, line: str, expected: dict | None):
        assert TranscriptParser.parse_line(line) == expected


class TestExtractTextOnly:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("plain string", "plain string"),
            (
                [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
                "hello\nworld",
            ),
            (
                [
                    {"type": "text", "text": "keep"},
                    {"type": "tool_use", "name": "Read"},
                ],
                "keep",
            ),
            ([], ""),
            (42, ""),
        ],
        ids=["string", "text_blocks", "mixed", "empty_list", "non_list_non_string"],
    )
    def test_extract_text_only(self, content: list | str | int, expected: str):
        assert TranscriptParser.extract_text_only(content) == expected  # type: ignore[arg-type]

    def test_ansi_stripped_from_extract_text_only(self):
        content = [
            {"type": "text", "text": "\x1b[32mgreen\x1b[0m and \x1b[1;31mred\x1b[0m"}
        ]
        assert TranscriptParser.extract_text_only(content) == "green and red"


class TestAnsiStripping:
    def test_ansi_stripped_from_assistant_text_block(self):
        entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "\x1b[32mhello\x1b[0m world"}]
                },
            }
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].text == "hello world"

    def test_ansi_stripped_from_user_text_block(self):
        entries = [
            {
                "type": "user",
                "message": {
                    "content": [{"type": "text", "text": "\x1b[1;34muser input\x1b[0m"}]
                },
            }
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].text == "user input"


class TestFormatToolUseSummary:
    @pytest.mark.parametrize(
        "name, input_data, expected",
        [
            (
                "Read",
                {"file_path": "src/main.py"},
                "\U0001f4d6 **read**: `src/main.py`",
            ),
            ("Write", {"file_path": "out.txt"}, "\U0001f4dd **write**: `out.txt`"),
            ("Bash", {"command": "ls -la"}, "\U0001f4bb **bash**: `ls -la`"),
            ("Grep", {"pattern": "TODO"}, "\U0001f50e **grep**: `TODO`"),
            ("Glob", {"pattern": "*.py"}, "\U0001f4c2 **glob**: `*.py`"),
            (
                "Task",
                {"description": "analyze code"},
                "\U0001f916 **task**: `analyze code`",
            ),
            (
                "TaskCreate",
                {"subject": "Understand the problem domain"},
                "\U0001f4cb **taskcreate**: `Understand the problem domain`",
            ),
            (
                "TaskUpdate",
                {"subject": "Understand the problem domain", "status": "completed"},
                "\U0001f4cb **taskupdate**: `Understand the problem domain -> completed`",
            ),
            ("TaskList", {}, "\U0001f4cb **tasklist**: `refresh`"),
            (
                "WebFetch",
                {"url": "https://example.com"},
                "\U0001f310 **webfetch**: `https://example.com`",
            ),
            (
                "WebSearch",
                {"query": "python async"},
                "\U0001f50e **websearch**: `python async`",
            ),
            (
                "TodoWrite",
                {"todos": [1, 2, 3]},
                "\U0001f4cb **todowrite**: `3 item(s)`",
            ),
            ("TodoRead", {}, "\U0001f4cb **todoread**"),
            (
                "AskUserQuestion",
                {"questions": [{"question": "Continue?"}]},
                "\u2753 **askuserquestion**: `Continue?`",
            ),
            ("ExitPlanMode", {}, "\U0001f4cb **exitplanmode**"),
            (
                "Skill",
                {"skill": "code-review"},
                "\U0001f4da **skill**: `code-review`",
            ),
            (
                "CustomTool",
                {"first_key": "value1"},
                "\U0001f527 **customtool**: `value1`",
            ),
        ],
        ids=[
            "Read",
            "Write",
            "Bash",
            "Grep",
            "Glob",
            "Task",
            "TaskCreate",
            "TaskUpdate",
            "TaskList",
            "WebFetch",
            "WebSearch",
            "TodoWrite",
            "TodoRead",
            "AskUserQuestion",
            "ExitPlanMode",
            "Skill",
            "unknown_tool",
        ],
    )
    def test_tool_summary(self, name: str, input_data: dict, expected: str):
        assert TranscriptParser.format_tool_use_summary(name, input_data) == expected

    def test_non_dict_input(self):
        assert (
            TranscriptParser.format_tool_use_summary("Read", "not a dict")
            == "\U0001f4d6 **read**"
        )

    def test_truncation_at_50_chars(self):
        long_value = "x" * 100
        result = TranscriptParser.format_tool_use_summary(
            "Bash", {"command": long_value}
        )
        assert len(long_value) > 50
        assert result == f"\U0001f4bb **bash**: `{'x' * 50}\u2026`"


class TestExtractToolResultText:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("raw string", "raw string"),
            (
                [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
                "line1\nline2",
            ),
            (
                [{"type": "text", "text": "keep"}, {"type": "image", "data": "..."}],
                "keep",
            ),
            (None, ""),
        ],
        ids=["string", "text_blocks", "mixed", "none"],
    )
    def test_extract_tool_result_text(self, content: str | list | None, expected: str):
        assert TranscriptParser.extract_tool_result_text(content) == expected


class TestParseMessage:
    def test_user_text(self):
        data = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="user", text="hello")

    def test_assistant_text(self):
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hi there"}]},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="assistant", text="hi there")

    def test_local_command_with_stdout(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<command-name>/help</command-name>"
                            "<local-command-stdout>Available commands</local-command-stdout>"
                        ),
                    }
                ]
            },
        }
        result = TranscriptParser.parse_message(data)
        assert result is not None
        assert result.message_type == "local_command"
        assert result.text == "Available commands"
        assert result.tool_name == "/help"

    def test_local_command_invoke(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "<command-name>/clear</command-name>"}
                ]
            },
        }
        result = TranscriptParser.parse_message(data)
        assert result is not None
        assert result.message_type == "local_command_invoke"
        assert result.text == ""
        assert result.tool_name == "/clear"

    def test_non_user_assistant_returns_none(self):
        data = {
            "type": "summary",
            "message": {"content": "summary text"},
        }
        assert TranscriptParser.parse_message(data) is None

    def test_string_content(self):
        data = {
            "type": "assistant",
            "message": {"content": "plain response"},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="assistant", text="plain response")


class TestFormatEditDiff:
    @pytest.mark.parametrize(
        "old, new, check",
        [
            (
                "hello",
                "world",
                lambda r: "-hello" in r and "+world" in r,
            ),
            (
                "line1\nline2\nline3",
                "line1\nchanged\nline3",
                lambda r: "-line2" in r and "+changed" in r,
            ),
            (
                "same",
                "same",
                lambda r: r == "",
            ),
        ],
        ids=["single_line", "multi_line", "identical"],
    )
    def test_format_edit_diff(self, old: str, new: str, check):
        result = TranscriptParser._format_edit_diff(old, new)
        assert check(result), f"Check failed for ({old!r}, {new!r}): {result!r}"


class TestFormatToolResultText:
    @pytest.mark.parametrize(
        "text, tool_name, check",
        [
            (
                "line1\nline2\nline3",
                "Read",
                lambda r: r == "  \u23bf  3 lines",
            ),
            (
                "line1\nline2",
                "Write",
                lambda r: r == "  \u23bf  2 lines written",
            ),
            (
                "output line",
                "Bash",
                lambda r: (
                    r.startswith("  \u23bf  1 lines")
                    and EXPQUOTE_START in r
                    and EXPQUOTE_END in r
                ),
            ),
            (
                "file1.py\nfile2.py\n",
                "Grep",
                lambda r: "2 matches" in r and EXPQUOTE_START in r,
            ),
            (
                "a.py\nb.py\nc.py",
                "Glob",
                lambda r: "3 files" in r and EXPQUOTE_START in r,
            ),
            (
                "agent says hello",
                "Task",
                lambda r: "1 lines" in r and EXPQUOTE_START in r,
            ),
            (
                "page content here",
                "WebFetch",
                lambda r: (
                    f"{len('page content here')} chars" in r and EXPQUOTE_START in r
                ),
            ),
            (
                "",
                "Read",
                lambda r: r == "",
            ),
        ],
        ids=["Read", "Write", "Bash", "Grep", "Glob", "Task", "WebFetch", "empty"],
    )
    def test_format_tool_result_text(self, text: str, tool_name: str, check):
        result = TranscriptParser._format_tool_result_text(text, tool_name)
        assert check(result), f"Failed check for {tool_name!r}: {result!r}"


class TestParseEntries:
    def test_assistant_text(self, make_jsonl_entry, make_text_block):
        entries = [make_jsonl_entry("assistant", [make_text_block("Hello!")])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].text == "Hello!"
        assert result[0].content_type == "text"

    def test_user_text(self, make_jsonl_entry, make_text_block):
        entries = [make_jsonl_entry("user", [make_text_block("Hi bot")])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].text == "Hi bot"

    def test_tool_use_and_result_pairing(
        self,
        make_jsonl_entry,
        make_text_block,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "app.py"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "file contents line1\nline2\nline3")],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_use_entries = [e for e in result if e.content_type == "tool_use"]
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_use_entries) == 1
        assert tool_use_entries[0].tool_use_id == "t1"
        assert "\U0001f4d6 **read**" in tool_use_entries[0].text
        assert len(tool_result_entries) == 1
        assert tool_result_entries[0].tool_use_id == "t1"
        assert not pending

    def test_thinking_block(self, make_jsonl_entry, make_thinking_block):
        entries = [
            make_jsonl_entry("assistant", [make_thinking_block("reasoning here")])
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "thinking"
        assert EXPQUOTE_START in result[0].text
        assert EXPQUOTE_END in result[0].text
        assert "reasoning here" in result[0].text

    def test_local_command_with_stdout(self, make_jsonl_entry, make_text_block):
        xml = (
            "<command-name>/status</command-name>"
            "<local-command-stdout>all good</local-command-stdout>"
        )
        entries = [make_jsonl_entry("user", [make_text_block(xml)])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "local_command"
        assert "/status" in result[0].text
        assert "all good" in result[0].text

    def test_exit_plan_mode_emits_plan(self, make_jsonl_entry, make_tool_use_block):
        block = make_tool_use_block(
            "t1", "ExitPlanMode", {"plan": "Step 1: do X\nStep 2: do Y"}
        )
        entries = [make_jsonl_entry("assistant", [block])]
        result, pending = TranscriptParser.parse_entries(entries)
        texts = [e for e in result if e.content_type == "text"]
        tool_uses = [e for e in result if e.content_type == "tool_use"]
        assert len(texts) == 1
        assert "Step 1: do X" in texts[0].text
        assert len(tool_uses) >= 1

    def test_edit_tool_diff_stats(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        edit_input = {
            "file_path": "main.py",
            "old_string": "old line",
            "new_string": "new line",
        }
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Edit", edit_input)],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "OK")],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        tr = tool_result_entries[0]
        assert "+1" in tr.text
        assert "\u22121" in tr.text
        assert EXPQUOTE_START in tr.text

    def test_error_tool_result(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Bash", {"command": "rm -rf /"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "Permission denied", is_error=True)],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        assert "\u26a0\ufe0f Permission denied" in tool_result_entries[0].text

    def test_interrupted_tool_result(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "x.py"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", TranscriptParser._INTERRUPTED_TEXT)],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        assert "Interrupted" in tool_result_entries[0].text

    def test_pending_tools_carry_over(self, make_jsonl_entry, make_tool_use_block):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "a.py"})],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries, pending_tools={})
        assert "t1" in pending
        flushed = [
            e for e in result if e.content_type == "tool_use" and e.tool_use_id == "t1"
        ]
        assert len(flushed) == 1

    def test_pending_tools_flushed_without_carry_over(
        self, make_jsonl_entry, make_tool_use_block
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "a.py"})],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries, pending_tools=None)
        tool_entries = [e for e in result if e.tool_use_id == "t1"]
        assert len(tool_entries) == 2
        assert tool_entries[0].content_type == "tool_use"
        assert tool_entries[1].content_type == "tool_use"

    def test_system_tag_filtered(self, make_jsonl_entry, make_text_block):
        entries = [
            make_jsonl_entry(
                "user",
                [
                    make_text_block(
                        "<system-reminder>secret instructions</system-reminder>"
                    )
                ],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        user_entries = [e for e in result if e.role == "user"]
        assert len(user_entries) == 0

    # ── Characterization tests: edge cases and cross-entry state ─────────────

    def test_local_command_invoke_carry_to_next_entry(
        self, make_jsonl_entry, make_text_block
    ):
        # Two-entry test: invoke sets last_cmd_name, stdout entry consumes it.
        invoke_xml = "<command-name>/foo</command-name>"
        stdout_xml = (
            "<command-name>/foo</command-name>"
            "<local-command-stdout>output here</local-command-stdout>"
        )
        entries = [
            make_jsonl_entry("user", [make_text_block(invoke_xml)]),
            make_jsonl_entry("user", [make_text_block(stdout_xml)]),
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        # The invoke entry is consumed (not emitted); only the stdout entry emits.
        assert len(result) == 1
        assert result[0].content_type == "local_command"
        assert "/foo" in result[0].text
        assert "output here" in result[0].text

    def test_local_command_stdout_uses_prior_invoke_name(
        self, make_jsonl_entry, make_text_block
    ):
        # Invoke entry has no stdout; following entry has stdout but tool_name
        # may differ — last_cmd_name from invoke should be carried forward.
        invoke_xml = "<command-name>/bar</command-name>"
        # Stdout entry without repeating command-name: parser falls back to last_cmd_name
        stdout_only_xml = "<local-command-stdout>result</local-command-stdout>"
        entries = [
            make_jsonl_entry("user", [make_text_block(invoke_xml)]),
            make_jsonl_entry("user", [make_text_block(stdout_only_xml)]),
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        cmd_entry = result[0]
        assert cmd_entry.content_type == "local_command"
        # cmd comes from last_cmd_name carry
        assert "/bar" in cmd_entry.text

    def test_flushed_pending_tools_have_null_timestamp_and_tool_name(
        self, make_jsonl_entry, make_tool_use_block
    ):
        # In one-shot mode (pending_tools=None), unmatched tool_use gets flushed
        # at the end with timestamp=None and tool_name=None (per lines 723-731).
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t99", "Bash", {"command": "echo hi"})],
                timestamp="2024-01-01T00:00:00.000Z",
            ),
        ]
        result, _ = TranscriptParser.parse_entries(entries, pending_tools=None)
        tool_entries = [e for e in result if e.tool_use_id == "t99"]
        # First entry: encounter-time (has timestamp and tool_name)
        encounter = tool_entries[0]
        assert encounter.timestamp is not None
        assert encounter.tool_name == "Bash"
        # Second entry: flushed copy (timestamp=None, tool_name=None)
        flushed = tool_entries[1]
        assert flushed.timestamp is None
        assert flushed.tool_name is None

    def test_thinking_empty_with_prior_text_emits_nothing(
        self, make_jsonl_entry, make_text_block, make_thinking_block
    ):
        # Empty thinking block in same entry that already has a text block:
        # has_text=True, so empty thinking is silently skipped.
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_text_block("real text"), make_thinking_block("")],
            )
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        thinking_entries = [e for e in result if e.content_type == "thinking"]
        assert len(thinking_entries) == 0
        text_entries = [e for e in result if e.content_type == "text"]
        assert len(text_entries) == 1
        assert text_entries[0].text == "real text"

    def test_thinking_empty_without_prior_text_emits_placeholder(
        self, make_jsonl_entry, make_thinking_block
    ):
        # Empty thinking with no prior text block in the entry → "(thinking)"
        entries = [make_jsonl_entry("assistant", [make_thinking_block("")])]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "thinking"
        assert result[0].text == "(thinking)"

    def test_tool_result_unknown_id_with_text_emits_formatted_result(
        self, make_jsonl_entry, make_tool_result_block
    ):
        # tool_result with no matching pending tool_use_id and non-empty text
        # → emits formatted result entry (tool_name=None branch, line 684).
        entries = [
            make_jsonl_entry(
                "user",
                [make_tool_result_block("unknown_id", "some output")],
            )
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "tool_result"
        assert result[0].tool_use_id == "unknown_id"

    def test_tool_result_unknown_id_empty_text_emits_nothing(
        self, make_jsonl_entry, make_tool_result_block
    ):
        # No pending match, empty result, not error/interrupted → nothing emitted.
        entries = [
            make_jsonl_entry(
                "user",
                [make_tool_result_block("unknown_id", "")],
            )
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 0

    def test_notebook_edit_no_diff(
        self, make_jsonl_entry, make_tool_use_block, make_tool_result_block
    ):
        # NotebookEdit stores input_data but the diff path only fires for "Edit".
        # NotebookEdit falls through to _format_tool_result_text, which wraps
        # the result text in an expandable quote — no +N/−N diff stats.
        nb_input = {
            "notebook_path": "nb.ipynb",
            "old_string": "old",
            "new_string": "new",
        }
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "NotebookEdit", nb_input)],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "OK")],
            ),
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        tr = next(e for e in result if e.content_type == "tool_result")
        # No diff stats (+N/−N) — NotebookEdit does not run the Edit diff path.
        assert "+1" not in tr.text
        assert "−1" not in tr.text

    def test_no_content_placeholder_skipped(self, make_jsonl_entry, make_text_block):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_text_block(TranscriptParser._NO_CONTENT_PLACEHOLDER)],
            )
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 0

    def test_multiple_text_blocks_produce_multiple_entries(
        self, make_jsonl_entry, make_text_block
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_text_block("first"), make_text_block("second")],
            )
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 2
        assert result[0].text == "first"
        assert result[1].text == "second"

    def test_unknown_entry_type_skipped(self, make_jsonl_entry, make_text_block):
        entries = [
            {
                "type": "summary",
                "message": {"content": [{"type": "text", "text": "x"}]},
            },
            make_jsonl_entry("assistant", [make_text_block("real")]),
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].text == "real"

    def test_all_whitespace_stripped_from_results(
        self, make_jsonl_entry, make_text_block
    ):
        entries = [
            make_jsonl_entry("assistant", [make_text_block("  hello  \n  ")]),
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert result[0].text == "hello"

    def test_carry_over_integrated_sequence(
        self,
        make_jsonl_entry,
        make_text_block,
        make_tool_use_block,
        make_tool_result_block,
    ):
        """Multi-entry two-call sequence exercising carry-over state threading."""
        # First call: assistant text + two tool_uses; user resolves one tool_result
        call1_entries = [
            make_jsonl_entry(
                "assistant",
                [
                    make_text_block("Starting work"),
                    make_tool_use_block("ta", "Read", {"file_path": "a.py"}),
                    make_tool_use_block("tb", "Bash", {"command": "ls"}),
                ],
                timestamp="2024-01-01T00:00:00.000Z",
            ),
            make_jsonl_entry(
                "user",
                [
                    make_tool_result_block("ta", "file contents"),
                    make_text_block("user follow-up"),
                ],
                timestamp="2024-01-01T00:00:01.000Z",
            ),
        ]
        result1, pending1 = TranscriptParser.parse_entries(
            call1_entries, pending_tools={}
        )

        # tb is unresolved → should still be pending
        assert "tb" in pending1
        # ta resolved → should NOT be pending
        assert "ta" not in pending1

        roles_types1 = [(e.role, e.content_type) for e in result1]
        assert ("assistant", "text") in roles_types1
        assert ("assistant", "tool_use") in roles_types1
        assert ("assistant", "tool_result") in roles_types1
        assert ("user", "text") in roles_types1

        # Second call: carries pending tb, resolves it
        call2_entries = [
            make_jsonl_entry(
                "user",
                [make_tool_result_block("tb", "ls output\nfile.py\nfoo.py")],
                timestamp="2024-01-01T00:00:02.000Z",
            ),
        ]
        result2, pending2 = TranscriptParser.parse_entries(
            call2_entries, pending_tools=pending1
        )
        assert not pending2
        tr = next(e for e in result2 if e.content_type == "tool_result")
        assert tr.tool_use_id == "tb"
