"""Tests for LLM completion summarizer."""

import json
from unittest.mock import AsyncMock, patch

from ccgram.llm.summarizer import (
    _build_summary_context,
    _extract_tool_summary,
    summarize_completion,
)


def _make_assistant_tool_use(name: str, input_data: dict) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": name, "input": input_data}
                ]
            },
        }
    )


def _make_assistant_text(text: str) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text}]},
        }
    )


def _make_user_tool_result(content: str) -> str:
    return json.dumps(
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": content}
                ]
            },
        }
    )


class TestExtractToolSummary:
    def test_bash_command(self):
        block = {"name": "Bash", "input": {"command": "make test"}}
        assert _extract_tool_summary(block) == "Bash: make test"

    def test_read_file(self):
        block = {"name": "Read", "input": {"file_path": "/src/foo.py"}}
        assert _extract_tool_summary(block) == "Read /src/foo.py"

    def test_edit_file(self):
        block = {"name": "Edit", "input": {"file_path": "/src/bar.py"}}
        assert _extract_tool_summary(block) == "Edit /src/bar.py"

    def test_grep_pattern(self):
        block = {"name": "Grep", "input": {"pattern": "TODO"}}
        assert _extract_tool_summary(block) == "Grep TODO"

    def test_unknown_tool(self):
        block = {"name": "CustomTool", "input": {"key": "val"}}
        assert _extract_tool_summary(block) == "CustomTool"

    def test_bash_long_command_truncated(self):
        block = {"name": "Bash", "input": {"command": "x" * 200}}
        result = _extract_tool_summary(block)
        assert result is not None
        assert len(result) <= 106


class TestBuildSummaryContext:
    def test_tool_use_and_result(self):
        lines = [
            _make_assistant_tool_use("Bash", {"command": "make test"}),
            _make_user_tool_result("23 passed, 0 failed"),
        ]
        context = _build_summary_context(lines)
        assert "\u2192 Bash: make test" in context
        assert "= 23 passed, 0 failed" in context

    def test_assistant_text_captured(self):
        lines = [
            _make_assistant_text("All tests pass. The fix is complete."),
        ]
        context = _build_summary_context(lines)
        assert "Final response:" in context
        assert "All tests pass" in context

    def test_last_assistant_text_wins(self):
        lines = [
            _make_assistant_text("First message"),
            _make_assistant_text("Second message"),
        ]
        context = _build_summary_context(lines)
        assert "Second message" in context
        assert "First message" not in context

    def test_empty_lines_skipped(self):
        lines = ["", "  ", "\n"]
        context = _build_summary_context(lines)
        assert context == ""

    def test_invalid_json_skipped(self):
        lines = ["not json at all", _make_assistant_text("valid")]
        context = _build_summary_context(lines)
        assert "valid" in context

    def test_summary_type_skipped(self):
        lines = [json.dumps({"type": "summary", "summary": "ignored"})]
        context = _build_summary_context(lines)
        assert context == ""

    def test_list_type_tool_result_content(self):
        line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "23 passed, 0 failed"},
                                {"type": "text", "text": "all tests ok"},
                            ],
                        }
                    ]
                },
            }
        )
        context = _build_summary_context([line])
        assert "23 passed" in context


class TestSummarizeCompletion:
    async def test_returns_none_when_path_empty(self):
        result = await summarize_completion("")
        assert result is None

    async def test_returns_none_when_file_missing(self, tmp_path):
        result = await summarize_completion(str(tmp_path / "nonexistent.jsonl"))
        assert result is None

    async def test_returns_none_when_no_llm(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text(_make_assistant_text("hello"))
        with patch("ccgram.llm.get_text_completer", return_value=None):
            result = await summarize_completion(str(f))
        assert result is None

    async def test_calls_completer_with_context(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text(
            _make_assistant_tool_use("Bash", {"command": "make test"})
            + "\n"
            + _make_user_tool_result("5 passed")
            + "\n"
            + _make_assistant_text("All tests pass.")
        )
        mock_completer = AsyncMock()
        mock_completer.complete.return_value = "Ran tests, all 5 pass"
        with patch(
            "ccgram.llm.get_text_completer",
            return_value=mock_completer,
        ):
            result = await summarize_completion(str(f))

        assert result == "Ran tests, all 5 pass"
        mock_completer.complete.assert_called_once()
        call_args = mock_completer.complete.call_args
        assert "Bash: make test" in call_args[0][1]
        assert "5 passed" in call_args[0][1]

    async def test_returns_none_on_llm_error(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text(_make_assistant_text("hello"))
        mock_completer = AsyncMock()
        mock_completer.complete.side_effect = RuntimeError("API error")
        with patch(
            "ccgram.llm.get_text_completer",
            return_value=mock_completer,
        ):
            result = await summarize_completion(str(f))
        assert result is None

    async def test_truncates_long_summary(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text(_make_assistant_text("hello"))
        mock_completer = AsyncMock()
        mock_completer.complete.return_value = "x" * 200
        with patch(
            "ccgram.llm.get_text_completer",
            return_value=mock_completer,
        ):
            result = await summarize_completion(str(f))
        assert result is not None
        assert len(result) <= 150

    async def test_returns_none_when_context_empty(self, tmp_path):
        f = tmp_path / "transcript.jsonl"
        f.write_text(json.dumps({"type": "summary", "summary": "ignored"}))
        mock_completer = AsyncMock()
        with patch(
            "ccgram.llm.get_text_completer",
            return_value=mock_completer,
        ):
            result = await summarize_completion(str(f))
        assert result is None
        mock_completer.complete.assert_not_called()
