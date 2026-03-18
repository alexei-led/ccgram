"""Tests for Markdown → Telegram MarkdownV2 conversion."""

import pytest

from ccgram.markdown_v2 import (
    _escape_mdv2,
    _strip_indented_code_blocks,
    convert_markdown,
)
from ccgram.providers.base import EXPANDABLE_QUOTE_END as EXP_END
from ccgram.providers.base import EXPANDABLE_QUOTE_START as EXP_START


class TestEscapeMdv2:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            (
                "_*[]()~>#+\\-=|{}.!",
                "\\_\\*\\[\\]\\(\\)\\~\\>\\#\\+\\\\\\-\\=\\|\\{\\}\\.\\!",
            ),
            ("hello world 123", "hello world 123"),
            ("", ""),
        ],
        ids=["special-chars", "alphanumeric-unchanged", "empty-string"],
    )
    def test_escape(self, input_text: str, expected: str) -> None:
        assert _escape_mdv2(input_text) == expected


class TestConvertMarkdown:
    def test_plain_text(self) -> None:
        result = convert_markdown("hello world")
        assert "hello world" in result

    def test_bold(self) -> None:
        result = convert_markdown("**bold text**")
        assert "*bold text*" in result
        assert "**bold text**" not in result

    def test_code_block_preserved(self) -> None:
        result = convert_markdown("```python\nprint('hi')\n```")
        assert "```" in result
        assert "print" in result

    def test_expandable_quote_sentinels(self) -> None:
        text = f"{EXP_START}quoted content{EXP_END}"
        result = convert_markdown(text)
        assert EXP_START not in result
        assert EXP_END not in result
        assert ">quoted content||" in result

    def test_mixed_text_and_expandable_quote(self) -> None:
        text = f"before {EXP_START}inside quote{EXP_END} after"
        result = convert_markdown(text)
        assert EXP_START not in result
        assert EXP_END not in result
        assert ">inside quote||" in result
        assert "before" in result
        assert "after" in result

    def test_indented_text_not_treated_as_code(self) -> None:
        result = convert_markdown("Some text:\n\n    indented line\n\nMore text")
        assert "```" not in result
        assert "indented line" in result

    def test_fenced_code_block_indentation_preserved(self) -> None:
        text = "```python\ndef foo():\n    x = 1\n\n    y = 2\n    return x + y\n```"
        result = convert_markdown(text)
        assert "    x = 1" in result
        assert "    y = 2" in result
        assert "    return x + y" in result


class TestStripIndentedCodeBlocks:
    def test_strips_indented_block_after_blank_line(self) -> None:
        text = "hello\n\n    indented\n    block\n\nend"
        result = _strip_indented_code_blocks(text)
        assert "    indented" not in result
        assert "indented\nblock" in result

    def test_strips_indented_block_at_start(self) -> None:
        text = "    indented start\n\nrest"
        result = _strip_indented_code_blocks(text)
        assert "    indented" not in result
        assert "indented start" in result

    def test_preserves_fenced_block_indentation(self) -> None:
        text = "text\n\n```python\ndef foo():\n    x = 1\n\n    y = 2\n```\n\nend"
        result = _strip_indented_code_blocks(text)
        assert "    x = 1" in result
        assert "    y = 2" in result

    def test_preserves_fenced_block_at_start(self) -> None:
        text = "```\n    code\n```\n\ntext"
        result = _strip_indented_code_blocks(text)
        assert "    code" in result

    def test_mixed_fenced_and_indented(self) -> None:
        text = "```\n    keep\n```\n\n    strip this\n\nend"
        result = _strip_indented_code_blocks(text)
        assert "    keep" in result
        assert "    strip" not in result
        assert "strip this" in result

    def test_no_indentation_passthrough(self) -> None:
        text = "plain text\nno indentation"
        assert _strip_indented_code_blocks(text) == text

    def test_unclosed_fence_kept_verbatim(self) -> None:
        text = "before\n\n```python\n    indented code\n    more"
        result = _strip_indented_code_blocks(text)
        assert "    indented code" in result

    def test_nested_fence_longer_opening(self) -> None:
        text = "`````\n    keep\n```\n\n    also keep\n`````"
        result = _strip_indented_code_blocks(text)
        assert "    keep" in result
        assert "    also keep" in result

    def test_tilde_fence_preserved(self) -> None:
        text = "~~~\n    keep\n~~~\n\n    strip this\n\nend"
        result = _strip_indented_code_blocks(text)
        assert "    keep" in result
        assert "    strip" not in result
        assert "strip this" in result
