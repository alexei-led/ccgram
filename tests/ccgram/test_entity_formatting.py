import pytest
from telegram import MessageEntity
from telegramify_markdown import utf16_len as _utf16_len

from ccgram.entity_formatting import (
    _EXPQUOTE_MAX_RENDERED,
    _MIN_PARTIAL_LINE_LEN,
    _strip_indented_code_blocks,
    _truncate_quote_text,
    convert_to_entities,
)
from ccgram.expandable_quote import EXPANDABLE_QUOTE_END as EXP_END
from ccgram.expandable_quote import EXPANDABLE_QUOTE_START as EXP_START
from ccgram.expandable_quote import _EXPANDABLE_QUOTE_MAX_CHARS, format_expandable_quote


def _extract_utf16(text: str, offset: int, length: int) -> str:
    utf16 = text.encode("utf-16-le")
    return utf16[offset * 2 : (offset + length) * 2].decode("utf-16-le")


class TestConvertToEntities:
    def test_plain_text(self) -> None:
        text, entities = convert_to_entities("hello world")
        assert "hello world" in text
        assert isinstance(entities, list)

    def test_bold(self) -> None:
        text, entities = convert_to_entities("**bold text**")
        assert "bold text" in text
        bold_entities = [e for e in entities if e.type == MessageEntity.BOLD]
        assert len(bold_entities) == 1
        assert (
            text[
                bold_entities[0].offset : bold_entities[0].offset
                + bold_entities[0].length
            ]
            == "bold text"
        )

    def test_italic(self) -> None:
        text, entities = convert_to_entities("*italic text*")
        assert "italic text" in text
        italic_entities = [e for e in entities if e.type == MessageEntity.ITALIC]
        assert len(italic_entities) == 1

    def test_inline_code(self) -> None:
        text, entities = convert_to_entities("`inline code`")
        assert "inline code" in text
        code_entities = [e for e in entities if e.type == MessageEntity.CODE]
        assert len(code_entities) == 1

    def test_code_block_preserved(self) -> None:
        text, entities = convert_to_entities("```python\nprint('hi')\n```")
        assert "print" in text
        pre_entities = [e for e in entities if e.type == MessageEntity.PRE]
        assert len(pre_entities) == 1

    def test_code_block_with_language(self) -> None:
        text, entities = convert_to_entities("```python\nx = 1\n```")
        pre_entities = [e for e in entities if e.type == MessageEntity.PRE]
        assert len(pre_entities) == 1
        assert pre_entities[0].language == "python"

    def test_expandable_quote_sentinels(self) -> None:
        text, entities = convert_to_entities(f"{EXP_START}quoted content{EXP_END}")
        assert EXP_START not in text
        assert EXP_END not in text
        assert "quoted content" in text
        exp_entities = [
            e for e in entities if e.type == MessageEntity.EXPANDABLE_BLOCKQUOTE
        ]
        assert len(exp_entities) == 1

    def test_mixed_text_and_expandable_quote(self) -> None:
        text, entities = convert_to_entities(
            f"before {EXP_START}inside quote{EXP_END} after"
        )
        assert EXP_START not in text
        assert EXP_END not in text
        assert "inside quote" in text
        assert "before" in text
        assert "after" in text
        exp_entities = [
            e for e in entities if e.type == MessageEntity.EXPANDABLE_BLOCKQUOTE
        ]
        assert len(exp_entities) == 1

    def test_indented_text_not_treated_as_code(self) -> None:
        text, entities = convert_to_entities(
            "Some text:\n\n    indented line\n\nMore text"
        )
        assert "indented line" in text
        pre_entities = [e for e in entities if e.type == MessageEntity.PRE]
        assert len(pre_entities) == 0

    def test_fenced_code_block_indentation_preserved(self) -> None:
        md = "```python\ndef foo():\n    x = 1\n\n    y = 2\n    return x + y\n```"
        text, entities = convert_to_entities(md)
        assert "    x = 1" in text
        assert "    y = 2" in text
        assert "    return x + y" in text

    def test_emoji_in_text(self) -> None:
        text, entities = convert_to_entities("Hello 🌍 **world**")
        assert text == "Hello 🌍 world"
        bold_entities = [e for e in entities if e.type == MessageEntity.BOLD]
        assert len(bold_entities) == 1
        ent = bold_entities[0]
        utf16 = text.encode("utf-16-le")
        extracted = utf16[ent.offset * 2 : (ent.offset + ent.length) * 2].decode(
            "utf-16-le"
        )
        assert extracted == "world"

    def test_special_chars_no_parse_error(self) -> None:
        text, entities = convert_to_entities(
            "_var_name_ and `file-path.txt` and #heading"
        )
        assert "var_name" in text
        assert "file-path.txt" in text
        assert isinstance(entities, list)

    def test_link(self) -> None:
        text, entities = convert_to_entities("[click here](https://example.com)")
        assert "click here" in text
        link_entities = [e for e in entities if e.type == MessageEntity.TEXT_LINK]
        assert len(link_entities) == 1
        assert link_entities[0].url == "https://example.com"

    def test_empty_text(self) -> None:
        text, entities = convert_to_entities("")
        assert text == ""
        assert entities == []

    def test_multiple_expandable_quotes(self) -> None:
        md = f"text1\n{EXP_START}quote1{EXP_END}\nmiddle\n{EXP_START}quote2{EXP_END}\nend"
        text, entities = convert_to_entities(md)
        exp_entities = [
            e for e in entities if e.type == MessageEntity.EXPANDABLE_BLOCKQUOTE
        ]
        assert len(exp_entities) == 2
        assert "quote1" in text
        assert "quote2" in text

    def test_expandable_quote_with_formatting_around(self) -> None:
        md = f"**bold** {EXP_START}quoted{EXP_END} *italic*"
        text, entities = convert_to_entities(md)
        bold_entities = [e for e in entities if e.type == MessageEntity.BOLD]
        italic_entities = [e for e in entities if e.type == MessageEntity.ITALIC]
        exp_entities = [
            e for e in entities if e.type == MessageEntity.EXPANDABLE_BLOCKQUOTE
        ]
        assert len(bold_entities) >= 1
        assert len(italic_entities) >= 1
        assert len(exp_entities) == 1

    def test_returns_telegram_message_entity_types(self) -> None:
        text, entities = convert_to_entities("**bold** `code`")
        for ent in entities:
            assert isinstance(ent, MessageEntity)

    def test_bold_offset_after_expandable_quote(self) -> None:
        md = f"{EXP_START}quoted content{EXP_END}**bold**"
        text, entities = convert_to_entities(md)
        bold = [e for e in entities if e.type == MessageEntity.BOLD]
        assert len(bold) == 1
        assert _extract_utf16(text, bold[0].offset, bold[0].length) == "bold"

    def test_emoji_before_expandable_quote_offsets(self) -> None:
        md = f"Hello 🌍 {EXP_START}quoted{EXP_END}**bold**"
        text, entities = convert_to_entities(md)
        exp = [e for e in entities if e.type == MessageEntity.EXPANDABLE_BLOCKQUOTE]
        bold = [e for e in entities if e.type == MessageEntity.BOLD]
        assert len(exp) == 1
        assert _extract_utf16(text, exp[0].offset, exp[0].length) == "quoted"
        assert len(bold) == 1
        assert _extract_utf16(text, bold[0].offset, bold[0].length) == "bold"


class TestTruncateQuoteText:
    def test_short_text_not_truncated(self) -> None:
        text, truncated = _truncate_quote_text("short text")
        assert text == "short text"
        assert not truncated

    def test_long_text_truncated(self) -> None:
        long_text = "line\n" * 2000
        text, truncated = _truncate_quote_text(long_text)
        assert truncated
        assert _utf16_len(text) <= _EXPQUOTE_MAX_RENDERED + 50
        assert "truncated" in text

    def test_exactly_at_budget_not_truncated(self) -> None:
        text = "a" * _EXPQUOTE_MAX_RENDERED
        result, truncated = _truncate_quote_text(text)
        assert result == text
        assert not truncated

    def test_partial_line_skipped_when_too_short(self) -> None:
        short_line = "x" * (_MIN_PARTIAL_LINE_LEN - 1)
        lines = ["a" * 100] * 37 + ["b" * 44, short_line]
        long_text = "\n".join(lines)
        result, truncated = _truncate_quote_text(long_text)
        assert truncated
        assert short_line not in result


class TestStripIndentedCodeBlocks:
    @pytest.mark.parametrize(
        ("text", "present", "absent"),
        [
            pytest.param(
                "hello\n\n    indented\n    block\n\nend",
                ["indented\nblock"],
                ["    indented"],
                id="strips_block_after_blank_line",
            ),
            pytest.param(
                "    indented start\n\nrest",
                ["indented start"],
                ["    indented"],
                id="strips_block_at_start",
            ),
            pytest.param(
                "text\n\n```python\ndef foo():\n    x = 1\n\n    y = 2\n```\n\nend",
                ["    x = 1", "    y = 2"],
                [],
                id="preserves_backtick_fence_indentation",
            ),
            pytest.param(
                "```\n    code\n```\n\ntext",
                ["    code"],
                [],
                id="preserves_fence_at_start",
            ),
            pytest.param(
                "```\n    keep\n```\n\n    strip this\n\nend",
                ["    keep", "strip this"],
                ["    strip"],
                id="mixed_fenced_and_indented",
            ),
            pytest.param(
                "~~~\n    keep\n~~~\n\n    strip this\n\nend",
                ["    keep", "strip this"],
                ["    strip"],
                id="tilde_fence_behaves_like_backtick",
            ),
            pytest.param(
                "before\n\n```python\n    indented code\n    more",
                ["    indented code"],
                [],
                id="unclosed_fence_kept_verbatim",
            ),
            pytest.param(
                "`````\n    keep\n```\n\n    also keep\n`````",
                ["    keep", "    also keep"],
                [],
                id="shorter_fence_does_not_close_longer_one",
            ),
        ],
    )
    def test_strips_only_unfenced_indentation(
        self, text: str, present: list[str], absent: list[str]
    ) -> None:
        result = _strip_indented_code_blocks(text)
        for fragment in present:
            assert fragment in result
        for fragment in absent:
            assert fragment not in result

    def test_no_indentation_passthrough(self) -> None:
        text = "plain text\nno indentation"
        assert _strip_indented_code_blocks(text) == text


class TestExpandableQuoteTruncation:
    def test_short_text_passes_through(self):
        text = "short content"
        result = format_expandable_quote(text)
        assert result == f"{EXP_START}{text}{EXP_END}"

    def test_long_text_truncated(self):
        text = "x" * (_EXPANDABLE_QUOTE_MAX_CHARS + 500)
        result = format_expandable_quote(text)
        assert EXP_START in result
        assert EXP_END in result
        inner = result.removeprefix(EXP_START).removesuffix(EXP_END)
        assert "truncated" in inner
        assert str(len(text)) in inner

    def test_exact_limit_not_truncated(self):
        text = "x" * _EXPANDABLE_QUOTE_MAX_CHARS
        result = format_expandable_quote(text)
        assert "truncated" not in result
