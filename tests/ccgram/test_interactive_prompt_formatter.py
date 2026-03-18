"""Tests for interactive prompt text formatting."""

import pytest

from ccgram.interactive_prompt_formatter import format_codex_interactive_prompt


def _make_edit_prompt(diff_lines: list[str]) -> str:
    """Build a minimal edit prompt with given diff lines."""
    return (
        "Do you want to make this edit to foo.py?\n"
        + "\n".join(diff_lines)
        + "\n"
        + "\u203a 1. Yes, proceed (y)\n"
        + "Press enter to confirm\n"
    )


class TestFormatCodexInteractivePrompt:
    def test_edit_prompt_compacts_diff_and_splits_options(self) -> None:
        raw = (
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

        result = format_codex_interactive_prompt(raw, "SelectionUI")

        assert "Do you want to make this edit to src/ccgram/bot.py?" in result
        assert "File: src/ccgram/bot.py" in result
        assert "Changes: +" in result
        assert "Preview:" in result
        assert "› 1. Yes, proceed (y)" in result
        assert "  2. Yes, and don't ask again for these files (a)" in result
        assert "  3. No, and tell Codex what to do differently (esc)" in result
        assert "Press enter to confirm or esc to cancel" in result

    def test_non_edit_prompt_only_normalizes_options(self) -> None:
        raw = "Which option should I use?\n› 1. A  2. B  3. C\nEsc to cancel\n"

        result = format_codex_interactive_prompt(raw, "SelectionUI")

        assert "Which option should I use?" in result
        assert "Changes:" not in result
        assert "Preview:" not in result
        assert "› 1. A" in result
        assert "  2. B" in result
        assert "  3. C" in result
        assert "Esc to cancel" in result

    def test_idempotent_for_already_formatted_edit_prompt(self) -> None:
        raw = (
            "Do you want to make this edit to src/ccgram/bot.py?\n"
            "File: src/ccgram/bot.py\n"
            "Changes: +2 -2\n"
            "Preview:\n"
            "  - old line\n"
            "  + new line\n"
            "\n"
            "› 1. Yes, proceed (y)\n"
            "  2. Yes, and don't ask again for these files (a)\n"
            "  3. No, and tell Codex what to do differently (esc)\n"
            "Press enter to confirm or esc to cancel\n"
        )

        once = format_codex_interactive_prompt(raw, "SelectionUI")
        twice = format_codex_interactive_prompt(once, "SelectionUI")

        assert once == twice


class TestExtractPreviewsHeadTail:
    def test_short_diff_shows_all_lines(self) -> None:
        raw = _make_edit_prompt(["+ added line 1", "- removed line 1"])
        result = format_codex_interactive_prompt(raw, "SelectionUI")
        assert "Preview:" in result
        assert "+ added line 1" in result
        assert "- removed line 1" in result
        assert "more lines" not in result

    def test_long_diff_shows_head_tail_with_omitted(self) -> None:
        raw = _make_edit_prompt([f"+ added line {i}" for i in range(8)])
        result = format_codex_interactive_prompt(raw, "SelectionUI")
        assert "Preview:" in result
        assert "+ added line 0" in result
        assert "+ added line 1" in result
        assert "more lines" in result
        assert "+ added line 6" in result
        assert "+ added line 7" in result
        assert "+ added line 3" not in result

    @pytest.mark.parametrize(
        "line_count, expect_truncation",
        [
            pytest.param(4, False, id="4_lines_no_truncation"),
            pytest.param(5, True, id="5_lines_triggers_truncation"),
        ],
    )
    def test_truncation_boundary(
        self, line_count: int, expect_truncation: bool
    ) -> None:
        diff_lines = [f"+ line {i}" for i in range(line_count)]
        raw = _make_edit_prompt(diff_lines)
        result = format_codex_interactive_prompt(raw, "SelectionUI")

        if expect_truncation:
            omitted = line_count - 4
            assert f"{omitted} more lines" in result
            assert "+ line 0" in result
            assert "+ line 1" in result
            assert f"+ line {line_count - 2}" in result
            assert f"+ line {line_count - 1}" in result
            assert "+ line 2" not in result
        else:
            assert "more lines" not in result
            for i in range(line_count):
                assert f"+ line {i}" in result

    def test_mixed_add_remove_in_head_tail(self) -> None:
        diff_lines = (
            ["- removed 0", "+ added 0"]
            + [f"+ added {i}" for i in range(1, 5)]
            + ["- removed 5", "+ added 5"]
        )
        raw = _make_edit_prompt(diff_lines)
        result = format_codex_interactive_prompt(raw, "SelectionUI")
        assert "Preview:" in result
        assert "more lines" in result
        assert "- removed 0" in result
        assert "+ added 0" in result
        assert "- removed 5" in result
        assert "+ added 5" in result


class TestFormatExpandableQuote:
    def test_wraps_text_with_sentinel_markers(self) -> None:
        from ccgram.providers.base import (
            EXPANDABLE_QUOTE_END,
            EXPANDABLE_QUOTE_START,
            format_expandable_quote,
        )

        text = "some tool output\nwith multiple lines"
        result = format_expandable_quote(text)
        assert result.startswith(EXPANDABLE_QUOTE_START)
        assert result.endswith(EXPANDABLE_QUOTE_END)
        assert text in result
