"""Claude AskUserQuestion prompts: one-tap options and free-text answers.

Screens are trimmed captures of Claude Code 2.1 question prompts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.interactive.interactive_ui import (
    _interactive_contents,
    _interactive_key,
    parse_direct_choices,
    typed_answer_option,
)
from ccgram.handlers.text import text_handler

RULE = "─" * 40

SINGLE = f"""←  ☒ Fruit  ☐ Colour  ✔ Submit  →

Pick a colour.

❯ 1. Red
     warm
  2. Blue
     calm
  3. Green
     natural
  4. Type something.
{RULE}
  5. Chat about this

Enter to select · Tab/Arrow keys to navigate · Esc to cancel"""

MULTI = f"""←  ☐ Fruit  ☐ Colour  ✔ Submit  →

Pick fruit (several allowed).

  1. [✔] Apple
         red or green
❯ 2. [ ] Banana
         yellow
  3. [ ] Type something
     Next
{RULE}
  4. Chat about this

Enter to select · Tab/Arrow keys to navigate · Esc to cancel"""

SINGLE_QUESTION = f"""☐ Number

Pick a number.

  1. 1
     one
❯ 2. 2
     two
  3. Type something.
{RULE}
  4. Chat about this

Enter to select · ↑/↓ to navigate · Esc to cancel"""


class TestQuestionChoices:
    def test_single_select_offers_every_option(self) -> None:
        assert parse_direct_choices(SINGLE, question=True) == (
            ("1", "1. Red"),
            ("2", "2. Blue"),
            ("3", "3. Green"),
            ("4", "4. Type something."),
            ("5", "5. Chat about this"),
        )

    def test_multi_select_shows_checkbox_state(self) -> None:
        assert parse_direct_choices(MULTI, question=True) == (
            ("1", "1. ☑ Apple"),
            ("2", "2. ☐ Banana"),
            ("3", "3. ☐ Type something"),
            ("4", "4. Chat about this"),
        )

    def test_single_question_header_is_not_a_checkbox(self) -> None:
        keys = [key for key, _ in parse_direct_choices(SINGLE_QUESTION, question=True)]
        assert keys == ["1", "2", "3", "4"]

    def test_other_prompts_keep_the_strict_rules(self) -> None:
        assert parse_direct_choices(MULTI) == ()


class TestTypedAnswerOption:
    @pytest.fixture
    def stored(self):
        key = _interactive_key(1, 42, -100)
        yield lambda content: _interactive_contents.__setitem__(key, content)
        _interactive_contents.pop(key, None)

    def test_finds_the_free_text_option(self, stored) -> None:
        stored(SINGLE)
        assert typed_answer_option(1, 42, -100) == ("4", False)

    def test_knows_when_the_cursor_is_on_it(self, stored) -> None:
        stored(
            SINGLE.replace("❯ 1. Red", "  1. Red").replace(
                "  4. Type something.", "❯ 4. Type something."
            )
        )
        assert typed_answer_option(1, 42, -100) == ("4", True)

    def test_none_without_a_free_text_option(self, stored) -> None:
        stored("Allow?\n❯ 1. Yes\n  2. No\nEnter to select · ↑/↓ to navigate")
        assert typed_answer_option(1, 42, -100) is None


class TestFreeTextAnswer:
    async def _answer(self, option: tuple[str, bool] | None) -> tuple[bool, list]:
        adapter = MagicMock()
        adapter.send_text = AsyncMock(return_value=(True, ""))
        message = MagicMock(message_id=9)
        with (
            patch(
                "ccgram.handlers.interactive.typed_answer_option",
                return_value=option,
            ),
            patch(
                "ccgram.handlers.interactive.get_interactive_pane", return_value=None
            ),
            patch("ccgram.handlers.interactive.get_interactive_msg_id", return_value=7),
            patch.object(text_handler, "_TYPED_ANSWER_FOCUS_S", 0),
        ):
            handled = await text_handler._answer_free_text_option(
                1, 42, "hc-t5", -100, "my own answer", message, adapter
            )
        return handled, adapter.send_text.await_args_list

    async def test_moves_to_the_option_then_types_and_submits(self) -> None:
        handled, calls = await self._answer(("4", False))
        assert handled is True
        assert [(c.args[1], c.kwargs["submit"]) for c in calls] == [
            ("4", False),
            ("my own answer", True),
        ]

    async def test_skips_the_move_when_already_focused(self) -> None:
        _handled, calls = await self._answer(("4", True))
        assert [c.args[1] for c in calls] == ["my own answer"]

    async def test_falls_back_without_a_free_text_option(self) -> None:
        handled, calls = await self._answer(None)
        assert handled is False
        assert calls == []
