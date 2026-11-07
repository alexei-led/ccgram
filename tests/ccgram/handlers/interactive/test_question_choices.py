"""Claude AskUserQuestion prompts: one-tap options and free-text answers.

Screens are trimmed captures of Claude Code 2.1 question prompts.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.interactive.interactive_ui import (
    _interactive_contents,
    _interactive_key,
    parse_direct_choices,
    question_choice_keys,
    typed_answer_keys,
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


class TestQuestionKeys:
    """Claude's question prompt ignores digits: picks move the cursor instead."""

    @pytest.fixture
    def stored(self):
        key = _interactive_key(1, 42, -100)
        yield lambda content: _interactive_contents.__setitem__(key, content)
        _interactive_contents.pop(key, None)

    def test_pick_moves_down_then_enters(self, stored) -> None:
        stored(SINGLE)  # cursor on 1
        assert question_choice_keys(1, 42, -100, "3") == ["Down", "Down", "Enter"]

    def test_pick_moves_up(self, stored) -> None:
        stored(MULTI)  # cursor on 2
        assert question_choice_keys(1, 42, -100, "1") == ["Up", "Enter"]

    def test_pick_under_the_cursor_just_enters(self, stored) -> None:
        stored(SINGLE)
        assert question_choice_keys(1, 42, -100, "1") == ["Enter"]

    def test_free_text_option_only_gets_the_cursor(self, stored) -> None:
        stored(SINGLE)
        assert question_choice_keys(1, 42, -100, "4") == ["Down", "Down", "Down"]

    def test_other_prompts_keep_number_keys(self, stored) -> None:
        stored("Do you want to proceed?\n❯ 1. Yes\n  2. No\nEsc to cancel")
        assert question_choice_keys(1, 42, -100, "2") is None

    def test_free_text_focus_keys(self, stored) -> None:
        stored(SINGLE)
        assert typed_answer_keys(1, 42, -100) == ["Down", "Down", "Down"]
        stored(
            SINGLE.replace("❯ 1. Red", "  1. Red").replace(
                "  4. Type something.", "❯ 4. Type something."
            )
        )
        assert typed_answer_keys(1, 42, -100) == []

    def test_no_free_text_option(self, stored) -> None:
        stored("☐ Ok\n❯ 1. Yes\n  2. No\nEnter to select · ↑/↓ to navigate")
        assert typed_answer_keys(1, 42, -100) is None


class TestEditingAndNext:
    """Changing a typed answer, and leaving a multi-select from its text field."""

    ANSWERED = MULTI.replace("  3. [ ] Type something", "❯ 3. [✔] C++").replace(
        "❯ 2. [ ] Banana", "  2. [ ] Banana"
    )

    @pytest.fixture
    def stored(self):
        key = _interactive_key(1, 42, -100)
        yield lambda content: _interactive_contents.__setitem__(key, content)
        _interactive_contents.pop(key, None)

    def test_multi_select_takes_no_typed_answer(self, stored) -> None:
        stored(self.ANSWERED)
        assert typed_answer_keys(1, 42, -100) is None

    def test_a_single_select_answer_is_replaced(self, stored) -> None:
        stored(
            SINGLE.replace("❯ 1. Red", "  1. Red").replace(
                "  4. Type something.", "❯ 4. Mauve"
            )
        )
        assert typed_answer_keys(1, 42, -100) == [
            "End",
            *["BSpace"] * 5,
        ]

    def test_an_answered_field_is_picked_like_any_option(self, stored) -> None:
        stored(self.ANSWERED.replace("❯ 3.", "  3.").replace("  1. [✔]", "❯ 1. [✔]"))
        assert question_choice_keys(1, 42, -100, "3") == ["Down", "Down", "Enter"]

    def test_next_steps_out_of_the_text_field_first(self, stored) -> None:
        stored(self.ANSWERED)
        assert question_choice_keys(1, 42, -100, "x") == ["Up", "Right"]

    def test_next_from_an_ordinary_option(self, stored) -> None:
        stored(MULTI)
        assert question_choice_keys(1, 42, -100, "x") == ["Right"]

    def test_hint_line_marks_a_question_without_its_footer(self, stored) -> None:
        from ccgram.handlers.interactive.interactive_ui import (
            QUESTION_INSTRUCTION_LINE,
        )

        body = "\n".join(
            line for line in SINGLE_QUESTION.splitlines() if "to navigate" not in line
        )
        stored(f"{QUESTION_INSTRUCTION_LINE}\n{body}")
        assert question_choice_keys(1, 42, -100, "1") == ["Up", "Enter"]


class TestFreeTextAnswer:
    async def test_without_a_free_text_option_falls_back(self) -> None:
        with patch("ccgram.handlers.interactive.typed_answer_keys", return_value=None):
            handled = await text_handler._answer_free_text_option(
                1, 42, "hc-t5", -100, "hi", MagicMock(), MagicMock(), MagicMock()
            )
        assert handled is False

    async def test_answers_are_typed_one_at_a_time(self) -> None:
        order: list[str] = []

        async def typing(lock, *_args, text="", **_kw):
            async with lock:
                order.append(f"start {_args[4]}")
                await asyncio.sleep(0.01)
                order.append(f"end {_args[4]}")

        with (
            patch("ccgram.handlers.interactive.typed_answer_keys", return_value=[]),
            patch.object(text_handler, "_type_free_text_answer", typing),
        ):
            for answer in ("C++", "Kotlin"):
                assert await text_handler._answer_free_text_option(
                    1, 42, "hc-t5", -100, answer, MagicMock(), MagicMock(), MagicMock()
                )
            await asyncio.sleep(0.05)
        assert order == ["start C++", "end C++", "start Kotlin", "end Kotlin"]
        text_handler._typed_answer_locks.pop((1, 42), None)

    async def _send(self, moves: list[str]) -> MagicMock:
        adapter = MagicMock()
        adapter.send_text = AsyncMock(return_value=(True, ""))
        adapter.send_keys = AsyncMock(return_value=(True, ""))
        with patch.object(text_handler, "_TYPED_ANSWER_FOCUS_S", 0):
            assert await text_handler._send_free_text(
                1,
                42,
                "hc-t5",
                -100,
                "my own answer",
                MagicMock(message_id=9),
                adapter,
                moves,
                None,
                lambda *a, **k: 7,
            )
        return adapter

    async def test_moves_to_the_option_then_types_and_submits(self) -> None:
        adapter = await self._send(["Down", "Down"])
        assert adapter.send_keys.await_args.args[1] == ["Down", "Down"]
        assert adapter.send_text.await_args.args[1] == "my own answer"
        assert adapter.send_text.await_args.kwargs["submit"] is True

    async def test_skips_the_move_when_already_focused(self) -> None:
        adapter = await self._send([])
        adapter.send_keys.assert_not_awaited()
        adapter.send_text.assert_awaited_once()

    async def test_long_clears_are_split_into_actions(self) -> None:
        adapter = await self._send(["End", *["BSpace"] * 40])
        assert [len(c.args[1]) for c in adapter.send_keys.await_args_list] == [32, 9]


class TestSettleFreeTextAnswer:
    async def _settle(self, screens: list) -> int:
        from ccgram.handlers.interactive import interactive_ui as ui

        captures = AsyncMock(side_effect=screens + [None])
        with (
            patch.object(ui, "_capture_interactive_content", captures),
            patch.object(ui, "handle_interactive_ui", new_callable=AsyncMock),
        ):
            await ui.settle_free_text_answer(
                MagicMock(),
                1,
                "hc-t5",
                42,
                "Kotlin",
                chat_id=-100,
                timeout=1,
                interval=0.001,
            )
        return captures.await_count

    @staticmethod
    def _screen(field: str) -> tuple:
        return (
            "AskUserQuestion",
            MULTI.replace("[ ] Type something", field),
            False,
            None,
        )

    async def test_waits_until_the_answer_shows_checked(self) -> None:
        count = await self._settle(
            [
                self._screen("[ ] Type something"),
                self._screen("[ ] Kotlin"),
                self._screen("[✔] Kotlin"),
                self._screen("[✔] Kotlin"),
            ]
        )
        assert count == 3

    async def test_single_select_lands_when_the_question_moves_on(self) -> None:
        moved_on = ("AskUserQuestion", SINGLE, False, None)
        count = await self._settle([self._screen("[ ] Kotlin"), moved_on, moved_on])
        assert count == 2


class TestQuestionChoiceCallback:
    async def _tap(
        self, keys: list[str], result: tuple[bool, str] = (True, "")
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        from ccgram.handlers.interactive import interactive_callbacks as cb

        query = MagicMock()
        query.answer = AsyncMock()
        adapter = MagicMock()
        adapter.send_keys = AsyncMock(return_value=result)
        with (
            patch.object(cb, "advance_interactive_sequence") as advance,
            patch.object(cb, "handle_interactive_ui", new_callable=AsyncMock) as ui,
            patch.object(cb, "follow_prompt_changes", new_callable=AsyncMock) as follow,
        ):
            await cb._send_question_choice(
                query, 1, 42, -100, "hc-t5", None, keys, adapter, "a", "fp", MagicMock()
            )
            await asyncio.sleep(0)
        self.redrawn = ui.await_count
        self.followed = follow.await_count
        return query, adapter, advance

    async def test_tap_sends_cursor_keys_and_advances(self) -> None:
        _query, adapter, advance = await self._tap(["Down", "Enter"])
        assert adapter.send_keys.await_args.args[1] == ["Down", "Enter"]
        advance.assert_called_once()
        assert self.followed == 1

    async def test_tap_on_the_focused_free_text_option_changes_nothing(self) -> None:
        query, adapter, advance = await self._tap([])
        adapter.send_keys.assert_not_awaited()
        advance.assert_not_called()
        query.answer.assert_awaited_once()

    async def test_a_prompt_changed_at_the_computer_is_redrawn(self) -> None:
        from ccgram.session_protocol.models import PROMPT_CHANGED_MESSAGE

        query, _adapter, _advance = await self._tap(
            ["Down", "Enter"], (False, PROMPT_CHANGED_MESSAGE)
        )
        assert self.redrawn == 1
        assert "changed on the computer" in query.answer.await_args.args[0]


class TestQuestionKeyboard:
    @staticmethod
    def _labels(content: str, *, question: bool = True) -> list[str]:
        from ccgram.handlers.interactive.interactive_ui import (
            _build_interactive_keyboard,
        )

        choices = parse_direct_choices(content, question=True)
        keyboard = _build_interactive_keyboard(
            "hc-t5", "AskUserQuestion", direct_choices=choices, question=question
        )
        return [b.text for row in keyboard.inline_keyboard for b in row]

    def test_single_select_has_only_its_options(self) -> None:
        assert self._labels(SINGLE) == [
            "1. Red",
            "2. Blue",
            "3. Green",
            "4. ✏️ Type something.",
            "5. 💬 Chat about this",
        ]

    def test_multi_select_adds_a_next_button(self) -> None:
        labels = self._labels(MULTI)
        assert labels[-1] == "➡ Next"
        assert not any("Type something" in label for label in labels)
        assert "⏎ Enter" not in labels and "␣ Space" not in labels

    def test_other_prompts_keep_the_keypad(self) -> None:
        assert "⏎ Enter" in self._labels(SINGLE, question=False)


class TestFollowPromptChanges:
    async def _follow(self, captures: list) -> AsyncMock:
        from ccgram.handlers.interactive import interactive_ui as ui

        key = ui._interactive_key(1, 42, -100)
        ui._interactive_mode[key] = "hc-t5"
        ui._interactive_prompt_fingerprints[key] = "old"
        redraw = AsyncMock(
            side_effect=lambda *a, **k: ui._interactive_prompt_fingerprints.__setitem__(
                key, __import__("hashlib").sha256(b"next").hexdigest()
            )
        )
        try:
            with (
                patch.object(
                    ui, "_capture_interactive_content", AsyncMock(side_effect=captures)
                ),
                patch.object(ui, "handle_interactive_ui", redraw),
            ):
                await ui.follow_prompt_changes(
                    MagicMock(),
                    1,
                    "hc-t5",
                    42,
                    chat_id=-100,
                    timeout=0.05,
                    interval=0.01,
                )
        finally:
            ui._interactive_mode.pop(key, None)
            ui._interactive_prompt_fingerprints.pop(key, None)
        return redraw

    async def test_redraws_once_per_new_state(self) -> None:
        state = ("AskUserQuestion", "next", False, None)
        redraw = await self._follow([state] * 20)
        assert redraw.await_count == 1

    async def test_stops_when_the_prompt_is_gone(self) -> None:
        redraw = await self._follow(
            [None] + [("AskUserQuestion", "x", False, None)] * 20
        )
        redraw.assert_not_awaited()


class TestOneFollowerPerTopic:
    async def test_a_new_tap_cancels_the_previous_follower(self) -> None:
        from ccgram.handlers.interactive import interactive_callbacks as cb

        started: list[asyncio.Event] = []

        async def follow(*_a, **_k):
            started.append(asyncio.Event())
            await asyncio.sleep(10)

        adapter = MagicMock()
        adapter.send_keys = AsyncMock(return_value=(True, ""))
        with (
            patch.object(cb, "advance_interactive_sequence"),
            patch.object(cb, "follow_prompt_changes", follow),
        ):
            for _ in range(2):
                query = MagicMock()
                query.answer = AsyncMock()
                await cb._send_question_choice(
                    query,
                    1,
                    42,
                    -100,
                    "hc-t5",
                    None,
                    ["Enter"],
                    adapter,
                    "a",
                    "fp",
                    MagicMock(),
                )
                await asyncio.sleep(0)
            tasks = list(cb._follow_tasks.values())
            assert len(tasks) == 1
            tasks[0].cancel()
