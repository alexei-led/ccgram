"""Tests for one-tap selections in interactive terminal prompts."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_ASK_CHOICE
from ccgram.handlers.interactive.interactive_callbacks import (
    handle_interactive_callback,
    parse_direct_choice_callback,
)
from ccgram.handlers.interactive.interactive_ui import (
    _build_interactive_keyboard,
    _interactive_contexts,
    _interactive_contents,
    _interactive_mode,
    _interactive_msgs,
    _interactive_sequences,
    handle_interactive_ui,
    parse_direct_choices,
)


def _button_data(keyboard) -> list[str]:
    return [
        str(button.callback_data)
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    ]


class TestParseDirectChoices:
    def test_parses_bounded_numbered_single_select_options(self) -> None:
        choices = parse_direct_choices(
            "Pick a deployment:\n  ❯ 1. Staging\n    2. Production\n  Enter to select"
        )

        assert choices == (("1", "1. Staging"), ("2", "2. Production"))

    def test_parses_yes_no_options_on_a_single_line(self) -> None:
        assert parse_direct_choices("Would you like to proceed?\n  Yes     No\n") == (
            ("y", "Yes"),
            ("n", "No"),
        )

    @pytest.mark.parametrize(
        "content",
        [
            "Choose any:\n  ☐ 1. Alpha\n  ☐ 2. Beta\n  Enter to select",
            "Notes:\n  1. This is prose\n  2. This is also prose",
            "Pick one:\n" + "\n".join(f"  {i}. Option {i}" for i in range(1, 10)),
        ],
    )
    def test_falls_back_for_multi_select_prose_and_unbounded_lists(
        self, content: str
    ) -> None:
        assert parse_direct_choices(content) == ()


class TestDirectChoiceKeyboard:
    def test_prepends_direct_choice_buttons_before_navigation(self) -> None:
        keyboard = _build_interactive_keyboard(
            "@12",
            direct_choices=(("1", "1. Staging"), ("2", "2. Production")),
            sequence=7,
        )

        first_row = keyboard.inline_keyboard[0]
        assert [button.text for button in first_row] == ["1. Staging", "2. Production"]
        assert _button_data(keyboard)[0] == f"{CB_ASK_CHOICE}1:7:@12"

    async def test_rendered_prompt_wires_parsed_choices_to_the_current_sequence(
        self,
    ) -> None:
        client = AsyncMock()
        client.send_message.return_value = MagicMock(message_id=99)
        with (
            patch(
                "ccgram.handlers.interactive.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=(
                    "AskUserQuestion",
                    "Pick one:\n  1. Alpha\n  2. Beta\n  Enter to select",
                ),
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.thread_router.resolve_chat_id",
                return_value=-100,
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.rate_limit_send",
                new_callable=AsyncMock,
            ),
        ):
            assert await handle_interactive_ui(client, 10, "@12", thread_id=42)

        keyboard = client.send_message.call_args.kwargs["reply_markup"]
        assert _button_data(keyboard)[0] == f"{CB_ASK_CHOICE}1:1:@12"
        assert _interactive_contexts[(10, 42)] == (-100, 99)


def _callback_update(*, chat_id: int, thread_id: int | None, message_id: int):
    message = MagicMock()
    message.chat.id = chat_id
    message.message_id = message_id
    message.message_thread_id = thread_id
    query = AsyncMock()
    query.message = message
    update = MagicMock()
    update.message = None
    update.callback_query = query
    return query, update


@pytest.fixture(autouse=True)
def _clear_interactive_state():
    for state in (
        _interactive_contexts,
        _interactive_contents,
        _interactive_mode,
        _interactive_msgs,
        _interactive_sequences,
    ):
        state.clear()
    yield
    for state in (
        _interactive_contexts,
        _interactive_contents,
        _interactive_mode,
        _interactive_msgs,
        _interactive_sequences,
    ):
        state.clear()


class TestDirectChoiceCallbacks:
    def test_parses_choice_sequence_and_opaque_pane_target(self) -> None:
        assert parse_direct_choice_callback(f"{CB_ASK_CHOICE}2:7:w2:t1|w2:p1") == (
            "2",
            7,
            "w2:t1",
            "w2:p1",
        )

    async def test_sends_direct_choice_only_for_current_chat_thread_and_sequence(
        self,
    ) -> None:
        _interactive_mode[(10, 42)] = "@12"
        _interactive_msgs[(10, 42)] = 99
        _interactive_contexts[(10, 42)] = (-100, 99)
        _interactive_sequences[(10, 42)] = 7
        query, update = _callback_update(chat_id=-100, thread_id=42, message_id=99)

        multiplexer = MagicMock()
        multiplexer.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id="@12")
        )
        multiplexer.send_keys = AsyncMock(return_value=True)
        with (
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.interactive.interactive_callbacks.tmux_manager",
                multiplexer,
            ),
            patch(
                "ccgram.handlers.interactive.interactive_callbacks.PTBTelegramClient"
            ),
        ):
            await handle_interactive_callback(
                query,
                10,
                f"{CB_ASK_CHOICE}1:7:@12",
                update,
                MagicMock(),
            )

        multiplexer.send_keys.assert_awaited_once_with(
            "@12", "1", enter=False, literal=True
        )
        assert _interactive_sequences[(10, 42)] == 8

    @pytest.mark.parametrize(
        ("chat_id", "thread_id", "message_id", "sequence"),
        [(-101, 42, 99, 7), (-100, 43, 99, 7), (-100, 42, 99, 6)],
        ids=["wrong-chat", "wrong-thread", "stale-sequence"],
    )
    async def test_rejects_direct_choice_without_current_prompt_ownership(
        self, chat_id: int, thread_id: int, message_id: int, sequence: int
    ) -> None:
        _interactive_mode[(10, 42)] = "@12"
        _interactive_msgs[(10, 42)] = 99
        _interactive_contexts[(10, 42)] = (-100, 99)
        _interactive_sequences[(10, 42)] = 7
        query, update = _callback_update(
            chat_id=chat_id, thread_id=thread_id, message_id=message_id
        )

        multiplexer = MagicMock()
        multiplexer.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id="@12")
        )
        multiplexer.send_keys = AsyncMock()
        with (
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.interactive.interactive_callbacks.tmux_manager",
                multiplexer,
            ),
            patch(
                "ccgram.handlers.interactive.interactive_callbacks.PTBTelegramClient"
            ),
        ):
            await handle_interactive_callback(
                query,
                10,
                f"{CB_ASK_CHOICE}1:{sequence}:@12",
                update,
                MagicMock(),
            )

        multiplexer.send_keys.assert_not_awaited()
        query.answer.assert_awaited_once_with(
            "This prompt has expired", show_alert=True
        )
