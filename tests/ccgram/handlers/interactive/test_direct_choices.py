"""Tests for one-tap selections in interactive terminal prompts."""

from hashlib import sha256
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
    _interactive_prompt_fingerprints,
    _interactive_sequences,
    _next_interactive_sequence,
    handle_interactive_ui,
    is_current_interactive_prompt,
    parse_direct_choices,
)
from ccgram.providers.claude import ClaudeProvider
from ccgram.providers.codex import CodexProvider


def _button_data(keyboard) -> list[str]:
    return [
        str(button.callback_data)
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    ]


class TestParseDirectChoices:
    def test_claude_header_checkbox_is_not_multi_select(self) -> None:
        # Current Claude Code renders the AskUserQuestion header as
        # "☐ <header>" even for single-select: the checkbox is decorative
        # and must not trip the multi-select guard.
        capture = (
            " ☐ Next step\n"
            "│ What would you like to do next?\n"
            "❯ 1. Just testing the flow\n"
            "     Confirm this prompt rendered correctly and stop here.\n"
            "  2. Build a ccgram repro\n"
            "     Set up an actual reproduction for the bridge.\n"
            "  3. Ask another question\n"
            "     Fire another AskUserQuestion to exercise the surface.\n"
            "  4. Type something.\n"
            "Enter to select · ↑/↓ to navigate · Esc to cancel"
        )
        status = ClaudeProvider().parse_terminal_status(capture)
        assert status is not None and status.is_interactive
        choices = parse_direct_choices(status.raw_text)
        assert choices == (
            ("1", "1. Just testing the flow"),
            ("2", "2. Build a ccgram repro"),
            ("3", "3. Ask another question"),
            ("4", "4. Type something."),
        )

    def test_checkbox_after_the_label_rejects_multi_select(self) -> None:
        assert (
            parse_direct_choices(
                "Pick your fixes:\n  1. Alpha ☐\n  2. Beta ☐\nEnter to select"
            )
            == ()
        )

    def test_described_plan_prose_above_an_editor_hint_is_not_a_menu(self) -> None:
        # Plan bodies and markdown lists carry numbered steps with indented
        # continuations; a vim or pager hint below them is not a selection
        # footer, and tapping a step would type a digit into a prompt.
        assert (
            parse_direct_choices(
                "  1. Refactor the parser\n"
                "     extract a shared helper\n"
                "  2. Add regression tests\n"
                "     cover the case\n"
                "  3. Update docs\n"
                "ctrl-g to edit in vim"
            )
            == ()
        )

    def test_confirm_footer_anchors_cursorless_menu(self) -> None:
        # This cursorless input exercises the confirm-footer path; it does not
        # model Codex cursor rendering or normalization.
        assert parse_direct_choices(
            "1. Allow once\n"
            "2. Allow for this session\n"
            "3. Deny\n"
            "Press enter to confirm or esc to cancel"
        ) == (
            ("1", "1. Allow once"),
            ("2", "2. Allow for this session"),
            ("3", "3. Deny"),
        )

    def test_codex_inline_cursor_options_survive_provider_normalization(self) -> None:
        # This is the Codex edit-approval capture also used by
        # test_jsonl_providers.py; normalization must keep its real cursor.
        pane = (
            "Do you want to make this edit to src/ccgram/bot.py?\n"
            "947    936 -    await register_commands(application.bot, provider=get_provider())"
            "    948 +    await register_commands(application.bot, providers=_menu_providers())\n"
            "953          try:\n"
            "942 -            await register_commands(context.bot, provider=get_provider())"
            "    954 +    await register_commands(context.bot, providers=_menu_providers())\n"
            "› 1. Yes, proceed (y)  2. Yes, and don't ask again for these files (a)"
            "  3. No, and tell Codex what to do differently (esc)\n"
            "Press enter to confirm or esc to cancel\n"
        )
        status = CodexProvider().parse_terminal_status(pane)
        assert status is not None and status.is_interactive
        assert "› 1. Yes, proceed (y)" in status.raw_text
        assert parse_direct_choices(status.raw_text) == (
            ("1", "1. Yes, proceed (y)"),
            ("2", "2. Yes, and don't ask again for these files (a)"),
            ("3", "3. No, and tell Codex what to do differently (esc)"),
        )

    def test_codex_normalized_indented_options_are_not_descriptions(self) -> None:
        # The provider normalizes inline options to an unindented first item
        # followed by two-space-indented items; that spacing is not prose.
        pane = (
            "› Choose an action\n"
            "1. Allow once  2. Allow for this session  3. Deny\n"
            "Press enter to confirm\n"
        )
        status = CodexProvider().parse_terminal_status(pane)
        assert status is not None and status.is_interactive
        assert status.raw_text.splitlines()[1:4] == [
            "1. Allow once",
            "  2. Allow for this session",
            "  3. Deny",
        ]
        assert parse_direct_choices(status.raw_text) == (
            ("1", "1. Allow once"),
            ("2", "2. Allow for this session"),
            ("3", "3. Deny"),
        )

    def test_arrow_or_quoted_prose_line_is_not_a_cursor_anchor(self) -> None:
        # A blockquote marker or an arrow starting a prose line must not
        # adopt the numbered list below it; only a cursor on an option
        # line anchors a menu.
        assert (
            parse_direct_choices(
                "→ see the migration guide for details\n"
                "  1. Refactor the parser\n"
                "     extract a shared helper\n"
                "  2. Add regression tests"
            )
            == ()
        )

    def test_described_prose_above_enter_to_continue_is_not_a_menu(self) -> None:
        assert (
            parse_direct_choices(
                "Summary:\n"
                "  1. Fix the parser\n"
                "     it drops lines\n"
                "  2. Add tests\n"
                "Enter to continue"
            )
            == ()
        )

    def test_pi_midline_footer_does_not_anchor_numbered_prose(self) -> None:
        # Pi's actual Enter-select footer is mid-line; its provider also
        # requires a selected-option cursor, which is the real menu anchor.
        assert (
            parse_direct_choices(
                "1. Review the parser\n"
                "2. Add regression tests\n"
                "↑↓ navigate • Enter select • Esc cancel • number quick-select"
            )
            == ()
        )

    def test_trailing_checkbox_options_reject_multi_select(self) -> None:
        # A multi-select can render the checkbox after the number; those
        # option lines still match the numbered-option regex, so only the
        # multi-select guard stands between them and false buttons.
        assert (
            parse_direct_choices(
                "Pick several:\n  1. ☐ Alpha\n  2. ☐ Beta\nEnter to select"
            )
            == ()
        )

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(
                "Pick several:\n❯ 1. [ ] Alpha\n  2. [x] Beta\nEnter to select",
                id="request-checkbox-markers",
            ),
            pytest.param(
                "Pick several:\n❯ 1. ✔ Alpha\n  2. ☒ Beta\nEnter to select",
                id="terminal-checkbox-markers",
            ),
        ],
    )
    def test_known_checkbox_option_markers_reject_direct_choices(
        self, content: str
    ) -> None:
        # These token positions use markers already recognized by the request
        # guard or terminal parser; this does not assert a Claude rendering.
        assert parse_direct_choices(content) == ()

    def test_unnumbered_checkbox_options_reject_numbered_prose(self) -> None:
        # Claude's provider fixture uses checkbox-prefixed option rows. Keep
        # nearby numbered prose from becoming buttons for that prompt.
        assert (
            parse_direct_choices(
                "Which targets?\n"
                "1. Review staging\n"
                "2. Review production\n"
                "☐ First option\n"
                "☐ Second option\n"
                "Enter to select"
            )
            == ()
        )

    @pytest.mark.parametrize("selected", [1, 2])
    @pytest.mark.parametrize("indent", [1, 2, 4])
    def test_numbered_prose_never_labels_an_indented_cursor_menu(
        self, selected: int, indent: int
    ) -> None:
        menu = "\n".join(
            f"{' ' * indent}{'❯ ' if number == selected else '  '}{number}. {label}"
            for number, label in ((1, "Yes"), (2, "No"))
        )
        choices = parse_direct_choices(
            f"Steps:\n1. Read file\n2. Edit file\n{menu}\nEnter to select"
        )
        live_choices = (("1", "1. Yes"), ("2", "2. No"))
        assert choices in ((), live_choices)
        if selected == 1:
            assert choices == live_choices

    @pytest.mark.parametrize("selected", [1, 2, 3])
    @pytest.mark.parametrize(
        "cursor,footer",
        [
            ("→", "↑↓ navigate • Enter select • Esc cancel • number quick-select"),
            ("❯", "Enter to select · ↑/↓ to navigate · Esc to cancel"),
        ],
    )
    def test_cursor_selected_description_lists_never_become_choices(
        self, selected: int, cursor: str, footer: str
    ) -> None:
        rows = ["Which target?"]
        for number, label in ((1, "Alpha"), (2, "Beta"), (3, "Gamma")):
            marker = cursor + " " if selected == number else "  "
            rows.append(f"{marker}{number}. {label}")
            if selected == number:
                rows.extend(
                    ("      1. Step one", "      2. Step two", "      3. Step three")
                )
        rows.extend(("", footer))
        assert parse_direct_choices("\n".join(rows)) == (
            ("1", "1. Alpha"),
            ("2", "2. Beta"),
            ("3", "3. Gamma"),
        )

    def test_overlapping_blocks_sharing_a_cursor_keep_navigation(self) -> None:
        assert (
            parse_direct_choices(
                "  1. Alpha\n  2. Beta\n      1. Step one\n      2. Step two\n"
                "→ 3. Gamma\n↑↓ navigate • Enter select • Esc cancel"
            )
            == ()
        )

    def test_overlapping_cursorless_blocks_keep_navigation(self) -> None:
        assert (
            parse_direct_choices(
                "1. Alpha\n   Description\n   1. Step one\n   2. Step two\n"
                "2. Beta\nEnter to select"
            )
            == ()
        )

    def test_checkmark_in_header_does_not_mark_numbered_options(self) -> None:
        assert parse_direct_choices(
            "✔ Recommended\n❯ 1. Alpha\n  2. Beta\nEnter to select"
        ) == (("1", "1. Alpha"), ("2", "2. Beta"))

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

    def test_selects_menu_block_nearest_action_footer(self) -> None:
        choices = parse_direct_choices(
            "Previous output:\n"
            "  1. First prose item\n"
            "  2. Second prose item\n"
            "Pick a deployment:\n"
            "  1. Staging\n"
            "  2. Production\n"
            "  Enter to select"
        )

        assert choices == (("1", "1. Staging"), ("2", "2. Production"))

    def test_rejects_ambiguous_numbered_blocks_around_action_footer(self) -> None:
        assert (
            parse_direct_choices(
                "  1. Earlier\n  2. Menu\n  Enter to select\n  1. Later\n  2. Menu"
            )
            == ()
        )

    def test_rejects_textual_multi_select_prompt(self) -> None:
        assert (
            parse_direct_choices(
                "Choose all applicable environments:\n"
                "  ❯ 1. Staging\n"
                "    2. Production\n"
                "  Enter to select"
            )
            == ()
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
                    False,
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

        prompt = "Pick one:\n  1. Alpha\n  2. Beta\n  Enter to select"
        assert (
            _interactive_prompt_fingerprints[(10, -100, 42)]
            == sha256(prompt.encode()).hexdigest()
        )

        keyboard = client.send_message.call_args.kwargs["reply_markup"]
        assert _button_data(keyboard)[0] == f"{CB_ASK_CHOICE}1:1:@12"
        assert _interactive_contexts[(10, -100, 42)] == (-100, 99)


def _callback_update(*, chat_id: int, thread_id: int | None, message_id: int):
    message = MagicMock()
    message.chat.id = chat_id
    message.chat.type = "supergroup"
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
        _interactive_prompt_fingerprints,
        _interactive_sequences,
    ):
        state.clear()
    yield
    for state in (
        _interactive_contexts,
        _interactive_contents,
        _interactive_mode,
        _interactive_msgs,
        _interactive_prompt_fingerprints,
        _interactive_sequences,
    ):
        state.clear()


class TestDirectChoiceCallbacks:
    async def test_private_topic_callback_uses_chat_scoped_ownership(self) -> None:
        _interactive_mode[(10, 100, 42)] = "@12"
        _interactive_msgs[(10, 100, 42)] = 99
        _interactive_contexts[(10, 100, 42)] = (100, 99)
        _interactive_sequences[(10, 100, 42)] = 7
        query, update = _callback_update(chat_id=100, thread_id=42, message_id=99)
        query.message.chat.type = "private"
        query.message.is_topic_message = True

        with patch(
            "ccgram.handlers.callback_helpers.user_owns_window",
            return_value=False,
        ) as owns:
            await handle_interactive_callback(
                query,
                10,
                f"{CB_ASK_CHOICE}1:7:@12",
                update,
                MagicMock(),
            )

        owns.assert_called_once_with(10, "@12", 100)

    def test_parses_choice_sequence_and_opaque_pane_target(self) -> None:
        assert parse_direct_choice_callback(f"{CB_ASK_CHOICE}2:7:w2:t1|w2:p1") == (
            "2",
            7,
            "w2:t1",
            "w2:p1",
        )

    @pytest.mark.parametrize(
        ("choice_key", "expected_submit"),
        [("1", False), ("y", True), ("n", True)],
        ids=["number-does-not-enter", "yes-enters", "no-enters"],
    )
    async def test_sends_direct_choice_only_for_current_chat_thread_and_sequence(
        self, choice_key: str, expected_submit: bool
    ) -> None:
        _interactive_mode[(10, -100, 42)] = "@12"
        _interactive_msgs[(10, -100, 42)] = 99
        _interactive_contexts[(10, -100, 42)] = (-100, 99)
        _interactive_sequences[(10, -100, 42)] = 7
        _interactive_prompt_fingerprints[(10, -100, 42)] = "prompt-fingerprint"
        query, update = _callback_update(chat_id=-100, thread_id=42, message_id=99)

        adapter = MagicMock(send_text=AsyncMock(return_value=(True, "Sent")))
        with (
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window",
                return_value=True,
            ),
            patch("ccgram.session_protocol.telegram.adapter_for", return_value=adapter),
            patch(
                "ccgram.handlers.interactive.interactive_callbacks.PTBTelegramClient"
            ),
        ):
            await handle_interactive_callback(
                query,
                10,
                f"{CB_ASK_CHOICE}{choice_key}:7:@12",
                update,
                MagicMock(),
            )

        adapter.send_text.assert_awaited_once()
        assert adapter.send_text.await_args.args == ("@12", choice_key)
        assert adapter.send_text.await_args.kwargs["terminal_input"] is True
        assert adapter.send_text.await_args.kwargs["submit"] is expected_submit
        assert (
            adapter.send_text.await_args.kwargs["expected_prompt"]
            == "prompt-fingerprint"
        )
        assert _interactive_sequences[(10, -100, 42)] == 8

    @pytest.mark.parametrize(
        ("chat_id", "thread_id", "message_id", "sequence"),
        [(-101, 42, 99, 7), (-100, 43, 99, 7), (-100, 42, 99, 6)],
        ids=["wrong-chat", "wrong-thread", "stale-sequence"],
    )
    async def test_rejects_direct_choice_without_current_prompt_ownership(
        self, chat_id: int, thread_id: int, message_id: int, sequence: int
    ) -> None:
        _interactive_mode[(10, -100, 42)] = "@12"
        _interactive_msgs[(10, -100, 42)] = 99
        _interactive_contexts[(10, -100, 42)] = (-100, 99)
        _interactive_sequences[(10, -100, 42)] = 7
        query, update = _callback_update(
            chat_id=chat_id, thread_id=thread_id, message_id=message_id
        )

        adapter = MagicMock(send_text=AsyncMock())
        with (
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window",
                return_value=True,
            ),
            patch("ccgram.session_protocol.telegram.adapter_for", return_value=adapter),
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

        adapter.send_text.assert_not_awaited()
        query.answer.assert_awaited_once_with(
            "This prompt has expired", show_alert=True
        )

    async def test_keeps_choice_current_when_key_delivery_fails(self) -> None:
        _interactive_mode[(10, -100, 42)] = "@12"
        _interactive_msgs[(10, -100, 42)] = 99
        _interactive_contexts[(10, -100, 42)] = (-100, 99)
        _interactive_sequences[(10, -100, 42)] = 7
        query, update = _callback_update(chat_id=-100, thread_id=42, message_id=99)

        adapter = MagicMock(send_text=AsyncMock(return_value=(False, "busy")))
        with (
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window",
                return_value=True,
            ),
            patch("ccgram.session_protocol.telegram.adapter_for", return_value=adapter),
        ):
            await handle_interactive_callback(
                query,
                10,
                f"{CB_ASK_CHOICE}1:7:@12",
                update,
                MagicMock(),
            )

        assert _interactive_sequences[(10, -100, 42)] == 7
        adapter.send_text.assert_awaited_once()
        query.answer.assert_awaited_once_with(
            "Unable to send choice. Try again.", show_alert=True
        )

    def test_interactive_state_is_scoped_to_chat_as_well_as_thread(self) -> None:
        _interactive_mode[(10, -100, 42)] = "@12"
        _interactive_msgs[(10, -100, 42)] = 99
        _interactive_contexts[(10, -100, 42)] = (-100, 99)
        _interactive_sequences[(10, -100, 42)] = 7
        _interactive_mode[(10, -101, 42)] = "@13"
        _interactive_msgs[(10, -101, 42)] = 99
        _interactive_contexts[(10, -101, 42)] = (-101, 99)
        _interactive_sequences[(10, -101, 42)] = 7

        assert is_current_interactive_prompt(10, 42, "@12", -100, 99, 7)
        assert is_current_interactive_prompt(10, 42, "@13", -101, 99, 7)
        assert not is_current_interactive_prompt(10, 42, "@12", -101, 99, 7)

    def test_raw_prompt_change_advances_sequence_when_rendered_text_is_equal(self):
        key = (10, -100, 42)
        assert _next_interactive_sequence(key, "same rendered text", "first") == 1
        assert _next_interactive_sequence(key, "same rendered text", "second") == 2
