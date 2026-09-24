"""Pi's cc-thingz question menu must reach Telegram as live choices."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.interactive.interactive_ui import (
    INTERACTIVE_TOOL_NAMES,
    handle_interactive_ui,
    parse_direct_choices,
)
from ccgram.providers.pi import PiProvider

MENU = """────────────────────────────────────────────────────────────────
 Release scope

 Which repositories should I update?
 Preserve existing notes that already fit.

→ 1. Both routers
    Use one shared release format.
  2. pi-model-router only
    Update this repository's releases.
  3. Other / type something

 ↑↓ navigate • Enter select • Esc cancel • number quick-select
────────────────────────────────────────────────────────────────
"""
CHOICES = (
    ("1", "1. Both routers"),
    ("2", "2. pi-model-router only"),
    ("3", "3. Other / type something"),
)


def test_pi_question_tool_activates_existing_interactive_route() -> None:
    provider = PiProvider()
    entry = provider.parse_transcript_line(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "question-1",
                            "name": "ask_user_question",
                            "arguments": {
                                "questions": [
                                    {
                                        "question": "Which repositories?",
                                        "options": [
                                            {"label": "Both routers"},
                                            {"label": "pi-model-router only"},
                                        ],
                                    }
                                ]
                            },
                        }
                    ],
                },
            }
        )
    )
    assert entry is not None
    messages, pending = provider.parse_transcript_entries([entry], {})
    assert messages[0].tool_name == "AskUserQuestion"
    assert messages[0].tool_name in INTERACTIVE_TOOL_NAMES
    assert messages[0].content_type == "tool_use"
    assert "question-1" in pending


@pytest.mark.parametrize("selected", [1, 2, 3])
def test_pi_menu_with_descriptions_produces_all_choices(selected: int) -> None:
    screen = MENU.replace("→ 1.", "  1.").replace(f"  {selected}.", f"→ {selected}.")
    status = PiProvider().parse_terminal_status(
        "Old output:\n  1. old\n  2. prose\n" + screen
    )
    assert status is not None and status.is_interactive
    assert status.ui_type == "AskUserQuestion"
    assert "Which repositories" in status.raw_text
    assert "shared release format" in status.raw_text
    assert "old" not in status.raw_text
    assert parse_direct_choices(status.raw_text) == CHOICES


@pytest.mark.parametrize(
    "screen",
    [
        "1. Prose\n2. Not a question",
        MENU.replace("→", " "),
        MENU.replace("number quick-select", "Space toggle"),
        "Question editor\nEnter comma-separated option numbers or labels.",
        "This was an earlier prompt:\n" + MENU.splitlines()[0] + "\nWorking now",
    ],
)
def test_non_question_screens_do_not_offer_direct_choices(screen: str) -> None:
    assert PiProvider().parse_terminal_status(screen) is None


def test_ansi_pi_menu_and_numbered_description_are_safe() -> None:
    screen = MENU.replace(
        "Use one shared release format.", "2. This is a description, not an option."
    )
    status = PiProvider().parse_terminal_status("\x1b[32m" + screen + "\x1b[0m")
    assert status is not None
    assert parse_direct_choices(status.raw_text) == CHOICES


def test_described_numbered_prose_does_not_become_choices_for_other_providers() -> None:
    assert (
        parse_direct_choices(
            "Pick one:\n→ 1. Alpha\n    Description\n  2. Beta\nEnter to select"
        )
        == ()
    )


async def test_real_pi_provider_renders_telegram_choice_buttons() -> None:
    client = AsyncMock()
    client.send_message.return_value = MagicMock(message_id=7731)
    mux = AsyncMock()
    mux.find_window_by_id.return_value = MagicMock(window_id="pi-question")
    mux.capture_pane.return_value = MENU
    with (
        patch("ccgram.handlers.interactive.interactive_ui.tmux_manager", mux),
        patch(
            "ccgram.handlers.interactive.interactive_ui.get_window_provider",
            return_value="pi",
        ),
        patch(
            "ccgram.handlers.interactive.interactive_ui.rate_limit_send",
            new_callable=AsyncMock,
        ),
    ):
        assert await handle_interactive_ui(
            client, 7731, "pi-question", thread_id=7732, chat_id=-7731
        )
    markup = client.send_message.call_args.kwargs["reply_markup"]
    labels = [button.text for row in markup.inline_keyboard for button in row]
    assert all(label in labels for _, label in CHOICES)
