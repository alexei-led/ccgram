"""Ordinary prompt replies must not become command-menu input."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.handlers.interactive.interactive_ui import INTERACTIVE_INSTRUCTION_LINE
from ccgram.handlers.interactive.pi_controls import is_menu_reply, MENU_INSTRUCTION
from ccgram.handlers.text.text_handler import _handle_menu_reply


@pytest.mark.parametrize(
    "header",
    [
        INTERACTIVE_INSTRUCTION_LINE,
        "↑↓ select · Enter confirm · Esc cancel · type to enter text",
    ],
)
async def test_ordinary_prompt_reply_keeps_normal_text_routing(header):
    message = MagicMock()
    message.reply_to_message = SimpleNamespace(text=header + "\nAllow this command?")
    message.reply_text = AsyncMock()
    assert not is_menu_reply(message)
    assert not await _handle_menu_reply(83, 84, "hello", message, MagicMock())
    message.reply_text.assert_not_called()


def test_genuine_menu_header_remains_distinct():
    message = MagicMock()
    message.reply_to_message = SimpleNamespace(text=MENU_INSTRUCTION + "\nSettings")
    assert is_menu_reply(message)
