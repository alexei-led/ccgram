"""Replies are terminal input even when Telegram tags them as bot commands."""

from unittest.mock import AsyncMock, patch

import pytest
from telegram import Update
from telegram.ext import filters

from ccgram.handlers.registry import register_all

from ccgram.handlers.interactive.pi_controls import PI_MENU_INSTRUCTION

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("text", ["/tmp", "/history", "/settings"])
async def test_command_shaped_pi_menu_reply_uses_guarded_text_route(
    dispatch_app, make_text_update, text
):
    dispatch_app.handlers.clear()
    register_all(dispatch_app, filters.ALL)
    update = make_text_update(text, bot=dispatch_app.bot)
    raw = update.to_dict()
    raw["message"]["reply_to_message"] = {
        "message_id": 988,
        "date": raw["message"]["date"],
        "chat": raw["message"]["chat"],
        "from": {"id": 999, "is_bot": True, "first_name": "CCGram"},
        "text": PI_MENU_INSTRUCTION + "\nNon-secret input",
    }
    reply_update = Update.de_json(raw, dispatch_app.bot)
    with patch(
        "ccgram.handlers.text.text_handler.handle_text_message", new_callable=AsyncMock
    ) as handle_text:
        await dispatch_app.process_update(reply_update)
    handle_text.assert_awaited_once()
    assert handle_text.call_args.args[0].message.text == text
