"""Retired Pi menu input must not enter another topic workflow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.handlers.text.text_handler import handle_text_message
from ccgram.handlers.interactive.pi_controls import MENU_INSTRUCTION
from ccgram.thread_router import ThreadRouter
from ccgram.multiplexer.base import WindowRef


@pytest.mark.parametrize("binding", [None, "shell-window", "pi-window"])
async def test_expired_reply_is_rejected_before_topic_routing(monkeypatch, binding):
    router = ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: True)
    if binding is not None:
        router.bind_thread(72, 42, binding, chat_id=-10072)
    monkeypatch.setattr("ccgram.handlers.text.text_handler.thread_router", router)
    mux = AsyncMock()
    mux.find_window_by_id.return_value = WindowRef(
        binding or "unused", "test", "/tmp", pane_current_command="zsh"
    )
    monkeypatch.setattr("ccgram.handlers.text.text_handler.tmux_manager", mux)
    adapter = AsyncMock()
    monkeypatch.setattr(
        "ccgram.handlers.text.text_handler.protocol_adapter_for", lambda _: adapter
    )
    monkeypatch.setattr(
        "ccgram.session_protocol.telegram.adapter_for", lambda _: adapter
    )
    message = MagicMock()
    message.text = "pwd"
    message.message_id = 11
    message.message_thread_id = 42
    message.chat.id = -10072
    message.chat.type = "supergroup"
    message.is_topic_message = True
    message.reply_text = AsyncMock()
    message.reply_to_message = SimpleNamespace(
        message_id=10, text=MENU_INSTRUCTION + "\nOld menu"
    )
    update = MagicMock(
        effective_user=SimpleNamespace(id=72),
        message=message,
        effective_chat=message.chat,
    )
    context = MagicMock(bot=AsyncMock(), bot_data={}, user_data={})

    await handle_text_message(update, context)

    assert "expired" in message.reply_text.call_args.args[0].lower()
    mux.find_window_by_id.assert_not_called()
    adapter.send_text.assert_not_called()
    adapter.send_keys.assert_not_called()
    assert context.user_data == {}
