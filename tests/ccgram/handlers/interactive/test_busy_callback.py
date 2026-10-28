from unittest.mock import AsyncMock, MagicMock

from ccgram.handlers.callback_data import CB_ASK_ESC
from ccgram.handlers.interactive.interactive_callbacks import (
    handle_interactive_callback,
)
from ccgram.handlers.interactive.interactive_ui import (
    _interactive_msgs,
    _interactive_contexts,
    _interactive_sequences,
    _interactive_mode,
    _interactive_prompt_fingerprints,
)
from ccgram.session_protocol.models import ProtocolError


def _confirmed_prompt():
    key = (10, 100, 42)
    _interactive_msgs[key] = 77
    _interactive_contexts[key] = (100, 77)
    _interactive_sequences[key] = 1
    _interactive_mode[key] = "@12"
    _interactive_prompt_fingerprints[key] = "captured-prompt"


async def test_busy_callback_answers_without_sending_input(monkeypatch):
    query = MagicMock()
    query.answer = AsyncMock()
    query.message.chat.type = "supergroup"
    query.message.chat.id = 100
    query.message.message_thread_id = 42
    query.message.message_id = 77
    update = MagicMock(message=None, callback_query=query)
    _confirmed_prompt()
    monkeypatch.setattr(
        "ccgram.handlers.callback_helpers.user_owns_window", lambda *_: True
    )
    adapter = MagicMock()
    adapter.send_keys = AsyncMock(
        side_effect=ProtocolError(503, "session_busy", "Session is busy")
    )
    monkeypatch.setattr(
        "ccgram.session_protocol.telegram.adapter_for", lambda _: adapter
    )
    await handle_interactive_callback(
        query, 10, f"{CB_ASK_ESC}1:@12", update, MagicMock()
    )
    query.answer.assert_awaited_once_with(
        "Session is busy. Try again.", show_alert=True
    )
    adapter.send_keys.assert_awaited_once()


async def test_changed_prompt_is_reported_instead_of_success_label(monkeypatch):
    query = MagicMock()
    query.answer = AsyncMock()
    query.message.chat.type = "supergroup"
    query.message.chat.id = 100
    query.message.message_thread_id = 42
    query.message.message_id = 77
    update = MagicMock(message=None, callback_query=query)
    _confirmed_prompt()
    monkeypatch.setattr(
        "ccgram.handlers.callback_helpers.user_owns_window", lambda *_: True
    )
    adapter = MagicMock()
    adapter.send_keys = AsyncMock(
        return_value=(False, "Interactive prompt changed before input")
    )
    monkeypatch.setattr(
        "ccgram.session_protocol.telegram.adapter_for", lambda _: adapter
    )
    await handle_interactive_callback(
        query, 10, f"{CB_ASK_ESC}1:@12", update, MagicMock()
    )
    query.answer.assert_awaited_once_with(
        "Interactive prompt changed before input", show_alert=True
    )
    adapter.send_keys.assert_awaited_once()
