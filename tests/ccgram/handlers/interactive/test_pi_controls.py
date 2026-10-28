"""Exercise Pi menus through the shared Telegram callback and text boundaries."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import NetworkError

from ccgram.handlers.interactive.interactive_callbacks import (
    handle_interactive_callback,
)
from ccgram.handlers.interactive.interactive_ui import (
    clear_interactive_msg,
    get_interactive_window,
    handle_interactive_ui,
)
from ccgram.handlers.interactive.pi_controls import build_pi_keyboard, parse_pi_control
from ccgram.handlers.callback_tokens import compact_callback_data
from ccgram.handlers.text.text_handler import handle_text_message
from ccgram.thread_router import ThreadRouter

FIXTURES = Path(__file__).parents[2] / "providers" / "fixtures" / "pi_menus"
USER, CHAT, THREAD, MESSAGE = 9941, -9941, 9942, 9943
WINDOW, PANE = "pi:opaque:session", "%9"


@pytest.fixture
async def menu(monkeypatch):
    router = ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: True)
    monkeypatch.setattr("ccgram.handlers.text.text_handler.thread_router", router)
    client = AsyncMock()
    client.send_message.return_value = SimpleNamespace(message_id=MESSAGE)
    mux = AsyncMock()
    mux.capture_pane_by_id.return_value = (FIXTURES / "settings.txt").read_text()
    adapter = AsyncMock()
    adapter.send_keys.return_value = (True, "Sent")
    adapter.send_text.return_value = (True, "Sent")
    monkeypatch.setattr("ccgram.handlers.interactive.interactive_ui.tmux_manager", mux)
    monkeypatch.setattr(
        "ccgram.handlers.interactive.interactive_ui.get_window_provider", lambda _: "pi"
    )
    monkeypatch.setattr(
        "ccgram.handlers.interactive.interactive_ui.rate_limit_send", AsyncMock()
    )
    monkeypatch.setattr(
        "ccgram.handlers.callback_helpers.user_owns_window", lambda *_: True
    )
    monkeypatch.setattr(
        "ccgram.session_protocol.telegram.adapter_for", lambda _: adapter
    )
    monkeypatch.setattr(
        "ccgram.handlers.interactive.interactive_callbacks.asyncio.sleep", AsyncMock()
    )
    monkeypatch.setattr(
        "ccgram.handlers.text.text_handler.protocol_adapter_for", lambda _: adapter
    )
    await clear_interactive_msg(USER, thread_id=THREAD, chat_id=CHAT)
    assert await handle_interactive_ui(
        client, USER, WINDOW, THREAD, pane_id=PANE, chat_id=CHAT
    )
    sent = client.send_message.call_args.kwargs
    message = MagicMock()
    message.chat.id = CHAT
    message.chat.type = "supergroup"
    message.chat.send_action = AsyncMock()
    message.reply_text = AsyncMock()
    message.message_id = MESSAGE + 1
    message.message_thread_id = THREAD
    message.is_topic_message = True
    message.reply_to_message = SimpleNamespace(message_id=MESSAGE, text=sent["text"])
    query = MagicMock()
    query.id = "pi-callback"
    query.message = MagicMock()
    query.message.chat.id = CHAT
    query.message.chat.type = "supergroup"
    query.message.message_thread_id = THREAD
    query.message.message_id = MESSAGE
    query.answer = AsyncMock()
    update = MagicMock(message=None, callback_query=query)
    context = SimpleNamespace(bot=client, bot_data={}, user_data={})
    yield SimpleNamespace(
        client=client,
        router=router,
        mux=mux,
        adapter=adapter,
        message=message,
        query=query,
        update=update,
        context=context,
        sent=sent,
    )
    await clear_interactive_msg(USER, thread_id=THREAD, chat_id=CHAT)


async def _route_reply(menu, text: str, window_id: str = WINDOW) -> None:
    menu.router.bind_thread(USER, THREAD, window_id, chat_id=CHAT)
    menu.message.text = text
    update = MagicMock(
        effective_user=SimpleNamespace(id=USER),
        message=menu.message,
        effective_chat=menu.message.chat,
    )
    await handle_text_message(update, menu.context)


def _action(menu, action: str) -> str:
    for row in menu.sent["reply_markup"].inline_keyboard:
        for button in row:
            parsed = parse_pi_control(str(button.callback_data))
            if parsed is not None and parsed[0] == action:
                return str(button.callback_data)
    raise AssertionError(action)


@pytest.mark.parametrize(
    ("action", "keys"),
    [
        ("down5", ["Down"] * 5),
        ("clear", ["C-u"]),
        ("enter", ["Enter"]),
    ],
)
async def test_controls_target_owning_pane_and_prompt(menu, action, keys):
    await handle_interactive_callback(
        menu.query, USER, _action(menu, action), menu.update, menu.context
    )
    call = menu.adapter.send_keys.call_args
    assert call.args == (WINDOW, keys)
    assert call.kwargs["pane_id"] == PANE
    assert len(call.kwargs["expected_prompt"]) == 64


async def test_consumed_keyboard_cannot_send_a_second_key(menu):
    data = _action(menu, "down")
    await handle_interactive_callback(menu.query, USER, data, menu.update, menu.context)
    await handle_interactive_callback(menu.query, USER, data, menu.update, menu.context)
    menu.adapter.send_keys.assert_awaited_once()
    menu.query.answer.assert_awaited_with("This prompt has expired", show_alert=True)


async def test_refresh_recovers_after_key_was_sent_but_display_edit_failed(menu):
    menu.client.edit_message_text.side_effect = [NetworkError("offline"), MagicMock()]
    await handle_interactive_callback(
        menu.query, USER, _action(menu, "down"), menu.update, menu.context
    )
    await handle_interactive_callback(
        menu.query, USER, _action(menu, "refresh"), menu.update, menu.context
    )
    menu.adapter.send_keys.assert_awaited_once()
    assert menu.client.edit_message_text.await_count == 2


async def test_failed_key_does_not_claim_success_or_retire_menu(menu):
    menu.adapter.send_keys.return_value = (
        False,
        "Interactive prompt changed before input",
    )
    await handle_interactive_callback(
        menu.query, USER, _action(menu, "down"), menu.update, menu.context
    )
    menu.query.answer.assert_awaited_once_with(
        "Interactive prompt changed before input", show_alert=True
    )
    assert get_interactive_window(USER, THREAD, chat_id=CHAT) == WINDOW


async def test_escape_from_submenu_keeps_parent_menu(menu):
    await handle_interactive_callback(
        menu.query, USER, _action(menu, "esc"), menu.update, menu.context
    )
    menu.client.delete_message.assert_not_called()
    assert get_interactive_window(USER, THREAD, chat_id=CHAT) == WINDOW


async def test_enter_closing_menu_retires_keyboard(menu):
    menu.mux.capture_pane_by_id.return_value = "─" * 80 + "\n >\n" + "─" * 80
    await handle_interactive_callback(
        menu.query, USER, _action(menu, "enter"), menu.update, menu.context
    )
    menu.client.delete_message.assert_awaited_once_with(
        chat_id=CHAT, message_id=MESSAGE
    )
    assert get_interactive_window(USER, THREAD, chat_id=CHAT) is None


async def test_reply_pastes_without_enter_or_agent_message(menu):
    await _route_reply(menu, "theme")
    call = menu.adapter.send_text.call_args
    assert call.args == (WINDOW, "theme")
    assert call.kwargs["submit"] is False
    assert call.kwargs["pane_id"] == PANE
    assert len(call.kwargs["expected_prompt"]) == 64
    menu.adapter.send_keys.assert_not_called()


@pytest.mark.parametrize("change", ["message", "window", "chat", "text", "closed"])
async def test_stale_menu_reply_never_becomes_agent_input(menu, change):
    if change == "message":
        menu.message.reply_to_message.message_id -= 1
    elif change == "chat":
        menu.message.chat.id -= 1
    elif change == "text":
        menu.message.reply_to_message.text += "\nOld view"
    elif change == "closed":
        await clear_interactive_msg(USER, thread_id=THREAD, chat_id=CHAT)
    await _route_reply(menu, "theme", "other-window" if change == "window" else WINDOW)
    menu.adapter.send_text.assert_not_called()
    menu.adapter.send_keys.assert_not_called()


@pytest.mark.parametrize("text", ["hello\nworld", "\x1b[A", "theme\r", "\t"])
async def test_reply_cannot_inject_terminal_controls(menu, text):
    await _route_reply(menu, text)
    menu.adapter.send_text.assert_not_called()
    menu.adapter.send_keys.assert_not_called()


def test_opaque_target_and_pane_round_trip():
    keyboard = build_pi_keyboard(
        WINDOW, "PiScopedModels", PANE, 17, compact_callback_data
    )
    parsed = [
        parse_pi_control(str(button.callback_data))
        for row in keyboard.inline_keyboard
        for button in row
    ]
    assert all(item is not None and item[1:] == (17, WINDOW, PANE) for item in parsed)
    assert {"save", "all", "none", "provider", "reorder-up", "reorder-down"} <= {
        item[0] for item in parsed if item is not None
    }


@pytest.mark.parametrize(
    "data",
    [
        "aq:pi:shell:1:@1",
        "aq:pi:down:no:@1",
        "aq:pi:down:1:",
        "aq:pi:down:1:@1|",
    ],
)
def test_invalid_controls_are_rejected(data):
    assert parse_pi_control(data) is None
