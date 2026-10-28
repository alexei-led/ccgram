"""Cross-agent controls must use the captured prompt, not a newer display."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.handlers.callback_data import CB_ASK_ENTER
from ccgram.handlers.interactive.interactive_callbacks import (
    handle_interactive_callback,
    handle_menu_reply,
)
from ccgram.handlers.interactive.interactive_ui import (
    clear_interactive_msg,
    handle_interactive_ui,
)
from ccgram.providers.base import StatusUpdate, TerminalControl, TerminalMenu

USER, CHAT, THREAD, MESSAGE = 8120, -8120, 8121, 8122
WINDOW = "menu-safety"
MENU = TerminalMenu(
    (
        TerminalControl("Use this session", ("s",)),
        TerminalControl("Next", ("Down",)),
        TerminalControl("Cancel", ("Escape",)),
    )
)


@pytest.fixture
async def display(monkeypatch):
    client = AsyncMock()
    client.send_message.return_value = SimpleNamespace(message_id=MESSAGE)
    mux = AsyncMock()
    mux.find_window_by_id.return_value = SimpleNamespace(window_id=WINDOW)
    mux.capture_pane.return_value = "first screen"
    status = StatusUpdate("First prompt", "PermissionPrompt", True, "PermissionPrompt")
    provider = SimpleNamespace(
        capabilities=SimpleNamespace(uses_pane_title=False),
        parse_terminal_status=lambda *_args, **_kwargs: status,
    )
    monkeypatch.setattr("ccgram.handlers.interactive.interactive_ui.tmux_manager", mux)
    monkeypatch.setattr(
        "ccgram.handlers.interactive.interactive_ui.get_provider_for_window",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "ccgram.handlers.interactive.interactive_ui.rate_limit_send", AsyncMock()
    )
    monkeypatch.setattr(
        "ccgram.handlers.callback_helpers.user_owns_window", lambda *_: True
    )
    adapter = AsyncMock()
    adapter.send_keys.return_value = (True, "Sent")
    adapter.send_text.return_value = (True, "Sent")
    monkeypatch.setattr(
        "ccgram.session_protocol.telegram.adapter_for", lambda _: adapter
    )
    monkeypatch.setattr(
        "ccgram.handlers.interactive.interactive_callbacks.asyncio.sleep", AsyncMock()
    )
    query = MagicMock()
    query.id = "safety"
    query.message.message_id = MESSAGE
    query.message.message_thread_id = THREAD
    query.message.chat.id = CHAT
    query.message.chat.type = "supergroup"
    query.answer = AsyncMock()
    update = MagicMock(message=None, callback_query=query)
    context = SimpleNamespace(bot=client, bot_data={})
    await clear_interactive_msg(USER, thread_id=THREAD, chat_id=CHAT)
    yield SimpleNamespace(
        client=client,
        adapter=adapter,
        provider=provider,
        query=query,
        update=update,
        context=context,
    )
    await clear_interactive_msg(USER, thread_id=THREAD, chat_id=CHAT)


async def render(display, status=None):
    if status is not None:
        display.provider.parse_terminal_status = lambda *_args, **_kwargs: status
    assert await handle_interactive_ui(
        display.client, USER, WINDOW, THREAD, chat_id=CHAT
    )
    call = (
        display.client.edit_message_text.call_args
        or display.client.send_message.call_args
    )
    return call.kwargs["reply_markup"]


async def test_old_generic_enter_cannot_confirm_a_newer_prompt(display):
    first = await render(display)
    data = next(
        str(button.callback_data)
        for row in first.inline_keyboard
        for button in row
        if str(button.callback_data).startswith(CB_ASK_ENTER)
    )
    await render(
        display,
        StatusUpdate("Second prompt", "PermissionPrompt", True, "PermissionPrompt"),
    )
    await handle_interactive_callback(
        display.query, USER, data, display.update, display.context
    )
    display.adapter.send_keys.assert_not_called()


async def test_blocking_primary_prompt_is_not_replaced_by_sibling(display):
    await render(display)
    assert not await handle_interactive_ui(
        display.client, USER, WINDOW, THREAD, chat_id=CHAT, pane_id="%other"
    )
    display.client.edit_message_text.assert_not_called()


async def test_confirmed_sibling_replaces_advisory_with_correct_owner(display):
    advisory = StatusUpdate("First prompt", "SelectionUI", True, "SelectionUI", True)
    await render(display, advisory)
    confirmed = StatusUpdate(
        "First prompt", "PermissionPrompt", True, "PermissionPrompt"
    )
    display.provider.parse_terminal_status = lambda *_args, **_kwargs: confirmed
    assert await handle_interactive_ui(
        display.client, USER, WINDOW, THREAD, chat_id=CHAT, pane_id="%other"
    )
    from ccgram.handlers.interactive.interactive_ui import get_interactive_pane

    assert get_interactive_pane(USER, THREAD, CHAT) == "%other"


async def test_generic_button_cannot_be_retargeted_to_another_pane(display):
    first = await render(display)
    data = next(
        str(button.callback_data)
        for row in first.inline_keyboard
        for button in row
        if str(button.callback_data).startswith(CB_ASK_ENTER)
    )
    await handle_interactive_callback(
        display.query, USER, data + "|%other", display.update, display.context
    )
    display.adapter.send_keys.assert_not_called()


@pytest.mark.parametrize("ui_type", ["PiTest", "CodexTest", "ClaudeTest"])
async def test_native_menu_uses_declared_controls_not_provider_name(display, ui_type):
    keyboard = await render(
        display, StatusUpdate("Choose a value", ui_type, True, ui_type, menu=MENU)
    )
    assert [button.text for row in keyboard.inline_keyboard for button in row] == [
        "Use this session",
        "Next",
        "Cancel",
        "🔄",
    ]
    data = str(keyboard.inline_keyboard[0][0].callback_data)
    await handle_interactive_callback(
        display.query, USER, data, display.update, display.context
    )
    call = display.adapter.send_text.call_args
    assert call.args == (WINDOW, "s")
    assert call.kwargs["submit"] is False
    assert len(call.kwargs["expected_prompt"]) == 64
    display.adapter.send_keys.assert_not_called()


async def test_advisory_view_exposes_only_refresh(display):
    keyboard = await render(
        display, StatusUpdate("Unverified", "SelectionUI", True, "SelectionUI", True)
    )
    assert [b.text for row in keyboard.inline_keyboard for b in row] == ["🔄"]


@pytest.mark.parametrize("ui_type", ["PiTest", "CodexTest", "ClaudeTest"])
async def test_button_only_menu_replies_never_send_input(display, ui_type):
    await render(display, StatusUpdate("Pick", ui_type, True, ui_type, menu=MENU))
    message = MagicMock()
    message.chat.id = CHAT
    message.reply_text = AsyncMock()
    message.reply_to_message = SimpleNamespace(
        message_id=MESSAGE, text=display.client.send_message.call_args.kwargs["text"]
    )
    assert await handle_menu_reply(
        display.client, USER, WINDOW, THREAD, "hello", message, display.adapter
    )
    display.adapter.send_text.assert_not_called()
    display.adapter.send_keys.assert_not_called()
    message.reply_text.assert_awaited_once()
    assert message.reply_text.call_args.args[0] == (
        "This menu does not accept text. Use its buttons."
    )
