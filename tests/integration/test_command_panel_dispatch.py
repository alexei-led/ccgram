"""Real callback dispatch from a bot-authored panel to terminal delivery."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from structlog.testing import capture_logs
from telegram import CallbackQuery, Chat, Message, Update, User
from telegram.ext import CallbackContext

from ccgram.config import config
from ccgram.handlers import callback_registry, callback_tokens, telegram_origin
from ccgram.handlers.command_history import clear_history, get_history
from ccgram.handlers.commands import failure_probe, panel
from ccgram.multiplexer.base import WindowRef
from ccgram.thread_router import thread_router
from ccgram.window_state_store import window_store

pytestmark = pytest.mark.integration

USER_ID = 12345
CHAT_ID = -100999
THREAD_ID = 42
WINDOW_ID = "@1"


@pytest.fixture
def bound_panel(session_manager, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "group_id", CHAT_ID)
    monkeypatch.setattr(config, "claude_config_dir", tmp_path / "claude")
    monkeypatch.setattr(callback_tokens, "_TOKEN_STORE_PATH", tmp_path / "tokens.json")
    monkeypatch.setattr(callback_tokens, "_tokens", {})
    monkeypatch.setattr(telegram_origin, "_pending_injections", {})
    monkeypatch.setattr(failure_probe, "_COMMAND_ERROR_PROBE_DELAY_SECONDS", 0)
    command_dir = config.claude_config_dir / "commands" / "spec"
    command_dir.mkdir(parents=True)
    (command_dir / "work.md").write_text("---\ndescription: Run a spec\n---\n")
    skill_dir = config.claude_config_dir / "skills" / "committing-code"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: committing-code\nuser-invocable: true\n---\n"
    )
    state = window_store.get_window_state(WINDOW_ID)
    state.provider_name = "claude"
    state.initial_provider_name = "claude"
    state.window_name = "tf-reflex-readers"
    thread_router.bind_thread(
        USER_ID, THREAD_ID, WINDOW_ID, state.window_name, chat_id=CHAT_ID
    )
    clear_history(USER_ID, THREAD_ID)
    yield
    clear_history(USER_ID, THREAD_ID)


def panel_update(bot, command: str) -> Update:
    message = Message(
        12,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        Chat(CHAT_ID, "supergroup", is_forum=True),
        from_user=User(bot.id, "Bot", True),
        text="Commands for this topic",
        message_thread_id=THREAD_ID,
        is_topic_message=True,
    )
    message.set_bot(bot)
    target, provider = panel.resolve_panel_target(message, USER_ID)
    names = panel._agent_names(provider)
    _, keyboard = panel.build_command_panel(
        target, provider, group="agent", page=names.index(command) // 8
    )
    button = next(
        button
        for row in keyboard.inline_keyboard
        for button in row
        if button.text == command
    )
    assert isinstance(button.callback_data, str)
    query = CallbackQuery(
        "click",
        User(USER_ID, "Owner", False),
        "instance",
        message=message,
        data=button.callback_data,
    )
    query.set_bot(bot)
    update = Update(1, callback_query=query)
    update.set_bot(bot)
    return update


@pytest.mark.parametrize(
    "command", ["/cost", "/help", "/doctor", "/spec:work", "/committing-code"]
)
@pytest.mark.parametrize("outcome", ["success", "failure", "exception"])
async def test_panel_dispatch_uses_clicking_user_and_preserves_command(
    bound_panel, dispatch_app, command, outcome
):
    update = panel_update(dispatch_app.bot, command)
    context = CallbackContext.from_update(update, dispatch_app)
    probe_finished = asyncio.Event()
    captures = 0

    async def capture_pane(*_args, **_kwargs):
        nonlocal captures
        captures += 1
        if captures > 1:
            probe_finished.set()
        return ""

    send = AsyncMock(return_value=(outcome == "success", "terminal unavailable"))
    if outcome == "exception":
        send.side_effect = RuntimeError("terminal unavailable")
    with (
        patch.object(type(dispatch_app.bot), "answer_callback_query", AsyncMock()),
        patch.object(type(dispatch_app.bot), "send_chat_action", AsyncMock()),
        patch.object(type(dispatch_app.bot), "set_my_commands", AsyncMock()),
        patch.object(type(dispatch_app.bot), "send_message", AsyncMock()) as reply,
        patch(
            "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
            AsyncMock(
                return_value=WindowRef(
                    WINDOW_ID,
                    "tf-reflex-readers",
                    str(config.claude_config_dir),
                    "claude",
                )
            ),
        ),
        patch(
            "ccgram.multiplexer.tmux.tmux_manager.capture_pane",
            AsyncMock(side_effect=capture_pane),
        ),
        patch.object(telegram_origin, "send_to_window", send),
        capture_logs() as logs,
    ):
        await callback_registry.dispatch(update, context)
        send.assert_awaited_once_with(WINDOW_ID, command, raw=False)
        if outcome == "success":
            await asyncio.wait_for(probe_finished.wait(), timeout=2)

    assert reply.await_args is not None
    events = [record["event"] for record in logs]
    request = next(
        record
        for record in logs
        if record["event"] == "Provider command request received"
    )
    assert request["user_id"] == USER_ID
    assert request["thread_id"] == THREAD_ID
    assert request["chat_id"] == CHAT_ID
    assert request["command"] == command
    assert "Topic command panel callback received" in events
    assert "Topic command panel callback accepted" in events
    if outcome == "success":
        assert "Provider command delivered to terminal" in events
        assert get_history(USER_ID, THREAD_ID) == [command]
        assert command in reply.await_args.kwargs["text"]
    else:
        assert "Provider command delivered to terminal" not in events
        assert get_history(USER_ID, THREAD_ID) == []
        assert "❌" in reply.await_args.kwargs["text"]
        assert (
            "Provider command delivery failed"
            if outcome == "failure"
            else "Topic command panel dispatch failed"
        ) in events
