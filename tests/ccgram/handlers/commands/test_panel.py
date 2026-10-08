"""Topic-aware command panels and lossless agent command dispatch."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot, CallbackQuery, Chat, Message, Update, User
from telegram.ext import CommandHandler

from ccgram.cc_commands import CCCommand, register_commands
from ccgram.providers.base import ProviderCapabilities


def message(thread_id=42, chat_id=-100999):
    msg = Message(
        message_id=12,
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        chat=Chat(chat_id, "supergroup", is_forum=True),
        from_user=User(100, "Tester", False),
        text="/commands",
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
    )
    msg.set_bot(AsyncMock(spec=Bot))
    return msg


@pytest.fixture
def panel_env(tmp_path, monkeypatch):
    from ccgram.handlers.commands import panel
    from ccgram.handlers import callback_tokens

    monkeypatch.setattr(callback_tokens, "_TOKEN_STORE_PATH", tmp_path / "tokens.json")
    callback_tokens._tokens.clear()
    from ccgram.handlers.commands import panel as panel_module

    panel_module._panel_messages.clear()
    provider = SimpleNamespace(
        capabilities=ProviderCapabilities(
            name="claude",
            launch_command="claude",
            supports_structured_transcript=True,
            supports_resume=True,
            supports_resume_picker=True,
        )
    )
    with (
        patch.object(panel.config, "is_user_allowed", return_value=True),
        patch.object(panel.config, "group_id", None),
        patch.object(
            panel.session_query, "resolve_window_for_topic", return_value="@1"
        ) as router,
        patch.object(
            panel.window_query,
            "view_window",
            return_value=SimpleNamespace(window_name="project"),
        ),
        patch.object(
            panel, "get_provider_for_window", return_value=provider
        ) as resolve_provider,
        patch.object(panel.window_query, "get_window_provider", return_value="claude"),
        patch.object(
            panel,
            "discover_provider_commands",
            return_value=[
                CCCommand("compact", "compact", "Summarize context", "builtin"),
                CCCommand("spec:work", "spec_work", "Run a spec", "command"),
                CCCommand(
                    "committing-code", "committing_code", "Commit changes", "skill"
                ),
                CCCommand("clear", "clear", "Reset conversation", "builtin"),
                CCCommand("/cost", "cost", "Show session cost", "builtin"),
                CCCommand("/doctor", "doctor", "Diagnose the agent", "builtin"),
            ],
        ),
        patch.object(panel, "safe_reply", new_callable=AsyncMock) as reply,
        patch.object(panel, "interactive_edit", new_callable=AsyncMock) as edit,
        patch.object(panel, "safe_edit", new_callable=AsyncMock) as update_panel,
        patch.object(
            panel, "forward_command_handler", new_callable=AsyncMock
        ) as forward,
    ):
        yield SimpleNamespace(
            panel=panel,
            provider=provider,
            router=router,
            resolve_provider=resolve_provider,
            reply=reply,
            edit=edit,
            update_panel=update_panel,
            forward=forward,
        )
    callback_tokens._tokens.clear()
    panel_module._panel_messages.clear()


def labels(keyboard):
    return [button.text for row in keyboard.inline_keyboard for button in row]


def click(env, target, keyboard, label, context, user_id=100):
    button = next(
        button
        for row in keyboard.inline_keyboard
        for button in row
        if button.text == label
    )
    query = CallbackQuery(
        "click",
        User(user_id, "Tester", False),
        "instance",
        message=target,
        data=button.callback_data,
    )
    query.set_bot(target.get_bot())
    update = Update(1, callback_query=query)
    return env.panel.handle_command_panel(update, context)


def test_runtime_callback_registry_loads_the_command_panel():
    from ccgram.handlers.callback_registry import get_registry, load_handlers

    load_handlers()

    assert "cmdpanel:" in get_registry()


async def test_default_slash_menu_contains_only_four_controls():
    bot = AsyncMock()
    await register_commands(bot)
    assert [cmd.command for cmd in bot.set_my_commands.call_args.args[0]] == [
        "commands",
        "sessions",
        "sync",
        "upgrade",
    ]


@pytest.mark.parametrize("thread_id", [None, 1])
async def test_general_panel_has_no_session_or_agent_commands(panel_env, thread_id):
    env = panel_env
    await env.panel.send_command_panel(message(thread_id), 100)
    keyboard = env.reply.call_args.kwargs["reply_markup"]
    assert labels(keyboard) == ["Commands", "Sessions", "Audit state", "Update ccgram"]
    env.router.assert_not_called()
    env.resolve_provider.assert_not_called()


async def test_reopening_topic_commands_edits_the_existing_panel(panel_env):
    env = panel_env
    msg = message()
    sent = MagicMock()
    sent.pin = AsyncMock()
    sent.message_thread_id = msg.message_thread_id
    env.reply.return_value = sent

    await env.panel.send_command_panel(msg, 100)
    await env.panel.send_command_panel(msg, 100)

    env.reply.assert_awaited_once()
    env.update_panel.assert_awaited_once()
    assert env.update_panel.await_args.args[0] is sent
    sent.pin.assert_awaited_once_with(disable_notification=True)


async def test_original_agent_names_survive_labels_and_dispatch(panel_env):
    env = panel_env
    msg = message()
    target, provider = env.panel.resolve_panel_target(msg, 100)
    _, keyboard = env.panel.build_command_panel(target, provider, group="agent")
    assert "/spec:work" in labels(keyboard)
    assert "/committing-code" in labels(keyboard)
    assert "/spec_work" not in labels(keyboard)
    context = MagicMock(user_data={})
    await click(env, msg, keyboard, "/spec:work", context)
    forwarded = env.forward.call_args.args[0]
    assert forwarded.message.text == "/spec:work"
    assert forwarded.effective_user.id == 100
    assert forwarded.message.chat.id == msg.chat.id
    assert forwarded.message.message_thread_id == 42
    assert env.edit.await_count == 0


async def test_cost_and_doctor_buttons_keep_the_original_provider_names(panel_env):
    env = panel_env
    for command in ("/cost", "/doctor"):
        msg = message()
        target, provider = env.panel.resolve_panel_target(msg, 100)
        _, keyboard = env.panel.build_command_panel(target, provider, group="agent")
        await click(env, msg, keyboard, command, MagicMock(user_data={}))
        assert env.forward.await_args.kwargs["native_command"] == command
        env.forward.reset_mock()
        env.edit.reset_mock()


async def test_agent_dispatch_exception_is_reported(panel_env):
    from structlog.testing import capture_logs

    env = panel_env
    env.forward.side_effect = RuntimeError("agent transport failed")
    msg = message()
    _, keyboard = env.panel.build_command_panel(
        env.panel.resolve_panel_target(msg, 100)[0], env.provider, group="agent"
    )

    with capture_logs() as logs:
        await click(env, msg, keyboard, "/cost", MagicMock(user_data={}))

    assert (
        env.reply.await_args.args[1]
        == "❌ Could not send the command. Check the ccgram log."
    )
    assert any(
        record["event"] == "Topic command panel dispatch failed" for record in logs
    )


async def test_shell_hides_transcript_and_agent_commands(panel_env):
    env = panel_env
    provider = SimpleNamespace(
        capabilities=ProviderCapabilities(
            name="shell",
            launch_command="",
            chat_first_command_path=True,
        )
    )
    target = env.panel.PanelTarget(100, -100999, 42, "@1", "shell")
    _, home = env.panel.build_command_panel(target, provider)
    assert "History" not in labels(home)
    assert "/compact" not in labels(home)
    _, groups = env.panel.build_command_panel(target, provider, group="groups")
    assert "Agent commands" not in labels(groups)
    _, common = env.panel.build_command_panel(target, provider, group="ccgram")
    assert "History" not in labels(common)
    assert "Resume session" not in labels(common)
    assert "Split terminal" not in labels(common)


async def test_unbound_panel_offers_setup_not_terminal_actions(panel_env):
    env = panel_env
    env.router.return_value = None
    await env.panel.send_command_panel(message(), 100)
    keyboard = env.reply.call_args.kwargs["reply_markup"]
    assert "Set up session" in labels(keyboard)
    assert "Screenshot" not in labels(keyboard)


@pytest.mark.parametrize("change", ["owner", "window", "provider", "chat", "thread"])
async def test_stale_or_foreign_panel_cannot_dispatch(panel_env, change):
    env = panel_env
    msg = message()
    target, provider = env.panel.resolve_panel_target(msg, 100)
    _, keyboard = env.panel.build_command_panel(target, provider, group="agent")
    user_id = 200 if change == "owner" else 100
    if change == "window":
        env.router.return_value = "@2"
    if change == "provider":
        env.resolve_provider.return_value = SimpleNamespace(
            capabilities=ProviderCapabilities(name="codex", launch_command="codex")
        )
    if change == "chat":
        msg = message(chat_id=-100888)
    if change == "thread":
        msg = message(thread_id=43)
    await click(env, msg, keyboard, "/spec:work", MagicMock(user_data={}), user_id)
    env.forward.assert_not_called()


async def test_long_native_command_uses_bounded_callback_without_truncation(panel_env):
    env = panel_env
    name = "group:" + "x" * 100
    with patch.object(
        env.panel,
        "discover_provider_commands",
        return_value=[CCCommand(name, "group_x", "Long command", "command")],
    ):
        msg = message()
        target, provider = env.panel.resolve_panel_target(msg, 100)
        _, keyboard = env.panel.build_command_panel(target, provider, group="agent")
        assert labels(keyboard)[0] == "Command 1"
        text, _ = env.panel.build_command_panel(target, provider, group="agent")
        assert "/" + name in text
        assert all(
            len(button.callback_data.encode()) <= 64
            for row in keyboard.inline_keyboard
            for button in row
        )
        await click(env, msg, keyboard, "Command 1", MagicMock(user_data={}))
    assert env.forward.call_args.kwargs["native_command"] == "/" + name


async def test_clear_needs_confirmation_and_confirm_is_one_shot(panel_env):
    env = panel_env
    msg = message()
    target, provider = env.panel.resolve_panel_target(msg, 100)
    _, keyboard = env.panel.build_command_panel(target, provider, group="agent")
    context = MagicMock(user_data={})
    await click(env, msg, keyboard, "/clear", context)
    env.forward.assert_not_called()
    confirmation = env.edit.call_args.kwargs["reply_markup"]
    await click(env, msg, confirmation, "Confirm", context)
    restored = env.edit.await_args.kwargs["reply_markup"]
    assert "/cost" in labels(restored)
    await click(env, msg, confirmation, "Confirm", context)
    env.forward.assert_awaited_once()
    assert env.forward.call_args.args[0].message.text == "/clear"


async def test_commands_control_reopens_the_current_panel_in_place(panel_env):
    env = panel_env
    msg = message()
    target, provider = env.panel.resolve_panel_target(msg, 100)
    _, keyboard = env.panel.build_command_panel(target, provider, group="global")
    handler = AsyncMock()
    context = MagicMock(user_data={})
    context.application.handlers = {0: [CommandHandler("commands", handler)]}

    await click(env, msg, keyboard, "Commands", context)

    handler.assert_not_awaited()
    env.update_panel.assert_awaited_once()
    assert env.update_panel.await_args.kwargs["reply_markup"]


async def test_global_button_uses_registered_bot_handler(panel_env):
    env = panel_env
    msg = message(1)
    target, provider = env.panel.resolve_panel_target(msg, 100)
    _, keyboard = env.panel.build_command_panel(target, provider)
    handler = AsyncMock()
    context = MagicMock(user_data={})
    context.application.handlers = {0: [CommandHandler("sessions", handler)]}
    await click(env, msg, keyboard, "Sessions", context)
    handler.assert_awaited_once()
    update = handler.call_args.args[0]
    assert update.message.text == "/sessions"
    assert update.effective_user.id == 100
    assert context.args == []


async def test_agent_command_collision_bypasses_bot_handler(panel_env):
    env = panel_env
    with patch.object(
        env.panel,
        "discover_provider_commands",
        return_value=[CCCommand("/resume", "resume", "Resume agent", "builtin")],
    ):
        msg = message()
        target, provider = env.panel.resolve_panel_target(msg, 100)
        _, keyboard = env.panel.build_command_panel(target, provider, group="agent")
        handler = AsyncMock()
        context = MagicMock(user_data={})
        context.application.handlers = {0: [CommandHandler("resume", handler)]}
        await click(env, msg, keyboard, "/resume", context)
    env.forward.assert_awaited_once()
    handler.assert_not_called()


async def test_pin_permission_failure_keeps_panel_usable(panel_env):
    from telegram.error import TelegramError

    env = panel_env
    sent = AsyncMock()
    sent.pin.side_effect = TelegramError("not an administrator")
    env.reply.return_value = sent
    await env.panel.send_command_panel(message(1), 100)
    sent.pin.assert_awaited_once_with(disable_notification=True)
    assert len(labels(env.reply.call_args.kwargs["reply_markup"])) == 4
