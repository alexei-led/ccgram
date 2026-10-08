from unittest.mock import MagicMock

import pytest
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from ccgram.config import config
from ccgram.handlers.registry import (
    COMMAND_NAMES,
    CommandSpec,
    _log_command_update,
    register_all,
)
from ccgram.handlers.topics.new_command import new_command
from ccgram.handlers.dashboard_command import private_start_greeting


def _stub_handler():
    return MagicMock()


def _make_app():
    app = MagicMock()
    app.add_handler = MagicMock()
    return app


def test_command_spec_is_frozen():
    spec = CommandSpec("foo", _stub_handler())
    with pytest.raises(AttributeError):
        spec.name = "bar"  # type: ignore[misc]


def test_register_all_installs_expected_command_names():
    app = _make_app()
    register_all(app, filters.ALL)

    command_names: list[str] = []
    for call in app.add_handler.call_args_list:
        handler = call.args[0]
        if isinstance(handler, CommandHandler):
            command_names.extend(sorted(handler.commands))

    assert set(command_names) == set(COMMAND_NAMES)
    assert len(command_names) == len(COMMAND_NAMES)


def test_register_all_registers_grouped_private_start_without_changing_group_filters():
    app = _make_app()
    group_filter = filters.Chat(chat_id=-100123)
    register_all(app, group_filter, group_id=-100123)

    start_handlers = [
        call.args[0]
        for call in app.add_handler.call_args_list
        if isinstance(call.args[0], CommandHandler) and "start" in call.args[0].commands
    ]
    group_start = next(
        handler for handler in start_handlers if handler.callback is new_command
    )
    private_start = next(
        handler
        for handler in start_handlers
        if handler.callback is private_start_greeting
    )
    assert str(group_start.filters) == str(group_filter)
    expected_private_filter = filters.ChatType.PRIVATE & filters.User(
        user_id=config.allowed_users
    )
    assert str(private_start.filters) == str(expected_private_filter)
    assert all(
        str(handler.filters) == str(group_filter)
        for handler in start_handlers
        if handler is not private_start
    )


def test_register_all_registers_all_handler_kinds():
    app = _make_app()
    register_all(app, filters.ALL)

    by_kind: dict[type, int] = {}
    for call in app.add_handler.call_args_list:
        handler = call.args[0]
        by_kind[type(handler)] = by_kind.get(type(handler), 0) + 1

    assert by_kind.get(CommandHandler) == len(COMMAND_NAMES)
    assert by_kind.get(CallbackQueryHandler) == 1
    assert by_kind.get(InlineQueryHandler) == 1
    # 9 MessageHandlers: command trace, FORUM_TOPIC_CLOSED, FORUM_TOPIC_EDITED,
    # COMMAND fallback, TEXT, PHOTO, Document.ALL, VOICE, catch-all unsupported.
    assert by_kind.get(MessageHandler) == 9


async def test_command_trace_logs_general_topic_without_arguments():
    update = MagicMock()
    update.effective_message.text = "/sync secret-argument"
    update.effective_message.chat.id = -100123
    update.effective_message.message_thread_id = None
    update.update_id = 42

    with pytest.MonkeyPatch.context() as monkeypatch:
        mock_logger = MagicMock()
        monkeypatch.setattr("ccgram.handlers.registry.logger", mock_logger)
        await _log_command_update(update, MagicMock())

    mock_logger.info.assert_called_once_with(
        "Telegram command received",
        command="/sync",
        chat_id=-100123,
        thread_id=None,
        update_id=42,
    )


def test_register_all_command_handlers_precede_message_command_fallback():
    """CommandHandlers must be registered before the COMMAND-fallback MessageHandler.

    PTB dispatches the first matching handler — if the COMMAND fallback came
    first, /history would never reach history_command.
    """
    app = _make_app()
    register_all(app, filters.ALL)

    last_command_idx = -1
    first_message_idx = -1
    for idx, call in enumerate(app.add_handler.call_args_list):
        handler = call.args[0]
        if isinstance(handler, CommandHandler):
            last_command_idx = idx
        elif isinstance(handler, MessageHandler) and first_message_idx == -1:
            first_message_idx = idx

    assert last_command_idx >= 0 and first_message_idx >= 0
    assert last_command_idx < first_message_idx
