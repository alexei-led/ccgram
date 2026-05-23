"""Central handler registration for the Telegram bot Application.

Owns the command/message/callback/inline handler registration that used
to live inline in ``bot.py``. ``register_all()`` is the single entry
point; ``bot.py`` is a factory + lifecycle hooks only.

Every handler called below lives in a feature subpackage under
``handlers/`` — this module only assembles them in the order PTB
requires.
"""

from dataclasses import dataclass
from typing import TypeAlias

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    InlineQueryHandler,
    MessageHandler,
    filters,
)
from telegram.ext._utils.types import HandlerCallback

from .callback_registry import dispatch as _dispatch_callback
from .callback_registry import load_handlers as _load_callback_handlers
from .agent_command import agent_command
from .cleanup import unbind_command
from .command_history import recall_command
from .commands import (
    commands_command,
    forward_command_handler,
    toolbar_command,
)
from .file_handler import handle_document_message, handle_photo_message
from .inline import inline_query_handler, unsupported_content_handler
from .live import live_command, panes_command, screenshot_command
from .messaging_pipeline import toolcalls_command, verbose_command
from .last_reply import last_command
from .recovery import restore_command, resume_command
from .recovery.history import history_command
from .send import send_command
from .sessions_dashboard import sessions_command
from .sync_command import sync_command
from .text.text_handler import text_handler
from .topics import new_command
from .topics.topic_lifecycle import topic_closed_handler, topic_edited_handler
from .upgrade import upgrade_command
from .voice import handle_voice_message

HandlerFn: TypeAlias = HandlerCallback


@dataclass(frozen=True)
class CommandSpec:
    """Specification for a single PTB CommandHandler registration."""

    name: str
    handler: HandlerFn


def register_all(
    application: Application,
    group_filter: filters.BaseFilter,
) -> None:
    """Register every command, callback, message and inline-query handler.

    Order is significant: PTB dispatches the first matching handler, so
    explicit CommandHandlers must precede the COMMAND-fallback
    MessageHandler, which must precede the TEXT MessageHandler.
    """
    command_specs: list[CommandSpec] = [
        CommandSpec("new", new_command),
        CommandSpec("start", new_command),  # compat alias
        CommandSpec("history", history_command),
        CommandSpec("commands", commands_command),
        CommandSpec("sessions", sessions_command),
        CommandSpec("resume", resume_command),
        CommandSpec("unbind", unbind_command),
        CommandSpec("upgrade", upgrade_command),
        CommandSpec("recall", recall_command),
        CommandSpec("screenshot", screenshot_command),
        CommandSpec("live", live_command),
        CommandSpec("panes", panes_command),
        CommandSpec("sync", sync_command),
        CommandSpec("toolbar", toolbar_command),
        CommandSpec("send", send_command),
        CommandSpec("verbose", verbose_command),
        CommandSpec("toolcalls", toolcalls_command),
        CommandSpec("restore", restore_command),
        CommandSpec("last", last_command),
        CommandSpec("agent", agent_command),
        CommandSpec("provider", agent_command),  # alias
    ]

    for spec in command_specs:
        application.add_handler(
            CommandHandler(spec.name, spec.handler, filters=group_filter)
        )

    _load_callback_handlers()
    application.add_handler(CallbackQueryHandler(_dispatch_callback))

    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_CLOSED & group_filter,
            topic_closed_handler,
        )
    )
    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_EDITED & group_filter,
            topic_edited_handler,
        )
    )
    application.add_handler(
        MessageHandler(filters.COMMAND & group_filter, forward_command_handler)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & group_filter, text_handler)
    )
    application.add_handler(
        MessageHandler(filters.PHOTO & group_filter, handle_photo_message)
    )
    application.add_handler(
        MessageHandler(filters.Document.ALL & group_filter, handle_document_message)
    )
    application.add_handler(
        MessageHandler(filters.VOICE & group_filter, handle_voice_message)
    )
    application.add_handler(
        MessageHandler(
            ~filters.COMMAND
            & ~filters.TEXT
            & ~filters.PHOTO
            & ~filters.Document.ALL
            & ~filters.VOICE
            & ~filters.StatusUpdate.ALL
            & group_filter,
            unsupported_content_handler,
        )
    )

    application.add_handler(InlineQueryHandler(inline_query_handler))


COMMAND_NAMES: tuple[str, ...] = (
    "new",
    "start",
    "history",
    "commands",
    "sessions",
    "resume",
    "unbind",
    "upgrade",
    "recall",
    "screenshot",
    "live",
    "panes",
    "sync",
    "toolbar",
    "send",
    "verbose",
    "toolcalls",
    "restore",
    "last",
    "agent",
    "provider",
)
"""Sentinel for tests: the exact command names register_all installs, in order."""
