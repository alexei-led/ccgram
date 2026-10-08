"""Command forwarding, topic panels, shared menus, and status fallbacks.

The subpackage owns five distinct paths:

  - ``forward``: route typed or selected slash commands to the provider.
  - ``menu_sync``: keep the shared four-command Telegram menu in safe scopes.
  - ``panel``: build topic-scoped CCGram and native provider command buttons.
  - ``failure_probe``: detect provider rejection from transcript and pane changes.
  - ``status_snapshot``: synthesize status for providers without native replies.

This package also hosts the ``/commands`` and ``/toolbar`` entry points.
It re-exports the public surface used by ``bot.py``, ``bootstrap.py``,
``handlers/registry.py``, and ``handlers/text/text_handler.py``.
"""

from __future__ import annotations


from typing import TYPE_CHECKING

import structlog
from telegram import Update

from ...config import config
from ... import window_query
from ...thread_router import thread_router
from ...utils import handle_general_topic_message, is_general_topic
from ..callback_helpers import get_thread_id as _get_thread_id
from ..messaging_pipeline.message_sender import safe_reply
from ..toolbar import build_toolbar_keyboard, seed_button_states
from .forward import forward_command_handler
from .panel import send_command_panel
from .menu_sync import (
    get_global_provider_menu,
    set_global_provider_menu,
    setup_menu_refresh_job,
    sync_scoped_menu_for_text_context,
    sync_scoped_provider_menu,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()


async def commands_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/commands`` — open and pin the controls relevant to this topic."""

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return

    await sync_scoped_menu_for_text_context(update, user.id)
    await send_command_panel(update.message, user.id)


async def toolbar_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/toolbar`` — show the persistent action toolbar for the topic."""

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "❌ Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(
        user.id, thread_id, update.message.chat.id
    )
    if not window_id:
        await safe_reply(update.message, "❌ This topic is not bound to any session.")
        return

    provider_name = window_query.get_window_provider(window_id) or "claude"
    # Seed toggle-button labels with the actual current state so the
    # initial render shows "Edit"/"Plan"/"YOLO"/"Def" instead of "Mode".
    await seed_button_states(window_id)
    keyboard = build_toolbar_keyboard(window_id, provider_name)
    display = thread_router.get_display_name(window_id)
    await safe_reply(
        update.message,
        f"\U0001f39b `{display}` toolbar",
        reply_markup=keyboard,
    )


__all__ = [
    "commands_command",
    "forward_command_handler",
    "get_global_provider_menu",
    "set_global_provider_menu",
    "setup_menu_refresh_job",
    "sync_scoped_menu_for_text_context",
    "sync_scoped_provider_menu",
    "toolbar_command",
]
