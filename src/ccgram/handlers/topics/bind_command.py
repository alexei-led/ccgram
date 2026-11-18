"""``/bind`` — attach an already-running window to this topic.

The window picker used to run on every unbound topic, which put an extra
screen (and a wrong default) in front of every new session. It now runs only
when asked: ``/bind`` lists the windows that are running but unbound, and one
tap attaches the chosen one to this topic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from ...config import config
from ...multiplexer import multiplexer as tmux_manager
from ...multiplexer.base import canonical_window_id
from ...thread_router import thread_router
from ..callback_helpers import get_thread_id
from ..messaging_pipeline.message_sender import safe_reply
from .directory_browser import (
    STATE_KEY,
    STATE_SELECTING_WINDOW,
    UNBOUND_WINDOWS_KEY,
    build_window_picker,
)
from ..user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

__all__ = ["bind_command"]


async def bind_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List unbound windows for this topic to adopt."""
    user = update.effective_user
    message = update.message
    if not user or message is None:
        return
    if not config.is_user_allowed(user.id):
        await safe_reply(message, "You are not authorized to use this bot.")
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await safe_reply(message, "Use /bind inside a topic.")
        return
    if thread_router.get_window_for_thread(user.id, thread_id, message.chat.id):
        await safe_reply(
            message,
            "This topic already has a session. Use /unbind first, then /bind.",
        )
        return

    all_windows = await tmux_manager.list_windows()
    bound = {
        canonical_window_id(wid) for _, _, wid in thread_router.iter_thread_bindings()
    }
    unbound = [
        (w.window_id, w.window_name, w.cwd)
        for w in all_windows
        if canonical_window_id(w.window_id) not in bound
    ]
    if not unbound:
        await safe_reply(
            message,
            "No unbound windows. Send a session line to start one:\n"
            "`codex <project> <instruction>`",
        )
        return

    text, keyboard, window_ids = build_window_picker(unbound, include_new=False)
    if context.user_data is not None:
        context.user_data[STATE_KEY] = STATE_SELECTING_WINDOW
        context.user_data[UNBOUND_WINDOWS_KEY] = window_ids
        context.user_data[PENDING_THREAD_ID] = thread_id
        context.user_data.pop(PENDING_THREAD_TEXT, None)
    await safe_reply(message, text, reply_markup=keyboard)
