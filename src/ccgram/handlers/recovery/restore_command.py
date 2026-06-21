"""/restore command — re-show the recovery banner for a dead topic.

When a tmux window dies, the topic becomes stale and ccgram normally
surfaces a recovery banner once. /restore lets the user re-show that
banner on demand so they can pick how to recover (Fresh / Continue /
Resume) — the actual session creation is owned by the recovery
callbacks. The command is a thin re-render — see
``handlers.recovery.recovery_banner.render_banner``.

Key function: restore_command().
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import structlog
from telegram import Update
from ... import window_query
from ...config import config
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ..messaging_pipeline.message_sender import safe_reply
from ..user_state import PENDING_THREAD_ID, RECOVERY_WINDOW_ID
from .recovery_banner import RecoveryBanner, render_banner

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()


async def restore_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /restore — re-show the recovery banner for a dead topic.

    The previous behaviour auto-ran ``--continue``; Task 1.9 of the UX
    overhaul moved that decision back to the user via the unified
    recovery banner.
    """
    user = update.effective_user
    if not user or not update.message:
        return

    if not config.is_user_allowed(user.id):
        await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    thread_id = update.message.message_thread_id
    if thread_id is None:
        await safe_reply(update.message, "Use this command inside a topic.")
        return

    user_id = user.id
    window_id = thread_router.resolve_window_for_thread(user_id, thread_id)
    if not window_id:
        await safe_reply(update.message, "No session bound to this topic.")
        return

    window = await tmux_manager.find_window_by_id(window_id)
    if window is not None:
        await safe_reply(
            update.message, "Window is still running — nothing to restore."
        )
        return

    view = window_query.view_window(window_id)
    if view is None or not view.cwd or not Path(view.cwd).is_dir():
        await safe_reply(update.message, "Directory no longer exists.")
        return

    display = thread_router.get_display_name(window_id) or window_id
    user_data = getattr(context, "user_data", None) if context else None
    if user_data is not None:
        user_data[PENDING_THREAD_ID] = thread_id
        user_data[RECOVERY_WINDOW_ID] = window_id

    banner = RecoveryBanner(
        chat_id=update.message.chat.id,
        thread_id=thread_id,
        window_id=window_id,
        mode="restore",
        provider=window_query.get_window_provider(window_id),
        display=display,
        cwd=view.cwd,
    )
    text, keyboard = render_banner(banner)
    await safe_reply(update.message, text, reply_markup=keyboard)
