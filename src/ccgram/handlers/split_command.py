"""/split — add a sibling pane to the current topic's window.

Splits the bound window/tab into a new pane (the multi-pane "agent team"
shape). With an argument, the text is run in the new pane — e.g.
``/split claude`` spawns a sibling agent, ``/split npm test`` starts a watcher.
The new pane is discoverable via /panes and the existing multi-pane scanning.

Backend-neutral: rides ``multiplexer.split_window`` (herdr ``pane split``,
tmux ``window.split()``), so it works on both backends.
"""

from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

from ..multiplexer import multiplexer as tmux_manager
from ..thread_router import thread_router

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = structlog.get_logger()


async def split_command(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Handle /split [command] — split the topic's window into a new pane."""
    # Lazy: config singleton resolved at call time so tests can swap it
    from ..config import config

    # Lazy: messaging_pipeline ↔ handler cycle through status_bubble
    from .messaging_pipeline.message_sender import safe_reply

    # Lazy: callback_helpers only used when we have a real update
    from .callback_helpers import get_thread_id

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "❌ Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "❌ This topic is not bound to any session.")
        return

    if not await tmux_manager.find_window_by_id(window_id):
        await safe_reply(update.message, "❌ Window no longer exists.")
        return

    new_pane = await tmux_manager.split_window(window_id)
    if not new_pane:
        await safe_reply(update.message, "❌ Could not split the window.")
        return

    command = " ".join(context.args).strip() if context.args else ""
    if command:
        await tmux_manager.send_to_pane(new_pane, command, window_id=window_id)
        await safe_reply(
            update.message,
            f"✅ Split into pane `{new_pane}` and ran `{command}`. Use /panes to view.",
        )
    else:
        await safe_reply(
            update.message,
            f"✅ Split into pane `{new_pane}`. Use /panes to view.",
        )
