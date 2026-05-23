"""Per-topic message-pipeline toggles — ``/verbose`` and ``/toolcalls``.

These commands flip per-window state held by ``SessionManager``: the
``batch_mode`` (used by the tool-use batching state machine in
``tool_batch.py``) and the ``tool_call_visibility`` (used by the queue
worker when filtering tool messages).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from telegram import Update
from ...config import config
from ...thread_router import thread_router
from ...utils import handle_general_topic_message, is_general_topic
from ...window_state_ports import tool_state
from ..callback_helpers import get_thread_id as _get_thread_id
from .message_sender import safe_reply

if TYPE_CHECKING:
    from telegram.ext import ContextTypes


async def verbose_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle tool-call batching for this topic."""
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

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "❌ This topic is not bound to any session.")
        return

    new_mode = tool_state.cycle_batch_mode(window_id)
    if new_mode == "batched":
        await safe_reply(
            update.message,
            "⚡ Tool calls will be *batched* into a single message.",
        )
    elif new_mode == "ephemeral":
        await safe_reply(
            update.message,
            "🫧 Tool calls shown live, removed when the reply is ready (ephemeral).",
        )
    else:
        await safe_reply(
            update.message,
            "💬 Tool calls will be sent *individually* (verbose mode).",
        )


async def toolcalls_command(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Cycle tool-call visibility for this topic: default → shown → hidden → default."""
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

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "❌ This topic is not bound to any session.")
        return

    new_mode = tool_state.cycle_tool_call_visibility(window_id)
    if new_mode == "shown":
        await safe_reply(
            update.message,
            "⚡ Tool calls *shown* for this topic (overrides global default).",
        )
    elif new_mode == "hidden":
        await safe_reply(
            update.message,
            "🔇 Tool calls *hidden* for this topic (overrides global default).",
        )
    else:
        resolved = "hidden" if config.hide_tool_calls else "shown"
        await safe_reply(
            update.message,
            f"🔄 Tool calls follow the global default (currently *{resolved}*).",
        )
