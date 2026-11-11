"""Messages from guests: text into the owner's bound session, nothing else.

A guest may only write in a topic of a guest chat that the owner has already
bound to a session. No new sessions, no "!" shell commands, no recovery
buttons. The message is forwarded with the owner's identity (bindings are the
owner's) and the guest's speaker label.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog
from telegram import Message, Update

from ... import session_query
from ..callback_helpers import get_thread_id
from ..messaging_pipeline.message_sender import safe_reply
from ..presence import digest as desk_digest
from ..presence import presence

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

# A guest message older than this when it is handled was sent while the
# computer (or CCGram) was off; the guest gets told when it actually arrived.
LATE_AFTER_S = 120

NOT_BOUND = "This topic is not connected to a session yet. Ask the owner to set it up."
NO_TOPIC = "Please write inside the session's topic."
NO_SHELL = "Shell commands are not available to guests."


def late_notice(sent_at: datetime, now: datetime | None = None) -> str | None:
    """The delay note for a message handled long after it was sent."""
    now = now or datetime.now(timezone.utc)
    if (now - sent_at).total_seconds() < LATE_AFTER_S:
        return None
    sent_local = sent_at.astimezone()
    now_local = now.astimezone()
    fmt = "%H:%M" if sent_local.date() == now_local.date() else "%m-%d %H:%M"
    return (
        f"⏰ Sent {sent_local.strftime(fmt)}, delivered {now_local.strftime('%H:%M')}"
        " — the computer was offline in between."
    )


def bound_window(owner_id: int, message: Message, thread_id: int | None) -> str | None:
    if thread_id is None:
        return None
    return session_query.resolve_window_for_topic(owner_id, thread_id, message.chat.id)


def count_for_brief(owner_id: int, message: Message, thread_id: int | None) -> None:
    """Mention the guest's message in the owner's desk brief while at the desk."""
    if presence.is_quiet(owner_id):
        desk_digest.record_guest_traffic(
            owner_id, message.chat.id, thread_id, role="user", content_type="text"
        )


async def handle_guest_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE, owner_id: int
) -> None:
    """Forward a guest's text message to the owner's session in this topic."""
    # Lazy: text_handler imports this module's package for the guest branch.
    from ..text.text_handler import forward_guest_text

    message = update.message
    user = update.effective_user
    if message is None or user is None or not message.text:
        return
    thread_id = get_thread_id(update)
    logger.info(
        "Guest message",
        user_id=user.id,
        chat_id=message.chat.id,
        thread_id=thread_id,
        owner_id=owner_id,
    )
    if message.text.startswith("!"):
        await safe_reply(message, NO_SHELL)
        return
    if thread_id is None:
        await safe_reply(message, NO_TOPIC)
        return
    window_id = bound_window(owner_id, message, thread_id)
    if window_id is None:
        await safe_reply(message, NOT_BOUND)
        return
    count_for_brief(owner_id, message, thread_id)
    delivered = await forward_guest_text(
        window_id, owner_id, thread_id, message.text, message, context
    )
    note = late_notice(message.date) if delivered else None
    if note:
        await safe_reply(message, note)
