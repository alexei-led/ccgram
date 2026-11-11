"""``/guest``: the owner manages guest access.

- ``/guest`` — status: on/off, guest chats, guests.
- ``/guest allow [name]`` — as a reply to a person's message in a guest chat:
  let them talk to this group's sessions (``name`` is their speaker label).
- ``/guest remove`` — as a reply, or ``/guest remove <user id>``.
- ``/guest off`` / ``/guest on`` — emergency switch for every guest, kept
  across restarts. Works from any chat.

Only ``ALLOWED_USERS`` can run it; anyone else in a guest chat gets silence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import Message, Update

from ...config import config
from ..messaging_pipeline.message_sender import safe_reply
from ..text.reply_quote import explicit_reply
from . import access

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

USAGE = (
    "Usage: /guest · /guest allow [name] (reply to their message in a guest "
    "chat) · /guest remove (reply, or a user id) · /guest off · /guest on"
)


def status_text() -> str:
    lines = [f"Guest access: {'ON' if access.is_enabled() else 'OFF'}"]
    if config.guest_chats:
        chats = ", ".join(
            f"{label} ({cid})" for cid, label in config.guest_chats.items()
        )
        lines.append(f"Guest chats: {chats}")
    else:
        lines.append("Guest chats: none (set CCGRAM_GUEST_CHATS)")
    entries = access.guests()
    if entries:
        lines.append("Guests:")
        lines.extend(
            f"- {g.name} (id {g.user_id}) in {access.chat_label(g.chat_id)}"
            for g in entries
        )
    else:
        lines.append("Guests: none")
    return "\n".join(lines)


async def _allow(message: Message, owner_id: int, args: list[str]) -> str:
    if not access.is_guest_chat(message.chat.id):
        return "Run /guest allow inside a guest chat (CCGRAM_GUEST_CHATS)."
    target = explicit_reply(message)
    person = target.from_user if target is not None else None
    if person is None:
        return "Reply to the person's message with /guest allow."
    if person.is_bot or config.is_user_allowed(person.id):
        return "That person is not a guest candidate."
    name = " ".join(args).strip() or person.first_name or str(person.id)
    access.allow(owner_id, person.id, message.chat.id, name)
    return (
        f"✅ {name} can now talk to the sessions in this group. "
        "Their buttons stay disabled. Stop anytime: /guest off"
    )


async def _remove(message: Message, args: list[str]) -> str:
    target = explicit_reply(message)
    if target is not None and target.from_user is not None:
        removed = access.remove(target.from_user.id, message.chat.id)
    elif args and args[0].lstrip("-").isdigit():
        removed = access.remove(int(args[0]))
    else:
        return "Reply to the guest's message with /guest remove, or give their id."
    return "Removed." if removed else "No such guest."


async def guest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.message
    if user is None or message is None:
        return
    if not config.is_user_allowed(user.id):
        if not access.is_guest_chat(message.chat.id):
            await safe_reply(message, "You are not authorized to use this bot.")
        return
    args = list(context.args or [])
    sub = args.pop(0).lower() if args else "status"
    if sub == "off":
        access.set_enabled(False)
        reply = "⛔ Guest access is OFF. Guests' messages are ignored until /guest on."
    elif sub == "on":
        access.set_enabled(True)
        reply = "✅ Guest access is ON."
    elif sub == "allow":
        reply = await _allow(message, user.id, args)
    elif sub == "remove":
        reply = await _remove(message, args)
    elif sub in ("status", "list"):
        reply = status_text()
    else:
        reply = USAGE
    await safe_reply(message, reply)
