"""Speaker labels for messages that reach the agent from a guest chat.

In a guest chat both the owner and the guest talk to the same session, so
every forwarded message says who wrote it ("Jason: …", "QFY: …"). A guest's
message also carries a one-line reminder that its scope is limited; the
session's instructions file (AGENTS.md, or CLAUDE.md for Claude) spells out
that scope. The real boundary is the session's
permission settings: anything that needs approval still goes to the owner.
"""

from __future__ import annotations

from . import access

GUEST_NOTE = (
    "[Guest message from {name}. Keep to the guest scope described in "
    "AGENTS.md / CLAUDE.md; anything that needs approval goes to the owner, "
    "who decides.]"
)


def speaker_name(user_id: int, chat_id: int, first_name: str | None) -> str:
    for guest in access.guests():
        if guest.user_id == user_id and guest.chat_id == chat_id and guest.name:
            return guest.name
    return (first_name or "").strip() or str(user_id)


def frame_for_agent(
    text: str, *, chat_id: int, user_id: int, first_name: str | None, is_guest: bool
) -> str:
    """Label ``text`` with its speaker when it comes from a guest chat."""
    if not access.is_guest_chat(chat_id):
        return text
    name = speaker_name(user_id, chat_id, first_name)
    framed = f"{name}: {text}"
    if is_guest:
        framed += "\n\n" + GUEST_NOTE.format(name=name)
    return framed
