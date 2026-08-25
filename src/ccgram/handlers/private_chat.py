"""Private-chat greeting: the only handler allowed outside the group.

With CCGRAM_GROUP_ID set, every handler is filtered to the group, so a user
opening a private chat with the bot gets total silence (confusing right when
they are following the /dashboard DM flow, which requires having messaged the
bot first). This module answers /start in private chats with a short welcome
that explains where the bot actually lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import config
from .messaging_pipeline.message_sender import safe_reply

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes


async def private_start_command(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Greet the user in private chat and point back to the forum topics."""
    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return
    await safe_reply(
        update.message,
        "👋 This is ccgram's service chat. I work inside the forum topics of "
        "our group: open a topic and talk to me there.\n"
        "From a topic, /dashboard sends you a fresh Mini App button here.",
    )
