"""Guest-chat scope: a guest group only ever sees the sessions bound in it.

Everything the bot posts in a guest chat is visible to every member, and a
guest acts as the owner when pressing the shared buttons. So inside a guest
chat:

  - session listings (``/sessions``) show only windows bound to a topic of
    that chat (``windows_in_chat``);
  - commands that survey every session or every window (``/bind``, ``/sync``)
    are refused (``refuse_in_guest_chats``);
  - a guest's button press reaches only a window bound in that chat
    (``guest_target_allowed``), whatever the callback data says.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Coroutine, Iterable
from typing import TYPE_CHECKING, Any

import structlog

from ... import session_query
from ...config import config
from ..callback_helpers import parse_target, user_owns_window
from ..callback_tokens import resolve_callback_data
from ..messaging_pipeline.message_sender import safe_reply

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

GUEST_CHAT_REFUSED = "Not available in a guest chat. Use it in your own group."

type CommandHandler = Callable[
    ["Update", "ContextTypes.DEFAULT_TYPE"], Coroutine[Any, Any, None]
]


def is_guest_chat(chat_id: int | None) -> bool:
    return chat_id is not None and chat_id in config.guest_chats


def windows_in_chat(chat_id: int) -> set[str]:
    """Windows bound to a topic of *chat_id* (any user's binding)."""
    found: set[str] = set()
    for user_id, bound_chat, thread_id, window_id in session_query.iter_bound_topics():
        if bound_chat is None:
            bound_chat = session_query.resolve_chat_id(user_id, thread_id)
        if bound_chat == chat_id:
            found.add(window_id)
    return found


def guest_target_allowed(
    data: str, prefixes: Iterable[str], owner_id: int, chat_id: int | None
) -> bool:
    """True when a guest's callback targets a window bound in *chat_id*."""
    if chat_id is None:
        return False
    resolved = resolve_callback_data(data, owner_id, user_owns_window)
    if resolved is None:
        return False
    for prefix in prefixes:
        if resolved.startswith(prefix):
            window_id, _pane = parse_target(resolved[len(prefix) :])
            return bool(window_id) and window_id in windows_in_chat(chat_id)
    return False


def refuse_in_guest_chats(handler: CommandHandler) -> CommandHandler:
    """Wrap a command that lists every session so it says no in guest chats."""

    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if chat is not None and is_guest_chat(chat.id):
            logger.info(
                "Command refused in a guest chat",
                chat_id=chat.id,
                command=handler.__name__,
            )
            user = update.effective_user
            # Only the owner hears why; anyone else gets the usual silence.
            if (
                update.message is not None
                and user is not None
                and config.is_user_allowed(user.id)
            ):
                await safe_reply(update.message, GUEST_CHAT_REFUSED)
            return
        await handler(update, context)

    return wrapper
