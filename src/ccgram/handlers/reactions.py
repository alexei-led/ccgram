"""Telegram message reactions helper (Bot API 7.0+).

Provides `react()` for setting a single emoji reaction on a message, with:
- Allowed-emoji validation against the Telegram-allowed reaction set
  (sourced from `telegram.constants.ReactionEmoji`).
- Per-(chat_id, message_id) dedupe to skip redundant API calls.
- Graceful failure: BadRequest/TelegramError caught, optional callback toast
  is shown as a user-visible fallback.

Intended replacement for `query.answer(...)` toasts where a persistent
acknowledgement is more useful than a one-shot popup (voice send, /send
delivery, shell command lifecycle).

Note on allowed reactions: Telegram restricts bots to a fixed set of free
emoji reactions; the audit's preferred icons (✅ ❌ 📬 ⚙) are not on the
list. Semantic constants below map intent → an allowed emoji approximation.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import Final

import structlog
from telegram import CallbackQuery, ReactionTypeEmoji
from telegram.constants import ReactionEmoji
from telegram.error import TelegramError

from ..telegram_client import TelegramClient

logger = structlog.get_logger()


ALLOWED_REACTIONS: Final[frozenset[str]] = frozenset(e.value for e in ReactionEmoji)


# Semantic constants — guaranteed members of ALLOWED_REACTIONS.
# Pick values that map cleanly to the audit's intents while staying valid.
REACT_SEEN: Final[str] = "👀"
REACT_THINKING: Final[str] = "🤔"
REACT_DONE: Final[str] = "🔥"
REACT_FAIL: Final[str] = "💔"
REACT_INBOX: Final[str] = "✍"
REACT_RUNNING: Final[str] = "⚡"


# LRU cap on the dedupe cache. A long-lived bot reacts to many messages, so an
# unbounded dict would grow without bound (one entry per ever-reacted message).
_MAX_DEDUPE_ENTRIES: Final[int] = 2000

_last_reaction: OrderedDict[tuple[int, int], str] = OrderedDict()


async def react(
    client: TelegramClient,
    chat_id: int,
    message_id: int,
    emoji: str,
    *,
    fallback_query: CallbackQuery | None = None,
    fallback_toast: str | None = None,
) -> bool:
    """Set an emoji reaction on a message.

    Returns True on success (including dedupe skip). Returns False when
    the emoji is not in `ALLOWED_REACTIONS` or the API call fails. On
    failure, if `fallback_query` and `fallback_toast` are both provided,
    the callback is answered with the toast text as a user-visible
    fallback.
    """
    if emoji not in ALLOWED_REACTIONS:
        logger.warning("Disallowed reaction emoji %r — falling back", emoji)
        await _maybe_toast(fallback_query, fallback_toast)
        return False

    key = (chat_id, message_id)
    if _last_reaction.get(key) == emoji:
        _last_reaction.move_to_end(key)
        return True

    try:
        await client.set_message_reaction(
            chat_id=chat_id,
            message_id=message_id,
            reaction=[ReactionTypeEmoji(emoji=emoji)],
        )
    except TelegramError as exc:
        logger.warning("set_message_reaction failed: %s", exc)
        await _maybe_toast(fallback_query, fallback_toast)
        return False

    _last_reaction[key] = emoji
    _last_reaction.move_to_end(key)
    if len(_last_reaction) > _MAX_DEDUPE_ENTRIES:
        _last_reaction.popitem(last=False)
    return True


async def clear_reaction(client: TelegramClient, chat_id: int, message_id: int) -> bool:
    """Remove all reactions from a message."""
    try:
        await client.set_message_reaction(
            chat_id=chat_id,
            message_id=message_id,
            reaction=[],
        )
    except TelegramError as exc:
        logger.warning("clear_reaction failed: %s", exc)
        return False
    _last_reaction.pop((chat_id, message_id), None)
    return True


async def _maybe_toast(query: CallbackQuery | None, text: str | None) -> None:
    if query is None or text is None:
        return
    with contextlib.suppress(TelegramError):
        await query.answer(text, show_alert=False)
