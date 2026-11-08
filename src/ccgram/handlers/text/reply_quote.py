"""Turn a Telegram reply into quoted context for the agent.

Claude Code has no notion of replying to a message, so a reply in a topic
reaches the agent as a short quote in front of the typed text. The quote only
points at the message the user means (the agent normally has it in context
already), so long messages are cut to an excerpt.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Message

# Weighted length budget for the quoted excerpt. CJK and other wide characters
# count 3 (one carries about as much as three Latin characters), so the limit
# is ≈ 1000 Chinese characters or ≈ 3000 English characters.
QUOTE_BUDGET = 3000
_WIDE_WEIGHT = 3
_HEAD_SHARE = 0.6
_ELISION = "\n…\n"

LABEL_AGENT = "Quoting your earlier message:"
LABEL_SELF = "Quoting my earlier message:"
LABEL_OTHER = "Quoting an earlier message:"


def _weight(char: str) -> int:
    return _WIDE_WEIGHT if unicodedata.east_asian_width(char) in ("W", "F") else 1


def weighted_length(text: str) -> int:
    return sum(_weight(c) for c in text)


def _take(text: str, budget: int) -> str:
    used = 0
    for index, char in enumerate(text):
        used += _weight(char)
        if used > budget:
            return text[:index]
    return text


def truncate_weighted(text: str, budget: int = QUOTE_BUDGET) -> str:
    """Keep the head and tail of ``text`` within ``budget``, eliding the middle."""
    if weighted_length(text) <= budget:
        return text
    room = budget - weighted_length(_ELISION)
    head = _take(text, int(room * _HEAD_SHARE))
    tail = _take(text[::-1], room - weighted_length(head))[::-1]
    return head.rstrip() + _ELISION + tail.lstrip()


def _explicit_reply(message: Message) -> Message | None:
    """The message the user replied to, ignoring a forum topic's root message.

    Inside a forum topic every message carries ``reply_to_message`` pointing at
    the service message that created the topic, reply or not.
    """
    reply = message.reply_to_message
    if reply is None:
        return None
    if reply.forum_topic_created is not None:
        return None
    if message.message_thread_id is not None and (
        reply.message_id == message.message_thread_id
    ):
        return None
    return reply


def _label(message: Message, reply: Message) -> str:
    author = reply.from_user
    if author is None:
        return LABEL_OTHER
    if author.is_bot:
        return LABEL_AGENT
    sender = message.from_user
    if sender is not None and author.id == sender.id:
        return LABEL_SELF
    return LABEL_OTHER


def quoted_context(message: Message) -> str | None:
    """The quote block for a reply, or None when there is nothing to quote."""
    reply = _explicit_reply(message)
    if reply is None:
        return None
    partial = message.quote.text if message.quote is not None else None
    source = partial or reply.text or reply.caption or ""
    source = source.strip()
    if not source:
        return None
    excerpt = truncate_weighted(source)
    quoted = "\n".join(f"> {line}" if line else ">" for line in excerpt.splitlines())
    return f"{_label(message, reply)}\n{quoted}"


def compose_with_quote(text: str, message: Message) -> str:
    """``text`` with the quote of the replied message in front, if any."""
    context = quoted_context(message)
    if context is None:
        return text
    return f"{context}\n\n{text}"
