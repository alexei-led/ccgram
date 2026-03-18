"""Safe message sending helpers with MarkdownV2 fallback.

Provides utility functions for sending Telegram messages with automatic
conversion to MarkdownV2 format and fallback to plain text on failure.

Functions:
  - rate_limit_send: Rate limiter to avoid Telegram flood control
  - rate_limit_send_message: Combined rate limiting + send with fallback
  - safe_reply: Reply with MarkdownV2, fallback to plain text
  - safe_edit: Edit message with MarkdownV2, fallback to plain text
  - safe_send: Send message with MarkdownV2, fallback to plain text
"""

import asyncio
import re
import structlog
import time
from collections.abc import Awaitable, Callable
from typing import Any

from telegram import Bot, CallbackQuery, LinkPreviewOptions, Message
from telegram.error import BadRequest, RetryAfter, TelegramError

from ..markdown_v2 import convert_markdown

logger = structlog.get_logger()

# Disable link previews in all messages to reduce visual noise
NO_LINK_PREVIEW = LinkPreviewOptions(is_disabled=True)

# Regex to strip MarkdownV2 escape sequences from plain text fallback.
# Matches a backslash followed by any MarkdownV2 special character.
_MDV2_STRIP_RE = re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")
# Strip expandable blockquote syntax: leading ">" prefix and trailing "||"
_BLOCKQUOTE_PREFIX_RE = re.compile(r"^>", re.MULTILINE)
_BLOCKQUOTE_CLOSE_RE = re.compile(r"\|\|$", re.MULTILINE)


class _MessageGoneError(Exception):
    """Raised when the target message no longer exists (deleted topic)."""


def _retry_after_seconds(exc: RetryAfter) -> int:
    """Extract retry delay from RetryAfter, handling both int and timedelta."""
    ra = exc.retry_after
    return ra if isinstance(ra, int) else int(ra.total_seconds())


def strip_mdv2(text: str) -> str:
    """Strip MarkdownV2 formatting artifacts for clean plain text fallback.

    Removes backslash escapes before special chars and blockquote syntax
    so the fallback message is readable without formatting artifacts.
    """
    text = _MDV2_STRIP_RE.sub(r"\1", text)
    text = _BLOCKQUOTE_CLOSE_RE.sub("", text)
    return _BLOCKQUOTE_PREFIX_RE.sub("", text)


# Rate limiting: last send time per chat to avoid Telegram flood control
_last_send_time: dict[int, float] = {}
MESSAGE_SEND_INTERVAL = 1.1  # seconds between messages to same chat


async def rate_limit_send(chat_id: int) -> None:
    """Wait if necessary to avoid Telegram flood control (max 1 msg/sec per chat)."""
    now = time.monotonic()
    if chat_id in _last_send_time:
        elapsed = now - _last_send_time[chat_id]
        if elapsed < MESSAGE_SEND_INTERVAL:
            await asyncio.sleep(MESSAGE_SEND_INTERVAL - elapsed)
    _last_send_time[chat_id] = time.monotonic()


async def _with_mdv2_fallback(
    send_fn: Callable[..., Awaitable[Any]],
    text: str,
    context_label: str,
    **kwargs: Any,
) -> Message | None:
    """Try MarkdownV2, fall back to plain text, handle RetryAfter throughout.

    Generic helper that eliminates the repeated MarkdownV2 → plain text →
    RetryAfter pattern across all send/reply/edit functions.

    Args:
        send_fn: Async callable accepting (text, parse_mode=..., **kwargs).
        text: Raw markdown text (pre-conversion).
        context_label: Label for warning log messages (e.g. "send to 123").
        **kwargs: Extra keyword arguments forwarded to send_fn.

    Returns the result Message on success, None on failure.
    """
    mdv2_text = convert_markdown(text)
    plain_text = strip_mdv2(text)

    # Phase 1: try MarkdownV2
    try:
        return await send_fn(mdv2_text, parse_mode="MarkdownV2", **kwargs)
    except RetryAfter as e:
        await asyncio.sleep(_retry_after_seconds(e) + 1)
        try:
            return await send_fn(mdv2_text, parse_mode="MarkdownV2", **kwargs)
        except TelegramError as e2:
            logger.warning("Failed to %s after retry: %s", context_label, e2)
            return None
    except TelegramError:
        pass

    # Phase 2: fall back to plain text (stripped MarkdownV2 artifacts)
    try:
        return await send_fn(plain_text, **kwargs)
    except RetryAfter as e:
        await asyncio.sleep(_retry_after_seconds(e) + 1)
        try:
            return await send_fn(plain_text, **kwargs)
        except TelegramError as e2:
            logger.warning("Failed to %s after retry: %s", context_label, e2)
            return None
    except TelegramError as e:
        logger.warning("Failed to %s: %s", context_label, e)
        return None


async def _send_with_fallback(
    bot: Bot,
    chat_id: int,
    text: str,
    **kwargs: Any,
) -> Message | None:
    """Send message with MarkdownV2, falling back to plain text on failure.

    Internal helper that handles the MarkdownV2 → plain text fallback pattern.
    Handles RetryAfter with a single sleep+retry instead of propagating.
    Returns the sent Message on success, None on failure.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)

    async def _send(text: str, **kw: Any) -> Message:
        return await bot.send_message(chat_id=chat_id, text=text, **kw)

    return await _with_mdv2_fallback(
        _send, text, f"send message to {chat_id}", **kwargs
    )


async def rate_limit_send_message(
    bot: Bot,
    chat_id: int,
    text: str,
    **kwargs: Any,
) -> Message | None:
    """Rate-limited send with MarkdownV2 fallback.

    Combines rate_limit_send() + _send_with_fallback() for convenience.
    The chat_id should be the group chat ID for forum topics, or the user ID
    for direct messages.  Use session_manager.resolve_chat_id() to obtain it.
    Returns the sent Message on success, None on failure.
    """
    await rate_limit_send(chat_id)
    return await _send_with_fallback(bot, chat_id, text, **kwargs)


async def safe_reply(message: Message, text: str, **kwargs: Any) -> Message | None:
    """Reply with MarkdownV2, falling back to plain text on failure.

    Returns None if the original message no longer exists (e.g. deleted topic).
    Handles RetryAfter with a single sleep+retry instead of propagating.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)

    async def _reply(text: str, **kw: Any) -> Message:
        try:
            return await message.reply_text(text, **kw)
        except BadRequest as exc:
            if "not found" in str(exc).lower():
                logger.warning("Cannot reply: original message gone (%s)", exc)
                raise _MessageGoneError from exc
            raise

    try:
        return await _with_mdv2_fallback(_reply, text, "reply", **kwargs)
    except _MessageGoneError:
        return None


async def safe_edit(target: Message | CallbackQuery, text: str, **kwargs: Any) -> None:
    """Edit message with MarkdownV2, falling back to plain text on failure.

    Accepts either a CallbackQuery (edit_message_text) or a Message (edit_text).
    Handles RetryAfter with a single sleep+retry instead of propagating.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    # Message.edit_text vs CallbackQuery.edit_message_text
    raw_edit_fn = (
        target.edit_text if isinstance(target, Message) else target.edit_message_text
    )

    async def _edit(text: str, **kw: Any) -> Any:
        return await raw_edit_fn(text, **kw)

    await _with_mdv2_fallback(_edit, text, "edit message", **kwargs)


async def safe_send(
    bot: Bot,
    chat_id: int,
    text: str,
    message_thread_id: int | None = None,
    **kwargs: Any,
) -> None:
    """Send message with MarkdownV2, falling back to plain text on failure.

    Handles RetryAfter with a single sleep+retry instead of propagating.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    if message_thread_id is not None:
        kwargs.setdefault("message_thread_id", message_thread_id)

    async def _send(text: str, **kw: Any) -> Message:
        return await bot.send_message(chat_id=chat_id, text=text, **kw)

    await _with_mdv2_fallback(_send, text, f"send message to {chat_id}", **kwargs)
