"""Non-destructive forum-topic existence probe shared by repair flows."""

from collections.abc import Callable

import structlog
from telegram.error import RetryAfter, TelegramError

from ...telegram_client import TelegramClient
from ..messaging_pipeline.message_sender import is_thread_gone

logger = structlog.get_logger()

_TOPIC_PROBE_TEXT = "."


async def probe_topic_exists(
    client: TelegramClient,
    chat_id: int,
    thread_id: int,
    *,
    propagate_retry_after: bool = False,
    on_cleanup_retry_after: Callable[[RetryAfter], None] | None = None,
) -> bool | None:
    """Return True when a topic exists, False when deleted, and None on uncertainty.

    Send failures may propagate flood control. Cleanup never revokes a positive
    result; callers can observe its flood response without repeating the probe.
    """
    try:
        message = await client.send_message(
            chat_id,
            _TOPIC_PROBE_TEXT,
            message_thread_id=thread_id,
            disable_notification=True,
        )
    except RetryAfter:
        if propagate_retry_after:
            raise
        return None
    except TelegramError as exc:
        if is_thread_gone(exc):
            return False
        return None

    try:
        await client.delete_message(chat_id, message.message_id)
    except RetryAfter as exc:
        if on_cleanup_retry_after is not None:
            on_cleanup_retry_after(exc)
        logger.warning(
            "Failed to delete topic probe message",
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=message.message_id,
            error="retry after",
        )
    except TelegramError as exc:
        logger.warning(
            "Failed to delete topic probe message",
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=message.message_id,
            error=str(exc),
        )
    return True
