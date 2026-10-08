"""Verify a recorded recovery topic by applying a supplied title."""

from telegram.error import BadRequest, RetryAfter, TelegramError

from ...telegram_client import TelegramClient
from ...telegram_rate_limiter import NO_RETRY_RATE_LIMIT_ARGS
from ..messaging_pipeline.message_sender import is_thread_gone

_TOPIC_NAME_LIMIT = 128


async def probe_topic_exists(
    client: TelegramClient,
    chat_id: int,
    thread_id: int,
    *,
    topic_name: str,
    propagate_retry_after: bool = False,
) -> bool | None:
    """Apply the title; return True if present, False if gone, None if unknown.

    Only recovery uses this repair, never a sweep of active bindings. An empty
    edit is not an existence check: Telegram can accept it without fetching the
    topic. A real title edit (including TOPIC_NOT_MODIFIED) validates the topic.
    Recovery supplies the resolved session name, which can remove title badges.
    Changing the title emits a Telegram rename service message. A later status
    refresh may restore badges and emit another rename notice. No temporary
    probe message is sent, and the topic's open/closed state is unchanged.
    """
    if not topic_name:
        return None
    try:
        edited = await client.edit_forum_topic(
            chat_id,
            thread_id,
            name=topic_name[:_TOPIC_NAME_LIMIT],
            rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS,
        )
    except RetryAfter:
        if propagate_retry_after:
            raise
        return None
    except TelegramError as exc:
        if is_thread_gone(exc):
            return False
        if (
            isinstance(exc, BadRequest)
            and exc.message.lower().replace(" ", "_") == "topic_not_modified"
        ):
            return True
        return None
    return True if edited is not False else None
