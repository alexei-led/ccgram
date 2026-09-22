"""Delete recorded retired topics, preserving failures for later retries."""

from collections import Counter
from collections.abc import Awaitable, Callable
import time

import structlog
from telegram.error import BadRequest, RetryAfter, TelegramError

from ...session import session_manager
from ...multiplexer.base import canonical_window_id
from ...telegram_client import TelegramClient
from ...telegram_rate_limiter import NO_RETRY_RATE_LIMIT_ARGS, retry_after_seconds
from ...thread_router import RetiredTopic, ThreadRouter, thread_router
from ..messaging_pipeline.message_sender import is_thread_gone

logger = structlog.get_logger()

_RETRY_SECONDS = 60.0
_REVIEWABLE_CLOSED_REASONS = {"remote_closed", "remote_removed"}


async def retire_topic_binding(
    client: TelegramClient,
    user_id: int,
    thread_id: int,
    window_id: str,
    *,
    router: ThreadRouter = thread_router,
    chat_id: int | None = None,
    before_delete: Callable[[], Awaitable[None]] | None = None,
) -> str:
    """Retire an exact confirmed-dead binding before any asynchronous cleanup."""
    if router.has_target_provisioning(window_id) is True:
        return "protected_provisioning"
    if (
        chat_id is not None
        and router.has_topic_provisioning(chat_id, thread_id) is True
    ):
        return "protected_provisioning"
    if chat_id is None and any(
        claim.user_id == user_id and claim.thread_id == thread_id
        for claim in router.iter_topic_provisionings()
    ):
        return "protected_provisioning"
    candidates = [
        (chat, wid)
        for uid, chat, tid, wid in router.iter_thread_bindings_with_chat()
        if uid == user_id
        and tid == thread_id
        and canonical_window_id(wid) == canonical_window_id(window_id)
        and (chat_id is None or chat == chat_id)
    ]
    if len(candidates) != 1 or candidates[0][0] is None:
        return "protected_active"
    if thread_id == 1:
        return "protected_general"
    exact_chat, exact_window = candidates[0]
    scoped_window = router.get_window_for_thread(user_id, thread_id, exact_chat)
    binding_chat = exact_chat if scoped_window is not None else None
    if router.get_window_for_thread(user_id, thread_id, binding_chat) != exact_window:
        return "protected_active"
    router.unbind_thread(
        user_id,
        thread_id,
        chat_id=binding_chat,
        retirement_reason="session_closed",
        cleanup_eligible=True,
    )
    session_manager.flush_state()
    retired = next(
        topic
        for topic in router.iter_retired_topics()
        if topic.user_id == user_id
        and topic.chat_id == exact_chat
        and topic.thread_id == thread_id
    )
    return await cleanup_retired_topic(
        client, retired, router=router, before_delete=before_delete
    )


def is_cleanup_candidate(topic: RetiredTopic, *, include_closed: bool = False) -> bool:
    """Include old closed records only in an explicit Sync cleanup."""
    return topic.thread_id != 1 and (
        topic.cleanup_eligible
        or (include_closed and topic.reason in _REVIEWABLE_CLOSED_REASONS)
    )


def _still_retired(topic: RetiredTopic, router: ThreadRouter) -> bool:
    retained = any(
        (
            current.user_id,
            current.chat_id,
            current.thread_id,
            current.sequence,
        )
        == (topic.user_id, topic.chat_id, topic.thread_id, topic.sequence)
        for current in router.iter_retired_topics()
    )
    return retained and not (
        router.has_active_topic(topic.chat_id, topic.thread_id)
        or router.has_topic_provisioning(topic.chat_id, topic.thread_id) is True
        or (
            topic.target_id is not None
            and router.has_target_provisioning(topic.target_id) is True
        )
    )


def _save_retry(
    topic: RetiredTopic,
    router: ThreadRouter,
    *,
    closed: bool,
    delay: float = _RETRY_SECONDS,
) -> None:
    router.update_retired_topic(
        topic,
        retry_at=time.time() + max(_RETRY_SECONDS, delay),
        closed=closed,
        cleanup_eligible=True,
    )
    session_manager.flush_state()


async def _close_fallback(
    client: TelegramClient, topic: RetiredTopic, router: ThreadRouter
) -> str:
    if router.has_topic_provisioning(topic.chat_id, topic.thread_id) is True:
        return "protected_provisioning"
    if (
        topic.target_id is not None
        and router.has_target_provisioning(topic.target_id) is True
    ):
        return "protected_provisioning"
    if not _still_retired(topic, router):
        return "protected_active"
    closed = topic.closed
    if not closed:
        try:
            closed = (
                await client.close_forum_topic(
                    topic.chat_id,
                    topic.thread_id,
                    rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS,
                )
                is not False
            )
        except RetryAfter as exc:
            _save_retry(topic, router, closed=False, delay=retry_after_seconds(exc))
            return "rate_limited"
        except TelegramError as exc:
            if is_thread_gone(exc):
                router.discard_retired_topic(topic)
                session_manager.flush_state()
                return "already_gone"
            closed = isinstance(exc, BadRequest) and exc.message.lower().replace(
                " ", "_"
            ) in {"topic_not_modified", "topic_closed"}
    _save_retry(topic, router, closed=closed)
    return "closed" if closed else "failed"


async def cleanup_retired_topic(  # noqa: C901, PLR0911
    client: TelegramClient,
    topic: RetiredTopic,
    *,
    router: ThreadRouter = thread_router,
    before_delete: Callable[[], Awaitable[None]] | None = None,
) -> str:
    """Attempt one exact retired record; close is never completed deletion."""
    if router.has_topic_provisioning(topic.chat_id, topic.thread_id) is True:
        return "protected_provisioning"
    if (
        topic.target_id is not None
        and router.has_target_provisioning(topic.target_id) is True
    ):
        return "protected_provisioning"
    if topic.thread_id == 1:
        return "protected_general"
    if not _still_retired(topic, router):
        return "protected_active"
    if topic.retry_at > time.time():
        return "deferred"
    if not router.begin_topic_deletion(topic):
        if router.has_topic_provisioning(topic.chat_id, topic.thread_id) is True:
            return "protected_provisioning"
        return "deferred"
    try:
        session_manager.flush_state()
        if before_delete is not None:
            await before_delete()
        try:
            deleted = await client.delete_forum_topic(
                topic.chat_id,
                topic.thread_id,
                rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS,
            )
        except RetryAfter as exc:
            _save_retry(
                topic, router, closed=topic.closed, delay=retry_after_seconds(exc)
            )
            return "rate_limited"
        except TelegramError as exc:
            if is_thread_gone(exc):
                router.discard_retired_topic(topic)
                session_manager.flush_state()
                return "already_gone"
            logger.warning(
                "Topic deletion failed; will retry. Check Delete Messages permission",
                chat_id=topic.chat_id,
                thread_id=topic.thread_id,
                error=str(exc),
            )
        else:
            if deleted is not False:
                router.discard_retired_topic(topic)
                session_manager.flush_state()
                return "deleted"
        return await _close_fallback(client, topic, router)
    finally:
        router.end_topic_deletion(topic)


async def cleanup_retired_topics(
    client: TelegramClient,
    *,
    router: ThreadRouter = thread_router,
    include_closed: bool = False,
    limit: int = 20,
    exclude_reasons: frozenset[str] | None = None,
) -> dict[str, int]:
    """Drain a bounded batch; automatic sweeps never adopt retained history."""
    outcomes: Counter[str] = Counter()
    attempts = 0
    topics = sorted(
        router.iter_retired_topics(), key=lambda topic: (topic.retry_at, topic.sequence)
    )
    for topic in topics:
        if exclude_reasons and topic.reason in exclude_reasons:
            continue
        if not is_cleanup_candidate(topic, include_closed=include_closed):
            continue
        if attempts >= limit:
            break
        outcome = await cleanup_retired_topic(client, topic, router=router)
        outcomes[outcome] += 1
        if outcome not in {
            "deferred",
            "protected_active",
            "protected_general",
            "protected_provisioning",
        }:
            attempts += 1
        if outcome == "rate_limited":
            break
    return dict(outcomes)
