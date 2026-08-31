"""Telegram rate limiting with quiet, incremental RetryAfter backoff."""

import asyncio
import contextlib
import random
from collections.abc import Callable, Coroutine
from typing import Any

import structlog
from telegram.error import RetryAfter
from telegram.ext import AIORateLimiter

logger = structlog.get_logger()

# PTB drops falsy per-request rate_limit_args before invoking the limiter.
# A negative truthy sentinel survives ExtBot transport and is intercepted here.
NO_RETRY_RATE_LIMIT_ARGS = -1

_RETRY_BACKOFF_BASE_SECONDS = 1.0
_MAX_RETRY_BACKOFF_SECONDS = 8.0
_RETRY_JITTER_MAX_SECONDS = 1.0


def retry_after_seconds(exc: RetryAfter) -> float:
    """Return PTB 22.6's normalized delay without its deprecated public shim."""
    return exc._retry_after.total_seconds()  # pyright: ignore[reportPrivateUsage]


class CCGramAIORateLimiter(AIORateLimiter):
    """Apply PTB throttling without logging expected flood control as crashes.

    PTB's reference limiter logs ``RetryAfter`` with ``logger.exception`` when
    its retry budget is exhausted. ccgram owns that loop instead: each limit hit
    is a concise warning, retries wait for Telegram's required delay plus
    bounded exponential backoff and jitter, and exhaustion propagates so the queue or endpoint
    caller can apply its own deferred retry without a limiter traceback.

    Retry waits stay local to the limited request. PTB's proactive overall and
    group limiters protect shared budgets; a reactive global gate would stall
    unrelated chats and concurrent requests could release it prematurely.

    Non-positive ``rate_limit_args`` values propagate the first response so
    probes with endpoint-specific backoff do not stall all Telegram requests.
    """

    async def process_request(
        self,
        callback: Callable[..., Coroutine[Any, Any, Any]],
        args: Any,
        kwargs: dict[str, Any],
        endpoint: str,
        data: dict[str, Any],
        rate_limit_args: int | None,
    ) -> Any:
        chat_id = data.get("chat_id")
        if chat_id is not None:
            with contextlib.suppress(TypeError, ValueError):
                chat_id = int(chat_id)
        group: int | str | bool = False
        if (isinstance(chat_id, int) and chat_id < 0) or isinstance(chat_id, str):
            group = chat_id

        async def run_request() -> Any:
            return await self._run_request(
                chat=chat_id is not None,
                group=group,
                allow_paid_broadcast=data.get("allow_paid_broadcast", False),
                callback=callback,
                args=args,
                kwargs=kwargs,
            )

        if rate_limit_args is not None and rate_limit_args <= 0:
            return await run_request()

        max_retries = self._max_retries if rate_limit_args is None else rate_limit_args
        for retry in range(max_retries + 1):
            try:
                return await run_request()
            except RetryAfter as exc:
                retry_after = retry_after_seconds(exc)
                if retry == max_retries:
                    logger.warning(
                        "Telegram rate limit persisted; returning request to caller",
                        endpoint=endpoint,
                        attempts=retry + 1,
                        retry_after_seconds=retry_after,
                    )
                    raise

                backoff = min(
                    _MAX_RETRY_BACKOFF_SECONDS,
                    _RETRY_BACKOFF_BASE_SECONDS * (2**retry),
                )
                jitter = random.uniform(0, _RETRY_JITTER_MAX_SECONDS)
                retry_in = retry_after + backoff + jitter
                logger.warning(
                    "Telegram rate limited; retrying later",
                    endpoint=endpoint,
                    retry=retry + 1,
                    max_retries=max_retries,
                    retry_after_seconds=retry_after,
                    backoff_seconds=backoff,
                    jitter_seconds=jitter,
                    retry_in_seconds=retry_in,
                )
                await asyncio.sleep(retry_in)

        raise AssertionError("unreachable")
