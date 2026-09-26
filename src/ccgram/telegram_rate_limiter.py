"""Telegram rate limiting with quiet, incremental RetryAfter backoff.

Interactive priority: a per-group scheduler serves user-tap UI requests
ahead of background traffic WITHIN the same flood budget. The scheduler
wraps one aiolimiter token bucket per group at the unchanged group rate
and only reorders waiters, so no additional request is ever sent and no
Telegram limit is bypassed.
"""

import asyncio
import contextlib
import contextvars
import random
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any

import structlog
from telegram.error import RetryAfter
from telegram.ext import AIORateLimiter

logger = structlog.get_logger()

# PTB drops falsy per-request rate_limit_args before invoking the limiter.
# A negative truthy sentinel survives ExtBot transport and is intercepted here.
NO_RETRY_RATE_LIMIT_ARGS = -1

_interactive_request: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ccgram_interactive_request", default=False
)


@contextlib.contextmanager
def interactive_priority() -> Any:
    """Mark Telegram requests made inside this block as user-interactive.

    Latency-sensitive UI work (a tap on the directory browser, a picker
    page turn) runs inside this context; the group scheduler then serves
    those requests ahead of queued background sends. The flood budget,
    the group ceiling, and RetryAfter handling are unchanged: priority
    changes the order of service, never the rate.
    """
    token = _interactive_request.set(True)
    try:
        yield
    finally:
        _interactive_request.reset(token)


_RETRY_BACKOFF_BASE_SECONDS = 1.0
_MAX_RETRY_BACKOFF_SECONDS = 8.0
_RETRY_JITTER_MAX_SECONDS = 1.0


def retry_after_seconds(exc: RetryAfter) -> float:
    """Return PTB's normalized delay without its deprecated public shim."""
    return exc._retry_after.total_seconds()  # pyright: ignore[reportPrivateUsage]


class _PriorityGroupScheduler:
    """Serve waiters at the wrapped bucket's rate, interactive ones first.

    Wraps one aiolimiter ``AsyncLimiter`` (the group's token bucket,
    unchanged ceiling): a pump task acquires each token THROUGH that
    limiter and then resolves the first interactive waiter, falling
    back to the FIFO background queue. Because every release still
    consumes exactly one underlying token, the flood budget is
    untouched; only the order of service changes. Cancellation-safe:
    a waiter that went away is skipped without extra token cost beyond
    the one already spent.
    """

    def __init__(self, limiter: Any) -> None:
        self._limiter = limiter
        self._interactive: deque[Any] = deque()
        self._background: deque[Any] = deque()
        self._pump: asyncio.Task[None] | None = None

    # aiolimiter-compatible surface so PTB's pruning logic (which reads
    # has_capacity/max_rate) keeps working if this object is stored in
    # the inherited registry.
    @property
    def max_rate(self) -> float:
        return self._limiter.max_rate

    def has_capacity(self, *args: Any) -> bool:
        return self._limiter.has_capacity(*args)

    def _ensure_pump(self) -> None:
        if self._pump is None or self._pump.done():
            self._pump = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while self._interactive or self._background:
            await self._limiter.acquire()
            queue = self._interactive if self._interactive else self._background
            waiter = queue.popleft()
            if not waiter.done():
                waiter.set_result(None)
            else:
                # A cancelled waiter consumed this token; give it back to
                # the next in line by re-acquiring immediately is not
                # possible with aiolimiter, so the slot is simply spent.
                continue

    async def acquire(self, *, interactive: bool) -> None:
        self._ensure_pump()
        waiter = asyncio.get_running_loop().create_future()
        (self._interactive if interactive else self._background).append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            if not waiter.done():
                # The pump never saw us; nothing to clean.
                pass
            raise


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

    _priority_schedulers: dict[int | str, _PriorityGroupScheduler]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._priority_schedulers = {}
        # Own overall gate, decoupled from PTB privates: same default
        # ceiling (30/s) the base class applies.
        # Lazy: aiolimiter import kept local for parity with the group gate
        from aiolimiter import AsyncLimiter

        self._overall_gate = AsyncLimiter(max_rate=30, time_period=1)

    def _interactive_group_acquire(self, group: int | str) -> Any:
        """Priority-aware group gate, replacing the base FIFO limiter.

        The wrapped bucket is created with the base class's exact
        parameters (same max_rate, same time_period), so the group
        ceiling is preserved bit for bit; only waiter order changes.
        """
        scheduler = self._priority_schedulers.get(group)
        if scheduler is None:
            # Lazy: aiolimiter only needed when a group first appears
            from aiolimiter import AsyncLimiter

            scheduler = _PriorityGroupScheduler(
                AsyncLimiter(
                    max_rate=self._group_max_rate,
                    time_period=self._group_time_period,
                )
            )
            self._priority_schedulers[group] = scheduler
        return scheduler

    async def _run_request(
        self,
        chat: bool,
        group: int | str | bool,
        allow_paid_broadcast: bool,  # noqa: ARG002  # base-class signature
        callback: Callable[..., Coroutine[Any, Any, Any]],
        args: Any,
        kwargs: dict[str, Any],
        *,
        interactive: bool = False,
    ) -> Any:
        # Same gate order as the base class (group, then overall), with
        # the group side served by the priority scheduler.
        if group and self._group_max_rate and self._group_time_period:
            scheduler = self._interactive_group_acquire(group)  # type: ignore[arg-type]
            await scheduler.acquire(interactive=interactive)
        if chat:
            async with self._overall_gate:
                return await callback(*args, **kwargs)
        return await callback(*args, **kwargs)

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

        interactive = _interactive_request.get()

        async def run_request() -> Any:
            return await self._run_request(
                chat=chat_id is not None,
                group=group,
                allow_paid_broadcast=data.get("allow_paid_broadcast", False),
                callback=callback,
                args=args,
                kwargs=kwargs,
                interactive=interactive,
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
