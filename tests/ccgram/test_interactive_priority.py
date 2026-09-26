"""Interactive priority lane: order of service, never rate."""

from __future__ import annotations

import asyncio
from ccgram.telegram_rate_limiter import (
    CCGramAIORateLimiter,
    _PriorityGroupScheduler,
    interactive_priority,
)


class _FakeLimiter:
    """Deterministic bucket: one token every `interval` seconds."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.acquires = 0
        self.max_rate = 20
        self._gate = asyncio.Semaphore(0)
        self._pump_task: asyncio.Task[None] | None = None

    def has_capacity(self, *args: object) -> bool:
        return True

    async def acquire(self) -> None:
        self.acquires += 1
        await asyncio.sleep(self.interval)


class TestPriorityGroupScheduler:
    async def test_interactive_jumps_the_background_queue(self) -> None:
        sched = _PriorityGroupScheduler(_FakeLimiter(interval=0.05))
        order: list[str] = []

        async def waiter(name: str) -> None:
            await sched.acquire(interactive=False)
            order.append(name)

        tasks = [asyncio.create_task(waiter(f"bg{i}")) for i in range(5)]
        await asyncio.sleep(0.01)
        interactive = asyncio.create_task(_interactive(sched, order))
        await asyncio.gather(*tasks, interactive)
        # The interactive waiter must not be last: it lands before the
        # background waiters queued ahead of it.
        assert order[-1] != "interactive", order
        assert "interactive" in order[:3], order

    async def test_every_release_consumes_one_underlying_token(self) -> None:
        limiter = _FakeLimiter(interval=0.01)
        sched = _PriorityGroupScheduler(limiter)
        for i in range(7):
            asyncio.create_task(_bg(sched))
        asyncio.create_task(_interactive(sched, []))
        await asyncio.sleep(0.5)
        assert limiter.acquires >= 8

    async def test_cancelled_waiter_does_not_deadlock(self) -> None:
        sched = _PriorityGroupScheduler(_FakeLimiter(interval=0.01))
        t = asyncio.create_task(_bg(sched))
        await asyncio.sleep(0.001)
        t.cancel()
        with contextlib_suppress():
            await t
        done = asyncio.create_task(_bg(sched))
        await asyncio.wait_for(done, timeout=1.0)


async def _bg(sched: _PriorityGroupScheduler) -> None:
    await sched.acquire(interactive=False)


async def _interactive(sched: _PriorityGroupScheduler, order: list[str]) -> None:
    await sched.acquire(interactive=True)
    order.append("interactive")


def contextlib_suppress():
    import contextlib

    return contextlib.suppress(asyncio.CancelledError)


class TestLimiterIntegration:
    async def test_process_request_honors_the_marker(self) -> None:
        lim = CCGramAIORateLimiter(max_retries=0)
        calls: list[bool] = []

        async def spy_acquire(*, interactive: bool) -> None:
            calls.append(interactive)

        lim._interactive_group_acquire = lambda group: _SpyScheduler(spy_acquire)  # type: ignore[assignment]

        async def callback() -> dict:
            return {}

        data = {"chat_id": -100123}
        await lim.process_request(callback, (), {}, "editMessageText", data, None)
        assert calls == [False]
        with interactive_priority():
            await lim.process_request(callback, (), {}, "editMessageText", data, None)
        assert calls == [False, True]


class _SpyScheduler:
    def __init__(self, spy) -> None:
        self._spy = spy

    async def acquire(self, *, interactive: bool) -> None:
        await self._spy(interactive=interactive)
