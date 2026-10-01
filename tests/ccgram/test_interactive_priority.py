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
        tasks = [asyncio.create_task(_bg(sched)) for _ in range(7)]
        tasks.append(asyncio.create_task(_interactive(sched, [])))
        await asyncio.gather(*tasks)
        # Every request released exactly one acquired token: no token-free
        # releases and no double spending.
        assert limiter.acquires == 8

    async def test_interactive_burst_cannot_starve_background(self) -> None:
        from ccgram.telegram_rate_limiter import _INTERACTIVE_BURST_LIMIT

        sched = _PriorityGroupScheduler(_FakeLimiter(interval=0.01))
        order: list[str] = []

        async def background() -> None:
            await sched.acquire(interactive=False)
            order.append("bg")

        bg = asyncio.create_task(background())
        await asyncio.sleep(0)
        taps = [
            asyncio.create_task(_interactive(sched, order))
            for _ in range(_INTERACTIVE_BURST_LIMIT + 3)
        ]
        await asyncio.gather(bg, *taps)
        # Background was queued first, then a tap flood arrived: it is
        # served after one full interactive burst, not after every tap.
        assert (
            order
            == ["interactive"] * _INTERACTIVE_BURST_LIMIT + ["bg"] + ["interactive"] * 3
        )

    async def test_cancelled_waiter_does_not_deadlock(self) -> None:
        sched = _PriorityGroupScheduler(_FakeLimiter(interval=0.01))
        t = asyncio.create_task(_bg(sched))
        await asyncio.sleep(0.001)
        t.cancel()
        with contextlib_suppress():
            await t
        done = asyncio.create_task(_bg(sched))
        await asyncio.wait_for(done, timeout=1.0)

    async def test_cancelled_waiter_burst_leaves_no_corpses(self) -> None:
        # 2026-09-26 wedge: cancelled futures left in the deque made the
        # pump spend one token per corpse, freezing the background lane
        # while the interactive lane flowed. Cancellation must remove the
        # waiter, so later background service waits ~one interval, not
        # one interval per cancelled waiter.
        interval = 0.05
        sched = _PriorityGroupScheduler(_FakeLimiter(interval=interval))
        parked = [asyncio.create_task(_bg(sched)) for _ in range(40)]
        await asyncio.sleep(0.01)
        for t in parked:
            t.cancel()
        await asyncio.gather(*parked, return_exceptions=True)
        assert not sched._background and not sched._interactive
        # Total cancellation while the pump awaits its token must let the
        # pump exit cleanly, not die on popleft of an emptied deque.
        await asyncio.sleep(interval * 3)
        if sched._pump is not None and sched._pump.done():
            assert sched._pump.exception() is None
        fresh = asyncio.create_task(_bg(sched))
        # Old code: 40 corpses x 0.05s = 2s drain before `fresh`. New
        # code: roughly one interval; 1.0s leaves ample CI slack.
        await asyncio.wait_for(fresh, timeout=1.0)


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

    async def test_configured_overall_limiter_is_honored(self) -> None:
        lim = CCGramAIORateLimiter(max_retries=0)
        entered: list[str] = []

        class _SpyGate:
            async def __aenter__(self) -> None:
                entered.append("enter")

            async def __aexit__(self, *exc: object) -> None:
                entered.append("exit")

        lim._base_limiter = _SpyGate()  # type: ignore[assignment]

        async def callback() -> str:
            return "ok"

        result = await lim.process_request(
            callback, (), {}, "sendMessage", {"chat_id": 123}, None
        )
        assert result == "ok"
        assert entered == ["enter", "exit"]

    async def test_disabled_overall_limiter_stays_disabled(self) -> None:
        lim = CCGramAIORateLimiter(overall_max_rate=0, max_retries=0)
        assert lim._base_limiter is None

        async def callback() -> str:
            return "ok"

        result = await lim.process_request(
            callback, (), {}, "sendMessage", {"chat_id": 123}, None
        )
        assert result == "ok"


class _SpyScheduler:
    def __init__(self, spy) -> None:
        self._spy = spy

    async def acquire(self, *, interactive: bool) -> None:
        await self._spy(interactive=interactive)
