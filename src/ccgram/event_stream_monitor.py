"""Persistent consumer of the multiplexer push event stream (herdr).

Bridges the backend event stream (``capabilities.supports_event_stream``) into
ccgram's existing flows — an *augment*, not a replacement (JSONL polling still
owns transcript content):

- **agent status** pushes update the neutral ``agent_status_cache``, which
  Telegram status polling and shared-core bulk refreshes read before probing.
- **window death** pushes clear cached status, then await the callback supplied
  by bootstrap. Recipient routing and destructive topic policy belong to that
  adapter, not the stream consumer.

A supervisor task runs one ``watch_events`` iteration over the current bound
window set; herdr cannot add subscriptions to a live connection, so when the
bound set changes the supervisor cancels the iteration and restarts with the new
set. tmux has no event stream, so bootstrap only starts this on event-stream
backends.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

import structlog

from .multiplexer import agent_status_cache
from .multiplexer import multiplexer as mux
from .multiplexer.base import MuxEvent

logger = structlog.get_logger()

# How often the supervisor re-checks the bound set and the consume task health.
_SET_POLL_INTERVAL = 2.0

WindowIdsProvider = Callable[[], set[str]]


class EventStreamMonitor:
    """Supervises a herdr event-stream subscription over the bound window set."""

    def __init__(
        self,
        list_window_ids: WindowIdsProvider,
        *,
        on_window_died: Callable[[str], Awaitable[None]],
    ) -> None:
        self._list_window_ids = list_window_ids
        self._on_window_died = on_window_died
        self._task: asyncio.Task[None] | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._supervise())

    def stop(self) -> None:
        """Request consumer cancellation; use ``stop_and_wait`` before drain."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
        agent_status_cache.reset()

    async def stop_and_wait(self) -> None:
        """Cancel and await the event producer so it cannot enqueue after drain."""
        self.stop()
        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if self._task is task:
                self._task = None

    async def _supervise(self) -> None:
        """Run the stream over the current set; restart when the set changes."""
        while self._running:
            ids = self._list_window_ids()
            if not ids:
                await asyncio.sleep(_SET_POLL_INTERVAL)
                continue
            try:
                stream_ended = await self._run_until_set_change(ids)
                if stream_ended:
                    # A backend may end a stream normally (EOF). Treat that as
                    # a reconnect failure, not a successful tight-loop restart.
                    await asyncio.sleep(_SET_POLL_INTERVAL)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let the supervisor die  # noqa: BLE001
                logger.warning("event stream supervisor error: %s", exc)
                await asyncio.sleep(_SET_POLL_INTERVAL)

    async def _run_until_set_change(self, ids: set[str]) -> bool:
        """Consume ``watch_events(ids)`` until the bound set changes or we stop.

        Re-raises an unexpected ``_consume`` failure so ``_supervise`` logs it and
        backs off, instead of hot-looping a re-subscribe on a persistent error.
        """
        consume = asyncio.create_task(self._consume(ids))
        try:
            while self._running and not consume.done():
                if self._list_window_ids() != ids:
                    return False
                await asyncio.sleep(_SET_POLL_INTERVAL)
        finally:
            if not consume.done():
                consume.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await consume
        # Loop exited because consume finished on its own — watch_events only ends
        # on an unexpected error (it reconnects internally), so surface it.
        if not consume.cancelled() and (exc := consume.exception()) is not None:
            raise exc
        return True

    async def _consume(self, ids: set[str]) -> None:
        # aclosing → the watch_events generator (and its socket) is closed
        # deterministically when this task is cancelled, not at GC time.
        async with contextlib.aclosing(mux.watch_events(sorted(ids))) as stream:
            async for event in stream:
                try:
                    await self._dispatch(event)
                except Exception as exc:  # noqa: BLE001 — one bad event must not kill the stream
                    logger.warning("event-stream dispatch failed: %s", exc)

    async def _dispatch(self, event: MuxEvent) -> None:
        if event.kind == "agent_status" and event.status is not None:
            agent_status_cache.set_status(event.window_id, event.status)
        elif event.kind == "window_died":
            agent_status_cache.clear(event.window_id)
            await self._on_window_died(event.window_id)


# Module-level active instance (for bootstrap shutdown), mirroring
# session_monitor.set_active_monitor / get_active_monitor.
_active: EventStreamMonitor | None = None


def set_active_event_stream(monitor: EventStreamMonitor | None) -> None:
    global _active
    _active = monitor


def get_active_event_stream() -> EventStreamMonitor | None:
    return _active
