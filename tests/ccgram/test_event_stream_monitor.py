"""Unit tests for the neutral push event-stream consumer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.event_stream_monitor import EventStreamMonitor
from ccgram.multiplexer import agent_status_cache
from ccgram.multiplexer.base import AgentStatus, MuxEvent


async def test_dispatch_agent_status_updates_cache() -> None:
    agent_status_cache.reset()
    notify = AsyncMock()
    monitor = EventStreamMonitor(set, on_window_died=notify)
    status = AgentStatus("working", "codex", "compiling")
    await monitor._dispatch(MuxEvent("agent_status", "w2:t1", "w2:p1", status))
    assert agent_status_cache.get_status("w2:t1") == status
    notify.assert_not_awaited()


async def test_dispatch_window_died_clears_cache_before_awaiting_callback() -> None:
    agent_status_cache.reset()
    agent_status_cache.set_status("w2:t1", AgentStatus("working"))

    async def notify(window_id: str) -> None:
        assert agent_status_cache.get_status(window_id) is None

    callback = AsyncMock(side_effect=notify)
    monitor = EventStreamMonitor(lambda: {"w2:t1"}, on_window_died=callback)
    await monitor._dispatch(MuxEvent("window_died", "w2:t1"))

    callback.assert_awaited_once_with("w2:t1")


async def test_failed_death_callback_does_not_drop_later_events() -> None:
    agent_status_cache.reset()
    status = AgentStatus("idle")

    async def events(_ids):
        yield MuxEvent("window_died", "dead")
        yield MuxEvent("agent_status", "live", status=status)

    callback = AsyncMock(side_effect=RuntimeError("transport unavailable"))
    monitor = EventStreamMonitor(lambda: {"dead", "live"}, on_window_died=callback)
    with patch("ccgram.event_stream_monitor.mux") as mux:
        mux.watch_events = events
        await monitor._consume({"dead", "live"})

    callback.assert_awaited_once_with("dead")
    assert agent_status_cache.get_status("live") == status


async def test_stop_and_wait_cancels_inflight_death_callback() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def events(_ids):
        yield MuxEvent("window_died", "target")

    async def notify(_window_id: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monitor = EventStreamMonitor(lambda: {"target"}, on_window_died=notify)
    with patch("ccgram.event_stream_monitor.mux") as mux:
        mux.watch_events = events
        monitor.start()
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
        finally:
            await monitor.stop_and_wait()

    assert cancelled.is_set()
    assert monitor._task is None


async def test_supervisor_backs_off_after_normal_stream_eof(monkeypatch) -> None:
    starts = 0

    async def eof_watch(_window_ids):
        nonlocal starts
        starts += 1
        if False:  # pragma: no cover - async generator marker
            yield

    monkeypatch.setattr("ccgram.event_stream_monitor._SET_POLL_INTERVAL", 0.02)
    fake_mux = MagicMock()
    fake_mux.watch_events = eof_watch
    monkeypatch.setattr("ccgram.event_stream_monitor.mux", fake_mux)
    monitor = EventStreamMonitor(lambda: {"target"}, on_window_died=AsyncMock())
    monitor.start()
    try:
        await asyncio.sleep(0.055)
    finally:
        await monitor.stop_and_wait()

    assert 1 <= starts <= 3


async def test_supervisor_restarts_watch_on_set_change(monkeypatch) -> None:
    agent_status_cache.reset()
    started: list[list[str]] = []

    async def fake_watch(window_ids):
        started.append(list(window_ids))
        await asyncio.Event().wait()
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr("ccgram.event_stream_monitor._SET_POLL_INTERVAL", 0.01)
    fake_mux = MagicMock()
    fake_mux.watch_events = fake_watch
    monkeypatch.setattr("ccgram.event_stream_monitor.mux", fake_mux)

    holder = {"ids": {"w2:t1"}}
    monitor = EventStreamMonitor(lambda: holder["ids"], on_window_died=AsyncMock())
    monitor.start()
    try:
        await asyncio.sleep(0.05)
        holder["ids"] = {"w2:t1", "w3:t1"}
        await asyncio.sleep(0.05)
    finally:
        await monitor.stop_and_wait()

    assert ["w2:t1"] in started
    assert sorted(["w2:t1", "w3:t1"]) in [sorted(s) for s in started]
