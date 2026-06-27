"""Unit tests for the push event-stream consumer (EventStreamMonitor)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.event_stream_monitor import EventStreamMonitor
from ccgram.multiplexer import agent_status_cache
from ccgram.multiplexer.base import AgentStatus, MuxEvent


async def test_dispatch_agent_status_updates_cache() -> None:
    agent_status_cache.reset()
    monitor = EventStreamMonitor(MagicMock(), set)
    status = AgentStatus("working", "codex", "compiling")
    await monitor._dispatch(MuxEvent("agent_status", "w2:t1", "w2:p1", status))
    assert agent_status_cache.get_status("w2:t1") == status


async def test_dispatch_window_died_clears_cache_and_notifies_bound_users() -> None:
    agent_status_cache.reset()
    agent_status_cache.set_status("w2:t1", AgentStatus("working"))
    monitor = EventStreamMonitor(MagicMock(), lambda: {"w2:t1"})

    notify = AsyncMock()
    fake_router = MagicMock()
    fake_router.iter_thread_bindings.return_value = [
        (7, 42, "w2:t1"),
        (7, 99, "w9:t9"),  # different window — must not be notified
    ]
    with (
        patch(
            "ccgram.handlers.polling.window_tick.apply._handle_dead_window_notification",
            notify,
        ),
        patch("ccgram.thread_router.thread_router", fake_router),
        patch("ccgram.event_stream_monitor.unwrap_bot", return_value="BOT"),
    ):
        await monitor._dispatch(MuxEvent("window_died", "w2:t1"))

    assert agent_status_cache.get_status("w2:t1") is None
    notify.assert_awaited_once_with("BOT", 7, 42, "w2:t1")


async def test_supervisor_restarts_watch_on_set_change(monkeypatch) -> None:
    agent_status_cache.reset()
    started: list[list[str]] = []

    async def fake_watch(window_ids):
        started.append(list(window_ids))
        await asyncio.Event().wait()  # block until cancelled (live stream)
        yield  # pragma: no cover — unreachable; makes this an async generator

    monkeypatch.setattr("ccgram.event_stream_monitor._SET_POLL_INTERVAL", 0.01)
    fake_mux = MagicMock()
    fake_mux.watch_events = fake_watch
    monkeypatch.setattr("ccgram.event_stream_monitor.mux", fake_mux)

    holder = {"ids": {"w2:t1"}}
    monitor = EventStreamMonitor(MagicMock(), lambda: holder["ids"])
    monitor.start()
    try:
        await asyncio.sleep(0.05)  # start watch with {w2:t1}
        holder["ids"] = {"w2:t1", "w3:t1"}  # change the bound set
        await asyncio.sleep(0.05)  # supervisor detects + restarts
    finally:
        monitor.stop()
        await asyncio.sleep(0.02)

    assert ["w2:t1"] in started
    assert sorted(["w2:t1", "w3:t1"]) in [sorted(s) for s in started]
