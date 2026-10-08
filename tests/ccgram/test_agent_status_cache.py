"""Unit tests for the backend-neutral push-updated agent-status cache."""

from __future__ import annotations

import asyncio
import logging

import pytest

from ccgram.multiplexer import agent_status_cache
from ccgram.multiplexer.base import AgentStatus


@pytest.fixture(autouse=True)
def _empty_cache():
    agent_status_cache.reset()
    yield
    agent_status_cache.reset()


def test_cold_window_has_no_status() -> None:
    assert agent_status_cache.get_status("w2:t1") is None


def test_set_status_is_readable_and_scoped_to_its_window() -> None:
    working = AgentStatus("working", "codex", "compiling")
    agent_status_cache.set_status("w2:t1", working)

    assert agent_status_cache.get_status("w2:t1") == working
    assert agent_status_cache.get_status("w3:t1") is None


def test_set_status_overwrites_the_previous_value() -> None:
    agent_status_cache.set_status("w2:t1", AgentStatus("working"))
    agent_status_cache.set_status("w2:t1", AgentStatus("idle"))

    assert agent_status_cache.get_status("w2:t1") == AgentStatus("idle")


def test_clear_drops_only_the_named_window() -> None:
    agent_status_cache.set_status("w2:t1", AgentStatus("working"))
    agent_status_cache.set_status("w3:t1", AgentStatus("idle"))

    agent_status_cache.clear("w2:t1")

    assert agent_status_cache.get_status("w2:t1") is None
    assert agent_status_cache.get_status("w3:t1") == AgentStatus("idle")


def test_clear_of_a_cold_window_is_a_no_op() -> None:
    agent_status_cache.clear("never-seen")


def test_reset_empties_the_whole_cache() -> None:
    agent_status_cache.set_status("a", AgentStatus("working"))
    agent_status_cache.set_status("b", AgentStatus("idle"))

    agent_status_cache.reset()

    assert agent_status_cache.get_status("a") is None
    assert agent_status_cache.get_status("b") is None


@pytest.mark.parametrize("source", ["push", "probe-positive", "probe-negative"])
async def test_writes_sweep_expired_entries_from_unvisited_windows(
    source: str, monkeypatch
) -> None:
    now = [0.0]
    monkeypatch.setattr(agent_status_cache, "_clock", lambda: now[0])
    status = None if source == "probe-negative" else AgentStatus("working")

    async def probe() -> AgentStatus | None:
        return status

    for index in range(6):
        now[0] = index * 91.0
        window_id = f"unbound-{index}"
        if source == "push":
            agent_status_cache.set_status(window_id, AgentStatus("working"))
        else:
            assert (
                await agent_status_cache.get_status_or_probe(window_id, probe) == status
            )
        assert set(agent_status_cache._cache) == {window_id}


async def test_generation_metadata_waits_for_all_invalidated_probes() -> None:
    old_started = asyncio.Event()
    new_started = asyncio.Event()
    release_old = asyncio.Event()
    release_new = asyncio.Event()
    window_id = "rebound-window"

    async def old_probe() -> AgentStatus:
        old_started.set()
        await release_old.wait()
        return AgentStatus("working", custom_status="stale")

    async def new_probe() -> AgentStatus:
        new_started.set()
        await release_new.wait()
        return AgentStatus("idle")

    get_status = agent_status_cache.get_status_or_probe
    old_call = asyncio.create_task(get_status(window_id, old_probe))
    await old_started.wait()
    state = agent_status_cache._window_generations[window_id]

    agent_status_cache.clear(window_id)
    new_call = asyncio.create_task(get_status(window_id, new_probe))
    await new_started.wait()
    assert agent_status_cache._window_generations[window_id] is state

    release_old.set()
    assert await old_call is None
    assert agent_status_cache._window_generations[window_id] is state

    release_new.set()
    assert await new_call == AgentStatus("idle")
    assert window_id not in agent_status_cache._window_generations


async def test_generation_metadata_does_not_accumulate_for_churned_windows() -> None:
    get_status = agent_status_cache.get_status_or_probe

    for index in range(100):
        window_id = f"churn-{index}"
        agent_status_cache.set_status(window_id, AgentStatus("working"))
        agent_status_cache.clear(window_id)

        async def probe() -> AgentStatus:
            return AgentStatus("idle")

        assert await get_status(window_id, probe) == AgentStatus("idle")

    assert agent_status_cache._window_generations == {}


async def test_successful_probe_warms_cache_for_next_call() -> None:
    probe_calls = 0
    working = AgentStatus("working", "codex", "compiling")

    async def probe() -> AgentStatus:
        nonlocal probe_calls
        probe_calls += 1
        return working

    get_status = agent_status_cache.get_status_or_probe
    assert await get_status("w2:t1", probe) == working
    assert await get_status("w2:t1", probe) == working
    assert probe_calls == 1
    assert agent_status_cache.get_status("w2:t1") == working


@pytest.mark.parametrize("entry_source", ["push", "probe"])
async def test_positive_status_expires_after_90_seconds(
    entry_source: str, monkeypatch
) -> None:
    now = [0.0]
    monkeypatch.setattr(agent_status_cache, "_clock", lambda: now[0], raising=False)
    probe_calls = 0
    working = AgentStatus("working", "codex", "compiling")
    idle = AgentStatus("idle", "codex")

    async def probe() -> AgentStatus:
        nonlocal probe_calls
        probe_calls += 1
        if entry_source == "probe" and probe_calls == 1:
            return working
        return idle

    get_status = agent_status_cache.get_status_or_probe
    if entry_source == "push":
        agent_status_cache.set_status("positive-ttl", working)
    else:
        assert await get_status("positive-ttl", probe) == working

    now[0] = 90.001
    assert await get_status("positive-ttl", probe) == idle
    assert probe_calls == (1 if entry_source == "push" else 2)
    assert agent_status_cache.get_status("positive-ttl") == idle

    assert await get_status("positive-ttl", probe) == idle
    assert probe_calls == (1 if entry_source == "push" else 2)


async def test_each_push_refreshes_positive_ttl(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(agent_status_cache, "_clock", lambda: now[0], raising=False)
    working = AgentStatus("working", "codex")
    idle = AgentStatus("idle", "codex")
    probe_calls = 0

    async def probe() -> AgentStatus:
        nonlocal probe_calls
        probe_calls += 1
        return idle

    agent_status_cache.set_status("refreshed-push", working)
    now[0] = 80.0
    agent_status_cache.set_status("refreshed-push", working)

    now[0] = 90.001
    assert (
        await agent_status_cache.get_status_or_probe("refreshed-push", probe) == working
    )
    assert probe_calls == 0

    now[0] = 170.0
    assert await agent_status_cache.get_status_or_probe("refreshed-push", probe) == idle
    assert probe_calls == 1


async def test_negative_probe_result_is_cached_for_15_seconds(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(agent_status_cache, "_clock", lambda: now[0], raising=False)
    probe_calls = 0

    async def probe() -> AgentStatus | None:
        nonlocal probe_calls
        probe_calls += 1
        return None

    get_status = agent_status_cache.get_status_or_probe
    assert await get_status("w2:t1", probe) is None

    now[0] = 14.999
    assert await get_status("w2:t1", probe) is None
    assert probe_calls == 1

    now[0] = 15.0
    assert await get_status("w2:t1", probe) is None
    assert probe_calls == 2


async def test_push_during_probe_wins_for_current_caller() -> None:
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()
    stale = AgentStatus("idle", "codex")
    pushed = AgentStatus("working", "codex", "new push")

    async def probe() -> AgentStatus:
        probe_started.set()
        await release_probe.wait()
        return stale

    get_status = agent_status_cache.get_status_or_probe
    pending = asyncio.create_task(get_status("w2:t1", probe))
    await probe_started.wait()
    agent_status_cache.set_status("w2:t1", pushed)
    release_probe.set()

    assert await pending == pushed


async def test_coalesced_waiters_both_receive_push_during_probe() -> None:
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()
    second_waiter_started = asyncio.Event()
    probe_calls = 0
    stale = AgentStatus("idle", "codex")
    pushed = AgentStatus("working", "codex", "new push")
    window_id = "coalesced-push"

    async def probe() -> AgentStatus:
        nonlocal probe_calls
        probe_calls += 1
        probe_started.set()
        await release_probe.wait()
        return stale

    async def unexpected_probe() -> AgentStatus | None:
        raise AssertionError("coalesced waiter should share the first probe")

    get_status = agent_status_cache.get_status_or_probe
    first = asyncio.create_task(get_status(window_id, probe))
    await probe_started.wait()

    async def second_waiter() -> AgentStatus | None:
        second_waiter_started.set()
        return await get_status(window_id, unexpected_probe)

    second = asyncio.create_task(second_waiter())
    await second_waiter_started.wait()
    agent_status_cache.set_status(window_id, pushed)
    release_probe.set()

    assert await asyncio.gather(first, second) == [pushed, pushed]
    assert probe_calls == 1
    assert window_id not in agent_status_cache._window_generations


async def test_coalesced_successful_probes_warm_cache_for_both_waiters() -> None:
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()
    second_waiter_started = asyncio.Event()
    probe_calls = 0
    working = AgentStatus("working", "codex", "compiling")
    window_id = "coalesced-success"

    async def probe() -> AgentStatus:
        nonlocal probe_calls
        probe_calls += 1
        probe_started.set()
        await release_probe.wait()
        return working

    async def unexpected_probe() -> AgentStatus | None:
        raise AssertionError("coalesced waiter should share the successful probe")

    get_status = agent_status_cache.get_status_or_probe
    first = asyncio.create_task(get_status(window_id, probe))
    await probe_started.wait()

    async def second_waiter() -> AgentStatus | None:
        second_waiter_started.set()
        return await get_status(window_id, unexpected_probe)

    second = asyncio.create_task(second_waiter())
    await second_waiter_started.wait()
    release_probe.set()

    assert await asyncio.gather(first, second) == [working, working]
    assert probe_calls == 1
    assert agent_status_cache.get_status(window_id) == working
    assert window_id not in agent_status_cache._window_generations


async def test_concurrent_negative_probes_share_one_result() -> None:
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()
    second_call_started = asyncio.Event()
    probe_calls = 0

    async def probe() -> AgentStatus | None:
        nonlocal probe_calls
        probe_calls += 1
        probe_started.set()
        await release_probe.wait()
        return None

    async def unexpected_probe() -> AgentStatus | None:
        raise AssertionError("concurrent cache hit should share the first probe")

    get_status = agent_status_cache.get_status_or_probe
    first = asyncio.create_task(get_status("w2:t1", probe))
    await probe_started.wait()

    async def second_call() -> AgentStatus | None:
        second_call_started.set()
        return await get_status("w2:t1", unexpected_probe)

    second = asyncio.create_task(second_call())
    await second_call_started.wait()
    release_probe.set()

    assert await asyncio.gather(first, second) == [None, None]
    assert probe_calls == 1


async def test_cancelled_probe_failure_is_logged_and_releases_generation(
    caplog, monkeypatch
) -> None:
    window_id = "cancelled-probe"
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()
    generation_released = asyncio.Event()
    caplog.set_level(logging.ERROR, logger=agent_status_cache.__name__)
    release_generation = agent_status_cache._release_generation

    def signal_generation_release(
        released_window: str, state: agent_status_cache._WindowProbeState
    ) -> None:
        release_generation(released_window, state)
        if (
            released_window == window_id
            and window_id not in agent_status_cache._window_generations
        ):
            generation_released.set()

    monkeypatch.setattr(
        agent_status_cache, "_release_generation", signal_generation_release
    )

    async def probe() -> AgentStatus:
        probe_started.set()
        await release_probe.wait()
        raise RuntimeError("detached probe failed")

    pending = asyncio.create_task(
        agent_status_cache.get_status_or_probe(window_id, probe)
    )
    await probe_started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert window_id in agent_status_cache._window_generations
    release_probe.set()
    await generation_released.wait()

    assert window_id not in agent_status_cache._window_generations
    assert any(
        "Detached agent-status probe failed" in record.getMessage()
        and window_id in record.getMessage()
        and "detached probe failed" in record.getMessage()
        for record in caplog.records
    )


async def test_active_waiter_receives_probe_failure(caplog) -> None:
    window_id = "active-probe"
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()
    caplog.set_level(logging.ERROR, logger=agent_status_cache.__name__)

    async def probe() -> AgentStatus:
        probe_started.set()
        await release_probe.wait()
        raise RuntimeError("active probe failed")

    pending = asyncio.create_task(
        agent_status_cache.get_status_or_probe(window_id, probe)
    )
    await probe_started.wait()
    release_probe.set()

    with pytest.raises(RuntimeError, match="active probe failed"):
        await pending
    assert not any(
        "Detached agent-status probe failed" in record.getMessage()
        for record in caplog.records
    )
    assert window_id not in agent_status_cache._window_generations


@pytest.mark.parametrize("invalidation", ["clear", "reset"])
async def test_invalidation_during_cold_probe_discards_its_result(
    invalidation: str,
) -> None:
    probe_started = asyncio.Event()
    release_probe = asyncio.Event()
    stale = AgentStatus("working", "codex", "before invalidation")

    async def probe() -> AgentStatus:
        probe_started.set()
        await release_probe.wait()
        return stale

    get_status = agent_status_cache.get_status_or_probe
    pending = asyncio.create_task(get_status("w2:t1", probe))
    await probe_started.wait()
    if invalidation == "clear":
        agent_status_cache.clear("w2:t1")
    else:
        agent_status_cache.reset()
    release_probe.set()

    assert await pending is None
    assert agent_status_cache.get_status("w2:t1") is None
    assert "w2:t1" not in agent_status_cache._window_generations
