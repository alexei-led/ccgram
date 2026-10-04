"""Unit tests for the backend-neutral push-updated agent-status cache."""

from __future__ import annotations

import pytest

from ccgram.multiplexer import agent_status_cache
from ccgram.multiplexer.base import AgentStatus


@pytest.fixture(autouse=True)
def _empty_cache():
    agent_status_cache.reset()
    yield
    agent_status_cache.reset()


def test_cold_window_has_no_status() -> None:
    assert agent_status_cache._fresh_status("w2:t1") is None


def test_set_status_is_readable_and_scoped_to_its_window() -> None:
    working = AgentStatus("working", "codex", "compiling")
    agent_status_cache.set_status("w2:t1", working)

    assert agent_status_cache._fresh_status("w2:t1") == working
    assert agent_status_cache._fresh_status("w3:t1") is None


def test_set_status_overwrites_the_previous_value() -> None:
    agent_status_cache.set_status("w2:t1", AgentStatus("working"))
    agent_status_cache.set_status("w2:t1", AgentStatus("idle"))

    assert agent_status_cache._fresh_status("w2:t1") == AgentStatus("idle")


def test_clear_drops_only_the_named_window() -> None:
    agent_status_cache.set_status("w2:t1", AgentStatus("working"))
    agent_status_cache.set_status("w3:t1", AgentStatus("idle"))

    agent_status_cache.clear("w2:t1")

    assert agent_status_cache._fresh_status("w2:t1") is None
    assert agent_status_cache._fresh_status("w3:t1") == AgentStatus("idle")


def test_clear_of_a_cold_window_is_a_no_op() -> None:
    agent_status_cache.clear("never-seen")


def test_reset_empties_the_whole_cache() -> None:
    agent_status_cache.set_status("a", AgentStatus("working"))
    agent_status_cache.set_status("b", AgentStatus("idle"))

    agent_status_cache.reset()

    assert agent_status_cache._fresh_status("a") is None
    assert agent_status_cache._fresh_status("b") is None


class TestResolveAgentWorking:
    async def test_cold_cache_resolves_via_backend(self, monkeypatch):
        """TASK-47 cold-cache hole: post-restart, before any push, the gate
        must still see a working agent (2026-09-30 Mac live fire)."""
        agent_status_cache.reset()

        class Mux:
            async def agent_status(self, window_id):
                return AgentStatus(state="working")

        monkeypatch.setattr(agent_status_cache, "multiplexer", Mux())
        assert await agent_status_cache.resolve_agent_working("w1") is True
        # and the resolution warmed the cache (fresh on the fast path)
        assert agent_status_cache._fresh_status("w1") is not None

    async def test_stale_entry_beyond_ttl_reresolves(self, monkeypatch):
        agent_status_cache.reset()
        agent_status_cache.set_status("w2", AgentStatus(state="working"))
        # age the stamp past the TTL
        stamp, status = agent_status_cache._cache["w2"]
        agent_status_cache._cache["w2"] = (
            stamp - agent_status_cache._STATUS_TTL_S - 1,
            status,
        )
        calls = {"n": 0}

        class Mux:
            async def agent_status(self, window_id):
                calls["n"] += 1
                return AgentStatus(state="idle")

        monkeypatch.setattr(agent_status_cache, "multiplexer", Mux())
        assert await agent_status_cache.resolve_agent_working("w2") is False
        assert calls["n"] == 1

    async def test_backend_without_status_reports_false(self, monkeypatch):
        agent_status_cache.reset()

        class Mux:
            async def agent_status(self, window_id):
                return None

        monkeypatch.setattr(agent_status_cache, "multiplexer", Mux())
        assert await agent_status_cache.resolve_agent_working("w3") is False
