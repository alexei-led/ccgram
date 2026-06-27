"""Unit tests for the backend-neutral push-updated agent-status cache."""

from __future__ import annotations

from ccgram.multiplexer import agent_status_cache
from ccgram.multiplexer.base import AgentStatus


def test_set_get_clear_reset() -> None:
    agent_status_cache.reset()
    assert agent_status_cache.get_status("w2:t1") is None

    working = AgentStatus("working", "codex", "compiling")
    agent_status_cache.set_status("w2:t1", working)
    assert agent_status_cache.get_status("w2:t1") == working

    # set overwrites; other keys stay cold.
    agent_status_cache.set_status("w2:t1", AgentStatus("idle"))
    assert agent_status_cache.get_status("w2:t1") == AgentStatus("idle")
    assert agent_status_cache.get_status("w3:t1") is None

    agent_status_cache.clear("w2:t1")
    assert agent_status_cache.get_status("w2:t1") is None

    agent_status_cache.set_status("a", AgentStatus("working"))
    agent_status_cache.set_status("b", AgentStatus("idle"))
    agent_status_cache.reset()
    assert agent_status_cache.get_status("a") is None
    assert agent_status_cache.get_status("b") is None
