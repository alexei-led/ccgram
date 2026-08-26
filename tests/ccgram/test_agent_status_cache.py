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
