"""Live Herdr contract for guarded opaque agent targets.

These checks require a disposable Herdr server with detected agents. Published
``agent_session`` values provide stable targets; sessionless agents use opaque
terminal-derived fallback targets. Raw locators never authorize an action.
"""

from __future__ import annotations

import os

import pytest

from ccgram.multiplexer.herdr import HerdrError, HerdrManager

pytestmark = [pytest.mark.integration, pytest.mark.herdr]


@pytest.fixture
async def herdr() -> HerdrManager:
    socket = os.environ.get("HERDR_SOCKET_PATH", "")
    if not socket or not os.path.exists(socket):
        pytest.skip("herdr socket not available ($HERDR_SOCKET_PATH unset/missing)")
    manager = HerdrManager(socket_path=socket)
    try:
        await manager.ensure_session()
    except HerdrError as exc:
        pytest.skip(f"herdr server unavailable: {exc}")
    return manager


async def test_agent_list_exposes_only_guarded_agent_targets(
    herdr: HerdrManager,
) -> None:
    """Discovery exposes only opaque targets from the sole identity source."""
    targets = await herdr.list_windows()
    if not targets:
        pytest.skip("no detected agent.list record available")
    assert all(target.window_id.startswith("herdr-session-v1-") for target in targets)
    for target in targets:
        assert await herdr.find_window_by_id(target.window_id) is not None


async def test_non_target_identifier_fails_closed(herdr: HerdrManager) -> None:
    """A location-like identifier cannot authorize an action."""
    assert await herdr.find_window_by_id("not-a-herdr-session-target") is None
    assert await herdr.send("not-a-herdr-session-target", "ccgram guard probe") is False
