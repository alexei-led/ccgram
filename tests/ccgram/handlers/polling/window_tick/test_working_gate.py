"""The working-agent gate at the status-resolution seam (TASK-47).

A selection-shaped region on a WORKING pane is Claude Code's queued-input
block (messages typed mid-turn), not a prompt. Both scrapers' results pass
through ``_resolve_status``, so one suppression there covers the strategy
parse and the provider parse alike, falling through to the native working
status instead of latching a false interactive state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ccgram.handlers.polling.window_tick import observe
from ccgram.handlers.polling.window_tick.observe import _resolve_status
from ccgram.multiplexer import agent_status_cache
from ccgram.multiplexer.base import AgentStatus
from ccgram.providers.base import StatusUpdate

pytest = __import__("pytest")


@pytest.fixture(autouse=True)
def _clear_status_cache():
    agent_status_cache.reset()
    yield
    agent_status_cache.reset()


_WINDOW = SimpleNamespace(pane_width=100, pane_height=40)
_INTERACTIVE = StatusUpdate(
    raw_text="queued block",
    display_label="SelectionUI",
    is_interactive=True,
    ui_type="SelectionUI",
)


def _provider(return_value):
    provider = SimpleNamespace(
        capabilities=SimpleNamespace(
            uses_pyte_status_parsing=True, uses_pane_title=False
        ),
        parse_terminal_status=lambda *a, **k: return_value,
    )
    return provider


async def test_working_status_suppresses_strategy_interactive() -> None:
    agent_status_cache.set_status("w1", AgentStatus(state="working"))
    native = StatusUpdate(raw_text="working", display_label="working")
    with (
        patch.object(observe, "_parse_with_pyte", return_value=_INTERACTIVE),
        patch.object(observe, "_get_provider", return_value=_provider(None)),
        patch.object(observe, "_native_agent_status", AsyncMock(return_value=native)),
    ):
        result = await _resolve_status("w1", "pane", _WINDOW)
    assert result is not None and not result.is_interactive
    assert result.display_label == "working"


async def test_working_status_suppresses_provider_interactive() -> None:
    agent_status_cache.set_status("w2", AgentStatus(state="working"))
    with (
        patch.object(observe, "_parse_with_pyte", return_value=None),
        patch.object(observe, "_get_provider", return_value=_provider(_INTERACTIVE)),
        patch.object(
            observe,
            "_native_agent_status",
            AsyncMock(
                return_value=StatusUpdate(raw_text="working", display_label="working")
            ),
        ),
    ):
        result = await _resolve_status("w2", "pane", _WINDOW)
    assert result is not None and not result.is_interactive


async def test_idle_status_keeps_interactive() -> None:
    agent_status_cache.set_status("w3", AgentStatus(state="idle"))
    with patch.object(observe, "_parse_with_pyte", return_value=_INTERACTIVE):
        result = await _resolve_status("w3", "pane", _WINDOW)
    assert result is not None and result.is_interactive
