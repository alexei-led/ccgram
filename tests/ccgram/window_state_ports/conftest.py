from __future__ import annotations

import pytest

from ccgram.window_state_store import WindowStateStore


@pytest.fixture
def save_calls() -> list[int]:
    return []


@pytest.fixture
def store(monkeypatch, save_calls: list[int]) -> WindowStateStore:
    """Fresh store wired into every port module + window_state_store."""
    s = WindowStateStore(
        schedule_save=lambda: save_calls.append(1),
        on_hookless_provider_switch=lambda _wid: None,
    )
    targets = (
        "ccgram.window_state_store.window_store",
        "ccgram.window_state_ports.pane_state.window_store",
        "ccgram.window_state_ports.identity_state.window_store",
        "ccgram.window_state_ports.worktree_state.window_store",
        "ccgram.window_state_ports.tool_state.window_store",
        "ccgram.window_state_ports.lifecycle_state.window_store",
    )
    for target in targets:
        monkeypatch.setattr(target, s)
    return s
