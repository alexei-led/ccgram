from __future__ import annotations

import pytest

from ccgram.window_state_ports.terminal_identity import (
    TerminalIdentityProjection,
    get_backend,
    get_terminal_identity,
    get_unit_id,
    set_terminal_identity,
)
from ccgram.window_state_store import WindowState, WindowStateStore


class TestReads:
    def test_missing_window_returns_none(self, store: WindowStateStore) -> None:
        assert get_terminal_identity("@missing") is None

    def test_legacy_row_defaults_to_tmux_with_window_id(
        self, store: WindowStateStore
    ) -> None:
        # Legacy WindowState — no terminal_backend / terminal_unit_id set.
        store.window_states["@3"] = WindowState()
        proj = get_terminal_identity("@3")
        assert proj == TerminalIdentityProjection(
            window_id="@3",
            backend="tmux",
            unit_id="@3",
            legacy_window_id="@3",
        )

    def test_cmux_row_with_unit_id(self, store: WindowStateStore) -> None:
        store.window_states["@5"] = WindowState(
            terminal_backend="cmux",
            terminal_unit_id="workspace-uuid",
        )
        proj = get_terminal_identity("@5")
        assert proj is not None
        assert proj.backend == "cmux"
        assert proj.unit_id == "workspace-uuid"
        assert proj.legacy_window_id == "@5"

    def test_invalid_backend_falls_back_to_tmux(self, store: WindowStateStore) -> None:
        store.window_states["@7"] = WindowState(terminal_backend="garbage")
        proj = get_terminal_identity("@7")
        assert proj is not None
        assert proj.backend == "tmux"
        assert proj.unit_id == "@7"

    def test_get_backend_defaults_when_missing(self, store: WindowStateStore) -> None:
        assert get_backend("@nope") == "tmux"

    def test_get_unit_id_legacy_tmux(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState()
        assert get_unit_id("@1") == "@1"

    def test_get_unit_id_cmux(self, store: WindowStateStore) -> None:
        store.window_states["@2"] = WindowState(
            terminal_backend="cmux",
            terminal_unit_id="ws-abc",
        )
        assert get_unit_id("@2") == "ws-abc"

    def test_get_unit_id_missing_window(self, store: WindowStateStore) -> None:
        assert get_unit_id("@nope") == ""


class TestWrites:
    def test_set_terminal_identity_persists(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_terminal_identity("@1", backend="cmux", unit_id="ws-1")
        state = store.window_states["@1"]
        assert state.terminal_backend == "cmux"
        assert state.terminal_unit_id == "ws-1"
        assert len(save_calls) == 1

    def test_set_terminal_identity_noop_when_unchanged(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_terminal_identity("@1", backend="cmux", unit_id="ws-1")
        save_calls.clear()
        set_terminal_identity("@1", backend="cmux", unit_id="ws-1")
        assert save_calls == []

    def test_set_terminal_identity_persists_metadata(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_terminal_identity(
            "cmux:ws-1",
            backend="cmux",
            unit_id="ws-1",
            cwd="/repo",
            provider_name="claude",
            window_name="alpha",
        )
        state = store.window_states["cmux:ws-1"]
        assert state.cwd == "/repo"
        assert state.provider_name == "claude"
        assert state.window_name == "alpha"
        assert len(save_calls) == 1

    def test_rejects_unknown_backend(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            set_terminal_identity("@1", backend="nope", unit_id="x")

    def test_rejects_blank_unit_id(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            set_terminal_identity("@1", backend="cmux", unit_id="")

    def test_switching_backend_saves_once(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_terminal_identity("@1", backend="tmux", unit_id="@1")
        save_calls.clear()
        set_terminal_identity("@1", backend="cmux", unit_id="ws-1")
        state = store.window_states["@1"]
        assert state.terminal_backend == "cmux"
        assert state.terminal_unit_id == "ws-1"
        assert len(save_calls) == 1
