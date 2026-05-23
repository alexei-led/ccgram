from __future__ import annotations

import pytest

from ccgram.config import config
from ccgram.window_state_ports.tool_state import (
    ToolModeProjection,
    cycle_batch_mode,
    cycle_tool_call_visibility,
    get_batch_mode,
    get_tool_call_visibility,
    get_tool_modes,
    is_ephemeral_tools,
    is_tool_calls_hidden,
    set_batch_mode,
    set_tool_call_visibility,
)
from ccgram.window_state_store import WindowState, WindowStateStore


class TestReads:
    def test_get_tool_modes_missing(self, store: WindowStateStore) -> None:
        assert get_tool_modes("@missing") is None

    def test_get_tool_modes_full(
        self, store: WindowStateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "hide_tool_calls", True, raising=False)
        monkeypatch.setattr(config, "ephemeral_tools", False, raising=False)
        store.window_states["@1"] = WindowState(
            batch_mode="verbose", tool_call_visibility="shown"
        )
        modes = get_tool_modes("@1")
        assert modes == ToolModeProjection(
            window_id="@1",
            batch_mode="verbose",
            tool_call_visibility="shown",
            batch_mode_resolved="verbose",
            tool_calls_hidden_resolved=False,
        )

    def test_get_batch_mode_falls_back_to_config(
        self, store: WindowStateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "ephemeral_tools", True, raising=False)
        assert get_batch_mode("@missing") == "ephemeral"

    def test_get_batch_mode_per_window_wins(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(batch_mode="batched")
        assert get_batch_mode("@1") == "batched"

    def test_get_batch_mode_invalid_falls_back(
        self, store: WindowStateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "ephemeral_tools", False, raising=False)
        store.window_states["@1"] = WindowState(batch_mode="garbage")
        assert get_batch_mode("@1") == "ephemeral"

    def test_is_ephemeral_tools(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(batch_mode="ephemeral")
        store.window_states["@2"] = WindowState(batch_mode="verbose")
        assert is_ephemeral_tools("@1") is True
        assert is_ephemeral_tools("@2") is False

    def test_get_tool_call_visibility_default(self, store: WindowStateStore) -> None:
        assert get_tool_call_visibility("@missing") == "default"

    def test_get_tool_call_visibility_invalid_falls_back(
        self, store: WindowStateStore
    ) -> None:
        store.window_states["@1"] = WindowState(tool_call_visibility="garbage")
        assert get_tool_call_visibility("@1") == "default"

    def test_is_tool_calls_hidden_per_window_override(
        self, store: WindowStateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "hide_tool_calls", False, raising=False)
        store.window_states["@1"] = WindowState(tool_call_visibility="hidden")
        store.window_states["@2"] = WindowState(tool_call_visibility="shown")
        store.window_states["@3"] = WindowState(tool_call_visibility="default")
        assert is_tool_calls_hidden("@1") is True
        assert is_tool_calls_hidden("@2") is False
        assert is_tool_calls_hidden("@3") is False

    def test_is_tool_calls_hidden_default_uses_global(
        self, store: WindowStateStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "hide_tool_calls", True, raising=False)
        store.window_states["@1"] = WindowState(tool_call_visibility="default")
        assert is_tool_calls_hidden("@1") is True


class TestWrites:
    def test_set_batch_mode_persists(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_batch_mode("@1", "verbose")
        assert store.window_states["@1"].batch_mode == "verbose"
        assert len(save_calls) == 1

    def test_set_batch_mode_noop_no_save(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_batch_mode("@1", "verbose")
        save_calls.clear()
        set_batch_mode("@1", "verbose")
        assert save_calls == []

    def test_set_batch_mode_invalid(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            set_batch_mode("@1", "garbage")

    def test_cycle_batch_mode(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(batch_mode="batched")
        assert cycle_batch_mode("@1") == "ephemeral"
        assert cycle_batch_mode("@1") == "verbose"
        assert cycle_batch_mode("@1") == "batched"

    def test_set_tool_call_visibility_persists(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_tool_call_visibility("@1", "hidden")
        assert store.window_states["@1"].tool_call_visibility == "hidden"
        assert len(save_calls) == 1

    def test_set_tool_call_visibility_noop_no_save(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_tool_call_visibility("@1", "hidden")
        save_calls.clear()
        set_tool_call_visibility("@1", "hidden")
        assert save_calls == []

    def test_set_tool_call_visibility_invalid(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            set_tool_call_visibility("@1", "garbage")

    def test_cycle_tool_call_visibility(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(tool_call_visibility="default")
        assert cycle_tool_call_visibility("@1") == "shown"
        assert cycle_tool_call_visibility("@1") == "hidden"
        assert cycle_tool_call_visibility("@1") == "default"
