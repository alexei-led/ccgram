from __future__ import annotations

import pytest

from ccgram.window_state_ports.lifecycle_state import (
    LifecycleProjection,
    get_lifecycle,
    get_origin,
    set_window_origin,
)
from ccgram.window_state_store import WindowState, WindowStateStore


class TestReads:
    def test_get_lifecycle_missing(self, store: WindowStateStore) -> None:
        assert get_lifecycle("@missing") is None

    def test_get_lifecycle_default(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState()
        proj = get_lifecycle("@1")
        assert proj == LifecycleProjection(
            window_id="@1",
            origin="manual_discovered",
        )

    def test_get_lifecycle_ccgram_created(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(origin="ccgram_created")
        proj = get_lifecycle("@1")
        assert proj == LifecycleProjection(
            window_id="@1",
            origin="ccgram_created",
        )

    def test_get_origin_invalid_falls_back(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(origin="garbage")
        assert get_origin("@1") == "manual_discovered"

    def test_get_origin_missing(self, store: WindowStateStore) -> None:
        assert get_origin("@missing") == "manual_discovered"


class TestWrites:
    def test_set_window_origin_persists(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_window_origin("@1", "ccgram_created")
        assert store.window_states["@1"].origin == "ccgram_created"
        assert len(save_calls) == 1

    def test_set_window_origin_noop_no_save(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_window_origin("@1", "ccgram_created")
        save_calls.clear()
        set_window_origin("@1", "ccgram_created")
        assert save_calls == []

    def test_set_window_origin_invalid(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            set_window_origin("@1", "garbage")
