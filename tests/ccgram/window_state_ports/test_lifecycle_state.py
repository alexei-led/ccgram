from __future__ import annotations

import pytest

from ccgram.window_state_ports.lifecycle_state import (
    LifecycleProjection,
    get_lifecycle,
    get_origin,
    is_external,
    mark_gemini_external_warned,
    set_window_origin,
    was_gemini_external_warned,
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
            external=False,
            gemini_external_warned=False,
        )

    def test_get_lifecycle_external(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(
            origin="external", external=True, gemini_external_warned=True
        )
        proj = get_lifecycle("@1")
        assert proj == LifecycleProjection(
            window_id="@1",
            origin="external",
            external=True,
            gemini_external_warned=True,
        )

    def test_get_origin_invalid_falls_back(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(origin="garbage")
        assert get_origin("@1") == "manual_discovered"

    def test_get_origin_missing(self, store: WindowStateStore) -> None:
        assert get_origin("@missing") == "manual_discovered"

    def test_is_external(self, store: WindowStateStore) -> None:
        assert is_external("@missing") is False
        store.window_states["@1"] = WindowState(external=False)
        store.window_states["@2"] = WindowState(external=True)
        assert is_external("@1") is False
        assert is_external("@2") is True

    def test_was_gemini_warned(self, store: WindowStateStore) -> None:
        assert was_gemini_external_warned("@missing") is False
        store.window_states["@1"] = WindowState(gemini_external_warned=True)
        assert was_gemini_external_warned("@1") is True


class TestWrites:
    def test_set_window_origin_persists(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_window_origin("@1", "ccgram_created")
        assert store.window_states["@1"].origin == "ccgram_created"
        assert len(save_calls) == 1

    def test_set_window_origin_external_marks_external(
        self, store: WindowStateStore
    ) -> None:
        set_window_origin("@1", "external")
        assert store.window_states["@1"].external is True

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

    def test_mark_gemini_warned(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        mark_gemini_external_warned("@1")
        assert store.window_states["@1"].gemini_external_warned is True
        assert len(save_calls) == 1

    def test_mark_gemini_warned_noop_no_save(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        mark_gemini_external_warned("@1")
        save_calls.clear()
        mark_gemini_external_warned("@1")
        assert save_calls == []
