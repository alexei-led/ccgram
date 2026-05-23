from __future__ import annotations

import pytest

from ccgram.window_state_ports.pane_state import (
    PaneProjection,
    WindowPaneSnapshot,
    get_pane_lifecycle_notify,
    get_pane_projection,
    list_pane_projections,
    remove_pane,
    set_pane_lifecycle_notify,
    snapshot_window_panes,
    upsert_pane,
)
from ccgram.window_state_store import PaneInfo, WindowState, WindowStateStore


class TestReads:
    def test_get_pane_projection_missing_window(self, store: WindowStateStore) -> None:
        assert get_pane_projection("@missing", "%1") is None

    def test_get_pane_projection_missing_pane(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState()
        assert get_pane_projection("@1", "%missing") is None

    def test_get_pane_projection_returns_frozen(self, store: WindowStateStore) -> None:
        ws = WindowState()
        ws.panes["%5"] = PaneInfo(pane_id="%5", name="api", state="active")
        store.window_states["@1"] = ws
        proj = get_pane_projection("@1", "%5")
        assert proj == PaneProjection(
            pane_id="%5",
            name="api",
            provider="",
            last_active_ts=0.0,
            state="active",
            subscribed=False,
        )
        with pytest.raises(AttributeError):
            proj.name = "mutated"  # type: ignore[misc]

    def test_list_pane_projections_empty(self, store: WindowStateStore) -> None:
        assert list_pane_projections("@missing") == ()
        store.window_states["@1"] = WindowState()
        assert list_pane_projections("@1") == ()

    def test_list_pane_projections_preserves_insertion_order(
        self, store: WindowStateStore
    ) -> None:
        ws = WindowState()
        ws.panes["%5"] = PaneInfo(pane_id="%5")
        ws.panes["%3"] = PaneInfo(pane_id="%3", state="blocked")
        ws.panes["%9"] = PaneInfo(pane_id="%9", subscribed=True)
        store.window_states["@1"] = ws
        proj = list_pane_projections("@1")
        assert [p.pane_id for p in proj] == ["%5", "%3", "%9"]

    def test_snapshot_window_panes(self, store: WindowStateStore) -> None:
        ws = WindowState(pane_lifecycle_notify=True)
        ws.panes["%5"] = PaneInfo(pane_id="%5")
        store.window_states["@1"] = ws
        snap = snapshot_window_panes("@1")
        assert isinstance(snap, WindowPaneSnapshot)
        assert snap.window_id == "@1"
        assert snap.pane_lifecycle_notify is True
        assert len(snap.panes) == 1

    def test_snapshot_window_panes_missing(self, store: WindowStateStore) -> None:
        assert snapshot_window_panes("@missing") is None

    def test_get_pane_lifecycle_notify_default(self, store: WindowStateStore) -> None:
        assert get_pane_lifecycle_notify("@missing", default=True) is True
        store.window_states["@1"] = WindowState()
        assert get_pane_lifecycle_notify("@1", default=False) is False

    def test_get_pane_lifecycle_notify_override(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(pane_lifecycle_notify=False)
        assert get_pane_lifecycle_notify("@1", default=True) is False


class TestWrites:
    def test_upsert_pane_creates(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        proj = upsert_pane("@1", "%5", name="api", state="active")
        assert proj.pane_id == "%5"
        assert proj.name == "api"
        assert proj.state == "active"
        assert store.window_states["@1"].panes["%5"].name == "api"
        assert len(save_calls) == 1

    def test_upsert_pane_updates_partial(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        upsert_pane("@1", "%5", name="api", provider="claude")
        save_calls.clear()
        proj = upsert_pane("@1", "%5", state="blocked")
        assert proj.name == "api"
        assert proj.provider == "claude"
        assert proj.state == "blocked"
        assert len(save_calls) == 1

    def test_upsert_pane_invalid_state_rejected(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            upsert_pane("@1", "%5", state="garbage")  # type: ignore[arg-type]

    def test_upsert_pane_clear_name_with_none(self, store: WindowStateStore) -> None:
        upsert_pane("@1", "%5", name="api")
        proj = upsert_pane("@1", "%5", name=None)
        assert proj.name is None

    def test_remove_pane_existing(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        upsert_pane("@1", "%5")
        save_calls.clear()
        assert remove_pane("@1", "%5") is True
        assert "%5" not in store.window_states["@1"].panes
        assert len(save_calls) == 1

    def test_remove_pane_missing_no_save(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        assert remove_pane("@1", "%missing") is False
        assert save_calls == []

    def test_set_pane_lifecycle_notify(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_pane_lifecycle_notify("@1", True)
        assert store.window_states["@1"].pane_lifecycle_notify is True
        assert len(save_calls) == 1

    def test_set_pane_lifecycle_notify_noop_no_save(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_pane_lifecycle_notify("@1", True)
        save_calls.clear()
        set_pane_lifecycle_notify("@1", True)
        assert save_calls == []
