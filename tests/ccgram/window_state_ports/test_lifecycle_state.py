from __future__ import annotations

import pytest

from ccgram.window_state_ports.lifecycle_state import (
    LifecycleProjection,
    get_lifecycle,
    get_origin,
    set_window_origin,
)
from ccgram.window_state_store import WindowState, WindowStateStore


@pytest.mark.parametrize(
    ("detached", "pending"),
    [(False, None), (False, True), (True, False)],
)
def test_detach_intent_survives_restart(detached: bool, pending: bool | None) -> None:
    state = WindowState.from_dict(
        {
            "origin": "ccgram_created",
            "topic_detached": detached,
            "topic_mark_pending": pending,
        }
    )
    saved = state.to_dict()
    assert saved.get("topic_detached", False) is detached
    assert saved.get("topic_mark_pending") is pending
    assert saved["origin"] == "ccgram_created"


@pytest.mark.parametrize("raw", ["false", 0, 1, {}, []])
def test_invalid_pending_marker_does_not_become_intent(raw) -> None:
    state = WindowState.from_dict({"topic_mark_pending": raw})
    assert state.topic_mark_pending is None


def test_selection_port_is_casefold_aware(store: WindowStateStore) -> None:
    from ccgram.window_state_ports import lifecycle_state

    store.window_states["ABC-DEF"] = WindowState(origin="ccgram_created")
    lifecycle_state.set_topic_selection(
        "abc-def", detached=True, pending=False, owner=None
    )
    assert lifecycle_state.is_topic_detach_pending("abc-def", owner=None)
    assert lifecycle_state.is_topic_detached("abc-def")
    assert list(store.window_states) == ["ABC-DEF"]
    assert lifecycle_state.get_pending_topic_mark("abc-def") is False


def test_selection_state_updates_save_only_on_change(
    store: WindowStateStore, save_calls: list[int]
) -> None:
    from ccgram.window_state_ports import lifecycle_state

    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner=None)
    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner=None)
    assert len(save_calls) == 1
    lifecycle_state.set_topic_selection("@1", detached=True, pending=None, owner=None)
    assert len(save_calls) == 2
    assert not lifecycle_state.is_topic_detach_pending("@1", owner=None)


def test_session_clear_does_not_erase_unfinished_terminal_selection(
    store: WindowStateStore,
) -> None:
    store.window_states["@1"] = WindowState(
        session_id="previous-agent",
        transcript_path="/previous/transcript",
        topic_detached=True,
        topic_mark_pending=False,
    )
    store.clear_window_session("@1")
    assert store.window_states["@1"].session_id == ""
    assert store.window_states["@1"].transcript_path == ""
    assert store.window_states["@1"].topic_detached
    assert store.window_states["@1"].topic_mark_pending is False


def test_marker_owner_survives_restart_and_is_not_session_identity() -> None:
    restored = WindowState.from_dict(
        {
            "session_id": "conversation-a",
            "topic_mark_owner": "native-owner-a",
            "topic_mark_pending": False,
            "topic_detached": True,
        }
    )
    assert restored.to_dict().get("topic_mark_owner") == "native-owner-a"
    assert restored.to_dict()["session_id"] == "conversation-a"


def test_pending_clear_is_matched_by_owner_not_routing_alias(
    store: WindowStateStore,
) -> None:
    from ccgram.window_state_ports import lifecycle_state

    lifecycle_state.set_topic_selection(
        "old-route", detached=True, pending=False, owner="owner-a"
    )
    assert lifecycle_state.is_topic_detach_pending("new-route", owner="owner-a")
    assert not lifecycle_state.is_topic_detach_pending("new-route", owner="owner-b")
    assert lifecycle_state.is_topic_detach_pending("OLD-ROUTE", owner=None)
    assert "new-route" not in store.window_states


def test_observation_never_creates_unadmitted_rows_or_overwrites_pending_owner(
    store: WindowStateStore,
) -> None:
    from ccgram.window_state_ports import lifecycle_state

    assert not lifecycle_state.observe_topic_mark_owner("@new", "owner-new")
    assert not store.window_states
    assert lifecycle_state.record_topic_mark_owner("@bound", "owner-first")
    assert lifecycle_state.observe_topic_mark_owner("@bound", "owner-current")
    assert not lifecycle_state.observe_topic_mark_owner("@bound", None)
    lifecycle_state.set_topic_selection(
        "@bound", detached=True, pending=False, owner="owner-current"
    )
    assert not lifecycle_state.observe_topic_mark_owner("@bound", "other-owner")
    assert lifecycle_state.get_topic_mark_owner("@bound") == "owner-current"
    assert not lifecycle_state.record_topic_mark_owner("@new-route", "owner-current")
    assert "@new-route" not in store.window_states


def test_auto_ownerless_admission_records_no_fabricated_owner(
    store: WindowStateStore,
) -> None:
    from ccgram.window_state_ports import lifecycle_state

    assert lifecycle_state.record_topic_mark_owner("@unmarked", None)
    assert lifecycle_state.get_topic_mark_owner("@unmarked") is None
    assert lifecycle_state.get_pending_topic_mark("@unmarked") is None


def test_atomic_clear_cancels_older_same_owner_mark_on(store: WindowStateStore) -> None:
    from ccgram.window_state_ports import lifecycle_state

    lifecycle_state.set_topic_selection(
        "@on", detached=False, pending=True, owner="shared"
    )
    lifecycle_state.set_topic_selection(
        "@other", detached=False, pending=True, owner="independent"
    )
    lifecycle_state.set_topic_selection(
        "@off", detached=True, pending=False, owner="shared"
    )
    assert lifecycle_state.get_pending_topic_mark("@on") is None
    assert lifecycle_state.is_topic_detached("@on")
    assert lifecycle_state.get_pending_topic_mark("@other") is True


@pytest.mark.parametrize("desired", [False, True])
def test_store_pruning_retains_pending_owner_operations(
    store: WindowStateStore, desired: bool
) -> None:
    store.set_topic_selection(
        "@old", detached=not desired, pending=desired, owner="native-owner"
    )
    assert not store.prune_stale_window_states(set(), set(), set())
    assert store.window_states["@old"].topic_mark_pending is desired


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
