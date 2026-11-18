from __future__ import annotations

import pytest

from ccgram.topic_mark_journal import TopicMarkOperation
from ccgram.window_state_ports import lifecycle_state
from ccgram.window_state_ports.lifecycle_state import (
    LifecycleProjection,
    get_lifecycle,
    get_origin,
    set_window_origin,
)
from ccgram.window_state_store import WindowState, WindowStateStore


@pytest.mark.parametrize("detached", [False, True])
def test_observed_lifetime_survives_restart_without_pending_bit(detached):
    state = WindowState.from_dict(
        {
            "origin": "ccgram_created",
            "topic_detached": detached,
            "topic_mark_owner": "opaque-owner",
        }
    )
    saved = state.to_dict()
    assert saved.get("topic_detached", False) is detached
    assert saved["topic_mark_owner"] == "opaque-owner"
    assert "topic_mark_pending" not in saved
    assert saved["origin"] == "ccgram_created"


def test_selection_port_is_casefold_aware(store):
    store.window_states["ABC-DEF"] = WindowState(origin="ccgram_created")
    lifecycle_state.set_topic_selection(
        "abc-def", detached=True, pending=False, owner=None
    )
    assert lifecycle_state.is_topic_detach_pending("ABC-DEF", owner=None)
    assert lifecycle_state.is_topic_detached("abc-def")
    assert list(store.window_states) == ["ABC-DEF"]


def test_metadata_none_never_settles_a_native_operation(store, save_calls):
    lifecycle_state.set_topic_selection(
        "@1", detached=True, pending=False, owner="owner"
    )
    save_calls.clear()
    lifecycle_state.set_topic_selection(
        "@1", detached=True, pending=False, owner="owner"
    )
    assert save_calls == []
    lifecycle_state.set_topic_selection(
        "@1", detached=True, pending=None, owner="owner"
    )
    assert lifecycle_state.get_pending_topic_mark("@1") is False


def test_session_clear_preserves_journal_and_terminal_lifetime(store):
    store.window_states["@1"] = WindowState(
        session_id="previous-agent",
        transcript_path="/previous/transcript",
        topic_detached=True,
        topic_mark_owner="owner",
    )
    operation = lifecycle_state.queue_topic_mark("@1", "owner", False)
    store.clear_window_session("@1")
    assert store.window_states["@1"].session_id == ""
    assert store.window_states["@1"].transcript_path == ""
    assert store.window_states["@1"].topic_detached
    assert lifecycle_state.pending_topic_selections() == [operation]


def test_pending_match_uses_known_owner_before_original_route(store):
    lifecycle_state.set_topic_selection("@old", detached=True, pending=False, owner="O")
    assert lifecycle_state.is_topic_detach_pending("@new", owner="O")
    assert not lifecycle_state.is_topic_detach_pending("@old", owner="P")
    assert lifecycle_state.is_topic_detach_pending("@old", owner=None)
    assert "@new" not in store.window_states


def test_observation_does_not_create_workers_or_modify_an_old_operation(store):
    assert not lifecycle_state.observe_topic_mark_owner("@new", "P")
    assert not store.window_states
    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner="O")
    operation = lifecycle_state.pending_topic_selections()[0]
    assert lifecycle_state.observe_topic_mark_owner("@1", "P")
    assert lifecycle_state.get_topic_mark_owner("@1") == "P"
    assert lifecycle_state.pending_topic_selections() == [operation]
    assert not lifecycle_state.observe_topic_mark_owner("@1", None)


def test_reused_route_can_admit_new_owner_and_retain_old_retry(store):
    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner="O")
    old = lifecycle_state.pending_topic_selections()[0]
    assert lifecycle_state.record_topic_mark_owner("@1", "P")
    current = lifecycle_state.queue_topic_mark("@1", "P", True)
    assert lifecycle_state.get_pending_topic_mark("@1") is True
    assert lifecycle_state.get_pending_topic_mark("@1", owner="O") is False
    assert not lifecycle_state.is_topic_detach_pending("@1", owner="P")
    assert lifecycle_state.settle_topic_mark(old)
    assert lifecycle_state.get_topic_mark_owner("@1") == "P"
    assert lifecycle_state.pending_topic_selections() == [current]


def test_journal_round_trip_keeps_multiple_owners_for_one_route(store):
    store.queue_topic_mark("@1", "O", False)
    store.queue_topic_mark("@1", "P", True)
    snapshot = store.topic_mark_journal_to_dict()
    store.reset()
    store.load_topic_mark_journal(snapshot)
    assert list(store.topic_mark_operations.values()) == [
        TopicMarkOperation("@1", "O", False),
        TopicMarkOperation("@1", "P", True),
    ]


def test_stale_completion_and_rollback_never_overwrite_new_intent(store):
    old = store.queue_topic_mark("@1", "O", True)
    newer = store.queue_topic_mark("@new", "O", False)
    assert not store.settle_topic_mark(old)
    store.restore_topic_mark(old)
    assert list(store.topic_mark_operations.values()) == [newer]


def test_same_owner_intents_coalesce_and_other_owners_remain_independent(store):
    old = store.queue_topic_mark("@1", "O", True)
    assert store.queue_topic_mark("@new", "O", True) == old
    newer = store.queue_topic_mark("@new", "O", False)
    independent = store.queue_topic_mark("@1", "P", True)
    assert list(store.topic_mark_operations.values()) == [newer, independent]


@pytest.mark.parametrize("desired", [False, True])
def test_journal_survives_stale_window_pruning_and_removal(store, desired):
    lifecycle_state.set_topic_selection(
        "@old", detached=True, pending=desired, owner="O"
    )
    operation = lifecycle_state.pending_topic_selections()[0]
    assert not store.prune_stale_window_states(set(), set(), set())
    assert store.remove_window("@old")
    assert lifecycle_state.pending_topic_selections() == [operation]


def test_admission_capture_keeps_ttl_protection_until_success(store):
    lifecycle_state.set_topic_selection("@old", detached=True, pending=None, owner="O")
    assert lifecycle_state.record_topic_mark_owner("@new", "O")
    assert lifecycle_state.is_topic_detached("@old")
    assert lifecycle_state.complete_topic_admission("@new", "O")
    assert not lifecycle_state.is_topic_detached("@old")


def test_completion_preserves_pending_mark_on_and_other_owner_clear(store):
    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner="O")
    assert lifecycle_state.record_topic_mark_owner("@1", "P")
    current = lifecycle_state.queue_topic_mark("@1", "P", True)
    assert lifecycle_state.complete_topic_admission("@1", "P")
    assert not lifecycle_state.is_topic_detached("@1")
    assert lifecycle_state.topic_mark_operation_is_current(current)
    assert lifecycle_state.is_topic_detach_pending("@elsewhere", owner="O")


def test_completion_refuses_matching_clear_and_unknown_capture(store):
    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner="O")
    assert not lifecycle_state.complete_topic_admission("@1", "O")
    assert lifecycle_state.is_topic_detached("@1")
    assert lifecycle_state.complete_topic_admission("@absent", None)
    assert not lifecycle_state.complete_topic_admission("@absent", "missing")


def test_auto_ownerless_admission_does_not_fabricate_owner(store):
    assert lifecycle_state.record_topic_mark_owner("@unmarked", None)
    assert lifecycle_state.get_topic_mark_owner("@unmarked") is None
    assert lifecycle_state.get_pending_topic_mark("@unmarked") is None


class TestReads:
    def test_get_lifecycle_missing(self, store: WindowStateStore) -> None:
        assert get_lifecycle("@missing") is None

    def test_get_lifecycle_default(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState()
        assert get_lifecycle("@1") == LifecycleProjection("@1", "manual_discovered")

    def test_get_lifecycle_ccgram_created(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(origin="ccgram_created")
        assert get_lifecycle("@1") == LifecycleProjection("@1", "ccgram_created")

    def test_get_origin_invalid_falls_back(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(origin="garbage")
        assert get_origin("@1") == "manual_discovered"

    def test_get_origin_missing(self, store: WindowStateStore) -> None:
        assert get_origin("@missing") == "manual_discovered"


class TestWrites:
    def test_set_window_origin_persists(self, store, save_calls):
        set_window_origin("@1", "ccgram_created")
        assert store.window_states["@1"].origin == "ccgram_created"
        assert len(save_calls) == 1

    def test_set_window_origin_noop_no_save(self, store, save_calls):
        set_window_origin("@1", "ccgram_created")
        save_calls.clear()
        set_window_origin("@1", "ccgram_created")
        assert save_calls == []

    def test_set_window_origin_invalid(self, store):
        with pytest.raises(ValueError):
            set_window_origin("@1", "garbage")
