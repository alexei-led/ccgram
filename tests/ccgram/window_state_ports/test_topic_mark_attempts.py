"""Latest lifecycle intent cannot erase or settle another native attempt."""

from __future__ import annotations

from dataclasses import replace

import pytest

from _topic_mark_helpers import make_witness as _witness
from ccgram.window_state_ports import lifecycle_state


def test_exact_on_completion_keeps_newer_off_and_cannot_complete_new_attempt(store):
    initial = lifecycle_state.queue_topic_mark("@1", "owner", True)
    armed_a = lifecycle_state.begin_topic_mark_attempt(initial, _witness("A"))
    latest = lifecycle_state.queue_topic_mark("@rekey", "owner", False)
    assert latest.window_id == "@1"
    assert latest.desired is False
    assert latest.attempt == armed_a.attempt
    saved = store.topic_mark_journal_to_dict()
    store.reset()
    store.load_topic_mark_journal(saved)

    ready_off = lifecycle_state.finish_topic_mark_attempt(
        "@1", owner="owner", dispatch_id="A"
    )
    assert ready_off is not None
    assert ready_off.desired is False and ready_off.attempt is None
    armed_b = lifecycle_state.begin_topic_mark_attempt(ready_off, _witness("B"))

    assert (
        lifecycle_state.finish_topic_mark_attempt("@1", owner="owner", dispatch_id="A")
        is None
    )
    assert lifecycle_state.pending_topic_selections() == [armed_b]
    assert not lifecycle_state.settle_topic_mark(ready_off)
    finished = lifecycle_state.finish_topic_mark_attempt(
        "@1", owner="owner", dispatch_id="B"
    )
    assert finished is not None and finished.desired is False
    assert lifecycle_state.settle_topic_mark(finished)
    assert lifecycle_state.pending_topic_selections() == []


def test_same_desired_aba_cannot_let_old_completion_clear_new_attempt(store):
    original = lifecycle_state.queue_topic_mark("@1", "owner", True)
    lifecycle_state.begin_topic_mark_attempt(original, _witness("A"))
    ready = lifecycle_state.finish_topic_mark_attempt(
        "@1", owner="owner", dispatch_id="A"
    )
    assert ready is not None
    current = lifecycle_state.begin_topic_mark_attempt(ready, _witness("B"))

    assert (
        lifecycle_state.finish_topic_mark_attempt("@1", owner="owner", dispatch_id="A")
        is None
    )
    assert not lifecycle_state.settle_topic_mark(original)
    assert not lifecycle_state.settle_topic_mark(current)
    assert lifecycle_state.pending_topic_selections() == [current]


@pytest.mark.parametrize("desired", [False, True])
def test_armed_attempt_blocks_settlement_and_admission_not_observation(store, desired):
    assert lifecycle_state.record_topic_mark_owner("@1", "owner")
    operation = lifecycle_state.queue_topic_mark("@1", "owner", desired)
    armed = lifecycle_state.begin_topic_mark_attempt(operation, _witness("A"))

    assert not lifecycle_state.settle_topic_mark(armed)
    assert not lifecycle_state.complete_topic_admission("@1", "owner")
    assert lifecycle_state.observe_topic_mark_owner("@1", "replacement")
    assert lifecycle_state.get_topic_mark_owner("@1") == "replacement"
    assert lifecycle_state.pending_topic_selections() == [armed]
    assert lifecycle_state.is_topic_mark_quarantined("@elsewhere", owner="owner")
    assert not lifecycle_state.is_topic_mark_quarantined("@1", owner="replacement")


@pytest.mark.parametrize(
    "saved_owner,route,observed_owner,blocked",
    [
        ("owner", "@new-route", "owner", True),
        ("owner", "@1", "replacement", False),
        ("owner", "@1", None, True),
        ("owner", "@unrelated", None, False),
        (None, "@1", "replacement", True),
        (None, "@unrelated", "replacement", False),
    ],
)
def test_quarantine_uses_exact_owner_then_unknown_route_fallback(
    store, saved_owner, route, observed_owner, blocked
):
    operation = lifecycle_state.queue_topic_mark("@1", saved_owner, True)
    lifecycle_state.begin_topic_mark_attempt(operation, _witness("A"))

    assert (
        lifecycle_state.is_topic_mark_quarantined(route, owner=observed_owner)
        is blocked
    )
    assert "@new-route" not in store.window_states
    assert "@unrelated" not in store.window_states


def test_changed_intent_cannot_arm_a_prechecked_old_payload(store):
    original = lifecycle_state.queue_topic_mark("@1", "owner", True)
    latest = lifecycle_state.queue_topic_mark("@1", "owner", False)

    with pytest.raises(ValueError):
        lifecycle_state.begin_topic_mark_attempt(original, _witness("A"))

    assert lifecycle_state.pending_topic_selections() == [latest]
    assert not lifecycle_state.is_topic_mark_quarantined("@1", owner="owner")


def test_failed_terminal_checkpoint_rollback_preserves_newer_intent(store):
    original = lifecycle_state.queue_topic_mark("@1", "owner", True)
    armed = lifecycle_state.begin_topic_mark_attempt(original, _witness("A"))
    latest = lifecycle_state.queue_topic_mark("@1", "owner", False)
    finished = lifecycle_state.finish_topic_mark_attempt(
        "@1", owner="owner", dispatch_id="A"
    )
    assert finished is not None and finished.desired is False

    lifecycle_state.restore_topic_mark(armed)

    assert lifecycle_state.pending_topic_selections() == [
        replace(latest, attempt=armed.attempt)
    ]
    assert lifecycle_state.is_topic_mark_quarantined("@1", owner="owner")


def test_ownerless_armed_evidence_blocks_admission_without_a_window_state(store):
    operation = lifecycle_state.queue_topic_mark("@1", None, True)
    armed = lifecycle_state.begin_topic_mark_attempt(operation, _witness("A"))
    assert not store.window_states

    assert not lifecycle_state.complete_topic_admission("@1", None)
    assert lifecycle_state.complete_topic_admission("@unrelated", None)

    assert lifecycle_state.pending_topic_selections() == [armed]
    assert not store.window_states
