"""Lifecycle feature port for observed ownership and unfinished native effects."""

from __future__ import annotations

from dataclasses import dataclass

from ..multiplexer.base import canonical_window_id
from ..topic_mark_journal import TopicMarkOperation, operation_matches
from ..topic_mark_types import TopicMarkWriteWitness
from ..window_state_store import DEFAULT_WINDOW_ORIGIN, WINDOW_ORIGINS, window_store


@dataclass(frozen=True, slots=True)
class LifecycleProjection:
    window_id: str
    origin: str


def get_lifecycle(window_id: str) -> LifecycleProjection | None:
    state = window_store.window_states.get(window_id)
    if state is None:
        return None
    origin = state.origin if state.origin in WINDOW_ORIGINS else DEFAULT_WINDOW_ORIGIN
    return LifecycleProjection(window_id=window_id, origin=origin)


def get_origin(window_id: str) -> str:
    state = window_store.window_states.get(window_id)
    if state is None:
        return DEFAULT_WINDOW_ORIGIN
    return state.origin if state.origin in WINDOW_ORIGINS else DEFAULT_WINDOW_ORIGIN


def set_window_origin(window_id: str, origin: str) -> None:
    window_store.set_window_origin(window_id, origin)


def _selection_key(window_id: str) -> str:
    wanted = canonical_window_id(window_id)
    return next(
        (
            key
            for key in window_store.window_states
            if canonical_window_id(key) == wanted
        ),
        window_id,
    )


def is_topic_detach_pending(window_id: str, *, owner: str | None) -> bool:
    return any(
        not operation.desired and operation_matches(operation, window_id, owner)
        for operation in window_store.topic_mark_operations.values()
    )


def is_topic_mark_quarantined(window_id: str, *, owner: str | None) -> bool:
    return any(
        operation.attempt is not None and operation_matches(operation, window_id, owner)
        for operation in window_store.topic_mark_operations.values()
    )


def begin_topic_mark_attempt(
    operation: TopicMarkOperation, witness: TopicMarkWriteWitness
) -> TopicMarkOperation:
    return window_store.begin_topic_mark_attempt(operation, witness)


def finish_topic_mark_attempt(
    window_id: str, *, owner: str | None, dispatch_id: str
) -> TopicMarkOperation | None:
    return window_store.finish_topic_mark_attempt(
        window_id, owner=owner, dispatch_id=dispatch_id
    )


def is_topic_detached(window_id: str) -> bool:
    state = window_store.window_states.get(_selection_key(window_id))
    return state is not None and state.topic_detached


def get_topic_mark_owner(window_id: str) -> str | None:
    state = window_store.window_states.get(_selection_key(window_id))
    return state.topic_mark_owner if state is not None else None


def get_pending_topic_mark(window_id: str, *, owner: str | None = None) -> bool | None:
    """Read unfinished intent for the supplied or currently observed resource."""
    current_owner = owner if owner is not None else get_topic_mark_owner(window_id)
    matching = [
        operation.desired
        for operation in window_store.topic_mark_operations.values()
        if operation_matches(operation, window_id, current_owner)
    ]
    return min(matching) if matching else None


def queue_topic_mark(
    window_id: str, owner: str | None, desired: bool
) -> TopicMarkOperation:
    return window_store.queue_topic_mark(window_id, owner, desired)


def topic_mark_operation_is_current(operation: TopicMarkOperation) -> bool:
    current = window_store.topic_mark_operations.get(operation.key)
    return current == operation


def settle_topic_mark(operation: TopicMarkOperation) -> bool:
    return window_store.settle_topic_mark(operation)


def restore_topic_mark(operation: TopicMarkOperation) -> None:
    window_store.restore_topic_mark(operation)


def pending_topic_selections() -> list[TopicMarkOperation]:
    return list(window_store.topic_mark_operations.values())


def set_topic_selection(
    window_id: str, *, detached: bool, pending: bool | None, owner: str | None
) -> None:
    """Record current metadata or enqueue an immutable native side effect.

    None is metadata-only, never a command to discard an unfinished operation.
    """
    key = _selection_key(window_id)
    state = window_store.window_states.get(key)
    if pending is None or state is None or state.topic_mark_owner in (None, owner):
        window_store.set_topic_selection(key, detached=detached, owner=owner)
    if pending is not None:
        queue_topic_mark(window_id, owner, pending)


def observe_topic_mark_owner(window_id: str, owner: str | None) -> bool:
    state = window_store.window_states.get(_selection_key(window_id))
    if state is None or not owner:
        return False
    window_store.set_topic_selection(
        _selection_key(window_id), detached=state.topic_detached, owner=owner
    )
    return True


def record_topic_mark_owner(window_id: str, owner: str | None) -> bool:
    if is_topic_detach_pending(window_id, owner=owner) or is_topic_mark_quarantined(
        window_id, owner=owner
    ):
        return False
    state = window_store.window_states.get(_selection_key(window_id))
    window_store.set_topic_selection(
        _selection_key(window_id),
        detached=state.topic_detached if state is not None else False,
        owner=owner if owner is not None else get_topic_mark_owner(window_id),
    )
    return True


def complete_topic_admission(window_id: str, owner: str | None) -> bool:
    """Clear lifetime protection only after an actual synchronous bind."""
    if is_topic_mark_quarantined(window_id, owner=owner):
        return False
    state = window_store.window_states.get(_selection_key(window_id))
    if state is None:
        return owner is None
    if owner is not None and state.topic_mark_owner != owner:
        return False
    effective_owner = owner if owner is not None else state.topic_mark_owner
    if is_topic_detach_pending(
        window_id, owner=effective_owner
    ) or is_topic_mark_quarantined(window_id, owner=effective_owner):
        return False
    for key, candidate in list(window_store.window_states.items()):
        related = (
            candidate.topic_mark_owner == effective_owner
            if effective_owner is not None and candidate.topic_mark_owner is not None
            else canonical_window_id(key) == canonical_window_id(window_id)
        )
        if related and candidate.topic_detached:
            window_store.set_topic_selection(
                key, detached=False, owner=candidate.topic_mark_owner
            )
    return True


def record_topic_detached(window_id: str, owner: str | None) -> None:
    """Protect the detached lifetime without replacing another observed owner."""
    key = _selection_key(window_id)
    state = window_store.window_states.get(key)
    if state is None:
        window_store.set_topic_selection(key, detached=True, owner=owner)
    for candidate_key, candidate in list(window_store.window_states.items()):
        related = (
            candidate.topic_mark_owner == owner
            if owner is not None and candidate.topic_mark_owner is not None
            else canonical_window_id(candidate_key) == canonical_window_id(window_id)
        )
        if related:
            window_store.set_topic_selection(
                candidate_key, detached=True, owner=candidate.topic_mark_owner
            )


__all__ = [
    "LifecycleProjection",
    "get_lifecycle",
    "get_origin",
    "set_window_origin",
    "is_topic_detach_pending",
    "is_topic_mark_quarantined",
    "begin_topic_mark_attempt",
    "finish_topic_mark_attempt",
    "is_topic_detached",
    "get_pending_topic_mark",
    "get_topic_mark_owner",
    "observe_topic_mark_owner",
    "record_topic_mark_owner",
    "complete_topic_admission",
    "record_topic_detached",
    "set_topic_selection",
    "queue_topic_mark",
    "topic_mark_operation_is_current",
    "settle_topic_mark",
    "restore_topic_mark",
    "pending_topic_selections",
]
