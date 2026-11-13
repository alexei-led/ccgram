"""Lifecycle-state feature port — window origin.

Reads the project origin flag. Writes delegate to ``WindowStateStore``
setters which already validate input and schedule a single save per real
change.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..multiplexer.base import canonical_window_id
from ..window_state_store import (
    DEFAULT_WINDOW_ORIGIN,
    WINDOW_ORIGINS,
    window_store,
)


@dataclass(frozen=True, slots=True)
class LifecycleProjection:
    """Read-only snapshot of lifecycle/origin flags for a window."""

    window_id: str
    origin: str


def get_lifecycle(window_id: str) -> LifecycleProjection | None:
    """Lifecycle projection, or None if no state is tracked."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return None
    origin = state.origin if state.origin in WINDOW_ORIGINS else DEFAULT_WINDOW_ORIGIN
    return LifecycleProjection(window_id=window_id, origin=origin)


def get_origin(window_id: str) -> str:
    """Origin string for a window. Defaults to manual_discovered."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return DEFAULT_WINDOW_ORIGIN
    return state.origin if state.origin in WINDOW_ORIGINS else DEFAULT_WINDOW_ORIGIN


def set_window_origin(window_id: str, origin: str) -> None:
    """Set the lifecycle origin. Raises ValueError on unknown origin."""
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
    """Match unfinished clear by original routing key or exact native owner."""
    wanted = canonical_window_id(window_id)
    return any(
        state.topic_mark_pending is False
        and (
            canonical_window_id(key) == wanted
            or (owner is not None and state.topic_mark_owner == owner)
        )
        for key, state in window_store.window_states.items()
    )


def is_topic_detached(window_id: str) -> bool:
    """Whether the user intentionally left this terminal without a topic."""
    state = window_store.window_states.get(_selection_key(window_id))
    return state is not None and state.topic_detached


def get_pending_topic_mark(window_id: str) -> bool | None:
    """Desired native mark, or None when no write remains unfinished."""
    state = window_store.window_states.get(_selection_key(window_id))
    return state.topic_mark_pending if state is not None else None


def get_topic_mark_owner(window_id: str) -> str | None:
    """Return captured native selection identity, never a routing alias."""
    state = window_store.window_states.get(_selection_key(window_id))
    return state.topic_mark_owner if state is not None else None


def set_topic_selection(
    window_id: str, *, detached: bool, pending: bool | None, owner: str | None
) -> None:
    """Persist native ownership and lifecycle intent through their state owner."""
    window_store.set_topic_selection(
        _selection_key(window_id), detached=detached, pending=pending, owner=owner
    )


def observe_topic_mark_owner(window_id: str, owner: str | None) -> bool:
    """Refresh only an admitted, settled row from a confirmed native witness."""
    state = window_store.window_states.get(_selection_key(window_id))
    if state is None or not owner or state.topic_mark_pending is not None:
        return False
    set_topic_selection(
        window_id, detached=state.topic_detached, pending=None, owner=owner
    )
    return True


def record_topic_mark_owner(window_id: str, owner: str | None) -> bool:
    """Capture actual admission ownership without marking a native terminal."""
    if is_topic_detach_pending(window_id, owner=owner):
        return False
    state = window_store.window_states.get(_selection_key(window_id))
    if state is not None and state.topic_mark_pending is not None:
        return state.topic_mark_owner == owner
    set_topic_selection(
        window_id,
        detached=False,
        pending=None,
        owner=owner if owner is not None else get_topic_mark_owner(window_id),
    )
    return True


def pending_topic_selections() -> list[str]:
    """Snapshot targets whose native mark write remains unfinished."""
    return [
        window_id
        for window_id, state in window_store.window_states.items()
        if state.topic_mark_pending is not None
    ]


__all__ = [
    "LifecycleProjection",
    "get_lifecycle",
    "get_origin",
    "set_window_origin",
    "is_topic_detach_pending",
    "is_topic_detached",
    "get_pending_topic_mark",
    "get_topic_mark_owner",
    "observe_topic_mark_owner",
    "record_topic_mark_owner",
    "set_topic_selection",
    "pending_topic_selections",
]
