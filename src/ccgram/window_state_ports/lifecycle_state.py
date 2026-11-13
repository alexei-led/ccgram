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


def is_topic_detach_pending(window_id: str) -> bool:
    """Whether the native mark clear from a deliberate detach needs retry."""
    state = window_store.window_states.get(_selection_key(window_id))
    return state is not None and state.topic_mark_pending is False


def is_topic_detached(window_id: str) -> bool:
    """Whether the user intentionally left this terminal without a topic."""
    state = window_store.window_states.get(_selection_key(window_id))
    return state is not None and state.topic_detached


def get_pending_topic_mark(window_id: str) -> bool | None:
    """Desired native mark, or None when no write remains unfinished."""
    state = window_store.window_states.get(_selection_key(window_id))
    return state.topic_mark_pending if state is not None else None


def set_topic_selection(
    window_id: str, *, detached: bool, pending: bool | None
) -> None:
    """Persist lifecycle intent through its state owner."""
    window_store.set_topic_selection(
        _selection_key(window_id), detached=detached, pending=pending
    )


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
    "set_topic_selection",
    "pending_topic_selections",
]
