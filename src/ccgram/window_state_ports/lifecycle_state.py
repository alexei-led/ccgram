"""Lifecycle-state feature port — window origin.

Reads the project origin flag. Writes delegate to ``WindowStateStore``
setters which already validate input and schedule a single save per real
change.
"""

from __future__ import annotations

from dataclasses import dataclass

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


__all__ = [
    "LifecycleProjection",
    "get_lifecycle",
    "get_origin",
    "set_window_origin",
]
