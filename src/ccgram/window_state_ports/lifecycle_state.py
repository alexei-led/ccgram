"""Lifecycle-state feature port — origin/external + Gemini warning.

Reads project origin/external flags plus the persisted external-Gemini
warning bit. Writes delegate to ``WindowStateStore`` setters which
already validate input and schedule a single save per real change.
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
    external: bool
    gemini_external_warned: bool


def get_lifecycle(window_id: str) -> LifecycleProjection | None:
    """Lifecycle projection, or None if no state is tracked."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return None
    origin = state.origin if state.origin in WINDOW_ORIGINS else DEFAULT_WINDOW_ORIGIN
    return LifecycleProjection(
        window_id=window_id,
        origin=origin,
        external=state.external,
        gemini_external_warned=state.gemini_external_warned,
    )


def get_origin(window_id: str) -> str:
    """Origin string for a window. Defaults to manual_discovered."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return DEFAULT_WINDOW_ORIGIN
    return state.origin if state.origin in WINDOW_ORIGINS else DEFAULT_WINDOW_ORIGIN


def is_external(window_id: str) -> bool:
    """True if the window is owned by an external tool (e.g. emdash)."""
    state = window_store.window_states.get(window_id)
    return bool(state and state.external)


def was_gemini_external_warned(window_id: str) -> bool:
    """True if the external-Gemini shell-mode warning was already shown."""
    return window_store.was_gemini_external_warned(window_id)


def set_window_origin(window_id: str, origin: str) -> None:
    """Set the lifecycle origin. Raises ValueError on unknown origin."""
    window_store.set_window_origin(window_id, origin)


def mark_gemini_external_warned(window_id: str) -> None:
    """Mark that the external-Gemini warning was shown for this window."""
    window_store.mark_gemini_external_warned(window_id)


__all__ = [
    "LifecycleProjection",
    "get_lifecycle",
    "get_origin",
    "is_external",
    "mark_gemini_external_warned",
    "set_window_origin",
    "was_gemini_external_warned",
]
