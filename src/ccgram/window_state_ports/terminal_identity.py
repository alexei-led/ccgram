"""Terminal-identity feature port — backend-neutral identity projection.

Reads return a frozen ``TerminalIdentityProjection`` (backend + opaque
unit id + legacy window id). Writes route through ``WindowStateStore``
so persistence stays in the kernel.

Legacy state without ``terminal_backend`` / ``terminal_unit_id`` loads
as ``backend=tmux`` with ``unit_id=<window_id>``. Handlers never parse
``unit_id``; they pass the projection back to the backend router.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..window_state_store import (
    DEFAULT_TERMINAL_BACKEND,
    TERMINAL_BACKEND_TMUX,
    TERMINAL_BACKENDS,
    window_store,
)


@dataclass(frozen=True, slots=True)
class TerminalIdentityProjection:
    """Read-only snapshot of a window's backend identity.

    ``backend`` is one of the known terminal backends; ``unit_id`` is
    opaque to handlers. ``legacy_window_id`` is the tmux-shaped key the
    rest of the system uses today (``@N``) — kept for compatibility
    until handlers fully migrate to ``TerminalUnitRef``.
    """

    window_id: str
    backend: str
    unit_id: str
    legacy_window_id: str


def get_terminal_identity(window_id: str) -> TerminalIdentityProjection | None:
    """Frozen terminal identity projection, or None if no state is tracked.

    Legacy rows without an explicit backend resolve to
    ``backend=tmux`` and ``unit_id=<window_id>``.
    """
    state = window_store.window_states.get(window_id)
    if state is None:
        return None
    backend = (
        state.terminal_backend
        if state.terminal_backend in TERMINAL_BACKENDS
        else DEFAULT_TERMINAL_BACKEND
    )
    # Empty unit_id for a tmux row falls back to the window_id key —
    # this preserves "no backend identity stored" as the legacy
    # invariant while still giving handlers a usable opaque id.
    unit_id = state.terminal_unit_id or (
        window_id if backend == TERMINAL_BACKEND_TMUX else ""
    )
    return TerminalIdentityProjection(
        window_id=window_id,
        backend=backend,
        unit_id=unit_id,
        legacy_window_id=window_id,
    )


def get_backend(window_id: str) -> str:
    """Resolved backend name for a window. Defaults to ``tmux``."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return DEFAULT_TERMINAL_BACKEND
    return (
        state.terminal_backend
        if state.terminal_backend in TERMINAL_BACKENDS
        else DEFAULT_TERMINAL_BACKEND
    )


def get_unit_id(window_id: str) -> str:
    """Opaque backend unit id for a window. Falls back to ``window_id`` for tmux.

    Returns an empty string when no state is tracked or a cmux row has
    no unit id persisted yet.
    """
    state = window_store.window_states.get(window_id)
    if state is None:
        return ""
    if state.terminal_unit_id:
        return state.terminal_unit_id
    backend = (
        state.terminal_backend
        if state.terminal_backend in TERMINAL_BACKENDS
        else DEFAULT_TERMINAL_BACKEND
    )
    return window_id if backend == TERMINAL_BACKEND_TMUX else ""


def set_terminal_identity(window_id: str, *, backend: str, unit_id: str) -> None:
    """Persist the terminal backend identity for a window.

    Raises ``ValueError`` on unknown backend or blank unit id. No-op
    when both values already match the persisted state.
    """
    window_store.set_terminal_identity(window_id, backend=backend, unit_id=unit_id)


__all__ = [
    "TerminalIdentityProjection",
    "get_backend",
    "get_terminal_identity",
    "get_unit_id",
    "set_terminal_identity",
]
