"""Backend-neutral liveness state for queued message delivery.

A confirmed multiplexer listing is the authority for deciding whether a
queued task may still target a session.  Until the first confirmed listing,
and while a listing is unavailable, callers fail open so a transient backend
outage does not discard delivery work.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .base import WindowRef, canonical_window_id
from ..window_resolver import resolve_window_alias


_live_window_ids: set[str] | None = None
_known_window_ids: set[str] = set()


def note_live_windows(
    windows: Sequence[WindowRef], tracked_window_ids: Iterable[str] = ()
) -> None:
    """Record one confirmed listing and the tracked IDs it makes authoritative.

    Tracked IDs are included in the known set even when absent from the listing.
    This lets a stale topic binding become droppable, while newly-created IDs
    not seen by this listing remain unknown until the next reconciliation pass.
    """
    global _live_window_ids
    live_ids = {
        canonical_window_id(resolve_window_alias(window.window_id))
        for window in windows
    }
    _known_window_ids.update(live_ids)
    _known_window_ids.update(
        canonical_window_id(resolve_window_alias(window_id))
        for window_id in tracked_window_ids
        if window_id
    )
    _live_window_ids = live_ids


def is_window_live(window_id: str) -> bool:
    """Return whether *window_id* is live, failing open when its state is unknown."""
    if not window_id or _live_window_ids is None:
        return True
    canonical_id = canonical_window_id(resolve_window_alias(window_id))
    return canonical_id not in _known_window_ids or canonical_id in _live_window_ids


def reset_window_liveness() -> None:
    """Reset liveness state for shutdown and test isolation."""
    global _live_window_ids
    _live_window_ids = None
    _known_window_ids.clear()


__all__ = ["is_window_live", "note_live_windows", "reset_window_liveness"]
