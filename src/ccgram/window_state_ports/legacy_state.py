"""Read and transition the blocked legacy-Herdr migration state.

This narrow port keeps handlers away from the persisted ``WindowState`` model.
A legacy record is retained only for archive/rollback; it never becomes an
actionable target.
"""

from __future__ import annotations

from ..window_state_store import window_store


def is_legacy_herdr(window_id: str) -> bool:
    """Return whether this binding is migration-only and action-blocked."""
    return window_store.is_legacy_herdr(window_id)


def archive_legacy_herdr(window_id: str) -> bool:
    """Retain the record for rollback while callers remove its topic binding."""
    return window_store.archive_legacy_herdr(window_id)


def rollback_legacy_herdr_archive(window_id: str) -> bool:
    """Undo archive state without making the legacy target actionable."""
    return window_store.rollback_legacy_herdr_archive(window_id)
