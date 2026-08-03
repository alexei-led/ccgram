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


def archive_legacy_herdr(window_id: str, user_id: int, thread_id: int) -> bool:
    """Retain a legacy record with its owner/topic for safe rollback."""
    return window_store.archive_legacy_herdr(window_id, user_id, thread_id)


def get_archived_legacy_herdr_binding(user_id: int, thread_id: int) -> str | None:
    """Return this owner/topic's archived binding without selecting a live target."""
    return window_store.get_archived_legacy_herdr_binding(user_id, thread_id)


def rollback_legacy_herdr_archive(window_id: str) -> bool:
    """Undo archive state without making the legacy target actionable."""
    return window_store.rollback_legacy_herdr_archive(window_id)
