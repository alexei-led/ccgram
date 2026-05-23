"""Worktree-state feature port — git worktree path/branch projection.

Reads return a frozen projection; writes set or clear worktree metadata
through ``WindowStateStore`` (single save per real change). Behavior
parity with ``SessionManager.set_window_worktree`` is intentional —
that method remains the public write/admin facade for handlers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..window_state_store import window_store


@dataclass(frozen=True, slots=True)
class WorktreeProjection:
    """Read-only snapshot of a window's worktree metadata."""

    window_id: str
    worktree_path: str | None
    worktree_branch: str | None


def get_worktree(window_id: str) -> WorktreeProjection | None:
    """Worktree projection, or None if no state is tracked."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return None
    return WorktreeProjection(
        window_id=window_id,
        worktree_path=state.worktree_path,
        worktree_branch=state.worktree_branch,
    )


def set_worktree(window_id: str, worktree_path: str, branch: str) -> None:
    """Persist the git worktree path + branch for a window.

    Writes both fields atomically and schedules a single save. No-op if
    both values already match the current state.
    """
    state = window_store.get_window_state(window_id)
    if state.worktree_path == worktree_path and state.worktree_branch == branch:
        return
    state.worktree_path = worktree_path
    state.worktree_branch = branch
    window_store._schedule_save()


def clear_worktree(window_id: str) -> None:
    """Clear worktree path/branch metadata for a window."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return
    if state.worktree_path is None and state.worktree_branch is None:
        return
    state.worktree_path = None
    state.worktree_branch = None
    window_store._schedule_save()


__all__ = [
    "WorktreeProjection",
    "clear_worktree",
    "get_worktree",
    "set_worktree",
]
