"""Session lifecycle management — single write authority for claude_task_state mutations.

Owns the session_map diff logic: compares old vs. new session_map to detect
session changes and deleted windows. Returns a structured result so the
coordinator (SessionMonitor) can clean up its own per-session state.

All mutations to claude_task_state are routed through this module so there
is one place to audit for state consistency. hook_events.py and window_tick.py
may still *read* claude_task_state (format_completion_text, has_snapshot,
get_wait_header) but must not call mutating methods directly.

Key class: SessionLifecycle. Module-level singleton: session_lifecycle.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .claude_task_state import (
    add_subagent,
    clear_subagents,
    claude_task_state,
    remove_subagent,
)
from .window_state_store import window_store

if TYPE_CHECKING:
    from .idle_tracker import IdleTracker

logger = structlog.get_logger()


@dataclass
class ReconcileResult:
    """Result of a session_map reconciliation pass."""

    sessions_to_remove: set[str] = field(default_factory=set)
    new_windows: dict[str, dict[str, Any]] = field(default_factory=dict)
    changed_windows: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_map: dict[str, dict[str, Any]] = field(default_factory=dict)


class SessionLifecycle:
    """Detects session_map changes and consolidates task-state cleanup."""

    def __init__(self) -> None:
        self._last_session_map: dict[str, dict[str, str]] = {}

    @property
    def last_session_map(self) -> dict[str, dict[str, str]]:
        return self._last_session_map

    def resolve_session_id(self, window_id: str) -> str | None:
        """Return the session_id for window_id from the last known session_map."""
        for wid, details in self._last_session_map.items():
            if wid.endswith(f":{window_id}") or wid == window_id:
                return details.get("session_id")
        return None

    def reconcile(
        self,
        current_map: dict[str, dict[str, Any]],
        idle_tracker: IdleTracker,
    ) -> ReconcileResult:
        """Diff current_map against last known map; clean up stale sessions.

        Calls claude_task_state.clear_window() for changed/deleted windows.
        Returns sessions to remove and new windows so the coordinator can
        clean up its own per-session state dicts.
        """
        result = ReconcileResult(current_map=current_map)

        old_windows = set(self._last_session_map.keys())
        current_windows = set(current_map.keys())

        # Session changed: window in both maps but session_id differs
        for window_id, old_details in self._last_session_map.items():
            new_details = current_map.get(window_id)
            if new_details and new_details["session_id"] != old_details["session_id"]:
                logger.info(
                    "Window '%s' session changed: %s -> %s",
                    window_id,
                    old_details["session_id"],
                    new_details["session_id"],
                )
                result.sessions_to_remove.add(old_details["session_id"])
                result.changed_windows[window_id] = new_details
                idle_tracker.clear_session(old_details["session_id"])
                claude_task_state.clear_window(window_id)

        current_session_ids = {
            details.get("session_id")
            for details in current_map.values()
            if details.get("session_id")
        }

        # Deleted: window in old map but not current.  A window may also be
        # re-keyed while the underlying agent session stays the same (for
        # example during migration from window-name keys to stable window_id
        # keys, or when a tmux window rename refreshes session_map display
        # metadata).  In that case the transcript offset and pending-tool state
        # are still live for another current window, so do not clear them.
        for window_id in old_windows - current_windows:
            old_sid = self._last_session_map[window_id]["session_id"]
            if old_sid in current_session_ids:
                logger.debug(
                    "Window '%s' deleted/re-keyed but session %s remains active; "
                    "preserving per-session state",
                    window_id,
                    old_sid,
                )
                continue
            logger.info(
                "Window '%s' deleted, removing session %s",
                window_id,
                old_sid,
            )
            result.sessions_to_remove.add(old_sid)
            idle_tracker.clear_session(old_sid)
            claude_task_state.clear_window(window_id)

        # New windows
        for window_id in current_windows - old_windows:
            result.new_windows[window_id] = current_map[window_id]

        self._last_session_map = current_map
        return result

    def handle_subagent_start(self, window_id: str, subagent_id: str, name: str) -> int:
        """Single write authority for SubagentStart — add subagent to tracking."""
        return add_subagent(window_id, subagent_id, name)

    def handle_subagent_stop(self, window_id: str, subagent_id: str) -> tuple[str, int]:
        """Single write authority for SubagentStop — remove subagent from tracking."""
        return remove_subagent(window_id, subagent_id)

    def handle_session_end(self, window_id: str) -> None:
        """Single cleanup point for SessionEnd: task state, subagents, session mapping."""
        claude_task_state.clear_window(window_id)
        clear_subagents(window_id)
        window_store.clear_window_session(window_id)

    # ── Hook-event mutation authority ────────────────────────────────────────

    def handle_notification_wait(self, window_id: str, wait_header: str) -> None:
        """Set wait-state header when a Notification hook fires."""
        claude_task_state.set_wait_header(window_id, wait_header)

    def handle_stop_task_state(self, window_id: str) -> None:
        """Clear wait-state header when a Stop hook fires."""
        claude_task_state.clear_wait_header(window_id)

    def handle_task_completed(
        self,
        window_id: str,
        session_id: str,
        task_id: str,
        subject: str,
    ) -> bool:
        """Mark task as completed; return True if it was tracked in the task list."""
        return claude_task_state.mark_task_completed(
            window_id, session_id, task_id, subject=subject
        )

    def initialize(self, session_map: dict[str, dict[str, str]]) -> None:
        """Set initial session_map (called once at monitor startup)."""
        self._last_session_map = session_map


session_lifecycle = SessionLifecycle()
