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
from .multiplexer.base import canonical_window_id
from .window_state_store import window_store

if TYPE_CHECKING:
    from .idle_tracker import IdleTracker

logger = structlog.get_logger()


def _same_transcript_path(
    old_details: dict[str, Any],
    new_details: dict[str, Any],
) -> bool:
    old_path = old_details.get("transcript_path", "")
    new_path = new_details.get("transcript_path", "")
    return bool(old_path and new_path and old_path == new_path)


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
        wanted = canonical_window_id(window_id)
        for wid, details in self._last_session_map.items():
            candidate = wid.rsplit(":", 1)[-1]
            if canonical_window_id(candidate) == wanted:
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
        old_by_canonical = {
            canonical_window_id(wid.rsplit(":", 1)[-1]): wid
            for wid in self._last_session_map
        }
        converged_map = {
            old_by_canonical.get(
                canonical_window_id(wid.rsplit(":", 1)[-1]), wid
            ): details
            for wid, details in current_map.items()
        }
        result = ReconcileResult(current_map=converged_map)

        old_windows = set(self._last_session_map.keys())
        current_windows = set(converged_map.keys())

        # Session changed: window in both maps but session_id differs
        for window_id, old_details in self._last_session_map.items():
            new_details = converged_map.get(window_id)
            if new_details and new_details["session_id"] != old_details["session_id"]:
                if _same_transcript_path(old_details, new_details):
                    logger.debug(
                        "Window '%s' session metadata refreshed for same transcript: %s -> %s",
                        window_id,
                        old_details["session_id"],
                        new_details["session_id"],
                    )
                    continue
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

        # Deleted: window in old map but not current
        for window_id in old_windows - current_windows:
            old_sid = self._last_session_map[window_id]["session_id"]
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
            result.new_windows[window_id] = converged_map[window_id]

        self._last_session_map = converged_map
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
