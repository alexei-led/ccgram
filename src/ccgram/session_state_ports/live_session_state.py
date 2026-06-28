"""Live-session read port — frozen projections over volatile session state.

Thin adapter over ``claude_task_state`` (task snapshots, wait headers) and
``session_lifecycle`` (session-id resolution). Handlers must use this module
for reads; they must not import ``claude_task_state`` or ``session_lifecycle``
directly for read access.

Write authority stays in the respective owning modules:
  - ``session_lifecycle`` owns all mutation methods.
  - ``claude_task_state`` singleton owns task/wait-header mutations via
    ``session_lifecycle``'s handle_* methods.

Boundary enforced by ``tests/ccgram/test_session_state_ports_audit.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..claude_task_state import ClaudeTaskSnapshot


# ---------------------------------------------------------------------------
# Projection dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiveSessionSnapshot:
    """Read-only snapshot of volatile live-session state for a window.

    Fields
    ------
    window_id:
        The window this snapshot was built for.
    session_id:
        Active session id from ``session_lifecycle``, or ``None`` if the
        window has no known session (not yet in session_map, or hookless).
    task_snapshot:
        Current ``ClaudeTaskSnapshot`` if any task rows exist, else ``None``.
    wait_header:
        Hook-derived wait-state header (Notification hook), or ``None``.
    last_activity_ts:
        Monotonic timestamp of the last transcript activity for the window's
        session, or ``None`` if no activity has been recorded yet.
    """

    window_id: str
    session_id: str | None
    task_snapshot: "ClaudeTaskSnapshot | None"
    wait_header: str | None
    last_activity_ts: float | None


# ---------------------------------------------------------------------------
# Free functions — thin read adapters
# ---------------------------------------------------------------------------


def get_task_snapshot(window_id: str) -> "ClaudeTaskSnapshot | None":
    """Return the Claude task snapshot for *window_id*, or ``None``."""
    # Lazy: claude_task_state imports topic_state_registry; deferring avoids
    # loading the full task store at module import time.
    from ..claude_task_state import get_claude_task_snapshot  # Lazy:

    return get_claude_task_snapshot(window_id)


def has_task_snapshot(window_id: str) -> bool:
    """Return ``True`` if the window has at least one tracked task row."""
    # Lazy: claude_task_state imports topic_state_registry; defer load.
    from ..claude_task_state import claude_task_state  # Lazy:

    return claude_task_state.has_snapshot(window_id)


def get_wait_header(window_id: str) -> str | None:
    """Return the hook-derived wait-state header for *window_id*, or ``None``."""
    # Lazy: claude_task_state imports topic_state_registry; defer load.
    from ..claude_task_state import get_claude_wait_header  # Lazy:

    return get_claude_wait_header(window_id)


def get_session_id(window_id: str) -> str | None:
    """Return the active session_id for *window_id* from the last session map."""
    # Lazy: session_lifecycle imports window_store + claude_task_state; defer.
    from ..session_lifecycle import session_lifecycle  # Lazy:

    return session_lifecycle.resolve_session_id(window_id)


def get_last_activity_ts(window_id: str) -> float | None:
    """Return the monotonic last-activity timestamp for the window's session.

    Resolves the session_id via the session_lifecycle snapshot then reads
    from the active ``SessionMonitor``'s idle tracker.  Returns ``None`` when
    the window has no session or the monitor is not yet started.
    """
    # Lazy: avoid importing session_lifecycle + session_monitor (heavy) at module load.
    from ..session_lifecycle import session_lifecycle  # Lazy:

    # Lazy: session_monitor imports SessionMonitor with aiofiles; defer.
    from ..session_monitor import get_active_monitor  # Lazy:

    session_id = session_lifecycle.resolve_session_id(window_id)
    if not session_id:
        return None
    monitor = get_active_monitor()
    if monitor is None:
        return None
    return monitor.get_last_activity(session_id)


def get_live_session_snapshot(window_id: str) -> LiveSessionSnapshot:
    """Build a full ``LiveSessionSnapshot`` for *window_id*.

    Never raises — all fields degrade to ``None`` when the window has no
    tracked session or no volatile state has been recorded yet.
    """
    return LiveSessionSnapshot(
        window_id=window_id,
        session_id=get_session_id(window_id),
        task_snapshot=get_task_snapshot(window_id),
        wait_header=get_wait_header(window_id),
        last_activity_ts=get_last_activity_ts(window_id),
    )


__all__ = [
    "LiveSessionSnapshot",
    "get_last_activity_ts",
    "get_live_session_snapshot",
    "get_session_id",
    "get_task_snapshot",
    "get_wait_header",
    "has_task_snapshot",
]
