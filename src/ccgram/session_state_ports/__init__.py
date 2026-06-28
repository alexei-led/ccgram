"""Feature-port package for live session state — narrow read seam.

Each module exposes frozen projection dataclasses for reads and thin
free functions that delegate to the owning state stores
(``claude_task_state``, ``session_lifecycle``, ``session_monitor``).
Ports do not own state; they are the approved read boundary between
handler code and the volatile session-state modules.

Write authority stays exclusively in the owning modules:
  ``session_lifecycle`` owns all state mutations; handlers must not
  call mutating methods on ``claude_task_state`` directly.
"""

from __future__ import annotations

from .live_session_state import (
    LiveSessionSnapshot,
    get_last_activity_ts,
    get_live_session_snapshot,
    get_session_id,
    get_task_snapshot,
    get_wait_header,
    has_task_snapshot,
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
