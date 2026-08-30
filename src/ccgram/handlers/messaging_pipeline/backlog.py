"""Queue-backlog telemetry boundary for status presentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BacklogSnapshot:
    """Pending source work and its most recently observed delivery lag."""

    pending_count: int
    oldest_age_seconds: float
    delivery_lag_seconds: float | None


_snapshot_provider: Callable[[int, str, int | None], BacklogSnapshot] | None = None


def register_snapshot_provider(
    provider: Callable[[int, str, int | None], BacklogSnapshot],
) -> None:
    """Install the queue-owned telemetry provider without a status back-edge."""
    global _snapshot_provider
    _snapshot_provider = provider


def get_backlog_snapshot(
    user_id: int, window_id: str, thread_id: int | None
) -> BacklogSnapshot:
    """Return queue telemetry, or an empty snapshot before queue initialization."""
    if _snapshot_provider is None:
        return BacklogSnapshot(0, 0.0, None)
    return _snapshot_provider(user_id, window_id, thread_id)
