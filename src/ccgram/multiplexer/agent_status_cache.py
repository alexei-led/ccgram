"""Backend-neutral push-updated agent-status cache.

The herdr event-stream consumer writes the latest native ``AgentStatus`` per
window here; the status-polling layer (``observe._native_agent_status``) reads
it synchronously instead of forking a ``herdr`` subprocess every tick. This
bridges the push stream and the existing poll loop (the "augment" reconcile):
push keeps the cache fresh, the poll reads it, and a one-shot subprocess
``agent_status()`` is the cold-cache fallback.

Depends on the seam proxy (never a concrete backend), so the polling layer
can import it without crossing the F1 boundary (mirrors
``multiplexer.vim_state``). All access is from the single asyncio event-loop
thread, so a plain dict suffices.
"""

from __future__ import annotations

import time

import structlog

from . import multiplexer
from .base import AgentStatus

# One entry per window: (monotonic stamp, status). The stamp ages entries:
# a push stream that drops silently leaves "working" frozen, which would
# wrongly suppress real prompts (TASK-47, 2026-09-30).
# The value is None when a probe answered "no status" (negative cache).
_cache: dict[str, tuple[float, AgentStatus | None]] = {}

# A push entry older than this is treated as cold: the stream may have
# dropped without anyone noticing, and a frozen "working" would wrongly
# suppress real prompts (TASK-47 cold-cache incident, 2026-09-30).
_STATUS_TTL_S = 90.0

# A None probe answer is remembered for a short window so a backend that
# persistently answers None (or fails) cannot turn every gate evaluation
# into a probe.
_NEGATIVE_TTL_S = 15.0

logger = structlog.get_logger()


def set_status(window_id: str, status: AgentStatus) -> None:
    """Record the latest push-reported agent status for *window_id*."""
    _cache[window_id] = (time.monotonic(), status)


def clear(window_id: str) -> None:
    """Drop the cached status for *window_id* (e.g. on window death)."""
    _cache.pop(window_id, None)


def reset() -> None:
    """Clear the whole cache (consumer shutdown; test isolation)."""
    _cache.clear()


def _fresh_status(window_id: str) -> AgentStatus | None:
    """The cached answer when fresh, else None (cold, stale, or expired).

    A None answer (probe said no status) is remembered with a shorter
    negative TTL; a real status uses the full TTL.
    """
    entry = _cache.get(window_id)
    if entry is None:
        return None
    stamp, status = entry
    ttl = _NEGATIVE_TTL_S if status is None else _STATUS_TTL_S
    if time.monotonic() - stamp > ttl:
        return None
    return status


async def resolve_agent_status(window_id: str) -> AgentStatus | None:
    """The window's agent status, resolving cold or stale cache entries.

    Herdr pushes status CHANGES only: after a restart a window already
    working pushes nothing, and on a healthy quiet stream every entry
    ages past the TTL. Both residuals are closed with one backend
    ``agent_status()`` read whose result is written back, so the next
    resolution inside the TTL is a dict hit. Backends without native
    status (tmux) answer None. A failed probe never breaks the caller.
    """
    status = _fresh_status(window_id)
    if status is not None:
        return status
    # Ordering guard: a push landing during the probe is newer than the
    # probe's snapshot; writing the probe result back would clobber it and,
    # because pushes are change-only, nothing would correct it until the TTL
    # expires.
    stamp_before = _cache.get(window_id, (-1.0, None))[0]
    try:
        status = await multiplexer.agent_status(window_id)
    except Exception:  # noqa: BLE001  # a failed probe must not break ticks
        logger.debug("agent_status probe failed", window_id=window_id)
        status = None
    if stamp_before != _cache.get(window_id, (-1.0, None))[0]:
        return status
    if status is not None:
        set_status(window_id, status)
    else:
        _cache[window_id] = (time.monotonic(), None)
    return status


async def resolve_agent_working(window_id: str) -> bool:
    """Whether the window's agent is working, resolving cold or stale state.

    The TASK-47 gates use this: a selection-shaped region on a working
    pane is Claude Code's queued-input block, not a prompt, and a stale
    frozen "working" must not suppress real prompts either.
    """
    status = await resolve_agent_status(window_id)
    return status is not None and status.state == "working"
