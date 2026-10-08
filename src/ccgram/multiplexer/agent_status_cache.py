"""Backend-neutral push-updated agent-status cache.

The herdr event-stream consumer writes the latest native ``AgentStatus`` per
window here; the status-polling layer (``observe._native_agent_status``) reads
it synchronously instead of forking a ``herdr`` subprocess every tick. This
bridges the push stream and the existing poll loop (the "augment" reconcile):
push keeps the cache fresh, the poll reads it, and a one-shot subprocess
``agent_status()`` is the cold-cache fallback.

Pure module — depends only on the seam value type (``AgentStatus``), never on a
concrete backend, so the polling layer can import it without crossing the F1
boundary (mirrors ``multiplexer.vim_state``). All access is from the single
asyncio event-loop thread, so a plain dict suffices.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic

from .base import AgentStatus

logger = logging.getLogger(__name__)

_NEGATIVE_TTL = 15.0
_POSITIVE_TTL = 90.0


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    status: AgentStatus | None
    expires_at: float | None = None


@dataclass(slots=True)
class _WindowProbeState:
    generation: int = 0
    references: int = 0


@dataclass(slots=True)
class _InFlightProbe:
    task: asyncio.Task[AgentStatus | None]
    waiters: int = 0


_cache: dict[str, _CacheEntry] = {}
# This map only tracks generations for active callers/probe tasks. Entries are
# removed when their final reference is released, so window churn is bounded.
_window_generations: dict[str, _WindowProbeState] = {}
_reset_generation = 0
_inflight_probes: dict[tuple[str, int, int], _InFlightProbe] = {}


# Kept as a function so cache expiry can be tested without wall-clock waits.
def _clock() -> float:
    return monotonic()


def _retain_generation(window_id: str) -> _WindowProbeState:
    state = _window_generations.get(window_id)
    if state is None:
        state = _WindowProbeState()
        _window_generations[window_id] = state
    state.references += 1
    return state


def _release_generation(window_id: str, state: _WindowProbeState) -> None:
    state.references -= 1
    if state.references == 0 and _window_generations.get(window_id) is state:
        _window_generations.pop(window_id)


def _generation(state: _WindowProbeState) -> tuple[int, int]:
    return _reset_generation, state.generation


def _bump_generation(window_id: str) -> None:
    state = _window_generations.get(window_id)
    if state is not None:
        state.generation += 1


def _get_entry(window_id: str) -> tuple[bool, AgentStatus | None]:
    entry = _cache.get(window_id)
    if entry is None:
        return False, None
    if entry.expires_at is not None and _clock() >= entry.expires_at:
        _cache.pop(window_id, None)
        return False, None
    return True, entry.status


def set_status(window_id: str, status: AgentStatus) -> None:
    """Record the latest push-reported agent status for *window_id*."""
    _bump_generation(window_id)
    _cache[window_id] = _CacheEntry(status, _clock() + _POSITIVE_TTL)


def get_status(window_id: str) -> AgentStatus | None:
    """Return the cached push/probe status, or None when cold or negative."""
    _, status = _get_entry(window_id)
    return status


async def get_status_or_probe(
    window_id: str,
    probe: Callable[[], Awaitable[AgentStatus | None]],
) -> AgentStatus | None:
    """Read a fresh cache entry, probing once on a miss.

    Pushed and successful probe results expire after 90 seconds; each push
    refreshes that TTL. Empty probe results are cached for 15 seconds.
    Concurrent callers share a probe. A newer push wins over a probe already
    in flight, and clear/reset invalidates an older probe even when the cache
    was cold at invalidation.
    """
    hit, status = _get_entry(window_id)
    if hit:
        return status

    state = _retain_generation(window_id)
    try:
        generation = _generation(state)
        probe_key = (window_id, *generation)
        in_flight = _inflight_probes.get(probe_key)
        if in_flight is None:
            async def run_probe() -> AgentStatus | None:
                return await probe()

            task = asyncio.create_task(run_probe())
            in_flight = _InFlightProbe(task)
            _inflight_probes[probe_key] = in_flight
            state.references += 1  # Keep state until the shared task is done.

            def remove_finished_probe(completed: asyncio.Task[AgentStatus | None]) -> None:
                if _inflight_probes.get(probe_key) is in_flight:
                    _inflight_probes.pop(probe_key, None)
                _release_generation(window_id, state)
                if completed.cancelled():
                    return
                error = completed.exception()
                if error is not None and in_flight.waiters == 0:
                    logger.error(
                        "Detached agent-status probe failed for %s: %s",
                        window_id,
                        error,
                    )

            task.add_done_callback(remove_finished_probe)

        in_flight.waiters += 1
        try:
            result = await asyncio.shield(in_flight.task)
        finally:
            in_flight.waiters -= 1

        # A push, clear, or reset that happened during the await supersedes this
        # probe. In particular, don't resurrect a cold entry after clear/reset.
        if _generation(state) != generation:
            hit, status = _get_entry(window_id)
            return status if hit else None

        if result is None:
            entry = _CacheEntry(None, _clock() + _NEGATIVE_TTL)
        else:
            entry = _CacheEntry(result, _clock() + _POSITIVE_TTL)
        _cache[window_id] = entry
        _bump_generation(window_id)
        return result
    finally:
        _release_generation(window_id, state)


def clear(window_id: str) -> None:
    """Drop the cached status for *window_id* (e.g. on window death)."""
    _bump_generation(window_id)
    _cache.pop(window_id, None)


def reset() -> None:
    """Clear the whole cache (consumer shutdown; test isolation)."""
    global _reset_generation
    _reset_generation += 1
    _cache.clear()
