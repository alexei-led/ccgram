"""Periodic task orchestration for the polling subsystem.

Orchestrates time-gated tasks within the poll loop: topic lifecycle management,
live view ticking, and state pruning.

Key components:
  - run_periodic_tasks: time-gated live view tick and topic check
  - run_lifecycle_tasks: per-tick unbound window management
"""

import time
from typing import TYPE_CHECKING

import structlog

from ...config import config
from ...telegram_client import TelegramClient
from ...utils import log_throttle_sweep
from ..live.live_view import tick_live_views
from ..topics.topic_deletion import cleanup_retired_topics
from ..topics.topic_provisioning_recovery import recover_topic_provisioning
from ..topics.topic_lifecycle import (
    check_unbound_window_ttl,
    probe_topic_existence,
    prune_stale_state,
)
from .window_tick.apply import _AUTODELETE_DEAD_TOPICS

if TYPE_CHECKING:
    from ...multiplexer.base import WindowRef as TmuxWindow

logger = structlog.get_logger()

# ── Timing constants ──────────────────────────────────────────────────────

TOPIC_CHECK_INTERVAL = 60.0  # seconds


# ── Orchestration ──────────────────────────────────────────────────────────


async def run_periodic_tasks(
    client: TelegramClient,
    all_windows: list["TmuxWindow"],
    timers: dict[str, float],
) -> None:
    """Run time-gated periodic tasks (topic check, live view)."""
    now = time.monotonic()

    if now - timers["live_view"] >= config.live_view_interval:
        timers["live_view"] = now
        await tick_live_views(client)

    if now - timers["topic_check"] >= TOPIC_CHECK_INTERVAL:
        timers["topic_check"] = now
        recovery = await recover_topic_provisioning(client)
        await prune_stale_state(all_windows)
        if not recovery.get("rate_limited"):
            await probe_topic_existence(client)
            # Retired-topic records only exist when deletion ran; with
            # autodelete off the drain would still delete records left
            # over from before the flag flipped, against the operator's
            # intent, so it stays off with the knob.
            if _AUTODELETE_DEAD_TOPICS:
                await cleanup_retired_topics(client)
        log_throttle_sweep()


async def run_lifecycle_tasks(
    _client: TelegramClient, all_windows: list["TmuxWindow"]
) -> None:
    """Run per-tick unbound window TTL management."""
    await check_unbound_window_ttl(all_windows)
