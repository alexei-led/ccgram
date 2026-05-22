"""Periodic task orchestration for the polling subsystem.

Orchestrates time-gated tasks within the poll loop: message broker delivery,
mailbox sweep, spawn request processing, topic lifecycle management, live view
ticking, and state pruning.

Key components:
  - run_periodic_tasks: time-gated broker, sweep, live view tick, and topic check
  - run_lifecycle_tasks: per-tick autoclose and unbound window management
  - run_broker_cycle: message broker delivery (also called from hook_events)
"""

import time
from typing import TYPE_CHECKING

import structlog
from telegram.error import TelegramError

from ...config import config
from ...telegram_client import TelegramClient
from ...tmux_manager import tmux_manager
from ...utils import log_throttle_sweep
from ..live.live_view import tick_live_views
from ..messaging.msg_broker import BROKER_CYCLE_INTERVAL, SWEEP_INTERVAL
from ..topics.topic_lifecycle import (
    check_autoclose_timers,
    check_unbound_window_ttl,
    probe_topic_existence,
    prune_stale_state,
)

if TYPE_CHECKING:
    from ...tmux_manager import TmuxWindow

logger = structlog.get_logger()

# ── Timing constants ──────────────────────────────────────────────────────

TOPIC_CHECK_INTERVAL = 60.0  # seconds


# ── Broker integration ────────────────────────────────────────────────────


async def run_broker_cycle(
    client: TelegramClient | None = None,
    idle_windows: frozenset[str] = frozenset(),
) -> None:
    """Run one broker delivery cycle (called from poll loop and hook_events)."""
    # Lazy: msg_broker is registered as a callback target via the broker
    # registry; importing it at top of periodic_tasks pulls the
    # messaging subpackage into the polling package's cold path.
    # Lazy: imports resolved per-tick so tests can swap singletons
    from ... import window_query

    # Lazy: imports resolved per-tick so tests can swap singletons
    from ...mailbox import Mailbox

    # Lazy: messaging ↔ polling cycle through msg_telegram
    from ..messaging.msg_broker import broker_delivery_cycle

    mailbox = Mailbox(config.mailbox_dir)
    await broker_delivery_cycle(
        mailbox=mailbox,
        tmux_mgr=tmux_manager,
        window_ids=window_query.iter_window_ids(),
        tmux_session=config.tmux_session_name,
        msg_rate_limit=config.msg_rate_limit,
        client=client,
        idle_windows=idle_windows,
    )
    if client is not None:
        await _run_spawn_cycle(client)


async def _run_spawn_cycle(client: TelegramClient) -> None:
    """Scan for file-based spawn requests and post approval keyboards or auto-approve."""
    # Lazy: msg_spawn pulls topic_orchestration which sits inside the
    # sync_command cycle; keep at call site.
    # Lazy: spawn pipeline reaches back into polling
    from ...spawn_request import pop_pending, scan_spawn_requests

    # Lazy: spawn pipeline reaches back into polling
    from ..messaging.msg_spawn import (
        handle_spawn_approval,
        post_spawn_approval_keyboard,
    )

    new_requests = scan_spawn_requests(spawn_timeout=config.msg_spawn_timeout)
    for req in new_requests:
        try:
            if req.auto or config.msg_auto_spawn:
                await handle_spawn_approval(
                    req.id, client, spawn_timeout=config.msg_spawn_timeout
                )
            else:
                posted = await post_spawn_approval_keyboard(
                    client, req.requester_window, req
                )
                if not posted:
                    # Same lost-work case as the except below: the approval
                    # keyboard could not be posted (e.g. no topic bound to the
                    # requester window), so the spawn silently never happens.
                    pop_pending(req.id)
                    logger.warning(
                        "Dropped spawn request: approval keyboard not posted",
                        request_id=req.id,
                        requester_window=req.requester_window,
                    )
        except (OSError, TelegramError) as exc:
            # The request is discarded (pop_pending), so the spawn the user
            # asked for silently never happens — surface it at WARNING with
            # detail, not a swallowed debug line.
            pop_pending(req.id)
            logger.warning(
                "Dropped spawn request after error posting approval",
                request_id=req.id,
                requester_window=req.requester_window,
                error=str(exc),
            )


def _run_mailbox_sweep() -> None:
    """Run periodic mailbox sweep."""
    # Lazy: Mailbox is a leaf module; loading inside the sweep keeps
    # the polling subpackage's import surface narrow.
    # Lazy: imports resolved per-tick so tests can swap singletons
    from ...mailbox import Mailbox

    mailbox = Mailbox(config.mailbox_dir)
    removed = mailbox.sweep()
    if removed:
        logger.debug("Mailbox sweep removed %d messages", removed)


# ── Orchestration ──────────────────────────────────────────────────────────


async def run_periodic_tasks(
    client: TelegramClient,
    all_windows: list["TmuxWindow"],
    timers: dict[str, float],
) -> None:
    """Run time-gated periodic tasks (topic check, broker, sweep)."""
    now = time.monotonic()

    if now - timers["live_view"] >= config.live_view_interval:
        timers["live_view"] = now
        await tick_live_views(client)

    if now - timers["topic_check"] >= TOPIC_CHECK_INTERVAL:
        timers["topic_check"] = now
        await prune_stale_state(all_windows)
        await probe_topic_existence(client)
        log_throttle_sweep()

    if now - timers["broker"] >= BROKER_CYCLE_INTERVAL:
        timers["broker"] = now
        await run_broker_cycle(client)

    if now - timers["sweep"] >= SWEEP_INTERVAL:
        timers["sweep"] = now
        _run_mailbox_sweep()


async def run_lifecycle_tasks(
    client: TelegramClient, all_windows: list["TmuxWindow"]
) -> None:
    """Run per-tick lifecycle tasks (autoclose timers, unbound window TTL)."""
    await check_autoclose_timers(client)
    await check_unbound_window_ttl(all_windows)
