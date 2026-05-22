"""Polling coordinator for terminal status monitoring.

Orchestrates the background polling cycle: iterates thread bindings,
delegates per-window work to window_tick, and runs periodic/lifecycle tasks.

Key components:
  - status_poll_loop: Background polling task (entry point for bot.py)
"""

import asyncio
from typing import TYPE_CHECKING

import structlog
from telegram.error import TelegramError

from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from ..utils import log_throttled
from . import window_tick
from .periodic_tasks import run_lifecycle_tasks, run_periodic_tasks

if TYPE_CHECKING:
    from telegram import Bot

logger = structlog.get_logger()

# ── Timing constants ──────────────────────────────────────────────────────

_BACKOFF_MIN = 2.0
_BACKOFF_MAX = 30.0

_LoopError = (TelegramError, OSError, RuntimeError, ValueError)


# ── Main loop ─────────────────────────────────────────────────────────────


async def status_poll_loop(bot: "Bot") -> None:
    """Background task to poll terminal status for all thread-bound windows."""
    from ..config import config as _cfg

    poll_interval = _cfg.status_poll_interval
    logger.info("Status polling started (interval: %ss)", poll_interval)
    timers = {"topic_check": 0.0, "broker": 0.0, "sweep": 0.0, "live_view": 0.0}
    _error_streak = 0
    while True:
        try:
            all_windows = await tmux_manager.list_windows()
            external_windows = await tmux_manager.discover_external_sessions()
            all_windows.extend(external_windows)
            window_lookup = {w.window_id: w for w in all_windows}

            await run_periodic_tasks(bot, all_windows, timers)

            for user_id, thread_id, wid in list(thread_router.iter_thread_bindings()):
                from ..utils import is_local_qualified
                if not is_local_qualified(wid):
                    continue

                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(window_id=wid)
                try:
                    w = window_lookup.get(wid)
                    await window_tick.tick_window(bot, user_id, thread_id, wid, w)
                except (TelegramError, OSError) as e:
                    log_throttled(
                        logger,
                        f"status-update:{user_id}:{thread_id}",
                        "Status update error for user %s thread %s: %s",
                        user_id,
                        thread_id,
                        e,
                    )

            await run_lifecycle_tasks(bot, all_windows)

        except _LoopError:
            logger.exception("Status poll loop error")
            backoff_delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**_error_streak))
            _error_streak += 1
            await asyncio.sleep(backoff_delay)
            continue
        except Exception:
            logger.exception("Unexpected error in status poll loop")
            backoff_delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**_error_streak))
            _error_streak += 1
            await asyncio.sleep(backoff_delay)
            continue

        _error_streak = 0
        await asyncio.sleep(poll_interval)
