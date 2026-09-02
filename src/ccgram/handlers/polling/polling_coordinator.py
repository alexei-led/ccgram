"""Polling coordinator for terminal status monitoring."""

import asyncio
from typing import TYPE_CHECKING

import structlog
from telegram.error import TelegramError

from ...thread_router import chat_scope, thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...multiplexer.reconciliation import list_windows_for_reconciliation
from ...multiplexer.window_liveness import note_live_windows
from ...utils import log_throttled
from . import window_tick
from .polling_runtime import PollingRuntime

if TYPE_CHECKING:
    from telegram import Bot

    from ...multiplexer.base import WindowRef as TmuxWindow

logger = structlog.get_logger()

# ── Timing constants ──────────────────────────────────────────────────────

_BACKOFF_MIN = 2.0
_BACKOFF_MAX = 30.0

_LoopError = (TelegramError, OSError, RuntimeError, ValueError)
# ── Per-iteration tick helper ─────────────────────────────────────────────


async def _tick_bound_windows(
    bot: "Bot",
    window_lookup: "dict[str, TmuxWindow]",
    *,
    runtime: PollingRuntime | None = None,
) -> None:
    """Tick every thread-bound window once.

    Extracted so tests can drive a single iteration with an isolated
    ``PollingRuntime``. Production callers pass no runtime; default singletons.
    """
    bindings = list(thread_router.iter_thread_bindings_with_chat()) or [
        (uid, None, tid, wid) for uid, tid, wid in thread_router.iter_thread_bindings()
    ]
    for user_id, chat_id, thread_id, wid in bindings:
        if chat_id is None:
            chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(window_id=wid)
        try:
            with chat_scope(chat_id):
                w = window_lookup.get(wid)
                await window_tick.tick_window(
                    bot, user_id, thread_id, wid, w, runtime=runtime
                )
        except (TelegramError, OSError) as e:
            log_throttled(
                logger,
                f"status-update:{user_id}:{thread_id}",
                "Status update error for user %s thread %s: %s",
                user_id,
                thread_id,
                e,
            )


# ── Main loop ─────────────────────────────────────────────────────────────


async def status_poll_loop(bot: "Bot") -> None:
    """Background task to poll terminal status for all thread-bound windows."""
    # Lazy: imports keep PTB out of the polling package's cold path.
    from ...config import config as _cfg

    # Lazy: PTBTelegramClient wraps the live PTB bot — resolved per-tick
    from ...telegram_client import PTBTelegramClient

    # Lazy: periodic_tasks transitively imports topics.topic_lifecycle,
    # which imports polling_state. Hoisting forms a cycle through
    # polling/__init__.py whenever a module reaches polling_state
    # before polling_coordinator finishes loading.
    # Lazy: periodic_tasks ↔ coordinator cycle
    from .periodic_tasks import run_lifecycle_tasks, run_periodic_tasks

    poll_interval = _cfg.status_poll_interval
    client = PTBTelegramClient(bot)
    logger.info("Status polling started (interval: %ss)", poll_interval)
    timers = {"topic_check": 0.0, "live_view": 0.0}
    _error_streak = 0
    while True:
        try:
            all_windows = await list_windows_for_reconciliation(tmux_manager)
            if all_windows is None:
                logger.warning("Status poll skipped: window listing unavailable")
                await asyncio.sleep(poll_interval)
                continue
            note_live_windows(all_windows, thread_router.all_bound_window_ids())
            window_lookup = {w.window_id: w for w in all_windows}

            await run_periodic_tasks(client, all_windows, timers)
            await _tick_bound_windows(bot, window_lookup)
            await run_lifecycle_tasks(client, all_windows)

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
