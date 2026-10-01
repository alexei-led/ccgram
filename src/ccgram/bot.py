"""Telegram bot factory + lifecycle delegates.

This is the top-level wiring point for the PTB ``Application``. The
actual handler bodies live in feature subpackages under ``handlers/``;
``register_all`` composes them into the application, and ``bootstrap``
runs the post-init / post-shutdown sequence.

Responsibilities kept here:
  - ``create_bot`` — PTB ``Application.builder()`` factory
  - lifecycle delegates: ``post_init``, ``post_stop``, ``post_shutdown``
  - ``_error_handler`` — top-level error funnel for the application
  - ``_send_shutdown_notification`` — best-effort goodbye message
  - ``_group_filter`` / ``is_user_allowed`` — auth/filter helpers used
    by the message handler registry
"""

import logging
import os
import signal
import threading
import time

import structlog
from telegram.error import BadRequest, Conflict, NetworkError, RetryAfter
from telegram.ext import (
    Application,
    ContextTypes,
    filters,
)

from . import bootstrap
from .config import config
from .handlers.inline import inline_query_handler, unsupported_content_handler
from .handlers.commands import commands_command, toolbar_command
from .handlers.messaging_pipeline import toolcalls_command, verbose_command
from .handlers.messaging_pipeline.message_sender import safe_reply
from .handlers.recovery.history import history_command
from .handlers.registry import register_all
from .handlers.text.text_handler import handle_text_message, text_handler
from .handlers.topics import new_command
from .handlers.topics.directory_browser import clear_browse_state
from .session import session_manager
from .telegram_rate_limiter import CCGramAIORateLimiter, retry_after_seconds
from .telegram_request import ResilientPollingHTTPXRequest
from .thread_router import thread_router

# Re-export the moved handler callables and supporting singletons so
# existing tests and integration suites that import them from
# ``ccgram.bot`` keep working without churn. Canonical homes are the
# feature subpackages — these names are retained for ``patch`` targets.
__all__ = [
    "clear_browse_state",
    "commands_command",
    "create_bot",
    "handle_text_message",
    "history_command",
    "inline_query_handler",
    "is_user_allowed",
    "new_command",
    "post_init",
    "post_shutdown",
    "post_stop",
    "safe_reply",
    "session_manager",
    "text_handler",
    "thread_router",
    "toolbar_command",
    "toolcalls_command",
    "unsupported_content_handler",
    "verbose_command",
]

logger = structlog.get_logger()

# A graceful shutdown that never completes leaves the supervisor waiting on a
# live process forever. This watchdog is a plain thread on purpose: the very
# failure it guards against is a wedged event loop, so it must not be scheduled
# on that loop, and its callback may only do thread-safe teardown before
# forcing the exit. One watchdog per process; the first arm wins.
_SHUTDOWN_WATCHDOG_SECONDS = 600.0
# Cap on the best-effort log flush: shutdown may have wedged on a blocked log
# stream, and the escape hatch must not wait on the same write lock.
_WATCHDOG_FLUSH_SECONDS = 2.0
_shutdown_watchdog: threading.Timer | None = None
_shutdown_watchdog_lock = threading.Lock()
_shutdown_exit_code = 1


def _force_exit_after_wedged_shutdown() -> None:
    """Force the process out when graceful shutdown never completes.

    The flush runs in its own daemon thread with a bounded join: structlog's
    per-file lock lives in the main thread when shutdown wedged on a blocked
    log stream, and waiting on it here would defeat the watchdog.
    """

    def _best_effort_flush() -> None:
        try:
            os.write(2, b"ccgram: shutdown did not finish; forcing exit\n")
            logging.shutdown()
        except Exception:  # noqa: BLE001 - the process is about to die
            pass

    flusher = threading.Thread(target=_best_effort_flush, daemon=True)
    flusher.start()
    flusher.join(timeout=_WATCHDOG_FLUSH_SECONDS)
    os._exit(_shutdown_exit_code)


def arm_shutdown_watchdog(
    timeout: float = _SHUTDOWN_WATCHDOG_SECONDS, *, exit_code: int = 1
) -> None:
    """Arm the single shutdown watchdog. Idempotent: the first arm wins.

    ``exit_code`` is what the forced exit reports: signal-driven shutdowns
    pass ``128 + signum`` so a supervisor that distinguishes a stop request
    (SIGQUIT/131) does not read a wedged stop as a crash and restart.
    """
    global _shutdown_exit_code, _shutdown_watchdog
    with _shutdown_watchdog_lock:
        if _shutdown_watchdog is not None:
            return
        _shutdown_exit_code = exit_code
        timer = threading.Timer(timeout, _force_exit_after_wedged_shutdown)
        timer.daemon = True
        timer.start()
        _shutdown_watchdog = timer


def cancel_shutdown_watchdog() -> None:
    """Disarm the watchdog after shutdown completed normally."""
    global _shutdown_exit_code, _shutdown_watchdog
    with _shutdown_watchdog_lock:
        timer, _shutdown_watchdog = _shutdown_watchdog, None
        _shutdown_exit_code = 1
    if timer is not None:
        timer.cancel()


_CONFLICT_GRACE_PERIOD_S = 90.0
_GET_UPDATES_READ_TIMEOUT_S = 20.0


class _PollingConflictState:
    """Track consecutive getUpdates conflicts until a successful poll resets them."""

    def __init__(self) -> None:
        self._first_conflict_at: float | None = None
        self.shutdown_requested = False

    def reset(self) -> None:
        self._first_conflict_at = None
        self.shutdown_requested = False

    def record_success(self) -> None:
        self._first_conflict_at = None

    def record_conflict(self, now: float) -> bool:
        if self._first_conflict_at is None:
            self._first_conflict_at = now
            return False
        if now - self._first_conflict_at < _CONFLICT_GRACE_PERIOD_S:
            return False
        self.shutdown_requested = True
        return True


_polling_conflict_state = _PollingConflictState()


def _reset_polling_conflict_state() -> None:
    _polling_conflict_state.reset()


def polling_conflict_requires_restart() -> bool:
    """Return whether sustained polling conflicts stopped the application."""
    return _polling_conflict_state.shutdown_requested


def _record_successful_poll() -> None:
    _polling_conflict_state.record_success()


def is_user_allowed(user_id: int | None) -> bool:
    """Thin wrapper around ``config.is_user_allowed`` for None-safety."""
    return user_id is not None and config.is_user_allowed(user_id)


# Group filter: when CCGRAM_GROUP_ID is set, only process updates from that group.
# filters.ALL is a no-op — single-instance backward compat.
_group_filter: filters.BaseFilter = (
    filters.Chat(chat_id=config.group_id) if config.group_id else filters.ALL
)


# --- App lifecycle ---


async def post_init(application: Application) -> None:
    """Run the post_init wiring sequence — see ``bootstrap.bootstrap_application``."""
    await bootstrap.bootstrap_application(application)


async def _send_shutdown_notification(application: Application) -> None:
    """Send a shutdown notification to the General topic if a group is configured."""
    # Lazy: main imports bot at top to wire post_init/post_shutdown; hoisting
    # these forms a main ↔ bot cycle on cold import.
    # Lazy: bot ↔ main cycle
    from .main import _shutdown_signal

    if not config.group_id:
        return

    sig = _shutdown_signal
    reason = f"Received {signal.Signals(sig).name}" if sig else "Clean exit"

    # Lazy: __version__ is generated by hatch-vcs and only needed for the
    # shutdown banner; defer to avoid pulling it in the cold-start path.
    # Lazy: defer until startup banner runs
    from . import __version__

    # Lazy: only used inside the error handler
    from telegram.error import TelegramError

    text = f"🔌 ccgram stopped — {reason} (v{__version__})"
    try:
        await application.bot.send_message(
            chat_id=config.group_id,
            text=text,
        )
    except (TelegramError, RuntimeError) as exc:
        logger.debug("Shutdown notification skipped: %s", exc)


async def post_stop(application: Application) -> None:
    """Stop producers and drain pending deliveries while HTTP is alive.

    PTB runs post_stop before Application.shutdown (HTTPXRequest teardown),
    so this is the only place where queued Telegram sends can still succeed.
    Arms the shutdown watchdog first: a wedged drain must not keep the
    process alive forever, and this also covers shutdown paths that do not
    go through the signal handler.
    """
    arm_shutdown_watchdog()
    await bootstrap.stop_delivery_runtime()
    await _send_shutdown_notification(application)


async def post_shutdown(_application: Application) -> None:
    """Tear down runtime state — see ``bootstrap.shutdown_runtime``."""
    try:
        await bootstrap.shutdown_runtime()
    finally:
        cancel_shutdown_watchdog()


async def _error_handler(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle bot-level errors from updater and handlers."""
    if isinstance(context.error, Conflict):
        if _polling_conflict_state.record_conflict(time.monotonic()):
            logger.critical(
                "Telegram polling conflict persisted for %.0fs; stopping so the "
                "service supervisor can restart ccgram. Check for another bot instance.",
                _CONFLICT_GRACE_PERIOD_S,
            )
            # This stop path never reaches the signal handler, so arm here or
            # a wedge in stop() leaves the process alive and the supervisor
            # waiting on a conflict that was supposed to trigger a restart.
            arm_shutdown_watchdog()
            context.application.stop_running()
        else:
            logger.warning(
                "Telegram polling conflict; retrying for up to %.0fs before stopping. "
                "This can follow a network reconnect.",
                _CONFLICT_GRACE_PERIOD_S,
            )
        return
    if isinstance(context.error, BadRequest) and "too old" in str(context.error):
        logger.debug("Callback query expired (query too old)")
        return
    if isinstance(context.error, RetryAfter):
        logger.warning(
            "Telegram rate limit persisted after retries",
            retry_after_seconds=retry_after_seconds(context.error),
        )
        return
    if isinstance(context.error, NetworkError) and not isinstance(
        context.error, BadRequest
    ):
        # PTB will retry automatically — not actionable; demoted from warning.
        logger.info("Transient network error (PTB will retry): %s", context.error)
        return
    logger.error("Unhandled bot error", exc_info=context.error)


def create_bot() -> Application:
    _reset_polling_conflict_state()
    # Suppress PTBUserWarning about JobQueue (we intentionally don't use it for core tasks)
    # Lazy: only used inside the deprecation guard
    import warnings

    warnings.filterwarnings("ignore", message=".*JobQueue.*", category=UserWarning)
    application = (
        Application.builder()
        .token(config.telegram_bot_token)
        .rate_limiter(CCGramAIORateLimiter(max_retries=3))
        .request(
            ResilientPollingHTTPXRequest(
                read_timeout=10,
                request_name="Bot API",
            )
        )
        .get_updates_request(
            ResilientPollingHTTPXRequest(
                connection_pool_size=1,
                read_timeout=_GET_UPDATES_READ_TIMEOUT_S,
                request_name="getUpdates",
                on_success=_record_successful_poll,
            )
        )
        .post_init(post_init)
        .post_stop(post_stop)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_error_handler(_error_handler)
    register_all(application, _group_filter)

    return application
