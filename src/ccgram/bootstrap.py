"""Application bootstrap — wires post_init and post_shutdown lifecycle.

`bot.py` defines the PTB ``Application`` factory + lifecycle delegates;
the actual wiring (provider commands, runtime callbacks, session
monitor, status polling, mini-app) lives here as named functions so
each step is independently testable.

Ordering invariant: ``wire_runtime_callbacks`` must run before
``start_session_monitor`` because the monitor dispatches Stop events
to the registered Stop callback, and an unwired callback raises after
F2.6.

Module-level state (``session_monitor``, ``_status_poll_task``) is
created in post_init and torn down in post_shutdown — kept here, not
in ``bot.py``, so the lifecycle delegates stay one-liners.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

import structlog
from telegram.error import TelegramError

from .cc_commands import register_commands
from .handlers.commands import setup_menu_refresh_job
from .handlers.hook_events import dispatch_hook_event
from .handlers.messaging_pipeline.message_queue import shutdown_workers
from .handlers.messaging_pipeline.message_routing import handle_new_message
from .handlers.polling.polling_coordinator import status_poll_loop
from .handlers.shell import register_approval_callback, show_command_approval
from .handlers.topics.topic_orchestration import (
    adopt_unbound_windows as _adopt_unbound_windows,
)
from .handlers.topics.topic_orchestration import (
    handle_new_window as _handle_new_window,
)
from .providers import get_provider
from .session import session_manager
from .telegram_client import PTBTelegramClient
from .session_monitor import (
    NewMessage,
    NewWindowEvent,
    SessionMonitor,
    clear_active_monitor,
    set_active_monitor,
)
from .utils import task_done_callback

if TYPE_CHECKING:
    from telegram.ext import Application

    from .providers.base import HookEvent

logger = structlog.get_logger()

session_monitor: SessionMonitor | None = None
_status_poll_task: asyncio.Task[None] | None = None
_callbacks_wired = False


def install_global_exception_handler() -> None:
    """Install the asyncio last-resort exception handler."""
    asyncio.get_running_loop().set_exception_handler(_global_exception_handler)


def _global_exception_handler(
    _loop: asyncio.AbstractEventLoop, ctx: dict[str, object]
) -> None:
    """Last-resort handler for uncaught exceptions in asyncio tasks."""
    exc = ctx.get("exception")
    msg = ctx.get("message", "Unhandled exception in event loop")
    if isinstance(exc, BaseException):
        logger.error(
            "asyncio exception handler: %s",
            msg,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logger.error("asyncio exception handler: %s", msg)


async def register_provider_commands(application: Application) -> None:
    """Register the default provider's BotCommand list and schedule menu refresh."""
    default_provider = get_provider()
    try:
        await register_commands(application.bot, provider=default_provider)
    except TelegramError:
        logger.warning("Failed to register bot commands at startup, will retry later")
    setup_menu_refresh_job(application)


def verify_hooks_installed() -> None:
    """Warn if managed hooks are missing for the default provider."""
    provider = get_provider()
    if not provider.capabilities.supports_hook:
        return
    provider_name = provider.capabilities.name
    if provider_name != "claude":
        if provider.capabilities.hook_install_managed_by_ccgram:
            # DEBUG (not INFO/WARNING): Codex/Gemini fall back to transcript-scan
            # discovery when hooks are absent, so this is an opt-in latency tip,
            # not a degraded state — it should not greet every startup at INFO.
            logger.debug(
                "%s hooks can improve status tracking. Run: ccgram hook --provider %s --install",
                provider_name,
                provider_name,
            )
        return

    # Lazy: hook module is the Claude-Code subprocess entry point;
    # importing it eagerly drags `utils`/IO costs into bootstrap even
    # when the active provider has no hooks.
    # Lazy: hook helpers used only during the hook-verify step
    from .hook import _claude_settings_file, get_installed_events

    settings_file = _claude_settings_file()
    if not settings_file.exists():
        logger.warning(
            "Claude Code hooks not installed (%s missing). Run: ccgram hook --install",
            settings_file,
        )
        return

    try:
        settings = json.loads(settings_file.read_text())
    except json.JSONDecodeError, OSError:
        logger.warning("Claude Code hooks not installed. Run: ccgram hook --install")
        return

    events = get_installed_events(settings)
    missing = [e for e, ok in events.items() if not ok]
    if missing:
        logger.warning(
            "Claude Code hooks incomplete — %d missing: %s. Run: ccgram hook --install",
            len(missing),
            ", ".join(missing),
        )


def wire_runtime_callbacks() -> None:
    """Wire module-level callbacks that break cross-subsystem direct imports.

    Idempotent — safe to call multiple times. Must run before
    ``start_session_monitor`` — the monitor dispatches approval prompts to
    ``register_approval_callback``, which raises if not wired.
    """
    global _callbacks_wired

    if _callbacks_wired:
        return

    register_approval_callback(show_command_approval)
    _callbacks_wired = True


async def start_session_monitor(application: Application) -> SessionMonitor:
    """Build the SessionMonitor, set its callbacks, and start polling.

    Raises ``RuntimeError`` if ``wire_runtime_callbacks`` has not run —
    the monitor would dispatch Stop events to an unwired callback.
    """
    global session_monitor

    if not _callbacks_wired:
        raise RuntimeError(
            "wire_runtime_callbacks() must run before start_session_monitor()"
        )

    monitor = SessionMonitor()
    set_active_monitor(monitor)

    # Lazy: telegram_client wraps PTB Bot; bootstrap is otherwise free of
    # PTB types, so loading the adapter here keeps cold imports clean.

    client = PTBTelegramClient(application.bot)

    async def message_callback(msg: NewMessage) -> None:
        await handle_new_message(msg, client)

    monitor.set_message_callback(message_callback)

    async def new_window_callback(event: NewWindowEvent) -> None:
        await _handle_new_window(event, client)

    monitor.set_new_window_callback(new_window_callback)

    async def hook_event_callback(event: HookEvent) -> None:
        await dispatch_hook_event(event, client)

    monitor.set_hook_event_callback(hook_event_callback)

    monitor.start()
    session_monitor = monitor
    logger.info("Session monitor started")
    return monitor


def start_status_polling(application: Application) -> asyncio.Task[None]:
    """Spawn the status-polling background task."""
    global _status_poll_task

    _status_poll_task = asyncio.create_task(status_poll_loop(application.bot))
    _status_poll_task.add_done_callback(task_done_callback)
    logger.info("Status polling task started")
    return _status_poll_task


async def bootstrap_application(application: Application) -> None:
    """Run the full post_init sequence in the prescribed order."""
    install_global_exception_handler()
    await register_provider_commands(application)
    await session_manager.resolve_stale_ids()
    await _adopt_unbound_windows(PTBTelegramClient(application.bot))
    verify_hooks_installed()
    wire_runtime_callbacks()
    await start_session_monitor(application)
    start_status_polling(application)

    # Lazy: main imports bot at top, bot imports bootstrap; hoisting forms
    # main → bot → bootstrap → main on cold import.
    # Lazy: bootstrap ↔ main cycle
    from .main import start_miniapp_if_enabled

    await start_miniapp_if_enabled()


async def shutdown_runtime() -> None:
    """Run the post_shutdown teardown sequence."""
    global _status_poll_task, session_monitor

    if _status_poll_task is not None:
        _status_poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _status_poll_task
        _status_poll_task = None
        logger.info("Status polling stopped")

    if session_monitor is not None:
        session_monitor.stop()
        logger.info("Session monitor stopped")
        session_monitor = None
    clear_active_monitor()

    await shutdown_workers()

    # Lazy: main → bot → bootstrap cycle (same as start path).
    from .main import stop_miniapp_if_enabled

    await stop_miniapp_if_enabled()

    session_manager.flush_state()


def reset_for_testing() -> None:
    """Clear bootstrap module state and inner callback registrations.

    Each e2e/integration test that drives ``bootstrap_application`` must
    reset state between runs — F2.6 made the register_*_callbacks fail
    loud on double registration, and bootstrap caches its own
    ``_callbacks_wired`` flag too.
    """
    global _callbacks_wired, session_monitor, _status_poll_task

    # Lazy: each module's _reset_*_for_testing hook is only needed by the
    # test harness; production callers never reach reset_for_testing().
    from .handlers.shell import shell_capture

    shell_capture._reset_approval_callback_for_testing()

    _callbacks_wired = False
    session_monitor = None
    _status_poll_task = None
    clear_active_monitor()
