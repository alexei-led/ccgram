"""Callback query dispatch registry.

Provides self-registration for callback handler modules and longest-prefix
dispatch. Authorization (user allowlist, group_id) and group chat ID
recording happen once in dispatch(), not in individual handlers.

Key components:
  - register(*prefixes): decorator for handler self-registration
  - dispatch(update, context): longest-prefix match + authorization
  - load_handlers(): import all callback-bearing modules to trigger registration
"""

from __future__ import annotations


from typing import TYPE_CHECKING
from collections.abc import Awaitable, Callable

import structlog
from telegram import Update
from ..config import config
from ..thread_router import thread_router
from .callback_helpers import get_thread_id

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

type CallbackHandler = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]

_registry: dict[str, CallbackHandler] = {}


def register(
    *prefixes: str,
) -> Callable[[CallbackHandler], CallbackHandler]:
    """Register a callback handler for given prefix strings.

    The handler must accept ``(update: Update, context: ContextTypes.DEFAULT_TYPE)``.
    Returns the original function unchanged so existing call sites keep working.
    """

    def decorator(func: CallbackHandler) -> CallbackHandler:
        for prefix in prefixes:
            if prefix in _registry:
                raise ValueError(
                    f"Callback prefix {prefix!r} already registered "
                    f"(existing: {_registry[prefix].__qualname__}, "
                    f"new: {func.__qualname__})"
                )
            _registry[prefix] = func
        return func

    return decorator


def get_registry() -> dict[str, CallbackHandler]:
    """Return the mutable registry dict (for testing only)."""
    return _registry


async def dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route callback query to registered handler by longest-prefix match.

    Performs authorization, group_id filtering, and group chat ID recording
    before dispatching to the matched handler.
    """
    if config.group_id:
        chat = update.effective_chat
        if not chat or chat.id != config.group_id:
            return

    query = update.callback_query
    if not query or not query.data:
        return

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        await query.answer("Not authorized")
        return

    # Store group chat_id for forum topic message routing.
    if query.message and query.message.chat.type in ("group", "supergroup"):
        cb_thread_id = get_thread_id(update)
        if cb_thread_id is not None:
            thread_router.set_group_chat_id(
                user.id, cb_thread_id, query.message.chat.id
            )

    data = query.data

    if data == "noop":
        await query.answer()
        return

    handler = _find_handler(data)
    if handler is not None:
        await handler(update, context)


def _find_handler(data: str) -> CallbackHandler | None:
    """Find the handler for the longest matching prefix."""
    best_handler: CallbackHandler | None = None
    best_len = 0
    for prefix, handler in _registry.items():
        if data.startswith(prefix) and len(prefix) > best_len:
            best_handler = handler
            best_len = len(prefix)
    return best_handler


def load_handlers() -> None:
    """Import handler modules to trigger @register and @topic_state.register decorators.

    The imports below are intentionally inside this function: their sole
    purpose is the side effect of running the decorators at module-load
    time.  Hoisting them to the top of ``callback_registry`` would
    re-introduce import cycles back into the registry itself and defeat
    the explicit ``bootstrap.bootstrap_application`` lifecycle contract
    (callers control _when_ handler modules load).
    """
    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from . import (  # noqa: F401
        agent_command,
        command_history,
        hook_events,
        sessions_dashboard,
        sync_command,
    )

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .interactive import interactive_callbacks  # noqa: F401

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .voice import voice_callbacks  # noqa: F401

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .live import pane_callbacks, screenshot_callbacks  # noqa: F401

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .recovery import history_callbacks, recovery_callbacks, resume_command  # noqa: F401

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .send import send_callbacks  # noqa: F401

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .shell import shell_capture, shell_commands, shell_prompt_orchestrator  # noqa: F401

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .status import status_bar_actions  # noqa: F401

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .toolbar import toolbar_callbacks  # noqa: F401

    # Lazy: handler subpackage modules import callback_registry for the @register_callback decorator; importing them here at module load would cycle
    from .topics import directory_callbacks, window_callbacks  # noqa: F401
