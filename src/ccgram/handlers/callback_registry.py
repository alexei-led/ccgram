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
from telegram import CallbackQuery, Update
from ..config import config
from ..thread_router import thread_router
from .callback_data import (
    CB_SESSIONS_KILL,
    CB_SESSIONS_KILL_CONFIRM,
    CB_SESSIONS_NEW,
    CB_SESSIONS_REFRESH,
    CB_STATUS_ESC,
    CB_STATUS_LAST_REPLY,
    CB_STATUS_SCREENSHOT,
)
from .callback_helpers import acting_as, get_thread_id
from .guest import access as guest_access

# Buttons a guest may press (acting as the owner) for one session, which must
# be bound in the guest chat: Esc, screenshot, last reply, and ending it from
# /sessions (to free a slot under the chat's session limit).
GUEST_BUTTON_PREFIXES = (
    CB_STATUS_ESC,
    CB_STATUS_SCREENSHOT,
    CB_STATUS_LAST_REPLY,
    CB_SESSIONS_KILL_CONFIRM,
    CB_SESSIONS_KILL,
)
# Whole-chat /sessions buttons a guest may press; they act on the chat itself.
GUEST_CHAT_BUTTONS = frozenset(
    {
        CB_SESSIONS_REFRESH,
        CB_SESSIONS_NEW,
        f"{CB_SESSIONS_NEW}:claude",
        f"{CB_SESSIONS_NEW}:codex",
    }
)

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
    if not user:
        return
    acting_id = user.id
    if not config.is_user_allowed(user.id):
        # A guest may press only the buttons above, acting as the owner.
        # Approvals, keys and menus stay with the owner.
        chat_id = query.message.chat.id if query.message else None
        owner_id = guest_access.owner_for(user.id, chat_id)
        chat_button = query.data in GUEST_CHAT_BUTTONS
        if owner_id is None or not (
            chat_button or query.data.startswith(GUEST_BUTTON_PREFIXES)
        ):
            await query.answer("Only the owner can use these buttons.")
            return
        # Lazy: guest.scope imports the message sender, which reaches back here.
        from .guest.scope import guest_target_allowed

        # Only for a session bound in this guest chat, whatever the data says
        # (a /sessions list may show other buttons, and clients can forge data).
        if not chat_button and not guest_target_allowed(
            query.data, GUEST_BUTTON_PREFIXES, owner_id, chat_id
        ):
            logger.warning(
                "Guest button for a session outside this chat refused",
                user_id=user.id,
                chat_id=chat_id,
            )
            await query.answer("Only the owner can use these buttons.")
            return
        logger.info(
            "Guest button", user_id=user.id, chat_id=chat_id, data=query.data[:16]
        )
        acting_id = owner_id

    with acting_as(acting_id):
        await _dispatch_authorized(update, context, query, acting_id)


async def _dispatch_authorized(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: CallbackQuery,
    acting_id: int,
) -> None:
    assert query.data is not None
    # Store group chat_id for forum topic message routing.
    if query.message and query.message.chat.type in ("group", "supergroup"):
        cb_thread_id = get_thread_id(update)
        if cb_thread_id is not None:
            thread_router.set_group_chat_id(
                acting_id, cb_thread_id, query.message.chat.id
            )

    data = query.data
    if data.startswith("cmdpanel:"):
        logger.info(
            "Topic command panel callback received",
            user_id=acting_id,
            chat_id=query.message.chat.id if query.message else None,
            thread_id=get_thread_id(update),
        )

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

    # Lazy: command panels register callbacks and import this registry.
    from .commands import panel  # noqa: F401

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
