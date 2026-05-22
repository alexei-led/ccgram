"""Toolbar callback handlers — dispatch for /toolbar inline button clicks.

Callback data scheme: ``tb:<window_id>:<action_name>``. The action_name is
looked up in the loaded ``ToolbarConfig.actions`` and dispatched by
``action_type``:

  - ``key``    → ``tmux_manager.send_keys(payload, enter=False, literal=...)``
  - ``text``   → ``tmux_manager.send_keys(payload, enter=True, literal=True)``
  - ``builtin`` → dispatched via ``_BUILTIN_DISPATCH`` to a specialized handler

Keyboard construction lives in ``toolbar_keyboard``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

import structlog
from telegram import (
    CallbackQuery,
    Update,
)
from telegram.error import TelegramError

from ...window_query import view_window
from ...thread_router import thread_router
from ...tmux_manager import tmux_manager
from ...toolbar_config import ToolbarAction
from ..callback_data import CB_TOOLBAR
from ..callback_helpers import get_thread_id, user_owns_window
from ..callback_registry import register
from .toolbar_keyboard import get_toolbar_config, refresh_button_label

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()


async def _dispatch_key(
    action: ToolbarAction, query: CallbackQuery, window_id: str
) -> None:
    """Send a tmux key for a ``key`` action.

    Toggle actions (``read_state=True``) rewrite the clicked button's
    label in place to reflect the new state — the button text is the
    state indicator. No popups, no disruptive dialogs.
    """
    w = await tmux_manager.find_window_by_id(window_id)
    if w is None:
        await query.answer("Window not found", show_alert=True)
        return
    await tmux_manager.send_keys(
        w.window_id, action.payload, enter=False, literal=action.literal
    )
    if action.read_state:
        short_label = await refresh_button_label(action, query, window_id)
        await query.answer(f"{action.emoji} {short_label}")
    else:
        await query.answer(f"{action.emoji} {action.text}")


async def _dispatch_text(
    action: ToolbarAction, query: CallbackQuery, window_id: str
) -> None:
    """Send literal text + Enter for a ``text`` action."""
    w = await tmux_manager.find_window_by_id(window_id)
    if w is None:
        await query.answer("Window not found", show_alert=True)
        return
    await tmux_manager.send_keys(w.window_id, action.payload, enter=True, literal=True)
    if action.read_state:
        short_label = await refresh_button_label(action, query, window_id)
        await query.answer(f"{action.emoji} {short_label}")
    else:
        await query.answer(f"{action.emoji} {action.text}")


# ──────────────────────────────────────────────────────────────────────
# Built-in handlers
# ──────────────────────────────────────────────────────────────────────


async def _builtin_screenshot(
    _action: ToolbarAction,
    query: CallbackQuery,
    window_id: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Builtin: trigger the screenshot handler."""
    # Lazy: live.screenshot_callbacks is registered through the callback
    # registry; importing at top forms toolbar ↔ live cycle via shared
    # callback_data symbols.
    # Lazy: avoid pulling the full handlers.callback_data tree at module load
    from ..callback_data import CB_STATUS_SCREENSHOT

    # Lazy: live ↔ toolbar cycle
    from ..live.screenshot_callbacks import handle_screenshot_callback

    user = update.effective_user
    if user is None:
        await query.answer("No user context", show_alert=True)
        return
    fake_data = f"{CB_STATUS_SCREENSHOT}{window_id}"
    await handle_screenshot_callback(query, user.id, fake_data, update, context)


async def _builtin_ctrlc(
    _action: ToolbarAction,
    query: CallbackQuery,
    window_id: str,
    _update: Update,
    _context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Builtin: send Ctrl-C."""
    w = await tmux_manager.find_window_by_id(window_id)
    if w is None:
        await query.answer("Window not found", show_alert=True)
        return
    await tmux_manager.send_keys(w.window_id, "C-c", enter=False, literal=False)
    await query.answer("\u23f9 Ctrl-C sent")


async def _builtin_live(
    _action: ToolbarAction,
    query: CallbackQuery,
    window_id: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Builtin: start the live view via the existing screenshot dispatcher."""
    # Lazy: same toolbar ↔ live cycle as _builtin_screenshot.
    from ..callback_data import CB_LIVE_START

    # Lazy: live ↔ toolbar cycle
    from ..live.screenshot_callbacks import handle_screenshot_callback

    user = update.effective_user
    if user is None:
        await query.answer("No user context", show_alert=True)
        return
    fake_data = f"{CB_LIVE_START}{window_id}"
    await handle_screenshot_callback(query, user.id, fake_data, update, context)


async def _builtin_getfile(
    _action: ToolbarAction,
    query: CallbackQuery,
    window_id: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Builtin: open the file browser to upload a file from the agent CWD."""
    user = update.effective_user
    if user is None:
        await query.answer("No user context", show_alert=True)
        return
    user_id = user.id
    view = view_window(window_id)
    cwd = Path(view.cwd) if view and view.cwd else None
    if not cwd or not cwd.is_dir():
        await query.answer("Working directory not available", show_alert=True)
        return
    if context.user_data is None:
        await query.answer("State error", show_alert=True)
        return
    thread_id = get_thread_id(update)
    chat_id = thread_router.resolve_chat_id(user_id, thread_id) if thread_id else None
    if chat_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    # Lazy: telegram_client wraps PTB Bot; send subpackage in turn pulls
    # the directory browser machinery.  Keep both at call site.
    # Lazy: PTBTelegramClient resolved per-call
    from ...telegram_client import PTBTelegramClient

    # Lazy: handlers.send subpackage cycle
    from ..send import open_file_browser

    await open_file_browser(
        PTBTelegramClient(query.get_bot()),
        chat_id,
        thread_id,
        context.user_data,
        window_id,
        cwd,
    )
    await query.answer()


async def _builtin_last(
    _action: ToolbarAction,
    query: CallbackQuery,
    window_id: str,
    update: Update,
    _context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Builtin: send the last assistant reply (or shell output) to this topic."""
    user = update.effective_user
    if user is None:
        await query.answer("No user context", show_alert=True)
        return
    user_id = user.id
    thread_id = get_thread_id(update)
    chat_id = thread_router.resolve_chat_id(user_id, thread_id) if thread_id else None
    if chat_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    # Lazy: PTBTelegramClient resolved per-call
    from ...telegram_client import PTBTelegramClient

    # Lazy: last_reply ↔ toolbar cycle
    from ..last_reply import send_last_reply

    await send_last_reply(
        PTBTelegramClient(query.get_bot()),
        chat_id,
        thread_id,
        window_id,
    )
    await query.answer()


async def _builtin_dismiss(
    _action: ToolbarAction,
    query: CallbackQuery,
    _window_id: str,
    _update: Update,
    _context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Builtin: delete the toolbar message."""
    with contextlib.suppress(TelegramError):
        await query.delete_message()
    await query.answer()


type _BuiltinHandler = Callable[
    [ToolbarAction, CallbackQuery, str, Update, ContextTypes.DEFAULT_TYPE],
    Awaitable[None],
]

_BUILTIN_DISPATCH: dict[str, _BuiltinHandler] = {
    "screenshot": _builtin_screenshot,
    "ctrlc": _builtin_ctrlc,
    "live": _builtin_live,
    "getfile": _builtin_getfile,
    "lastreply": _builtin_last,
    "dismiss": _builtin_dismiss,
}


# ──────────────────────────────────────────────────────────────────────
# Top-level dispatcher
# ──────────────────────────────────────────────────────────────────────


def _parse_callback_data(data: str) -> tuple[str, str] | None:
    """Parse ``tb:<window_id>:<action_name>`` into ``(window_id, name)``.

    Returns None if the format is invalid. Window IDs may themselves
    contain a colon (foreign emdash IDs like ``emdash-claude-main-x:@0``),
    so the action_name is the substring after the LAST colon.
    """
    if not data.startswith(CB_TOOLBAR):
        return None
    suffix = data[len(CB_TOOLBAR) :]
    sep = suffix.rfind(":")
    if sep <= 0:
        return None
    return suffix[:sep], suffix[sep + 1 :]


async def handle_toolbar_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Single entry point for all toolbar button clicks."""
    parsed = _parse_callback_data(data)
    if parsed is None:
        await query.answer("Bad toolbar callback", show_alert=True)
        return
    window_id, action_name = parsed
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    cfg = get_toolbar_config()
    action = cfg.actions.get(action_name)
    if action is None:
        await query.answer(f"Unknown action: {action_name}", show_alert=True)
        return
    if action.action_type == "key":
        await _dispatch_key(action, query, window_id)
    elif action.action_type == "text":
        await _dispatch_text(action, query, window_id)
    elif action.action_type == "builtin":
        handler = _BUILTIN_DISPATCH.get(action.payload)
        if handler is None:
            await query.answer(f"Unknown builtin: {action.payload}", show_alert=True)
            return
        await handler(action, query, window_id, update, context)
    else:
        await query.answer("Unsupported action type", show_alert=True)


@register(CB_TOOLBAR)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Single registered handler for all CB_TOOLBAR clicks."""
    query = update.callback_query
    if query is None or query.data is None:
        return
    user = update.effective_user
    if user is None:
        return
    await handle_toolbar_callback(query, user.id, query.data, update, context)
