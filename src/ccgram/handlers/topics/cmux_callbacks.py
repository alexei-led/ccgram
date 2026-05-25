"""``/cmux`` workspace picker — bind an existing cmux workspace to a topic.

This is the first user-visible cmux vertical slice (plan Task 4). The
``/cmux`` command queries the cmux sidecar for workspaces, renders an
inline picker, and on selection persists the binding through the
terminal-identity feature port so subsequent send/capture operations
route via ``terminal_operations`` → ``CmuxBackend`` → sidecar.

Persistence shape: the bound row uses ``window_id = "cmux:<workspace_id>"``
so ``thread_router`` continues to treat the key as opaque and the
identity matches ``TerminalUnitRef.display_id``. The actual backend unit
id is the bare workspace id, persisted via
:func:`set_terminal_identity` — handlers never parse the prefix.

Failure modes are scoped to cmux topics only:

* cmux disabled by config → reply with setup hint.
* cmux backend not registered with the router (sidecar not wired) →
  same hint, no router lookup attempted.
* sidecar unreachable / handshake fails → user-safe error text, no
  partial bind state left behind.
* stale picker (workspace list changed between render and tap) → alert.

The module is import-light: all backend/router imports are lazy so
unrelated code paths never pay the cost of touching the cmux client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import TelegramError

from ...config import config
from ...telegram_client import PTBTelegramClient, TelegramClient
from ...thread_router import thread_router
from ..callback_data import CB_CMUX_BIND, CB_CMUX_CANCEL, CB_CMUX_LIST
from ..callback_helpers import get_thread_id
from ..callback_registry import register
from ..messaging_pipeline.message_sender import safe_edit, safe_reply
from ..user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

    from ...terminal_backends.base import TerminalUnit


logger = structlog.get_logger()

CMUX_WORKSPACES_KEY = "cmux_workspace_units"
_CMUX_DISABLED_TEXT = (
    "cmux backend is disabled.\n\n"
    "Set CCGRAM_CMUX_ENABLED=true and CCGRAM_CMUX_SIDECAR_SOCKET to enable."
)
_CMUX_NOT_REGISTERED_TEXT = (
    "cmux backend is not wired in this bot instance.\n\nRestart with cmux configured."
)
_CMUX_NO_WORKSPACES_TEXT = "cmux sidecar reports no workspaces yet."


def _qualified_window_id(workspace_id: str) -> str:
    """Build the persisted ``window_id`` key for a cmux workspace."""
    return f"cmux:{workspace_id}"


def _load_backend_config():
    """Return the typed terminal backend config (lazy import)."""
    # Lazy: pulls in config.py which reads env at module load.
    from ...terminal_backends.config import load_terminal_backend_config

    return load_terminal_backend_config(config_dir=config.config_dir)


async def _list_cmux_workspaces() -> tuple[list[TerminalUnit], str | None]:
    """Return ``(units, error_message)`` for the picker.

    ``error_message`` is None on success. Non-None values are
    user-displayable strings and units is guaranteed empty in that case.
    """
    # Lazy: backend modules transitively touch cmux client/socket
    # plumbing. Keep the cold path of the topic handlers free of them.
    from ...terminal_backends.base import (
        BACKEND_CMUX,
        TerminalBackendError,
    )

    # Lazy: terminal_backends.router pulls TmuxBackend/libtmux on first
    # access; keep this off the bare handler-package import cost.
    from ...terminal_backends.router import get_router

    backend_config = _load_backend_config()
    if not backend_config.cmux_active:
        return [], _CMUX_DISABLED_TEXT

    router = get_router()
    if BACKEND_CMUX not in router.known():
        return [], _CMUX_NOT_REGISTERED_TEXT

    backend = router.get(BACKEND_CMUX)
    try:
        units = await backend.list_units()
    except TerminalBackendError as exc:
        logger.debug("cmux list_units failed", code=exc.code, message=str(exc))
        return [], f"cmux sidecar error: {exc.code}"
    return units, None


def _label_for_unit(unit: TerminalUnit) -> str:
    """Human label for a cmux workspace in the picker."""
    title = unit.title or unit.ref.unit_id
    if unit.provider_name:
        return f"{title} [{unit.provider_name}]"
    return title


def build_cmux_picker(
    units: list[TerminalUnit],
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the workspace picker keyboard for a list of cmux units.

    Empty lists still render a usable picker with a cancel button.
    Cancel and refresh share the bottom row regardless of count.
    """
    lines: list[str] = ["*Bind cmux Workspace*\n"]
    if not units:
        lines.append("_No workspaces reported by the cmux sidecar._")
    else:
        lines.append("Pick an existing cmux workspace to bind here.")
        for unit in units:
            cwd_display = f" — `{unit.cwd}`" if unit.cwd else ""
            lines.append(f"• `{_label_for_unit(unit)}`{cwd_display}")

    buttons: list[list[InlineKeyboardButton]] = []
    for idx, unit in enumerate(units):
        buttons.append(
            [
                InlineKeyboardButton(
                    f"🪟 {_label_for_unit(unit)[:24]}",
                    callback_data=f"{CB_CMUX_BIND}{idx}",
                )
            ]
        )
    buttons.append(
        [
            InlineKeyboardButton("🔄 Refresh", callback_data=CB_CMUX_LIST),
            InlineKeyboardButton("Cancel", callback_data=CB_CMUX_CANCEL),
        ]
    )
    return "\n".join(lines), InlineKeyboardMarkup(buttons)


def _store_units(context: ContextTypes.DEFAULT_TYPE, units: list[TerminalUnit]) -> None:
    if context.user_data is not None:
        context.user_data[CMUX_WORKSPACES_KEY] = units


def _clear_units(context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data is not None:
        context.user_data.pop(CMUX_WORKSPACES_KEY, None)


async def cmux_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/cmux`` — show the cmux workspace picker.

    Authorization, topic context, and cmux-active config are checked
    here so callback handlers can assume a well-formed state.
    """
    user = update.effective_user
    message = update.message
    if user is None or message is None:
        return
    if not config.is_user_allowed(user.id):
        await safe_reply(message, "You are not authorized to use this bot.")
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await safe_reply(
            message, "Open this command inside a topic to bind cmux there."
        )
        return

    units, error = await _list_cmux_workspaces()
    if error is not None:
        await safe_reply(message, error)
        return

    _store_units(context, units)
    if context.user_data is not None:
        context.user_data[PENDING_THREAD_ID] = thread_id

    text, keyboard = build_cmux_picker(units)
    if not units:
        text = f"{text}\n\n_{_CMUX_NO_WORKSPACES_TEXT}_"
    await safe_reply(message, text, reply_markup=keyboard)


async def _handle_refresh(
    query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE
) -> None:
    units, error = await _list_cmux_workspaces()
    if error is not None:
        _clear_units(context)
        await safe_edit(query, error)
        await query.answer()
        return
    _store_units(context, units)
    text, keyboard = build_cmux_picker(units)
    await safe_edit(query, text, reply_markup=keyboard)
    await query.answer("Refreshed")


async def _handle_bind(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    thread_id = get_thread_id(update)
    if pending_tid is None or thread_id is None or thread_id != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    try:
        idx = int(data[len(CB_CMUX_BIND) :])
    except ValueError:
        await query.answer("Invalid selection")
        return

    cached: list[TerminalUnit] = (
        context.user_data.get(CMUX_WORKSPACES_KEY, []) if context.user_data else []
    )
    if idx < 0 or idx >= len(cached):
        await query.answer("Workspace list changed, please retry", show_alert=True)
        return

    unit = cached[idx]
    workspace_id = unit.ref.unit_id
    window_id = _qualified_window_id(workspace_id)
    display = unit.title or workspace_id

    # Lazy: terminal identity port pulls window_state_store.
    from ...window_state_ports.terminal_identity import set_terminal_identity

    set_terminal_identity(window_id, backend=unit.ref.backend, unit_id=workspace_id)
    # NOTE: cwd is intentionally not mirrored into WindowState. The cmux
    # sidecar owns the canonical workspace cwd; surfacing it through the
    # ``TerminalUnit`` projection is enough for dashboards.
    thread_router.bind_thread(user_id, thread_id, window_id, window_name=display)

    chat = query.message.chat if query.message else None
    if chat and chat.type in ("group", "supergroup"):
        thread_router.set_group_chat_id(user_id, thread_id, chat.id)

    client: TelegramClient = PTBTelegramClient(context.bot)
    try:
        await client.edit_forum_topic(
            chat_id=thread_router.resolve_chat_id(user_id, thread_id),
            message_thread_id=thread_id,
            name=display,
        )
    except TelegramError as exc:
        logger.debug("Failed to rename topic after cmux bind: %s", exc)

    _clear_units(context)
    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_ID, None)
        context.user_data.pop(PENDING_THREAD_TEXT, None)

    await safe_edit(query, f"✅ Bound cmux workspace `{display}`")
    await query.answer("Bound")


async def _handle_cancel(
    query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _clear_units(context)
    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_ID, None)
    await safe_edit(query, "Cancelled")
    await query.answer("Cancelled")


@register(CB_CMUX_LIST, CB_CMUX_BIND, CB_CMUX_CANCEL)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    if query is None or query.data is None or user is None:
        return
    data = query.data
    if data == CB_CMUX_LIST:
        await _handle_refresh(query, context)
    elif data.startswith(CB_CMUX_BIND):
        await _handle_bind(query, user.id, data, update, context)
    elif data == CB_CMUX_CANCEL:
        await _handle_cancel(query, context)


__all__ = [
    "CMUX_WORKSPACES_KEY",
    "build_cmux_picker",
    "cmux_command",
]
