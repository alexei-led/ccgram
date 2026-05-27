"""``/cmux`` terminal picker — bind a cmux terminal session to a topic.

The ``/cmux`` command queries the cmux sidecar for terminal tabs/panels,
renders an inline picker, and on selection persists the binding through
the terminal-identity feature port so subsequent send/capture operations
route via ``terminal_operations`` → ``CmuxBackend`` → sidecar.

Persistence shape: the bound row uses ``window_id = "cmux:<terminal_id>"``
so ``thread_router`` continues to treat the key as opaque and the
identity matches ``TerminalUnitRef.display_id``. cmux workspace data is
metadata for display only; it is never the routing key.

Failure modes are scoped to cmux topics only:

* cmux disabled by config → reply with setup hint.
* cmux backend not registered with the router (sidecar not wired) →
  same hint, no router lookup attempted.
* sidecar unreachable / handshake fails → user-safe error text, no
  partial bind state left behind.
* stale picker (terminal list changed between render and tap) → alert.

The module is import-light: all backend/router imports are lazy so
unrelated code paths never pay the cost of touching the cmux client.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

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
from .directory_browser import (
    clear_browse_state,
    clear_window_picker_state,
    clear_worktree_state,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

    from ...terminal_backends.base import TerminalUnit


logger = structlog.get_logger()

CMUX_TERMINAL_SESSIONS_KEY = "cmux_terminal_session_units"
_CMUX_DISABLED_TEXT = (
    "cmux backend is disabled.\n\n"
    "Set CCGRAM_CMUX_ENABLED=true and CCGRAM_CMUX_SIDECAR_SOCKET to enable."
)
_CMUX_NOT_REGISTERED_TEXT = (
    "cmux backend is not wired in this bot instance.\n\nRestart with cmux configured."
)
_CMUX_NO_TERMINAL_SESSIONS_TEXT = "cmux sidecar reports no terminal sessions yet."


def _qualified_window_id(terminal_id: str) -> str:
    """Build the persisted ``window_id`` key for a cmux terminal session."""
    return f"cmux:{terminal_id}"


def _load_backend_config():
    """Return the typed terminal backend config (lazy import)."""
    # Lazy: pulls in config.py which reads env at module load.
    from ...terminal_backends.config import load_terminal_backend_config

    return load_terminal_backend_config(config_dir=config.config_dir)


async def _list_cmux_terminal_sessions() -> tuple[list[TerminalUnit], str | None]:
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
    """Human label for a cmux terminal session in the picker."""
    title = unit.title or unit.ref.unit_id
    if unit.provider_name:
        return f"{title} [{unit.provider_name}]"
    return title


def _workspace_label_for_unit(unit: TerminalUnit) -> str:
    raw = unit.backend_metadata.get("workspace_title") or unit.backend_metadata.get(
        "workspace_id"
    )
    if isinstance(raw, str) and raw:
        return raw
    return ""


def build_cmux_picker(
    units: list[TerminalUnit],
    *,
    picker_id: str = "default",
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the terminal-session picker keyboard for cmux units.

    Empty lists still render a usable picker with a cancel button.
    Cancel and refresh share the bottom row regardless of count. Bind callbacks
    include a picker token so old messages cannot bind against a newer list.
    """
    lines: list[str] = ["*Bind cmux Terminal Session*\n"]
    if not units:
        lines.append("_No terminal sessions reported by the cmux sidecar._")
    else:
        lines.append("Pick an existing cmux terminal tab/panel to bind here.")
        for unit in units:
            cwd_display = f" — `{unit.cwd}`" if unit.cwd else ""
            workspace = _workspace_label_for_unit(unit)
            workspace_display = f" — workspace `{workspace}`" if workspace else ""
            unavailable = "" if unit.supports_send_text else " — unavailable"
            lines.append(
                f"• `{_label_for_unit(unit)}`{workspace_display}{cwd_display}{unavailable}"
            )

    buttons: list[list[InlineKeyboardButton]] = []
    for idx, unit in enumerate(units):
        if not unit.supports_send_text:
            continue
        buttons.append(
            [
                InlineKeyboardButton(
                    f"💻 {_label_for_unit(unit)[:24]}",
                    callback_data=f"{CB_CMUX_BIND}{picker_id}:{idx}",
                )
            ]
        )
    buttons.append(
        [
            InlineKeyboardButton(
                "🔄 Refresh", callback_data=f"{CB_CMUX_LIST}{picker_id}"
            ),
            InlineKeyboardButton(
                "Cancel", callback_data=f"{CB_CMUX_CANCEL}{picker_id}"
            ),
        ]
    )
    return "\n".join(lines), InlineKeyboardMarkup(buttons)


def _new_picker_id() -> str:
    return secrets.token_hex(4)


def _raw_session_store(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    if context.user_data is None:
        return {}
    raw = context.user_data.get(CMUX_TERMINAL_SESSIONS_KEY)
    if not isinstance(raw, dict):
        return {}
    return raw


def _session_store(context: ContextTypes.DEFAULT_TYPE) -> dict[str, dict[str, Any]]:
    if context.user_data is None:
        return {}
    raw = context.user_data.setdefault(CMUX_TERMINAL_SESSIONS_KEY, {})
    if not isinstance(raw, dict):
        raw = {}
        context.user_data[CMUX_TERMINAL_SESSIONS_KEY] = raw
    return raw


def _store_units(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    picker_id: str,
    thread_id: int,
    units: list[TerminalUnit],
) -> None:
    if context.user_data is not None:
        _session_store(context)[picker_id] = {"thread_id": thread_id, "units": units}


def _load_picker(
    context: ContextTypes.DEFAULT_TYPE, picker_id: str
) -> tuple[int | None, list[TerminalUnit]]:
    store = _raw_session_store(context)
    entry = store.get(picker_id)
    if not isinstance(entry, dict):
        return None, []
    thread_id = entry.get("thread_id")
    units = entry.get("units", [])
    if not isinstance(thread_id, int) or not isinstance(units, list):
        return None, []
    return thread_id, units


def _clear_picker(context: ContextTypes.DEFAULT_TYPE, picker_id: str) -> None:
    if context.user_data is not None:
        store = _raw_session_store(context)
        store.pop(picker_id, None)
        if not store:
            context.user_data.pop(CMUX_TERMINAL_SESSIONS_KEY, None)


async def cmux_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/cmux`` — show the cmux terminal-session picker.

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

    units, error = await _list_cmux_terminal_sessions()
    if error is not None:
        await safe_reply(message, error)
        return

    picker_id = _new_picker_id()
    _store_units(context, picker_id=picker_id, thread_id=thread_id, units=units)

    text, keyboard = build_cmux_picker(units, picker_id=picker_id)
    if not units:
        text = f"{text}\n\n_{_CMUX_NO_TERMINAL_SESSIONS_TEXT}_"
    await safe_reply(message, text, reply_markup=keyboard)


async def _handle_refresh(
    query: CallbackQuery,
    data: str,
    context: ContextTypes.DEFAULT_TYPE,
    update: Update,
) -> None:
    old_picker_id = data[len(CB_CMUX_LIST) :]
    if not old_picker_id:
        await query.answer("Stale picker", show_alert=True)
        return
    thread_id = get_thread_id(update)
    pending_tid, _ = _load_picker(context, old_picker_id)
    if pending_tid is None or thread_id is None or thread_id != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    units, error = await _list_cmux_terminal_sessions()
    if error is not None:
        _clear_picker(context, old_picker_id)
        await safe_edit(query, error)
        await query.answer()
        return
    picker_id = _new_picker_id()
    _clear_picker(context, old_picker_id)
    _store_units(context, picker_id=picker_id, thread_id=thread_id, units=units)
    text, keyboard = build_cmux_picker(units, picker_id=picker_id)
    await safe_edit(query, text, reply_markup=keyboard)
    await query.answer("Refreshed")


async def _resolve_live_bind_unit(
    cached_unit: TerminalUnit,
) -> tuple[TerminalUnit | None, str | None]:
    if not cached_unit.supports_send_text:
        return None, "Terminal session is unavailable"

    units, error = await _list_cmux_terminal_sessions()
    if error is not None:
        return None, error
    live_by_id = {unit.ref.unit_id: unit for unit in units}
    unit = live_by_id.get(cached_unit.ref.unit_id)
    if unit is None:
        return None, "Terminal session no longer exists"
    if not unit.supports_send_text:
        return None, "Terminal session is unavailable"
    return unit, None


async def _handle_bind(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    thread_id = get_thread_id(update)
    try:
        picker_id, idx_raw = data[len(CB_CMUX_BIND) :].split(":", 1)
        idx = int(idx_raw)
    except ValueError:
        await query.answer("Invalid selection")
        return

    pending_tid, cached = _load_picker(context, picker_id)
    if pending_tid is None or thread_id is None or thread_id != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    if idx < 0 or idx >= len(cached):
        await query.answer(
            "Terminal session list changed, please retry", show_alert=True
        )
        return

    unit, alert = await _resolve_live_bind_unit(cached[idx])
    if alert is not None:
        await query.answer(alert, show_alert=True)
        return
    assert unit is not None

    terminal_id = unit.ref.unit_id
    window_id = _qualified_window_id(terminal_id)
    display = unit.title or terminal_id

    # Lazy: terminal identity port pulls window_state_store.
    from ...window_state_ports.terminal_identity import set_terminal_identity

    set_terminal_identity(
        window_id,
        backend=unit.ref.backend,
        unit_id=terminal_id,
        cwd=unit.cwd,
        provider_name=unit.provider_name,
        window_name=display,
    )
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

    if context.user_data is not None:
        context.user_data.pop(CMUX_TERMINAL_SESSIONS_KEY, None)
        clear_browse_state(context.user_data)
        clear_window_picker_state(context.user_data)
        clear_worktree_state(context.user_data)
        context.user_data.pop(PENDING_THREAD_ID, None)
        context.user_data.pop(PENDING_THREAD_TEXT, None)

    await safe_edit(query, f"✅ Bound cmux terminal session `{display}`")
    await query.answer("Bound")


async def _handle_cancel(
    query: CallbackQuery,
    data: str,
    context: ContextTypes.DEFAULT_TYPE,
    update: Update,
) -> None:
    picker_id = data[len(CB_CMUX_CANCEL) :]
    if not picker_id:
        await query.answer("Stale picker", show_alert=True)
        return
    thread_id = get_thread_id(update)
    pending_tid, _ = _load_picker(context, picker_id)
    if pending_tid is None or thread_id is None or thread_id != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    _clear_picker(context, picker_id)
    await safe_edit(query, "Cancelled")
    await query.answer("Cancelled")


@register(CB_CMUX_LIST, CB_CMUX_BIND, CB_CMUX_CANCEL)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    if query is None or query.data is None or user is None:
        return
    data = query.data
    if data.startswith(CB_CMUX_LIST):
        await _handle_refresh(query, data, context, update)
    elif data.startswith(CB_CMUX_BIND):
        await _handle_bind(query, user.id, data, update, context)
    elif data.startswith(CB_CMUX_CANCEL):
        await _handle_cancel(query, data, context, update)


__all__ = [
    "CMUX_TERMINAL_SESSIONS_KEY",
    "build_cmux_picker",
    "cmux_command",
]
