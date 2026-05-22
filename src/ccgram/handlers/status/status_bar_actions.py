"""Status-bubble button callbacks (recall, last reply, get file, esc, keys).

Handles inline keyboard callbacks originating from the status-bubble keyboard
built by status_bubble.py:
  - CB_STATUS_RECALL: Send one of the last shown commands directly
  - CB_STATUS_LAST_REPLY: Show last assistant reply (or last shell output)
  - CB_STATUS_GET_FILE: Open the file browser to download a file from agent CWD
  - CB_STATUS_ESC: Send Escape key from status message
  - CB_STATUS_KEY: Quick key dispatch (arrow keys, enter, esc, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
import contextlib
import io

import structlog

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InputMediaDocument,
    Update,
    WebAppInfo,
)
from telegram.error import TelegramError

from ...config import config
from ...miniapp.auth import sign_token
from ...screenshot import text_to_image
from ... import window_query
from ...telegram_client import PTBTelegramClient
from ...thread_router import thread_router
from ...tmux_manager import send_to_window, tmux_manager
from ...topic_state_registry import topic_state
from ..callback_data import (
    CB_KEYS_PREFIX,
    CB_STATUS_ESC,
    CB_STATUS_GET_FILE,
    CB_STATUS_LAST_REPLY,
    CB_STATUS_RECALL,
)
from ..callback_helpers import get_thread_id, parse_target, user_owns_window
from ..callback_registry import register
from ..live.screenshot_callbacks import (
    KEY_LABELS,
    KEYS_SEND_MAP,
    build_screenshot_keyboard,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_KEY_REFRESH_DELAY = 0.3  # seconds — debounce window for rapid key taps
_pending_key_refreshes: dict[tuple[int, str], asyncio.Task[None]] = {}


def build_dashboard_button(window_id: str, user_id: int) -> InlineKeyboardButton | None:
    """Return the 🪟 Dashboard WebApp button, or None when Mini App is disabled.

    Mints a short-lived signed token scoped to ``(window_id, user_id)`` and
    embeds it in the URL so the Mini App can verify the request without an
    extra round-trip. Returns ``None`` (button hidden) when
    ``CCGRAM_MINIAPP_BASE_URL`` is unset.
    """
    base_url = config.miniapp_base_url
    if not base_url:
        return None
    token = sign_token(
        bot_token=config.telegram_bot_token,
        window_id=window_id,
        user_id=user_id,
    )
    url = f"{base_url.rstrip('/')}/app/{token}"
    return InlineKeyboardButton("\U0001fa9f Dashboard", web_app=WebAppInfo(url=url))


@topic_state.register("window")
def _clear_key_refreshes(window_id: str) -> None:
    """Cancel in-flight debounced key-refresh tasks for a closing window."""
    stale = [
        k
        for k in _pending_key_refreshes
        if k[1] == window_id or k[1].startswith(f"{window_id}:%")
    ]
    for k in stale:
        task = _pending_key_refreshes.pop(k, None)
        if task and not task.done():
            task.cancel()


async def _handle_status_recall(
    query: CallbackQuery, user_id: int, data: str, update: Update
) -> None:
    """Handle CB_STATUS_RECALL: send one of the last shown commands directly."""
    rest = data[len(CB_STATUS_RECALL) :]
    if ":" not in rest:
        await query.answer("Invalid data")
        return
    window_id, idx_raw = rest.rsplit(":", 1)
    try:
        idx = int(idx_raw)
        if idx < 0:
            raise ValueError  # noqa: TRY301
    except ValueError:
        await query.answer("Invalid data")
        return
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    if thread_router.resolve_window_for_thread(user_id, thread_id) != window_id:
        await query.answer("Stale status button", show_alert=True)
        return

    # Lazy: command_history → messaging_pipeline → status → status_bar_actions
    # forms a cycle when imported at module top. Keep lazy.
    # Lazy: command_history ↔ status cycle
    from ..command_history import get_history, record_command

    history = get_history(user_id, thread_id, limit=idx + 1)
    if idx >= len(history):
        await query.answer("Command not found", show_alert=True)
        return

    command = history[idx]

    # Lazy: providers/__init__ pulls in process_detection / shell_infra.
    from ...providers import get_provider_for_window

    provider = get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    )
    if not provider.capabilities.supports_mailbox_delivery:
        # Lazy: shell.shell_commands ↔ status via the approval callback
        # wired in bootstrap.
        from ..shell.shell_commands import handle_shell_message

        await handle_shell_message(
            PTBTelegramClient(query.get_bot()),
            user_id,
            thread_id,
            window_id,
            command,
        )
        await query.answer("\u21a9 Recalled")
        return

    ok, err = await send_to_window(window_id, command)
    if not ok:
        await query.answer(err or "Failed to send command", show_alert=True)
        return

    record_command(user_id, thread_id, command)
    await query.answer("\u21a9 Sent")


async def _handle_status_esc(query: CallbackQuery, user_id: int, data: str) -> None:
    """Handle CB_STATUS_ESC: send Escape key from status message."""
    window_id = data[len(CB_STATUS_ESC) :]
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    w = await tmux_manager.find_window_by_id(window_id)
    if w:
        await tmux_manager.send_keys(w.window_id, "Escape", enter=False, literal=False)
        await query.answer("\u238b Sent Escape")
    else:
        await query.answer("Window not found", show_alert=True)


async def _handle_keys(
    query: CallbackQuery, user_id: int, data: str, update: Update
) -> None:
    """Handle CB_KEYS_PREFIX: send a quick key from screenshot keyboard."""
    rest = data[len(CB_KEYS_PREFIX) :]
    colon_idx = rest.find(":")
    if colon_idx < 0:
        await query.answer("Invalid data")
        return
    key_id = rest[:colon_idx]
    target = rest[colon_idx + 1 :]
    window_id, pane_id = parse_target(target)

    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return

    key_info = KEYS_SEND_MAP.get(key_id)
    if not key_info:
        await query.answer("Unknown key")
        return

    tmux_key, enter, literal = key_info
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await query.answer("Window not found", show_alert=True)
        return

    if pane_id:
        await tmux_manager.send_keys_to_pane(
            pane_id, tmux_key, enter=enter, literal=literal, window_id=window_id
        )
    else:
        await tmux_manager.send_keys(
            w.window_id, tmux_key, enter=enter, literal=literal
        )
    await query.answer(KEY_LABELS.get(key_id, key_id))

    # Lazy: live_view ↔ screenshot_callbacks ↔ status pair through the
    # screenshot keyboard wiring.
    # Lazy: live ↔ status cycle
    from ..live.live_view import get_live_view

    thread_id = get_thread_id(update)
    if thread_id is not None and get_live_view(user_id, thread_id) is not None:
        return

    _schedule_key_refresh(user_id, target, query, window_id, pane_id)


def _schedule_key_refresh(
    user_id: int,
    target: str,
    query: CallbackQuery,
    window_id: str,
    pane_id: str | None,
) -> None:
    """Schedule a debounced screenshot refresh after a key press.

    Cancels any pending refresh for the same target so rapid key taps
    (e.g. Down Down Down) only render the final terminal state.
    """
    refresh_key = (user_id, target)
    prev = _pending_key_refreshes.pop(refresh_key, None)
    if prev and not prev.done():
        prev.cancel()

    async def _do_refresh() -> None:
        try:
            await asyncio.sleep(_KEY_REFRESH_DELAY)
            if pane_id:
                text = await tmux_manager.capture_pane_by_id(
                    pane_id, with_ansi=True, window_id=window_id
                )
            else:
                text = await tmux_manager.capture_pane(window_id, with_ansi=True)
            if text:
                png_bytes = await text_to_image(text, with_ansi=True)
                keyboard = build_screenshot_keyboard(window_id, pane_id=pane_id)
                with contextlib.suppress(TelegramError):
                    await query.edit_message_media(
                        media=InputMediaDocument(
                            media=io.BytesIO(png_bytes),
                            filename="screenshot.png",
                        ),
                        reply_markup=keyboard,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("debounced screenshot refresh failed")
        finally:
            _pending_key_refreshes.pop(refresh_key, None)

    _pending_key_refreshes[refresh_key] = asyncio.create_task(_do_refresh())


# --- Dispatch for status-bar action callbacks ---


async def _handle_last_reply(
    query: CallbackQuery, user_id: int, data: str, update: Update
) -> None:
    """Handle CB_STATUS_LAST_REPLY: show last assistant reply or shell output."""
    window_id = data[len(CB_STATUS_LAST_REPLY) :]
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    # Lazy: last_reply ↔ status_bar_actions cycle
    from ..last_reply import send_last_reply

    await send_last_reply(
        PTBTelegramClient(query.get_bot()), chat_id, thread_id, window_id
    )
    await query.answer("\U0001f4c4 Last reply")


async def _handle_get_file(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_STATUS_GET_FILE: open the file browser for the agent CWD."""
    # Lazy: pathlib used only for the CWD existence check
    from pathlib import Path

    window_id = data[len(CB_STATUS_GET_FILE) :]
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    view = window_query.view_window(window_id)
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
    # Lazy: send subpackage ↔ status_bar_actions cycle
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


async def _handle_status_bar_action(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Route status-bar action callbacks to the correct handler."""
    with_update = {
        CB_STATUS_RECALL: _handle_status_recall,
        CB_KEYS_PREFIX: _handle_keys,
        CB_STATUS_LAST_REPLY: _handle_last_reply,
    }
    for prefix, handler in with_update.items():
        if data.startswith(prefix):
            await handler(query, user_id, data, update)
            return

    if data.startswith(CB_STATUS_ESC):
        await _handle_status_esc(query, user_id, data)
        return

    if data.startswith(CB_STATUS_GET_FILE):
        await _handle_get_file(query, user_id, data, update, context)
        return


@register(
    CB_STATUS_RECALL,
    CB_STATUS_ESC,
    CB_KEYS_PREFIX,
    CB_STATUS_LAST_REPLY,
    CB_STATUS_GET_FILE,
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    await _handle_status_bar_action(query, user.id, query.data, update, context)
