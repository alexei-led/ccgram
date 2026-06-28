"""Callback handlers for /send file browser navigation.

Handles all inline keyboard callbacks for the /send file browser UI:
  - CB_SEND_FILE: select and upload a file
  - CB_SEND_DIR: navigate into a subdirectory
  - CB_SEND_PAGE: paginate the directory listing
  - CB_SEND_UP: navigate to the parent directory (clamped at CWD)
  - CB_SEND_CANCEL: cancel the file browser

All handlers share a stale guard: the stored SEND_WINDOW_ID_KEY must match the
window currently bound to the topic. If it doesn't, the state is cleared and the
user receives a "Session expired" alert.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram import CallbackQuery, Message, Update
from telegram.error import TelegramError

from ...config import config
from ...telegram_client import PTBTelegramClient, TelegramClient
from ...thread_router import thread_router
from ..callback_data import (
    CB_SEND_CANCEL,
    CB_SEND_DIR,
    CB_SEND_FILE,
    CB_SEND_PAGE,
    CB_SEND_UP,
)
from ..callback_helpers import get_thread_id
from ..callback_registry import register
from ..messaging_pipeline.message_sender import REACT_DONE, react
from .send_command import upload_file, build_file_browser
from .send_security import is_path_contained, validate_sendable

if TYPE_CHECKING:
    from telegram.ext import ContextTypes
from ..user_state import (
    SEND_CWD_KEY,
    SEND_ITEMS_KEY,
    SEND_PAGE_KEY,
    SEND_PATH_KEY,
    SEND_WINDOW_ID_KEY,
)


async def _react_uploaded(client: TelegramClient, sent: Message | None) -> None:
    """Best-effort persistent ✅ on the just-uploaded file message."""
    if sent is None:
        return
    await react(client, sent.chat_id, sent.message_id, REACT_DONE)


logger = structlog.get_logger()

_SEND_STATE_KEYS = (
    SEND_PATH_KEY,
    SEND_PAGE_KEY,
    SEND_ITEMS_KEY,
    SEND_WINDOW_ID_KEY,
    SEND_CWD_KEY,
)


def _clear_send_state(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Pop all SEND_* keys from context.user_data."""
    if context.user_data is None:
        return
    for key in _SEND_STATE_KEYS:
        context.user_data.pop(key, None)


async def _render_send_browser(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    path: Path,
    cwd: Path,
    *,
    page: int = 0,
) -> None:
    assert context.user_data is not None

    display_text, markup, new_items = build_file_browser(path, cwd, page)
    context.user_data[SEND_ITEMS_KEY] = new_items
    context.user_data[SEND_PATH_KEY] = str(path)
    context.user_data[SEND_PAGE_KEY] = page

    await query.answer()
    msg = query.message if isinstance(query.message, Message) else None
    if msg is not None:
        with contextlib.suppress(TelegramError):
            await msg.edit_text(display_text, reply_markup=markup)


@register(CB_SEND_FILE, CB_SEND_DIR, CB_SEND_PAGE, CB_SEND_UP, CB_SEND_CANCEL)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route CB_SEND_* callbacks to the appropriate handler."""

    query = update.callback_query
    user = update.effective_user
    if query is None or query.data is None or user is None:
        return

    if not config.is_user_allowed(user.id):
        await query.answer("Not authorized", show_alert=True)
        return

    assert context.user_data is not None

    # --- Stale guard ---
    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Not in a topic", show_alert=True)
        return

    stored_window_id = context.user_data.get(SEND_WINDOW_ID_KEY)
    current_window_id = thread_router.resolve_window_for_thread(user.id, thread_id)
    if stored_window_id is None or stored_window_id != current_window_id:
        _clear_send_state(context)
        await query.answer("Browser expired — use /send to restart", show_alert=True)
        return

    data = query.data

    if data.startswith(CB_SEND_FILE):
        await _handle_file(update, context, data)
    elif data.startswith(CB_SEND_DIR):
        await _handle_dir(update, context, data)
    elif data.startswith(CB_SEND_PAGE):
        await _handle_page(update, context, data)
    elif data == CB_SEND_UP:
        await _handle_up(update, context)
    elif data == CB_SEND_CANCEL:
        await _handle_cancel(update, context)


async def _handle_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    data: str,
) -> None:
    """Select and upload the file at the given index."""
    query = update.callback_query
    assert query is not None and context.user_data is not None

    try:
        idx = int(data[len(CB_SEND_FILE) :])
    except ValueError:
        await query.answer("Invalid selection")
        return

    items: list[Path] = context.user_data.get(SEND_ITEMS_KEY, [])
    if idx < 0 or idx >= len(items):
        await query.answer("Item not found", show_alert=True)
        return

    path = items[idx]
    cwd_str = context.user_data.get(SEND_CWD_KEY, "")
    cwd = Path(cwd_str) if cwd_str else path.parent

    error = validate_sendable(path, cwd)
    if error:
        await query.answer(f"Cannot send: {error}", show_alert=True)
        return

    thread_id = get_thread_id(update)
    chat_id: int | None = None
    user = update.effective_user
    if user and thread_id:
        chat_id = thread_router.resolve_chat_id(user.id, thread_id)
    msg = query.message if isinstance(query.message, Message) else None
    if chat_id is None and msg is not None:
        chat_id = msg.chat_id
    if chat_id is None or thread_id is None:
        await query.answer("Cannot determine target chat", show_alert=True)
        return

    client: TelegramClient = PTBTelegramClient(context.bot)
    try:
        sent = await upload_file(client, chat_id, thread_id, path)
    except TelegramError as exc:
        await query.answer(f"Upload failed: {exc}", show_alert=True)
        return

    _clear_send_state(context)
    await _react_uploaded(client, sent)
    await query.answer()
    if msg is not None:
        with contextlib.suppress(TelegramError):
            await msg.delete()


async def _handle_dir(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    data: str,
) -> None:
    """Navigate into a subdirectory and rebuild the browser."""
    query = update.callback_query
    assert query is not None and context.user_data is not None

    try:
        idx = int(data[len(CB_SEND_DIR) :])
    except ValueError:
        await query.answer("Invalid data")
        return

    items: list[Path] = context.user_data.get(SEND_ITEMS_KEY, [])
    if idx < 0 or idx >= len(items):
        await query.answer("Item not found", show_alert=True)
        return

    target_dir = items[idx]
    cwd_str = context.user_data.get(SEND_CWD_KEY, "")
    cwd = Path(cwd_str) if cwd_str else target_dir.parent

    if not is_path_contained(target_dir, cwd):
        await query.answer("Directory is outside project root", show_alert=True)
        return

    await _render_send_browser(query, context, target_dir, cwd)


async def _handle_page(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    data: str,
) -> None:
    """Paginate to the requested page and rebuild the browser."""
    query = update.callback_query
    assert query is not None and context.user_data is not None

    try:
        page = int(data[len(CB_SEND_PAGE) :])
    except ValueError:
        await query.answer("Invalid page")
        return

    path_str = context.user_data.get(SEND_PATH_KEY, "")
    cwd_str = context.user_data.get(SEND_CWD_KEY, "")
    if not path_str or not cwd_str:
        await query.answer("Browser state lost", show_alert=True)
        return

    current_path = Path(path_str)
    cwd = Path(cwd_str)

    await _render_send_browser(query, context, current_path, cwd, page=page)


async def _handle_up(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Navigate to the parent directory, clamped at CWD."""
    query = update.callback_query
    assert query is not None and context.user_data is not None

    path_str = context.user_data.get(SEND_PATH_KEY, "")
    cwd_str = context.user_data.get(SEND_CWD_KEY, "")
    if not path_str or not cwd_str:
        await query.answer("Browser state lost", show_alert=True)
        return

    current_path = Path(path_str)
    cwd = Path(cwd_str)

    if current_path == cwd:
        await query.answer("Already at project root")
        return

    parent = current_path.parent
    if not is_path_contained(parent, cwd):
        parent = cwd

    await _render_send_browser(query, context, parent, cwd)


async def _handle_cancel(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Cancel the file browser and clean up state."""
    query = update.callback_query
    assert query is not None

    _clear_send_state(context)
    await query.answer("Cancelled")
    msg = query.message if isinstance(query.message, Message) else None
    if msg is not None:
        try:
            await msg.delete()
        except TelegramError:
            with contextlib.suppress(TelegramError):
                await msg.edit_text("Cancelled")
