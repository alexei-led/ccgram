"""Screenshot and status button callback handlers.

Handles inline keyboard callbacks for screenshot UI and status message buttons:
  - CB_SCREENSHOT_REFRESH: Refresh an existing screenshot
  - CB_STATUS_RECALL: Send one of the two recent commands from status row
  - CB_STATUS_ESC: Send Escape key from status message
  - CB_STATUS_SCREENSHOT: Take a screenshot from status message
  - CB_KEYS_PREFIX: Send a quick key from screenshot keyboard

Key function: handle_screenshot_callback (uniform callback handler signature).
"""

import asyncio
import contextlib
import io
import structlog

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaDocument,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..screenshot import text_to_image
from ..session import session_manager
from ..tmux_manager import tmux_manager
from .callback_data import (
    CB_KEYS_PREFIX,
    CB_PANE_SCREENSHOT,
    CB_SCREENSHOT_REFRESH,
    CB_STATUS_ESC,
    CB_STATUS_RECALL,
    CB_STATUS_NOTIFY,
    CB_STATUS_SCREENSHOT,
    NOTIFY_MODE_LABELS,
)
from .callback_helpers import get_thread_id, user_owns_window

logger = structlog.get_logger()

# key_id -> (tmux_key, enter, literal)
KEYS_SEND_MAP: dict[str, tuple[str, bool, bool]] = {
    "up": ("Up", False, False),
    "dn": ("Down", False, False),
    "lt": ("Left", False, False),
    "rt": ("Right", False, False),
    "esc": ("Escape", False, False),
    "ent": ("Enter", False, False),
    "spc": ("Space", False, False),
    "tab": ("Tab", False, False),
    "cc": ("C-c", False, False),
}

# key_id -> display label (shown in callback answer toast)
KEY_LABELS: dict[str, str] = {
    "up": "\u2191",
    "dn": "\u2193",
    "lt": "\u2190",
    "rt": "\u2192",
    "esc": "\u238b Esc",
    "ent": "\u23ce Enter",
    "spc": "\u2423 Space",
    "tab": "\u21e5 Tab",
    "cc": "^C",
}


def build_screenshot_keyboard(
    window_id: str, pane_id: str | None = None
) -> InlineKeyboardMarkup:
    """Build inline keyboard for screenshot: control keys + refresh.

    When *pane_id* is given, keys and refresh target that specific pane
    instead of the window's active pane.
    """
    target = f"{window_id}:{pane_id}" if pane_id else window_id

    def btn(label: str, key_id: str) -> InlineKeyboardButton:
        return InlineKeyboardButton(
            label,
            callback_data=f"{CB_KEYS_PREFIX}{key_id}:{target}"[:64],
        )

    return InlineKeyboardMarkup(
        [
            [btn("\u2423 Space", "spc"), btn("\u2191", "up"), btn("\u21e5 Tab", "tab")],
            [btn("\u2190", "lt"), btn("\u2193", "dn"), btn("\u2192", "rt")],
            [btn("\u238b Esc", "esc"), btn("^C", "cc"), btn("\u23ce Enter", "ent")],
            [
                InlineKeyboardButton(
                    "\U0001f504 Refresh",
                    callback_data=f"{CB_SCREENSHOT_REFRESH}{target}"[:64],
                )
            ],
        ]
    )


async def _handle_pane_screenshot(
    query: CallbackQuery, user_id: int, data: str, update: Update
) -> None:
    """Handle CB_PANE_SCREENSHOT: screenshot a specific pane."""
    rest = data[len(CB_PANE_SCREENSHOT) :]
    # Format: <window_id>:<pane_id> e.g. "@0:%3"
    colon_idx = rest.find(":")
    if colon_idx < 0:
        await query.answer("Invalid data")
        return
    window_id = rest[:colon_idx]
    pane_id = rest[colon_idx + 1 :]

    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return

    text = await tmux_manager.capture_pane_by_id(
        pane_id, with_ansi=True, window_id=window_id
    )
    if not text:
        await query.answer("Failed to capture pane", show_alert=True)
        return

    png_bytes = await text_to_image(text, with_ansi=True)
    keyboard = build_screenshot_keyboard(window_id, pane_id=pane_id)
    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    try:
        await query.get_bot().send_document(
            chat_id=chat_id,
            document=io.BytesIO(png_bytes),
            filename=f"pane_{pane_id}.png",
            reply_markup=keyboard,
            message_thread_id=thread_id,
        )
        await query.answer(f"\U0001f4f8 Pane {pane_id}")
    except TelegramError as e:
        logger.error("Failed to send pane screenshot: %s", e)
        await query.answer("Failed to send screenshot", show_alert=True)


async def handle_screenshot_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    _context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle screenshot, status button, and quick-key callbacks."""
    if data.startswith(CB_SCREENSHOT_REFRESH):
        await _handle_refresh(query, user_id, data)
    elif data.startswith(CB_STATUS_RECALL):
        await _handle_status_recall(query, user_id, data, update)
    elif data.startswith(CB_STATUS_ESC):
        await _handle_status_esc(query, user_id, data)
    elif data.startswith(CB_STATUS_SCREENSHOT):
        await _handle_status_screenshot(query, user_id, data, update)
    elif data.startswith(CB_STATUS_NOTIFY):
        await _handle_notify_toggle(query, user_id, data)
    elif data.startswith(CB_PANE_SCREENSHOT):
        await _handle_pane_screenshot(query, user_id, data, update)
    elif data.startswith(CB_KEYS_PREFIX):
        await _handle_keys(query, user_id, data)


async def _handle_refresh(query: CallbackQuery, user_id: int, data: str) -> None:
    """Handle CB_SCREENSHOT_REFRESH: refresh an existing screenshot."""
    target = data[len(CB_SCREENSHOT_REFRESH) :]
    window_id, pane_id = _parse_target(target)
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await query.answer("Window no longer exists", show_alert=True)
        return

    if pane_id:
        text = await tmux_manager.capture_pane_by_id(
            pane_id, with_ansi=True, window_id=window_id
        )
    else:
        text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not text:
        await query.answer("Failed to capture pane", show_alert=True)
        return

    png_bytes = await text_to_image(text, with_ansi=True)
    keyboard = build_screenshot_keyboard(window_id, pane_id=pane_id)
    try:
        await query.edit_message_media(
            media=InputMediaDocument(
                media=io.BytesIO(png_bytes), filename="screenshot.png"
            ),
            reply_markup=keyboard,
        )
        await query.answer("Refreshed")
    except TelegramError as e:
        logger.error("Failed to refresh screenshot: %s", e)
        await query.answer("Failed to refresh", show_alert=True)


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
    except ValueError:
        await query.answer("Invalid data")
        return
    if idx < 0:
        await query.answer("Invalid data")
        return
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    if session_manager.resolve_window_for_thread(user_id, thread_id) != window_id:
        await query.answer("Stale status button", show_alert=True)
        return

    from .command_history import get_history, record_command

    history = get_history(user_id, thread_id, limit=idx + 1)
    if idx >= len(history):
        await query.answer("Command not found", show_alert=True)
        return

    command = history[idx]
    ok, err = await session_manager.send_to_window(window_id, command)
    if not ok:
        await query.answer(err or "Failed to send command", show_alert=True)
        return

    record_command(user_id, thread_id, command)
    await query.answer("\u21a9 Sent")


async def _handle_status_screenshot(
    query: CallbackQuery, user_id: int, data: str, update: Update
) -> None:
    """Handle CB_STATUS_SCREENSHOT: take screenshot from status message."""
    window_id = data[len(CB_STATUS_SCREENSHOT) :]
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await query.answer("Window not found", show_alert=True)
        return
    text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not text:
        await query.answer("Failed to capture", show_alert=True)
        return
    png_bytes = await text_to_image(text, with_ansi=True)
    keyboard = build_screenshot_keyboard(window_id)
    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    try:
        await query.get_bot().send_document(
            chat_id=chat_id,
            document=io.BytesIO(png_bytes),
            filename="screenshot.png",
            reply_markup=keyboard,
            message_thread_id=thread_id,
        )
        await query.answer("\U0001f4f8")
    except TelegramError as e:
        logger.error("Failed to send screenshot: %s", e)
        await query.answer("Failed to send screenshot", show_alert=True)


async def _handle_notify_toggle(query: CallbackQuery, user_id: int, data: str) -> None:
    """Handle CB_STATUS_NOTIFY: cycle notification mode for a window."""
    window_id = data[len(CB_STATUS_NOTIFY) :]
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    new_mode = session_manager.cycle_notification_mode(window_id)
    label = NOTIFY_MODE_LABELS.get(new_mode, new_mode)
    # Update keyboard in-place to reflect new bell icon (all modes keep the
    # message so users can always toggle back)
    from .message_queue import build_status_keyboard

    keyboard = build_status_keyboard(window_id)
    with contextlib.suppress(TelegramError):
        await query.edit_message_reply_markup(reply_markup=keyboard)
    await query.answer(label)


def _parse_target(target: str) -> tuple[str, str | None]:
    """Parse window_id and optional pane_id from target string.

    Target format: ``@0`` (window only) or ``@0:%3`` (window + pane).
    """
    if ":%" in target:
        idx = target.index(":%")
        return target[:idx], target[idx + 1 :]
    return target, None


async def _handle_keys(query: CallbackQuery, user_id: int, data: str) -> None:
    """Handle CB_KEYS_PREFIX: send a quick key from screenshot keyboard."""
    rest = data[len(CB_KEYS_PREFIX) :]
    colon_idx = rest.find(":")
    if colon_idx < 0:
        await query.answer("Invalid data")
        return
    key_id = rest[:colon_idx]
    target = rest[colon_idx + 1 :]
    window_id, pane_id = _parse_target(target)

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

    # Refresh screenshot after key press
    await asyncio.sleep(0.5)
    if pane_id:
        text = await tmux_manager.capture_pane_by_id(
            pane_id, with_ansi=True, window_id=window_id
        )
    else:
        text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
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
