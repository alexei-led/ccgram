"""Screenshot callback handlers.

Handles inline keyboard callbacks for screenshot UI:
  - CB_SCREENSHOT_REFRESH: Refresh an existing screenshot
  - CB_LIVE_START: Start auto-refreshing live terminal view
  - CB_LIVE_STOP: Stop live view and revert to screenshot keyboard
  - CB_STATUS_SCREENSHOT: Take a screenshot from status message
  - CB_PANE_SCREENSHOT: Take a screenshot of a specific pane

Status-bar button callbacks (notify, recall, esc, remote, keys) are in
status_bar_actions.py. Toolbar callbacks (CB_TOOLBAR_*) are in
toolbar_callbacks.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import contextlib
import io
import time

import structlog

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaDocument,
    InputMediaPhoto,
    Update,
)
from telegram.error import TelegramError
from ...screenshot import text_to_image
from ...telegram_client import PTBTelegramClient
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager

from ..callback_data import (
    CB_KEYS_PREFIX,
    CB_LIVE_START,
    CB_LIVE_STOP,
    CB_PANE_DELIMITER,
    CB_PANE_SCREENSHOT,
    CB_SCREENSHOT_REFRESH,
    CB_STATUS_SCREENSHOT,
)
from ..callback_helpers import get_thread_id, parse_target, user_owns_window
from ..callback_registry import register

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

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
    target = f"{window_id}{CB_PANE_DELIMITER}{pane_id}" if pane_id else window_id

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
                    "\U0001f4fa Live",
                    callback_data=f"{CB_LIVE_START}{target}"[:64],
                ),
                InlineKeyboardButton(
                    "\U0001f504 Refresh",
                    callback_data=f"{CB_SCREENSHOT_REFRESH}{target}"[:64],
                ),
            ],
        ]
    )


async def _handle_live_start(
    query: CallbackQuery, user_id: int, data: str, update: Update
) -> None:
    """Handle CB_LIVE_START: start auto-refreshing live view."""
    target = data[len(CB_LIVE_START) :]
    window_id, pane_id = parse_target(target)
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return

    # Lazy: live_view ↔ screenshot_callbacks bidirectional cycle.
    from .live_view import (
        LiveViewState,
        build_live_keyboard,
        content_hash,
        is_live,
        start_live_view,
    )

    if is_live(user_id, thread_id):
        await query.answer("Already live")
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await query.answer("Window not found", show_alert=True)
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

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    png_bytes = await text_to_image(text, with_ansi=True, live_mode=True)
    keyboard = build_live_keyboard(window_id, pane_id=pane_id)

    try:
        await query.edit_message_media(
            media=InputMediaPhoto(
                media=io.BytesIO(png_bytes),
                caption=f"Live \u00b7 {time.strftime('%H:%M:%S')}",
            ),
            reply_markup=keyboard,
        )
    except TelegramError as e:
        logger.error("Failed to start live view: %s", e)
        await query.answer("Failed to start live view", show_alert=True)
        return

    if query.message is None:
        await query.answer("Message lost")
        return
    start_live_view(
        LiveViewState(
            chat_id=chat_id,
            message_id=query.message.message_id,
            thread_id=thread_id,
            user_id=user_id,
            window_id=window_id,
            pane_id=pane_id,
            last_hash=content_hash(text),
        )
    )
    await query.answer("\U0001f4fa Live started")


async def _handle_live_stop(
    query: CallbackQuery, user_id: int, data: str, update: Update
) -> None:
    """Handle CB_LIVE_STOP: stop live view and revert to screenshot keyboard."""
    target = data[len(CB_LIVE_STOP) :]
    window_id, pane_id = parse_target(target)
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return

    # Lazy: live_view ↔ screenshot_callbacks bidirectional cycle.
    from .live_view import stop_live_view

    stop_live_view(user_id, thread_id)
    keyboard = build_screenshot_keyboard(window_id, pane_id=pane_id)
    with contextlib.suppress(TelegramError):
        await query.edit_message_caption(caption="Screenshot", reply_markup=keyboard)
    await query.answer("\u23f9 Stopped")


async def _handle_pane_screenshot(
    query: CallbackQuery, user_id: int, data: str, update: Update
) -> None:
    """Handle CB_PANE_SCREENSHOT: screenshot a specific pane."""
    rest = data[len(CB_PANE_SCREENSHOT) :]
    # Format: <window_id>|<pane_id> — delimiter is | so herdr ids (w2:t1, w2:p1) round-trip
    delim_idx = rest.find(CB_PANE_DELIMITER)
    if delim_idx < 0:
        await query.answer("Invalid data")
        return
    window_id = rest[:delim_idx]
    pane_id = rest[delim_idx + 1 :]

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
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    client = PTBTelegramClient(query.get_bot())
    try:
        await client.send_document(
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
    """Handle screenshot-related callbacks only.

    Status-bar actions (notify, recall, esc, remote, keys) are dispatched
    separately by status_bar_actions.py.
    """
    with_update = {
        CB_LIVE_START: _handle_live_start,
        CB_LIVE_STOP: _handle_live_stop,
        CB_STATUS_SCREENSHOT: _handle_status_screenshot,
        CB_PANE_SCREENSHOT: _handle_pane_screenshot,
    }
    for prefix, handler in with_update.items():
        if data.startswith(prefix):
            await handler(query, user_id, data, update)
            return

    without_update = {
        CB_SCREENSHOT_REFRESH: _handle_refresh,
    }
    for prefix, handler in without_update.items():
        if data.startswith(prefix):
            await handler(query, user_id, data)
            return


async def _handle_refresh(query: CallbackQuery, user_id: int, data: str) -> None:
    """Handle CB_SCREENSHOT_REFRESH: refresh an existing screenshot."""
    target = data[len(CB_SCREENSHOT_REFRESH) :]
    window_id, pane_id = parse_target(target)
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
        text = await tmux_manager.capture_pane(window_id, with_ansi=True)
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
    text = await tmux_manager.capture_pane(window_id, with_ansi=True)
    if not text:
        await query.answer("Failed to capture", show_alert=True)
        return
    png_bytes = await text_to_image(text, with_ansi=True)
    keyboard = build_screenshot_keyboard(window_id)
    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    client = PTBTelegramClient(query.get_bot())
    try:
        await client.send_document(
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


# ------------------------------------------------------------------
# Command handlers (moved from bot.py)
# ------------------------------------------------------------------


async def screenshot_command(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Capture and send a terminal screenshot for the current topic."""
    # Lazy: command-handler entry; loaded by callback_registry.  Hoisting
    # config / utils / message_sender pulls handler registration into
    # this module's import graph and breaks reset-for-testing.
    # Lazy: config singleton resolved at call time so tests can swap it
    from ...config import config

    # Lazy: utils pulls in chat-id helpers that reach back into handlers
    from ...utils import handle_general_topic_message, is_general_topic

    # Lazy: messaging_pipeline ↔ live cycle through status_bubble
    from ..messaging_pipeline.message_sender import safe_reply

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await safe_reply(update.message, "\u274c Window no longer exists.")
        return

    pane_text = await tmux_manager.capture_pane(window_id, with_ansi=True)
    if not pane_text:
        await safe_reply(update.message, "\u274c Failed to capture terminal.")
        return

    # Lazy: only needed when rendering the screenshot payload
    import io

    png_bytes = await text_to_image(pane_text, with_ansi=True)
    keyboard = build_screenshot_keyboard(window_id)
    chat_id = thread_router.resolve_chat_id(user.id, thread_id)
    client = PTBTelegramClient(update.message.get_bot())
    try:
        await client.send_document(
            chat_id=chat_id,
            document=io.BytesIO(png_bytes),
            filename="screenshot.png",
            reply_markup=keyboard,
            message_thread_id=thread_id,
        )
    except TelegramError as e:
        logger.error("Failed to send screenshot: %s", e)
        await safe_reply(update.message, "\u274c Failed to send screenshot.")


async def live_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Open auto-refreshing live terminal view directly, skipping the screenshot step."""
    # Lazy: same callback-registry contract as screenshot_command, plus
    # the live_view ↔ screenshot_callbacks cycle.
    # Lazy: config singleton resolved at call time so tests can swap it
    from ...config import config

    # Lazy: utils pulls in chat-id helpers that reach back into handlers
    from ...utils import handle_general_topic_message, is_general_topic

    # Lazy: messaging_pipeline ↔ live cycle through status_bubble
    from ..messaging_pipeline.message_sender import safe_reply

    # Lazy: live_view registers callbacks against this module on import
    from .live_view import (
        LiveViewState,
        build_live_keyboard,
        content_hash,
        is_live,
        start_live_view,
    )

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "❌ Use this command inside a topic.")
        return

    if is_live(user.id, thread_id):
        await safe_reply(update.message, "\U0001f4fa Live view already running.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "❌ This topic is not bound to any session.")
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await safe_reply(update.message, "❌ Window no longer exists.")
        return

    text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not text:
        await safe_reply(update.message, "❌ Failed to capture terminal.")
        return

    chat_id = thread_router.resolve_chat_id(user.id, thread_id)
    png_bytes = await text_to_image(text, with_ansi=True, live_mode=True)
    keyboard = build_live_keyboard(window_id)
    client = PTBTelegramClient(update.message.get_bot())
    try:
        sent = await client.send_photo(
            chat_id=chat_id,
            photo=io.BytesIO(png_bytes),
            caption=f"Live · {time.strftime('%H:%M:%S')}",
            reply_markup=keyboard,
            message_thread_id=thread_id,
        )
    except TelegramError as e:
        logger.error("Failed to start live view: %s", e)
        await safe_reply(update.message, "❌ Failed to start live view.")
        return

    start_live_view(
        LiveViewState(
            chat_id=chat_id,
            message_id=sent.message_id,
            thread_id=thread_id,
            user_id=user.id,
            window_id=window_id,
            pane_id=None,
            last_hash=content_hash(text),
        )
    )


async def panes_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: C901
    """List all panes in the current topic's window."""
    # Lazy: same callback-registry contract as screenshot_command.
    from telegram import InlineKeyboardMarkup

    # Lazy: config singleton resolved at call time so tests can swap it
    from ...config import config

    # Lazy: utils pulls in chat-id helpers that reach back into handlers
    from ...utils import handle_general_topic_message, is_general_topic

    # Lazy: window_state_ports proxy not yet wired at module load
    from ...window_state_ports.pane_state import (
        get_pane_lifecycle_notify,
        get_pane_projection,
    )

    # Lazy: messaging_pipeline ↔ live cycle through status_bubble
    from ..messaging_pipeline.message_sender import safe_reply

    # Lazy: pane_callbacks ↔ screenshot_callbacks cycle through the toolbar
    from .pane_callbacks import build_pane_buttons, build_pane_lifecycle_button

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        if (
            update.message
            and update.effective_chat
            and is_general_topic(update.message)
        ):
            await handle_general_topic_message(
                update.get_bot(), update.message, update.effective_chat.id
            )
        else:
            await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    panes = await tmux_manager.list_panes(window_id)
    if len(panes) <= 1:
        await safe_reply(
            update.message,
            "\U0001f4d0 Single pane \u2014 no multi-pane layout detected.",
        )
        return

    # Lazy: polling_state imports many handler symbols transitively.
    from ..polling.polling_state import interactive_strategy

    lines = [f"\U0001f4d0 {len(panes)} panes in window\n"]
    rows: list[list] = []
    for pane in panes:
        prefix = "\U0001f4cd" if pane.active else "  "
        label = f"Pane {pane.index} ({pane.command})"
        suffix_parts: list[str] = []
        if pane.active:
            suffix_parts.append("active")
        if interactive_strategy.has_pane_alert(pane.pane_id):
            prefix = "\u26a0\ufe0f"
            suffix_parts.append("blocked")
        elif not pane.active:
            suffix_parts.append("running")
        stored = get_pane_projection(window_id, pane.pane_id)
        if stored and stored.name:
            suffix_parts.insert(0, stored.name)
        if stored and stored.subscribed:
            suffix_parts.append("subscribed")
        suffix = f" \u2014 {', '.join(suffix_parts)}" if suffix_parts else ""
        lines.append(f"{prefix} {label}{suffix}")
        rows.append(
            build_pane_buttons(
                window_id,
                pane.pane_id,
                subscribed=bool(stored and stored.subscribed),
            )
        )

    lifecycle_on = get_pane_lifecycle_notify(window_id, config.pane_lifecycle_notify)
    rows.append([build_pane_lifecycle_button(window_id, enabled=lifecycle_on)])
    keyboard = InlineKeyboardMarkup(rows) if rows else None
    await safe_reply(update.message, "\n".join(lines), reply_markup=keyboard)


# --- Registry dispatch entry point ---


@register(
    CB_LIVE_START,
    CB_LIVE_STOP,
    CB_SCREENSHOT_REFRESH,
    CB_STATUS_SCREENSHOT,
    CB_PANE_SCREENSHOT,
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    await handle_screenshot_callback(query, user.id, query.data, update, context)
