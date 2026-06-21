"""Sessions dashboard — /sessions command showing all bound sessions.

Displays a summary of all thread-bound sessions for the current user
with alive/dead status indicators, per-session action buttons (Esc,
Screenshot, Kill with two-step confirmation), cwd details, and
refresh/new-session actions.

Key functions:
  - sessions_command(): /sessions command handler
  - handle_sessions_refresh(): refresh button callback
  - handle_sessions_kill(): first Kill tap — show confirmation
  - handle_sessions_kill_confirm(): second tap — kill and unbind
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
from ..config import config
from ..telegram_client import PTBTelegramClient, TelegramClient
from ..thread_router import thread_router
from ..multiplexer import multiplexer as tmux_manager
from ..window_query import view_window
from .callback_data import (
    CB_SESSIONS_KILL,
    CB_SESSIONS_KILL_CONFIRM,
    CB_SESSIONS_NEW,
    CB_SESSIONS_REFRESH,
    CB_STATUS_ESC,
    CB_STATUS_SCREENSHOT,
)
from .callback_helpers import user_owns_window
from .callback_registry import register
from .cleanup import clear_topic_state
from .messaging_pipeline.message_sender import safe_edit, safe_reply

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_REFRESH_BTN = InlineKeyboardButton(
    "\U0001f504 Refresh", callback_data=CB_SESSIONS_REFRESH
)
_NEW_BTN = InlineKeyboardButton("\u2795 New Session", callback_data=CB_SESSIONS_NEW)


async def _build_dashboard(user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Build dashboard text and keyboard for a user's sessions."""
    bindings = thread_router.get_all_thread_windows(user_id)

    if not bindings:
        keyboard = InlineKeyboardMarkup([[_REFRESH_BTN, _NEW_BTN]])
        return (
            "No active sessions.\n\nCreate a new topic to start a session.",
            keyboard,
        )

    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}

    lines: list[str] = []
    action_rows: list[list[InlineKeyboardButton]] = []
    for _thread_id, window_id in sorted(bindings.items()):
        display_name = thread_router.get_display_name(window_id)
        view = view_window(window_id)
        alive = window_id in live_ids
        status = "\U0001f7e2" if alive else "\u26ab"

        # Session line with provider + mode tags and cwd detail
        provider_tag = f" [{view.provider_name}]" if view and view.provider_name else ""
        mode_tag = " [YOLO]" if view and view.approval_mode == "yolo" else ""
        line = f"{status} {display_name}{provider_tag}{mode_tag}"
        if view and view.cwd:
            line += f"\n    {view.cwd}"
        lines.append(line)

        if alive:
            row: list[InlineKeyboardButton] = [
                InlineKeyboardButton(
                    "\u238b Esc",
                    callback_data=f"{CB_STATUS_ESC}{window_id}"[:64],
                ),
                InlineKeyboardButton(
                    "\U0001f4f8",
                    callback_data=f"{CB_STATUS_SCREENSHOT}{window_id}"[:64],
                ),
                InlineKeyboardButton(
                    f"\U0001f5d1 Kill {display_name}",
                    callback_data=f"{CB_SESSIONS_KILL}{window_id}"[:64],
                ),
            ]
            action_rows.append(row)

    text = "Sessions\n\n" + "\n".join(lines)
    rows = action_rows + [[_REFRESH_BTN, _NEW_BTN]]
    return text, InlineKeyboardMarkup(rows)


async def sessions_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sessions — show dashboard of all bound sessions."""
    user = update.effective_user
    if not user or not update.message:
        return

    if not config.is_user_allowed(user.id):
        await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    text, keyboard = await _build_dashboard(user.id)
    await safe_reply(update.message, text, reply_markup=keyboard)


async def handle_sessions_refresh(query: CallbackQuery, user_id: int) -> None:
    """Handle refresh button — re-render the dashboard in-place."""
    text, keyboard = await _build_dashboard(user_id)
    await safe_edit(query, text, reply_markup=keyboard)


async def handle_sessions_kill(
    query: CallbackQuery, _user_id: int, window_id: str
) -> None:
    """First Kill tap — show confirmation prompt."""
    display = thread_router.get_display_name(window_id)
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"\u26a0 Confirm kill {display}",
                    callback_data=f"{CB_SESSIONS_KILL_CONFIRM}{window_id}"[:64],
                ),
            ],
            [_REFRESH_BTN],
        ]
    )
    await safe_edit(
        query,
        f"Kill session '{display}'?\n\nThis will terminate the Claude Code process.",
        reply_markup=keyboard,
    )


async def handle_sessions_kill_confirm(
    query: CallbackQuery, user_id: int, window_id: str, client: TelegramClient
) -> None:
    """Second tap — kill the tmux window, unbind all users, refresh dashboard."""
    display = thread_router.get_display_name(window_id)

    w = await tmux_manager.find_window_by_id(window_id)
    if w:
        await tmux_manager.kill_window(w.window_id)

    # Clean up BEFORE unbind — resolve_chat_id needs group_chat_ids
    # which unbind_thread deletes
    for uid, tid, bound_wid in list(thread_router.iter_thread_bindings()):
        if bound_wid == window_id:
            await clear_topic_state(uid, tid, client, window_id=window_id)
            thread_router.unbind_thread(uid, tid)

    logger.info(
        "sessions_kill_confirm: killed window %s (%s), user=%d",
        window_id,
        display,
        user_id,
    )

    # Re-render dashboard
    text, keyboard = await _build_dashboard(user_id)
    await safe_edit(
        query, f"\U0001f5d1 Killed '{display}'\n\n{text}", reply_markup=keyboard
    )


@register(
    CB_SESSIONS_REFRESH,
    CB_SESSIONS_NEW,
    CB_SESSIONS_KILL_CONFIRM,
    CB_SESSIONS_KILL,
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    user = update.effective_user
    if not user:
        return

    data = query.data

    if data == CB_SESSIONS_REFRESH:
        await handle_sessions_refresh(query, user.id)
        await query.answer("Refreshed")
    elif data == CB_SESSIONS_NEW:
        await query.answer("Create a new topic to start a session.")
    elif data.startswith(CB_SESSIONS_KILL_CONFIRM):
        window_id = data[len(CB_SESSIONS_KILL_CONFIRM) :]
        if not user_owns_window(user.id, window_id):
            await query.answer("Not your session", show_alert=True)
            return
        await handle_sessions_kill_confirm(
            query, user.id, window_id, PTBTelegramClient(context.bot)
        )
        await query.answer("Killed")
    elif data.startswith(CB_SESSIONS_KILL):
        window_id = data[len(CB_SESSIONS_KILL) :]
        if not user_owns_window(user.id, window_id):
            await query.answer("Not your session", show_alert=True)
            return
        await handle_sessions_kill(query, user.id, window_id)
        await query.answer()
