"""Interactive UI callback handlers.

Handles inline keyboard callbacks for AskUserQuestion/ExitPlanMode/Permission UIs:
  - CB_ASK_* direction/action keys: navigate interactive UI via tmux keys
  - CB_ASK_REFRESH: refresh the interactive UI display

Key function: handle_interactive_callback (uniform callback handler signature).
"""

import asyncio
import structlog

from telegram import CallbackQuery, Update
from telegram.ext import ContextTypes

from ..tmux_manager import tmux_manager
from .callback_data import (
    CB_ASK_DOWN,
    CB_ASK_ENTER,
    CB_ASK_ESC,
    CB_ASK_LEFT,
    CB_ASK_REFRESH,
    CB_ASK_RIGHT,
    CB_ASK_SPACE,
    CB_ASK_TAB,
    CB_ASK_UP,
)
from .interactive_ui import clear_interactive_msg, handle_interactive_ui

logger = structlog.get_logger()

# cb_prefix -> (tmux_key, refresh_ui_after)
INTERACTIVE_KEY_MAP: dict[str, tuple[str, bool]] = {
    CB_ASK_UP: ("Up", True),
    CB_ASK_DOWN: ("Down", True),
    CB_ASK_LEFT: ("Left", True),
    CB_ASK_RIGHT: ("Right", True),
    CB_ASK_ESC: ("Escape", False),
    CB_ASK_ENTER: ("Enter", True),
    CB_ASK_SPACE: ("Space", True),
    CB_ASK_TAB: ("Tab", True),
}

# Answer-toast labels for interactive key callbacks
INTERACTIVE_KEY_LABELS: dict[str, str] = {
    CB_ASK_ESC: "\u238b Esc",
    CB_ASK_ENTER: "\u23ce Enter",
    CB_ASK_SPACE: "\u2423 Space",
    CB_ASK_TAB: "\u21e5 Tab",
}

# All interactive prefixes (key map + refresh)
INTERACTIVE_PREFIXES: tuple[str, ...] = (
    *INTERACTIVE_KEY_MAP,
    CB_ASK_REFRESH,
)


def match_interactive_prefix(data: str) -> tuple[str, str, str | None] | None:
    """Match callback data against interactive UI prefixes.

    Returns (cb_prefix, window_id, pane_id_or_None) or None.

    Callback data format:
      - ``"aq:enter:@12"``      → window @12, active pane
      - ``"aq:enter:@12:%5"``   → window @12, specific pane %5
    """
    for prefix in INTERACTIVE_PREFIXES:
        if data.startswith(prefix):
            remainder = data[len(prefix) :]
            # Check for pane_id suffix: "@12:%5"
            if ":%" in remainder:
                window_id, pane_id = remainder.split(":%", 1)
                return prefix, window_id, f"%{pane_id}"
            return prefix, remainder, None
    return None


async def handle_interactive_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle interactive UI callbacks (AskUserQuestion/ExitPlanMode navigation)."""
    matched = match_interactive_prefix(data)
    if not matched:
        return

    cb_prefix, window_id, pane_id = matched
    from .callback_helpers import get_thread_id, user_owns_window

    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return

    thread_id = get_thread_id(update)

    if cb_prefix == CB_ASK_REFRESH:
        await handle_interactive_ui(
            context.bot, user_id, window_id, thread_id, pane_id=pane_id
        )
        await query.answer("\U0001f504")
    else:
        tmux_key, refresh_ui = INTERACTIVE_KEY_MAP[cb_prefix]
        if pane_id:
            sent = await tmux_manager.send_keys_to_pane(
                pane_id, tmux_key, enter=False, literal=False, window_id=window_id
            )
        else:
            w = await tmux_manager.find_window_by_id(window_id)
            sent = bool(w) and await tmux_manager.send_keys(
                w.window_id, tmux_key, enter=False, literal=False
            )
        if sent and refresh_ui:
            await asyncio.sleep(0.5)
            await handle_interactive_ui(
                context.bot, user_id, window_id, thread_id, pane_id=pane_id
            )
        elif sent and not refresh_ui:
            await clear_interactive_msg(user_id, context.bot, thread_id)
        await query.answer(INTERACTIVE_KEY_LABELS.get(cb_prefix, ""))
