"""Status-bubble rendering, send/edit/clear, and task-status formatting.

Owns the per-topic status message lifecycle: keyboard layout, send/edit/clear
I/O, Claude task-list formatting, and status-to-content conversion.  The queue
worker in ``message_queue`` delegates ``StatusUpdateTask`` / ``StatusClearTask``
here; ``convert_status_to_content`` is defined here and imported by
``message_queue._process_content_task``.

Status-bar Row 1: [⎋ Esc] [📸 Screenshot] [📄 Last] [📥 Get File].
"""

from __future__ import annotations

import contextlib
import time

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

from ...claude_task_state import get_claude_task_snapshot, get_claude_wait_header
from ...expandable_quote import format_expandable_quote
from ...telegram_client import TelegramClient
from ...thread_router import thread_router
from ...window_state_store import PaneInfo, window_store

from ..callback_data import (
    CB_STATUS_ESC,
    CB_STATUS_GET_FILE,
    CB_STATUS_LAST_REPLY,
    CB_STATUS_RECALL,
    CB_STATUS_SCREENSHOT,
    IDLE_STATUS_TEXT,
)
from ..messaging_pipeline.message_sender import edit_with_fallback, safe_send
from ..messaging_pipeline.message_task import (
    StatusClearTask,
    StatusUpdateTask,
    thread_key,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

# Status message tracking: (user_id, thread_key) -> (message_id, window_id, last_text, chat_id)
_status_msg_info: dict[tuple[int, int], tuple[int, str, str, int]] = {}


# ---------------------------------------------------------------------------
# Keyboard builder
# ---------------------------------------------------------------------------


def build_status_keyboard(
    window_id: str,
    history: list[str] | None = None,
    *,
    user_id: int | None = None,
) -> InlineKeyboardMarkup:
    """Build inline keyboard for status messages.

    Layout:
      Row 1 (optional): up to 2 history-recall buttons
      Row 2: [⎋ Esc] [📸 Screenshot] [📄 Last] [📥 Get File]
      Row 3 (optional): [🪟 Dashboard] when Mini App is enabled and user_id is set
    """
    # Lazy: command_history → messaging_pipeline → status → status_bubble
    # forms a cycle when imported at module top. Keep lazy.
    # Lazy: command_history ↔ status cycle
    from ..command_history import truncate_for_display

    # Lazy: status_bubble ↔ status_bar_actions sibling cycle
    from .status_bar_actions import build_dashboard_button

    rows: list[list[InlineKeyboardButton]] = []

    if history:
        hist_row: list[InlineKeyboardButton] = []
        for idx, cmd in enumerate(history[:2]):
            label = truncate_for_display(cmd, 20)
            hist_row.append(
                InlineKeyboardButton(
                    f"\u2191 {label}",
                    callback_data=f"{CB_STATUS_RECALL}{window_id}:{idx}"[:64],
                )
            )
        rows.append(hist_row)

    rows.append(
        [
            InlineKeyboardButton(
                "\u238b Esc",
                callback_data=f"{CB_STATUS_ESC}{window_id}"[:64],
            ),
            InlineKeyboardButton(
                "\U0001f4f8",
                callback_data=f"{CB_STATUS_SCREENSHOT}{window_id}"[:64],
            ),
            InlineKeyboardButton(
                "\U0001f4c4 Last",
                callback_data=f"{CB_STATUS_LAST_REPLY}{window_id}"[:64],
            ),
            InlineKeyboardButton(
                "\U0001f4e5 Get File",
                callback_data=f"{CB_STATUS_GET_FILE}{window_id}"[:64],
            ),
        ]
    )
    if user_id is not None:
        dashboard = build_dashboard_button(window_id, user_id)
        if dashboard is not None:
            rows.append([dashboard])
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_idle_history(
    user_id: int, thread_id_or_0: int, status_text: str
) -> list[str] | None:
    """Return history list if the status is idle, else None."""
    # Lazy: command_history → messaging_pipeline → status forms a cycle.
    from ..command_history import get_history

    first_line = status_text.split("\n", 1)[0]
    if first_line != IDLE_STATUS_TEXT:
        return None
    return get_history(user_id, thread_id_or_0, limit=2) or None


# ---------------------------------------------------------------------------
# Per-pane status block
# ---------------------------------------------------------------------------

# When a window has this many or more visible panes, the per-pane block is
# rendered as an expandable blockquote so the bubble stays compact.
_PANE_BLOCK_EXPAND_THRESHOLD = 4
_PANE_BLOCKED_GLYPH = "⏸"  # ⏸
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60


def _format_pane_idle_age(pane: PaneInfo, now_wall: float) -> str:
    """Format a pane's idle duration relative to ``now_wall``."""
    if not pane.last_active_ts:
        return "idle"
    delta = max(0.0, now_wall - pane.last_active_ts)
    if delta < _SECONDS_PER_MINUTE:
        return "idle"
    minutes = int(delta // _SECONDS_PER_MINUTE)
    if minutes < _MINUTES_PER_HOUR:
        return f"idle {minutes}m"
    hours = int(minutes // _MINUTES_PER_HOUR)
    return f"idle {hours}h"


def _format_pane_item(pane: PaneInfo, now_wall: float) -> str:
    """Render a single pane as ``"<label> <state>"``."""
    label = pane.name.strip() if pane.name and pane.name.strip() else pane.pane_id
    if pane.state == "active":
        return f"{label} active"
    if pane.state == "blocked":
        return f"{label} {_PANE_BLOCKED_GLYPH} blocked"
    if pane.state == "dead":
        return f"{label} dead"
    return f"{label} {_format_pane_idle_age(pane, now_wall)}"


def format_pane_block(window_id: str) -> str | None:
    """Render a per-pane status block for windows with multiple panes.

    Returns ``None`` for single-pane windows (or windows without recorded
    panes). For 2-3 panes returns a single ``└``-prefixed line listing each
    pane's state. For 4+ panes wraps the list in an expandable blockquote
    so the status bubble stays compact while still letting users tap to
    reveal every pane.

    Dead panes are excluded from the rendered list — pane lifecycle
    notifications are owned by ``PaneStatusStrategy`` and Theme 5 Task 2.5.
    """
    state = window_store.window_states.get(window_id)
    if state is None or len(state.panes) <= 1:
        return None
    visible = [p for p in state.panes.values() if p.state != "dead"]
    if len(visible) <= 1:
        return None
    # Sort numerically so %2 comes before %10 — lexicographic puts %10 first.
    visible.sort(key=lambda p: (int(p.pane_id.lstrip("%")), p.pane_id))
    now_wall = time.time()
    items = [_format_pane_item(p, now_wall) for p in visible]
    if len(visible) >= _PANE_BLOCK_EXPAND_THRESHOLD:
        body = "\n".join(f"└ {item}" for item in items)
        return format_expandable_quote(body)
    return "└ " + " · ".join(items)


# ---------------------------------------------------------------------------
# Claude task-status formatting
# ---------------------------------------------------------------------------


_TASK_STATUS_GLYPHS = {
    "completed": "\u2714",
    "in_progress": "\u25d4",
}
_TASK_DEFAULT_GLYPH = "\u25fb"
_VISIBLE_TASK_LIMIT = 8


def _format_task_lines(snapshot: object) -> list[str]:
    """Render the task snapshot into status-bubble lines."""
    total = getattr(snapshot, "total_count", 0)
    done = getattr(snapshot, "done_count", 0)
    open_count = getattr(snapshot, "open_count", 0)
    items = list(getattr(snapshot, "items", []))
    visible_items = items[:_VISIBLE_TASK_LIMIT]
    lines: list[str] = [f"{total} tasks ({done} done, {open_count} open)"]
    for item in visible_items:
        glyph = _TASK_STATUS_GLYPHS.get(item.status, _TASK_DEFAULT_GLYPH)
        label = (
            item.active_form
            if item.status == "in_progress" and item.active_form
            else item.subject
        )
        if item.owner:
            label = f"{label} ({item.owner})"
        line = f"{glyph} #{item.task_id} {label}".rstrip()
        if item.blocked_by:
            blocked = ", ".join(f"#{task_id}" for task_id in item.blocked_by)
            line = f"{line} blocked by {blocked}"
        lines.append(line)
    hidden_count = total - len(visible_items)
    if hidden_count > 0:
        lines.append(f"+{hidden_count} more")
    return lines


def format_claude_task_status(window_id: str, base_text: str | None) -> str | None:
    """Compose Claude wait/task state plus the per-pane block (if any)."""
    snapshot = get_claude_task_snapshot(window_id)
    wait_header = get_claude_wait_header(window_id)
    pane_block = format_pane_block(window_id)
    if snapshot is None and not wait_header and pane_block is None:
        return base_text

    lines: list[str] = []
    header = wait_header or base_text
    if header:
        lines.append(header)
    if pane_block is not None:
        lines.append(pane_block)
    if snapshot is not None:
        lines.extend(_format_task_lines(snapshot))
    return "\n".join(lines) if lines else base_text


# ---------------------------------------------------------------------------
# Status I/O — send / edit / clear
# ---------------------------------------------------------------------------


async def send_status_text(
    client: TelegramClient,
    user_id: int,
    thread_id_or_0: int,
    window_id: str,
    text: str,
) -> None:
    """Send a new status message with action buttons and track it.

    If a status message already exists for this (user, thread), edit it
    in-place via ``edit_with_fallback`` (entity-formatted, plain-text fallback
    on TelegramError).  Same-window same-text calls are a no-op.
    """
    skey = (user_id, thread_id_or_0)
    thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)

    history = _get_idle_history(user_id, thread_id_or_0, text)
    keyboard = build_status_keyboard(
        window_id,
        history=history,
        user_id=user_id,
    )

    existing = _status_msg_info.get(skey)
    if existing:
        msg_id, stored_wid, last_text, stored_chat_id = existing
        if stored_wid == window_id and text == last_text:
            return
        if stored_wid == window_id:
            success = await edit_with_fallback(
                client, stored_chat_id, msg_id, text, reply_markup=keyboard
            )
            if success:
                _status_msg_info[skey] = (msg_id, window_id, text, stored_chat_id)
                return
            # Edit failed — original message may still exist server-side.
            # Best-effort delete to avoid an orphan before creating a replacement.
            with contextlib.suppress(TelegramError):
                await client.delete_message(chat_id=stored_chat_id, message_id=msg_id)
            _status_msg_info.pop(skey, None)
        else:
            await clear_status_message(client, user_id, thread_id_or_0)

    msg = await safe_send(
        client,
        chat_id,
        text,
        message_thread_id=thread_id,
        reply_markup=keyboard,
    )
    if msg is not None:
        _status_msg_info[skey] = (msg.message_id, window_id, text, chat_id)


async def clear_status_message(
    client: TelegramClient,
    user_id: int,
    thread_id_or_0: int = 0,
) -> None:
    """Delete the status message for a user."""
    skey = (user_id, thread_id_or_0)
    info = _status_msg_info.pop(skey, None)
    if info:
        msg_id, _, _, chat_id = info
        try:
            await client.delete_message(chat_id=chat_id, message_id=msg_id)
        except TelegramError as e:
            logger.debug("Failed to delete status message %s: %s", msg_id, e)


async def convert_status_to_content(
    client: TelegramClient,
    user_id: int,
    thread_id_or_0: int,
    window_id: str,
    content_text: str,
) -> int | None:
    """Convert status message to content message by editing it.

    Returns the message_id if converted successfully, None otherwise.
    """
    skey = (user_id, thread_id_or_0)
    info = _status_msg_info.pop(skey, None)
    if not info:
        return None

    msg_id, stored_wid, _, chat_id = info
    if stored_wid != window_id:
        with contextlib.suppress(TelegramError):
            await client.delete_message(chat_id=chat_id, message_id=msg_id)
        return None

    success = await edit_with_fallback(
        client,
        chat_id,
        msg_id,
        content_text,
        reply_markup=None,
    )
    if success:
        return msg_id
    return None


# ---------------------------------------------------------------------------
# Status task processors (called by message_queue worker)
# ---------------------------------------------------------------------------


async def process_status_update(
    client: TelegramClient,
    user_id: int,
    task: StatusUpdateTask,
) -> None:
    """Update the status bubble in place."""
    tkey = thread_key(task.thread_id)
    status_text = format_claude_task_status(task.window_id, task.text)

    if not status_text:
        await clear_status_message(client, user_id, tkey)
        return

    await send_status_text(client, user_id, tkey, task.window_id, status_text)


async def process_status_clear(
    client: TelegramClient,
    user_id: int,
    task: StatusClearTask,
) -> None:
    """Clear the status bubble — re-render with task list or delete."""
    window_id = task.window_id or ""
    tkey = thread_key(task.thread_id)
    status_text = format_claude_task_status(window_id, None)
    if status_text and window_id:
        await send_status_text(client, user_id, tkey, window_id, status_text)
        return
    await clear_status_message(client, user_id, tkey)


# ---------------------------------------------------------------------------
# Cleanup (non-registry — see docstring)
# ---------------------------------------------------------------------------


def clear_status_msg_info(user_id: int, thread_id: int | None = None) -> None:
    """Clear status message tracking for a user (and optionally a specific thread).

    NOT registered with TopicStateRegistry — must only be called explicitly
    from cleanup.py in the ``client is None`` path.  When a client is available,
    ``clear_status_message`` (via the queued ``status_clear`` task) pops
    the entry *and* deletes the Telegram message.  Registering this function
    with the registry would pop the entry before the worker runs, preventing
    the actual Telegram delete.
    """
    skey = (user_id, thread_key(thread_id))
    _status_msg_info.pop(skey, None)
