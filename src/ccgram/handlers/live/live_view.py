"""Live terminal view — auto-refreshing screenshot via editMessageMedia.

Manages active live view sessions: one per topic (user_id, thread_id).
The tick function is called from the polling loop every config.live_view_interval
seconds. Each tick captures the pane, hashes the content, and edits the
Telegram message only when the terminal has changed.

Key functions:
  - start_live_view / stop_live_view / get_live_view: state management
  - tick_live_views: periodic refresh (called from periodic_tasks)
  - build_live_keyboard: inline keyboard shown during live view
  - content_hash: md5 digest for change detection
"""

import contextlib
import hashlib
import io
import time
from dataclasses import dataclass, field
from datetime import timedelta

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.error import RetryAfter, TelegramError

from ...config import config
from ...screenshot import text_to_image
from ...telegram_client import TelegramClient
from ...multiplexer import multiplexer as tmux_manager
from ...topic_state_registry import topic_state
from ..callback_data import CB_KEYS_PREFIX, CB_LIVE_STOP
from ..messaging_pipeline.message_sender import rate_limit_send

logger = structlog.get_logger()


@dataclass
class LiveViewState:
    """State for one active live view session."""

    chat_id: int
    message_id: int
    thread_id: int
    user_id: int
    window_id: str
    pane_id: str | None = None
    last_hash: str = ""
    next_edit_after: float = 0.0
    start_time: float = field(default_factory=time.monotonic)


_active_views: dict[tuple[int, int], LiveViewState] = {}


def start_live_view(state: LiveViewState) -> None:
    """Register a new live view session for a topic."""
    _active_views[(state.user_id, state.thread_id)] = state


def stop_live_view(user_id: int, thread_id: int) -> LiveViewState | None:
    """Stop and return the live view for a topic, or None if not active."""
    return _active_views.pop((user_id, thread_id), None)


def get_live_view(user_id: int, thread_id: int) -> LiveViewState | None:
    """Look up the live view for a topic."""
    return _active_views.get((user_id, thread_id))


def is_live(user_id: int, thread_id: int) -> bool:
    """Check if a topic has an active live view."""
    return (user_id, thread_id) in _active_views


def content_hash(text: str) -> str:
    """Compute a fast content hash for change detection."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


def build_live_keyboard(
    window_id: str, pane_id: str | None = None
) -> InlineKeyboardMarkup:
    """Build inline keyboard for live view: quick keys + stop button."""
    # Lazy: pane delimiter constant
    from ..callback_data import CB_PANE_DELIMITER

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
                    "\u23f9 Stop Live",
                    callback_data=f"{CB_LIVE_STOP}{target}"[:64],
                )
            ],
        ]
    )


@topic_state.register("topic")
def _clear_live_view(user_id: int, thread_id: int) -> None:
    _active_views.pop((user_id, thread_id), None)


async def tick_live_views(client: TelegramClient) -> None:
    """Refresh all active live views. Called from periodic_tasks."""

    effective_timeout = config.live_view_timeout
    interval = config.live_view_interval
    now = time.monotonic()

    for key, view in list(_active_views.items()):
        await _tick_one_view(client, key, view, now, effective_timeout, interval)


async def _tick_one_view(
    client: TelegramClient,
    key: tuple[int, int],
    view: LiveViewState,
    now: float,
    timeout: int,
    interval: int,
) -> None:
    """Refresh a single live view."""
    try:
        if now - view.start_time > timeout:
            _active_views.pop(key, None)
            await _edit_caption(client, view, "Live view ended (timeout)")
            return

        if now < view.next_edit_after:
            return

        window = await tmux_manager.find_window_by_id(view.window_id)
        if window is None:
            _active_views.pop(key, None)
            await _edit_caption(client, view, "Live view ended (window closed)")
            return

        text = await _capture_pane(view, window.window_id)
        if not text:
            return

        h = content_hash(text)
        if h == view.last_hash:
            return

        png_bytes = await text_to_image(text, with_ansi=True, live_mode=True)
        ts = time.strftime("%H:%M:%S")

        await rate_limit_send(view.chat_id)

        keyboard = build_live_keyboard(view.window_id, pane_id=view.pane_id)
        await client.edit_message_media(
            chat_id=view.chat_id,
            message_id=view.message_id,
            media=InputMediaPhoto(
                media=io.BytesIO(png_bytes),
                caption=f"Live \u00b7 {ts}",
            ),
            reply_markup=keyboard,
        )
        view.last_hash = h
        view.next_edit_after = time.monotonic() + interval

    except RetryAfter as exc:
        ra = exc.retry_after
        wait = ra.total_seconds() if isinstance(ra, timedelta) else float(ra)
        view.next_edit_after = time.monotonic() + wait
        logger.warning("live_view_retry_after", key=key, wait=wait)
    except TelegramError as exc:
        logger.warning("live_view_tick_error", key=key, error=str(exc))
        _active_views.pop(key, None)


async def _capture_pane(view: LiveViewState, window_id: str) -> str | None:
    """Capture pane text for a live view."""
    if view.pane_id:
        return await tmux_manager.capture_pane_by_id(
            view.pane_id, with_ansi=True, window_id=view.window_id
        )
    return await tmux_manager.capture_pane(window_id, with_ansi=True)


async def _edit_caption(client: TelegramClient, view: LiveViewState, text: str) -> None:
    """Best-effort edit of the live view message caption on stop."""
    # Lazy: live_view ↔ screenshot_callbacks bidirectional cycle.
    from .screenshot_callbacks import build_screenshot_keyboard

    keyboard = build_screenshot_keyboard(view.window_id, pane_id=view.pane_id)
    with contextlib.suppress(TelegramError):
        await client.edit_message_caption(
            chat_id=view.chat_id,
            message_id=view.message_id,
            caption=text,
            reply_markup=keyboard,
        )
