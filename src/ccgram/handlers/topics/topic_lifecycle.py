"""Topic lifecycle management — autoclose timers, unbound window TTL, probing.

Periodic tasks that manage topic and window lifecycle:
  - Autoclose: expire done/dead topics after configurable timeout
  - Unbound window TTL: kill orphaned tmux windows without topic bindings
  - Topic existence probing: detect deleted Telegram topics via API
  - State pruning: sync display names and remove stale entries
"""

from __future__ import annotations
import time
from typing import TYPE_CHECKING

import structlog
from telegram import Update
from telegram.error import BadRequest, TelegramError
from ... import window_query
from ...config import config
from ...session import session_manager
from ...session_map import session_map_prefix
from ...telegram_client import PTBTelegramClient, TelegramClient
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...utils import log_throttled
from ...window_state_store import CCGRAM_CREATED_WINDOW_ORIGIN
from ..cleanup import clear_topic_state
from ..messaging_pipeline.message_sender import is_thread_gone
from ..polling.polling_state import (
    lifecycle_strategy,
    terminal_poll_state,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes
    from ...multiplexer.base import WindowRef as TmuxWindow

logger = structlog.get_logger()


# ── Autoclose timer management ────────────────────────────────────────────


async def check_autoclose_timers(client: TelegramClient) -> None:
    """Close topics whose done/dead timers have expired."""
    all_topics = lifecycle_strategy.iter_topic_states()
    if not all_topics:
        return

    now = time.monotonic()
    expired: list[tuple[int, int, str]] = []
    for user_id, thread_id, ts in all_topics:
        if ts.autoclose is None:
            continue
        state, entered_at = ts.autoclose
        if state == "done":
            timeout = config.autoclose_done_minutes * 60
        elif state == "dead":
            timeout = config.autoclose_dead_minutes * 60
        else:
            continue
        if timeout > 0 and now - entered_at >= timeout:
            expired.append((user_id, thread_id, state))

    for user_id, thread_id, state in expired:
        await _close_expired_topic(client, user_id, thread_id, state)


async def _close_expired_topic(
    client: TelegramClient, user_id: int, thread_id: int, state: str
) -> None:
    """Attempt to close/delete an expired topic and clean up state."""
    window_id = thread_router.get_window_for_thread(user_id, thread_id)
    if state == "dead" and window_id is not None:
        live_window = await tmux_manager.find_window_by_id(window_id)
        if live_window is not None:
            lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
            logger.info(
                "stale_dead_autoclose_cleared",
                thread_id=thread_id,
                user_id=user_id,
                window_id=window_id,
            )
            return

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    removed = False
    try:
        await client.delete_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
        removed = True
    except TelegramError as e:
        if is_thread_gone(e):
            removed = True
        else:
            try:
                await client.close_forum_topic(
                    chat_id=chat_id, message_thread_id=thread_id
                )
                removed = True
            except TelegramError as close_err:
                if is_thread_gone(close_err):
                    removed = True
                else:
                    logger.debug(
                        "autoclose_failed", thread_id=thread_id, error=str(close_err)
                    )
    if removed:
        lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
        logger.info(
            "auto_removed_topic", chat_id=chat_id, thread_id=thread_id, user_id=user_id
        )
        await clear_topic_state(
            user_id,
            thread_id,
            client=client,
            window_id=window_id,
            window_dead=True,
        )
        thread_router.unbind_thread(user_id, thread_id)


# ── Unbound window TTL ────────────────────────────────────────────────────


async def check_unbound_window_ttl(
    live_windows: "list[TmuxWindow] | None" = None,
) -> None:
    """Kill unbound tmux windows whose TTL has expired."""
    timeout = config.autoclose_done_minutes * 60
    if timeout <= 0:
        return

    bound_ids: set[str] = set()
    for _, _, wid in thread_router.iter_thread_bindings():
        bound_ids.add(wid)

    if live_windows is None:
        live_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in live_windows}

    terminal_poll_state.clear_unbound_timers(bound_ids, live_ids)

    now = time.monotonic()
    for w in live_windows:
        if w.window_id in bound_ids:
            continue
        view = window_query.view_window(w.window_id)
        if view is None or view.origin != CCGRAM_CREATED_WINDOW_ORIGIN:
            terminal_poll_state.clear_unbound_timer(w.window_id)
            continue
        ws = terminal_poll_state.get_state(w.window_id)
        if ws.unbound_timer is None:
            terminal_poll_state.set_unbound_timer(w.window_id, now)

    await _kill_expired_unbound(now, timeout)
    _prune_orphaned_poll_state(live_ids, bound_ids)


async def _kill_expired_unbound(now: float, timeout: float) -> None:
    """Find and kill unbound windows past their TTL."""
    expired = terminal_poll_state.get_expired_unbound(now, timeout)
    for wid in expired:
        await tmux_manager.kill_window(wid)

        # Lazy: topic_state_registry is wired during bootstrap; importing
        # at top dragged registration side effects into the polling
        # subpackage's import path.
        from ...topic_state_registry import topic_state

        topic_state.clear_window(wid)
        qualified_id = f"{session_map_prefix()}{wid}"
        topic_state.clear_qualified(qualified_id)
        logger.info("auto_killed_unbound_window", window_id=wid)


def _prune_orphaned_poll_state(live_ids: set[str], bound_ids: set[str]) -> None:
    """Remove poll state for windows that are neither live nor bound."""
    for wid in terminal_poll_state.get_orphaned_window_ids(live_ids, bound_ids):
        terminal_poll_state.clear_state(wid)


# ── Display name sync / state pruning ─────────────────────────────────────


async def prune_stale_state(live_windows: "list[TmuxWindow]") -> None:
    """Sync display names and prune orphaned state entries."""
    live_ids = {w.window_id for w in live_windows}
    live_pairs = [(w.window_id, w.window_name) for w in live_windows]
    session_manager.sync_display_names(live_pairs)
    session_manager.prune_stale_state(live_ids)


# ── Topic existence probing ───────────────────────────────────────────────


# Windows whose chat lacks can_pin_messages: the unpin-based probe can never
# succeed there, so disable it permanently (per process) instead of counting it
# as a probe failure (which would suspend deleted-topic detection and re-arm on
# every inbound message). Reset on restart; mirrors _disabled_chats in
# handlers/status/topic_emoji.py.
_probe_pin_disabled: set[str] = set()


async def probe_topic_existence(client: TelegramClient) -> None:
    """Probe all bound topics via Telegram API; detect deleted topics."""
    for user_id, thread_id, wid in list(thread_router.iter_thread_bindings()):
        if wid in _probe_pin_disabled or lifecycle_strategy.should_skip_probe(wid):
            continue
        try:
            await client.unpin_all_forum_topic_messages(
                chat_id=thread_router.resolve_chat_id(user_id, thread_id),
                message_thread_id=thread_id,
            )
            terminal_poll_state.reset_probe_failures(wid)
        except TelegramError as e:
            if isinstance(e, BadRequest) and (
                "Topic_id_invalid" in e.message
                or "thread not found" in e.message.lower()
            ):
                w = await tmux_manager.find_window_by_id(wid)
                view = window_query.view_window(wid)
                killed = False
                if w and view and view.origin == CCGRAM_CREATED_WINDOW_ORIGIN:
                    await tmux_manager.kill_window(w.window_id)
                    killed = True
                terminal_poll_state.reset_probe_failures(wid)
                await clear_topic_state(user_id, thread_id, client, window_id=wid)
                thread_router.unbind_thread(user_id, thread_id)
                action = "killed" if killed else "unbound"
                logger.info(
                    "Topic deleted: %s window_id '%s' and unbound thread %d for user %d",
                    action,
                    wid,
                    thread_id,
                    user_id,
                )
            elif isinstance(e, BadRequest) and "not enough rights" in e.message.lower():
                _probe_pin_disabled.add(wid)
                logger.info(
                    "Topic probe disabled for window_id '%s': bot lacks pin rights",
                    wid,
                )
            else:
                lifecycle_strategy.record_probe_failure(wid)
                if not lifecycle_strategy.should_skip_probe(wid):
                    log_throttled(
                        logger,
                        f"topic-probe:{wid}",
                        "Topic probe error for %s: %s",
                        wid,
                        e,
                    )


# ------------------------------------------------------------------
# Telegram topic event handlers (moved from bot.py)
# ------------------------------------------------------------------


async def topic_closed_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle topic closure — unbind thread but keep the tmux window alive.

    The window becomes "unbound" and is available for rebinding via the window
    picker when a new topic is created. Unbound windows are auto-killed after
    the configured TTL (autoclose_done_minutes) by the status polling loop.
    """
    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return

    # Lazy: callback_helpers ↔ topic_lifecycle through bootstrap wiring.
    from ..callback_helpers import get_thread_id

    thread_id = get_thread_id(update)
    if thread_id is None:
        return

    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    if window_id:
        display = thread_router.get_display_name(window_id)
        await clear_topic_state(
            user.id,
            thread_id,
            PTBTelegramClient(context.bot),
            context.user_data,
            window_id=window_id,
            window_dead=False,
        )
        thread_router.unbind_thread(user.id, thread_id)
        logger.info(
            "Topic closed: window %s unbound (kept alive for rebinding, user=%d, thread=%d)",
            display,
            user.id,
            thread_id,
        )
    else:
        logger.debug(
            "Topic closed: no binding (user=%d, thread=%d)", user.id, thread_id
        )


async def topic_edited_handler(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle topic rename — sync new name to tmux window and emoji cache.

    Ignores icon-only edits (name is None) and emoji-only changes from the bot
    itself (clean name unchanged after stripping prefixes).
    """
    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message or not update.message.forum_topic_edited:
        return

    new_name = update.message.forum_topic_edited.name
    if not new_name:
        return

    # Lazy: same callback_helpers cycle plus status.topic_emoji ↔ topics
    # cycle through emoji refresh callbacks.
    # Lazy: handlers.callback_helpers / handlers.status cycle
    from ..callback_helpers import get_thread_id

    # Lazy: handlers.callback_helpers / handlers.status cycle
    from ..status.topic_emoji import strip_emoji_prefix, update_stored_topic_name

    thread_id = get_thread_id(update)
    if thread_id is None:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    window_id = thread_router.get_window_for_chat_thread(chat_id, thread_id)
    if not window_id:
        logger.debug("Topic edited: no binding (thread=%d)", thread_id)
        return

    clean_name = strip_emoji_prefix(new_name)

    current_display = thread_router.get_display_name(window_id)
    if current_display and strip_emoji_prefix(current_display) == clean_name:
        logger.debug(
            "Topic edited: name unchanged after strip, skipping (thread=%d)", thread_id
        )
        return

    renamed = await tmux_manager.rename_window(window_id, clean_name)
    if renamed:
        session_manager.set_display_name(window_id, clean_name)
        update_stored_topic_name(chat_id, thread_id, clean_name)
        logger.info(
            "Topic renamed: window %s → %r (thread=%d)",
            window_id,
            clean_name,
            thread_id,
        )
