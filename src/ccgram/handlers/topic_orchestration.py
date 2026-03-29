"""Auto-create Telegram forum topics for newly detected tmux windows.

Handles topic creation with flood-control backoff, provider auto-detection,
and post-restart adoption of unbound windows.

Core responsibilities:
  - handle_new_window(): create a topic when a new tmux window appears
  - adopt_unbound_windows(): post-restart recovery of orphaned windows
  - Rate-limited topic creation with per-chat exponential backoff
"""

from __future__ import annotations

import time
from pathlib import Path

import structlog
from telegram import Bot
from telegram.error import RetryAfter, TelegramError

from ..config import config
from ..providers import (
    detect_provider_from_pane,
    detect_provider_from_runtime,
    should_probe_pane_title_for_provider_detection,
)
from ..session import session_manager
from ..session_monitor import NewWindowEvent
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager

logger = structlog.get_logger()

# Per-chat backoff for auto topic creation after Telegram flood control.
# chat_id -> monotonic timestamp when next attempt is allowed.
_topic_create_retry_until: dict[int, float] = {}
_TOPIC_CREATE_RETRY_BUFFER_SECONDS = 1


def _is_window_already_bound(window_id: str) -> bool:
    """Check if a window is already bound to any topic."""
    for _, _, bound_wid in thread_router.iter_thread_bindings():
        if bound_wid == window_id:
            return True
    return False


async def _auto_detect_provider(window_id: str) -> None:
    """Auto-detect provider from the running process if not already set.

    detect_provider_from_command returns "" for unrecognized commands (shells),
    so we only persist when a known CLI is confidently identified.
    """
    existing_provider = session_manager.get_window_state(window_id).provider_name
    if existing_provider:
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w or not w.pane_current_command:
        return

    detected = await detect_provider_from_pane(
        w.pane_current_command,
        pane_tty=w.pane_tty,
        window_id=window_id,
    )
    if not detected and should_probe_pane_title_for_provider_detection(
        w.pane_current_command
    ):
        pane_title = await tmux_manager.get_pane_title(window_id)
        detected = detect_provider_from_runtime(
            w.pane_current_command,
            pane_title=pane_title,
        )
    if detected:
        session_manager.set_window_provider(window_id, detected)
        logger.info(
            "Auto-detected provider %r for window %s (command=%s)",
            detected,
            window_id,
            w.pane_current_command,
        )


def _collect_target_chats(window_id: str) -> set[int]:
    """Collect unique group chat IDs for topic creation."""
    seen_chats: set[int] = set()
    for user_id, thread_id, _ in thread_router.iter_thread_bindings():
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        if chat_id != user_id:
            seen_chats.add(chat_id)

    if not seen_chats:
        seen_chats.update(
            cid for cid in thread_router.group_chat_ids.values() if cid < 0
        )

    if not seen_chats:
        if config.group_id:
            seen_chats.add(config.group_id)
            logger.info(
                "Cold-start: using CCBOT_GROUP_ID=%d for auto-topic (window %s)",
                config.group_id,
                window_id,
            )
        else:
            logger.debug(
                "No group chats found for auto-topic creation (window %s)",
                window_id,
            )

    return seen_chats


def _bind_topic_to_user(
    thread_id: int, window_id: str, chat_id: int, topic_name: str
) -> None:
    """Bind a newly created topic to a user in the given chat."""
    for user_id, tid, _ in thread_router.iter_thread_bindings():
        if thread_router.resolve_chat_id(user_id, tid) == chat_id:
            thread_router.bind_thread(
                user_id, thread_id, window_id, window_name=topic_name
            )
            thread_router.set_group_chat_id(user_id, thread_id, chat_id)
            return

    if config.allowed_users:
        first_user_id = next(iter(config.allowed_users))
        thread_router.bind_thread(
            first_user_id, thread_id, window_id, window_name=topic_name
        )
        thread_router.set_group_chat_id(first_user_id, thread_id, chat_id)


async def _create_topic_in_chat(
    bot: Bot, chat_id: int, window_id: str, topic_name: str
) -> None:
    """Create a forum topic in one chat with backoff handling."""
    retry_until = _topic_create_retry_until.get(chat_id, 0.0)
    now = time.monotonic()
    if now < retry_until:
        wait_seconds = max(1, int(retry_until - now))
        logger.debug(
            "Skipping auto-topic creation for chat %d (window %s), "
            "backoff active for %ss",
            chat_id,
            window_id,
            wait_seconds,
        )
        return

    try:
        topic = await bot.create_forum_topic(chat_id=chat_id, name=topic_name)
        _topic_create_retry_until.pop(chat_id, None)
        logger.info(
            "Auto-created topic '%s' (thread=%d) in chat %d for window %s",
            topic_name,
            topic.message_thread_id,
            chat_id,
            window_id,
        )
        _bind_topic_to_user(topic.message_thread_id, window_id, chat_id, topic_name)
    except RetryAfter as e:
        retry_after_seconds = (
            e.retry_after
            if isinstance(e.retry_after, int)
            else int(e.retry_after.total_seconds())
        )
        retry_after_seconds = max(1, retry_after_seconds)
        _topic_create_retry_until[chat_id] = (
            time.monotonic() + retry_after_seconds + _TOPIC_CREATE_RETRY_BUFFER_SECONDS
        )
        logger.warning(
            "Flood control creating topic for window %s in chat %d, backing off %ss",
            window_id,
            chat_id,
            retry_after_seconds,
        )
    except TelegramError:
        logger.exception(
            "Failed to create topic for window %s in chat %d",
            window_id,
            chat_id,
        )


async def handle_new_window(event: NewWindowEvent, bot: Bot) -> None:
    """Create a Telegram forum topic for a newly detected tmux window.

    Skips if the window is already bound to a topic. Creates one topic per
    unique group chat, binds all users in that chat.
    """
    if _is_window_already_bound(event.window_id):
        logger.debug(
            "New window %s already bound, skipping topic creation", event.window_id
        )
        return

    await _auto_detect_provider(event.window_id)

    topic_name = event.window_name or Path(event.cwd).name or event.window_id
    seen_chats = _collect_target_chats(event.window_id)
    if not seen_chats:
        return

    for chat_id in seen_chats:
        await _create_topic_in_chat(bot, chat_id, event.window_id, topic_name)


async def adopt_unbound_windows(bot: Bot) -> None:
    """Auto-adopt known-but-unbound windows (post-restart recovery)."""
    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]
    audit = session_manager.audit_state(live_ids, live_pairs)
    orphaned = [i for i in audit.issues if i.category == "orphaned_window"]
    if orphaned:
        from .sync_command import _adopt_orphaned_windows

        await _adopt_orphaned_windows(bot, orphaned)
        logger.info("Startup: adopted %d unbound window(s)", len(orphaned))
