"""Auto-create Telegram forum topics for newly detected tmux windows.

Handles topic creation with flood-control backoff, provider auto-detection,
and post-restart adoption of unbound windows.

Core responsibilities:
  - handle_new_window(): create a topic when a new tmux window appears
  - adopt_unbound_windows(): post-restart recovery of orphaned windows
  - Rate-limited topic creation with per-chat exponential backoff
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import structlog
from telegram.error import NetworkError, RetryAfter, TelegramError, TimedOut

from ... import window_query
from ...config import config
from ...providers import (
    detect_provider_from_pane,
    detect_provider_from_runtime,
    should_probe_pane_title_for_provider_detection,
)
from ...session import session_manager
from ...session_monitor import NewWindowEvent
from ...telegram_client import TelegramClient
from ...thread_router import thread_router
from ...tmux_manager import tmux_manager
from ..messaging_pipeline.message_sender import is_thread_gone
from ..status.topic_emoji import strip_emoji_prefix

logger = structlog.get_logger()

# Per-chat backoff for auto topic creation after Telegram flood control.
# chat_id -> monotonic timestamp when next attempt is allowed.
_topic_create_retry_until: dict[int, float] = {}
_TOPIC_CREATE_RETRY_BUFFER_SECONDS = 1

# Transient retry on transport timeouts / network errors when creating a topic.
# Two attempts (one retry) with a short backoff is enough for brief blips —
# longer outages are caught by the existing flood-control backoff.
_TOPIC_CREATE_TRANSIENT_RETRIES = 1
_TOPIC_CREATE_TRANSIENT_BACKOFF_S = 1.0


async def _create_forum_topic_with_retry(
    client: TelegramClient, chat_id: int, topic_name: str
):
    """Call ``client.create_forum_topic`` with one retry on TimedOut/NetworkError."""
    last_exc: TelegramError | None = None
    for attempt in range(_TOPIC_CREATE_TRANSIENT_RETRIES + 1):
        try:
            return await client.create_forum_topic(chat_id=chat_id, name=topic_name)
        except (TimedOut, NetworkError) as exc:
            last_exc = exc
            if attempt < _TOPIC_CREATE_TRANSIENT_RETRIES:
                logger.debug("create_forum_topic transient error, retrying: %s", exc)
                await asyncio.sleep(_TOPIC_CREATE_TRANSIENT_BACKOFF_S)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def clear_topic_create_retry(chat_id: int, _thread_id: int = 0) -> None:
    """Clear topic creation retry backoff for this chat.

    NOT registered in TopicStateRegistry — the backoff is self-managing:
    entries expire via the time check in create_topic_in_chat() and are
    cleared on successful topic creation.  Clearing on every topic teardown
    would bypass Telegram's flood-control gate prematurely.
    """
    _topic_create_retry_until.pop(chat_id, None)


def _is_window_already_bound(window_id: str) -> bool:
    """Check if a window is already bound to any topic."""
    return thread_router.has_window(window_id)


# In-flight directory-flow window creations. Keyed by tmux window_id, value is
# the monotonic expiry timestamp. Set by directory_callbacks._create_window_and_bind
# right after `tmux_manager.create_window` returns and BEFORE any subsequent
# `await` (so the SessionMonitor poll cycle can't slip in and fire
# handle_new_window before the bind completes).
#
# Without this, the bug at MC-2967 / MC-3015 / MC-3030 fires:
#   1. directory flow calls tmux_manager.create_window → tmux pane spawns
#   2. claude-code (or other provider) boots inside, fires SessionStart hook
#   3. SessionMonitor reads events.jsonl on its 1s poll, builds NewWindowEvent
#   4. handle_new_window checks _is_window_already_bound — still False, because
#      directory_callbacks hasn't reached `thread_router.bind_thread()` yet
#   5. handle_new_window auto-creates a *new* Telegram topic and binds the
#      window to it
#   6. directory_callbacks then runs `bind_thread` for the original topic, but
#      thread_router only stores one binding per window — original topic stays
#      unbound forever ("Will deliver once the agent starts")
#
# The pending set lets handle_new_window detect "this window is owned by a
# directory flow that's about to bind, do not create a duplicate topic." A
# 30s TTL is the safety net in case directory_callbacks crashes before
# clearing the entry — handle_new_window will eventually reclaim the window.
_pending_user_creations: dict[str, float] = {}
_PENDING_CREATION_TTL_S = 30.0


def register_pending_creation(window_id: str) -> None:
    """Mark a tmux window as owned by an in-flight directory-flow bind.

    Call BEFORE any `await` between tmux window creation and
    `thread_router.bind_thread`. Pair with `clear_pending_creation` (or rely on
    the TTL) once the bind is complete.
    """
    if not window_id:
        return
    _pending_user_creations[window_id] = time.monotonic() + _PENDING_CREATION_TTL_S


def clear_pending_creation(window_id: str) -> None:
    """Remove a window's pending-creation marker (idempotent)."""
    _pending_user_creations.pop(window_id, None)


def _is_pending_user_creation(window_id: str) -> bool:
    """Return True iff a directory flow is mid-creation for this window.

    Expired entries are evicted lazily on read so a crashed directory flow
    can't permanently shadow a window from auto-topic-creation.
    """
    expires_at = _pending_user_creations.get(window_id)
    if expires_at is None:
        return False
    if time.monotonic() >= expires_at:
        _pending_user_creations.pop(window_id, None)
        return False
    return True


async def _auto_detect_provider(window_id: str) -> None:
    """Auto-detect provider from the running process if not already set.

    detect_provider_from_command returns "" for unrecognized commands (shells),
    so we only persist when a known CLI is confidently identified.
    """
    view = window_query.view_window(window_id)
    if view and view.provider_name:
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


def collect_target_chats(window_id: str) -> set[int]:
    """Collect unique group chat IDs for topic creation."""
    seen_chats: set[int] = set()
    for user_id, thread_id, _ in thread_router.iter_thread_bindings():
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        if isinstance(chat_id, int) and chat_id < 0:
            seen_chats.add(chat_id)

    if not seen_chats:
        seen_chats.update(
            cid for cid in thread_router.group_chat_ids.values() if cid < 0
        )

    if not seen_chats:
        if config.group_id:
            seen_chats.add(config.group_id)
            logger.info(
                "Cold-start: using CCGRAM_GROUP_ID=%d for auto-topic (window %s)",
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


async def create_topic_in_chat(
    client: TelegramClient, chat_id: int, window_id: str, topic_name: str
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
        topic = await _create_forum_topic_with_retry(client, chat_id, topic_name)
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


async def _topic_exists(
    client: TelegramClient, chat_id: int, thread_id: int
) -> bool | None:
    """Probe a Telegram topic. True=exists, False=gone, None=unknown."""
    try:
        msg = await client.send_message(
            chat_id,
            "​",
            message_thread_id=thread_id,
            disable_notification=True,
        )
    except TelegramError as exc:
        if is_thread_gone(exc):
            return False
        return None
    with contextlib.suppress(TelegramError):
        await client.delete_message(chat_id, msg.message_id)
    return True


async def _rebind_existing_topic_by_name(
    event: NewWindowEvent, client: TelegramClient, topic_name: str
) -> bool:
    """Bind a stale same-name topic to a newly discovered manual window."""
    clean_topic_name = strip_emoji_prefix(topic_name)
    matches: list[tuple[int, int, str, int]] = []
    bindings = list(thread_router.iter_thread_bindings())
    for user_id, thread_id, old_window_id in bindings:
        if old_window_id == event.window_id:
            continue
        display_name = strip_emoji_prefix(thread_router.get_display_name(old_window_id))
        if display_name != clean_topic_name:
            continue
        if await tmux_manager.find_window_by_id(old_window_id):
            continue
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        if chat_id == user_id:
            continue
        matches.append((user_id, thread_id, old_window_id, chat_id))

    if len(matches) != 1:
        if len(matches) > 1:
            logger.warning(
                "Multiple stale same-name topics for window %s (%s); not rebinding",
                event.window_id,
                clean_topic_name,
            )
        return False

    user_id, thread_id, old_window_id, chat_id = matches[0]
    exists = await _topic_exists(client, chat_id, thread_id)
    if exists is False:
        thread_router.unbind_thread(user_id, thread_id)
        logger.info(
            "Dropped dead same-name topic thread %d for stale window %s",
            thread_id,
            old_window_id,
        )
        return False
    if exists is None:
        logger.info(
            "Could not probe same-name topic thread %d for stale window %s; not rebinding",
            thread_id,
            old_window_id,
        )
        return False

    thread_router.bind_thread(
        user_id, thread_id, event.window_id, window_name=topic_name
    )
    thread_router.set_group_chat_id(user_id, thread_id, chat_id)
    logger.info(
        "Rebound existing topic thread %d from stale window %s to new window %s (%s)",
        thread_id,
        old_window_id,
        event.window_id,
        clean_topic_name,
    )
    return True


async def handle_new_window(event: NewWindowEvent, client: TelegramClient) -> None:
    """Create or bind a Telegram forum topic for a newly detected tmux window.

    Skips if the window is already bound. Reuses one stale same-name topic when
    it still exists; otherwise creates one topic per unique group chat.
    """
    if _is_window_already_bound(event.window_id):
        logger.debug(
            "New window %s already bound, skipping topic creation", event.window_id
        )
        return

    if _is_pending_user_creation(event.window_id):
        logger.debug(
            "New window %s creation pending in directory flow — "
            "skipping auto topic creation",
            event.window_id,
        )
        return

    await _auto_detect_provider(event.window_id)

    topic_name = event.window_name or Path(event.cwd).name or event.window_id
    if await _rebind_existing_topic_by_name(event, client, topic_name):
        return

    seen_chats = collect_target_chats(event.window_id)
    if not seen_chats:
        return

    for chat_id in seen_chats:
        await create_topic_in_chat(client, chat_id, event.window_id, topic_name)


async def adopt_unbound_windows(client: TelegramClient) -> None:
    """Auto-adopt known-but-unbound windows (post-restart recovery)."""
    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]
    audit = session_manager.audit_state(live_ids, live_pairs)
    orphaned = [i for i in audit.issues if i.category == "orphaned_window"]
    if orphaned:
        # Lazy: bidirectional cycle with sync_command (see
        # sync_command._adopt_orphaned_windows for details).
        from ..sync_command import _adopt_orphaned_windows

        await _adopt_orphaned_windows(client, orphaned)
        logger.info("Startup: adopted %d unbound window(s)", len(orphaned))
