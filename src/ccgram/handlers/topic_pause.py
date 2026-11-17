"""Flag-driven Telegram topic lifecycle, independent of native marker writes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4
from weakref import WeakValueDictionary

import structlog
from telegram.error import BadRequest, RetryAfter, TelegramError

from ..config import config
from ..multiplexer import multiplexer
from ..multiplexer.base import Multiplexer, WindowRef, canonical_window_id
from ..session_query import resolve_session_for_window
from ..session_monitor import capture_transcript_generation
from ..telegram_client import TelegramClient, delivery_guard
from ..telegram_rate_limiter import NO_RETRY_RATE_LIMIT_ARGS, retry_after_seconds
from ..thread_router import ThreadRouter, TopicPause, thread_router
from ..window_state_ports import lifecycle_state

logger = structlog.get_logger()
_locks: WeakValueDictionary[tuple[int, int, int], asyncio.Lock] = WeakValueDictionary()


def enabled() -> bool:
    return config.topic_discovery_mode == "marked"


def _flush() -> None:
    # Lazy: SessionManager constructs the router before lifecycle reconciliation.
    from ..session import session_manager

    session_manager.flush_state()


def _lock(key: tuple[int, int, int]) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock


def is_missing(exc: TelegramError) -> bool:
    return isinstance(exc, BadRequest) and (
        "thread not found" in exc.message.lower()
        or "topic_id_invalid" in exc.message.lower()
    )


def paused(user_id: int, thread_id: int | None, chat_id: int | None = None) -> bool:
    state = thread_router.get_topic_pause(user_id, thread_id, chat_id)
    return state is not None and state.paused


def revision(
    user_id: int, thread_id: int | None, chat_id: int | None = None
) -> str | None:
    state = thread_router.get_topic_pause(user_id, thread_id, chat_id)
    return (state.revision or None) if state else None


def guard(user_id: int, thread_id: int | None, chat_id: int | None):
    expected = revision(user_id, thread_id, chat_id)
    return delivery_guard(
        lambda: (
            not paused(user_id, thread_id, chat_id)
            and revision(user_id, thread_id, chat_id) == expected
        )
    )


def has_paused_target(window_id: str) -> bool:
    return any(
        state.paused
        and canonical_window_id(state.window_id) == canonical_window_id(window_id)
        for state in thread_router.topic_pauses.values()
    )


def accepts_message(
    user_id: int,
    thread_id: int | None,
    chat_id: int | None,
    session_id: str,
    source_offset: int | None,
    source_generation: int | None = None,
    source_identity: tuple[int, int] | None = None,
) -> bool:
    state = thread_router.get_topic_pause(user_id, thread_id, chat_id)
    if state is None:
        return True
    if state.paused:
        return False
    if state.session_id != session_id:
        return True
    if source_generation is not None:
        if source_generation != state.source_generation:
            return source_generation > state.source_generation
        if (
            state.source_identity is not None
            and source_identity != state.source_identity
        ):
            return False
    return source_offset is None or source_offset >= state.offset


def _current(router: ThreadRouter, state: TopicPause) -> bool:
    return (
        router.topic_pauses.get(state.key) == state
        and router.get_window_for_thread(state.user_id, state.thread_id, state.chat_id)
        == state.window_id
    )


def _verified(
    window: WindowRef | None, owner: str | None = None, window_id: str | None = None
) -> bool:
    return (
        window is not None
        and (
            window_id is None
            or canonical_window_id(window.window_id) == canonical_window_id(window_id)
        )
        and window.topic_markable
        and bool(window.topic_mark_owner)
        and window.topic_marked is not None
        and (owner is None or window.topic_mark_owner == owner)
        and not lifecycle_state.is_topic_mark_quarantined(
            window.window_id, owner=window.topic_mark_owner
        )
        and not lifecycle_state.is_topic_detach_pending(
            window.window_id, owner=window.topic_mark_owner
        )
    )


async def _frontier(state: TopicPause) -> TopicPause:
    session = await resolve_session_for_window(state.window_id)
    if session is None or not session.file_path:
        return replace(state, session_id=None, offset=0)
    stat = Path(session.file_path).stat()
    return replace(
        state,
        session_id=session.session_id,
        offset=stat.st_size,
        source_generation=capture_transcript_generation(session.session_id),
        source_identity=(stat.st_dev, stat.st_ino),
    )


def note_missing(user_id: int, chat_id: int | None, thread_id: int) -> bool:
    state = thread_router.get_topic_pause(user_id, thread_id, chat_id)
    if not enabled() or state is None:
        return False
    thread_router.set_topic_pause(
        replace(state, paused=True, missing=True, closed=True, revision=str(uuid4()))
    )
    _flush()
    return True


def note_closed(user_id: int, chat_id: int | None, thread_id: int) -> bool:
    """Native flags win over delayed own-close and human-close events alike."""
    state = thread_router.get_topic_pause(user_id, thread_id, chat_id)
    if not enabled() or state is None:
        return False
    thread_router.set_topic_pause(replace(state, paused=True, closed=True))
    _flush()
    return True


def _chat_cooling(router: ThreadRouter, chat_id: int) -> bool:
    now = time.time()
    return any(
        state.chat_id == chat_id and state.retry_at > now
        for state in router.topic_pauses.values()
    )


def _defer_flood(
    router: ThreadRouter,
    state: TopicPause,
    exc: TelegramError | OSError,
    flush: Callable[[], None],
) -> None:
    if not isinstance(exc, RetryAfter):
        return
    current = router.topic_pauses.get(state.key)
    if current is not None and _current(router, current):
        router.set_topic_pause(
            replace(current, retry_at=time.time() + retry_after_seconds(exc))
        )
        flush()


async def _close(
    client: TelegramClient,
    state: TopicPause,
    router: ThreadRouter,
    flush: Callable[[], None],
) -> None:
    if state.closed or state.missing or _chat_cooling(router, state.chat_id):
        return
    original = state
    try:
        result = await client.close_forum_topic(
            state.chat_id, state.thread_id, rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS
        )
        if result is False:
            return
    except TelegramError as exc:
        if is_missing(exc):
            state = replace(state, missing=True)
        elif not (
            isinstance(exc, BadRequest)
            and exc.message.lower().replace(" ", "_")
            in {"topic_not_modified", "topic_closed"}
        ):
            raise
    if _current(router, original):
        router.set_topic_pause(replace(state, closed=True))
        flush()


def _prepare_replacement(
    state: TopicPause, router: ThreadRouter, flush: Callable[[], None]
) -> None:
    if router.has_target_provisioning(state.window_id):
        return
    claim = router.begin_topic_provisioning(
        state.user_id,
        state.chat_id,
        thread_id=state.thread_id,
        target_id=state.window_id,
        kind="topic_for_target",
        topic_mark_owner=state.owner,
    )
    router.prepare_topic_recreation(claim.claim_id)
    # Carry the resume fence across an absent binding and uncertain creation.
    router.topic_pauses[state.key] = state
    router.defer_topic_recreation(claim.claim_id, retry_at=0.0)
    flush()


def _can_resume(
    router: ThreadRouter, state: TopicPause, window: WindowRef | None
) -> bool:
    return (
        _current(router, state)
        and _verified(window, state.owner, state.window_id)
        and (not enabled() or (window is not None and window.topic_marked is True))
    )


async def _resume(
    client: TelegramClient,
    state: TopicPause,
    router: ThreadRouter,
    backend: Multiplexer,
    flush: Callable[[], None],
) -> None:
    if _chat_cooling(router, state.chat_id):
        return
    frontier = await _frontier(state)
    window = await backend.find_window_by_id(state.window_id)
    if not _can_resume(router, state, window):
        return
    if state.missing:
        _prepare_replacement(frontier, router, flush)
        return
    # An in-flight reopen may succeed even if its acknowledgement is lost.
    # Fence forwarding, but do not retain a claim that the topic is closed.
    frontier = replace(frontier, closed=False)
    router.set_topic_pause(frontier)
    flush()
    state = frontier
    try:
        reopened = await client.reopen_forum_topic(
            state.chat_id, state.thread_id, rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS
        )
        if reopened is False:
            return
    except TelegramError as exc:
        if is_missing(exc):
            missing = replace(frontier, missing=True, closed=True)
            router.set_topic_pause(missing)
            flush()
            _prepare_replacement(missing, router, flush)
            return
        if not (
            isinstance(exc, BadRequest)
            and exc.message.lower().replace(" ", "_") == "topic_not_modified"
        ):
            raise
    window = await backend.find_window_by_id(state.window_id)
    if not _can_resume(router, state, window):
        return
    router.set_topic_pause(
        replace(
            frontier, paused=False, closed=False, revision=str(uuid4()), retry_at=0.0
        )
    )
    flush()


async def reconcile_topic_pauses(
    client: TelegramClient,
    windows: Iterable[WindowRef],
    *,
    router: ThreadRouter = thread_router,
    backend: Multiplexer = multiplexer,
    flush: Callable[[], None] = _flush,
) -> None:
    lookup = {canonical_window_id(w.window_id): w for w in windows}
    for uid, chat, tid, wid in list(router.iter_thread_bindings_with_chat()):
        if chat is None or tid == 1:
            continue
        window = lookup.get(canonical_window_id(wid))
        if not _verified(window):
            continue
        assert window is not None and window.topic_mark_owner is not None
        async with _lock((uid, chat, tid)):
            await _reconcile_one(
                client, uid, chat, tid, wid, window, router, backend, flush
            )


async def _reconcile_one(
    client: TelegramClient,
    uid: int,
    chat: int,
    tid: int,
    wid: str,
    window: WindowRef,
    router: ThreadRouter,
    backend: Multiplexer,
    flush: Callable[[], None],
) -> None:
    fresh = await backend.find_window_by_id(wid)
    if not _verified(fresh, window.topic_mark_owner, wid):
        return
    assert fresh is not None and fresh.topic_mark_owner is not None
    window = fresh
    assert window.topic_mark_owner is not None
    state = router.get_topic_pause(uid, tid, chat)
    if state is not None and state.owner != window.topic_mark_owner:
        return
    if not enabled() and state is None:
        return
    if enabled() and window.topic_marked is False:
        if state is None or not state.paused:
            state = TopicPause(
                uid,
                chat,
                tid,
                wid,
                window.topic_mark_owner,
                revision=str(uuid4()),
            )
            router.set_topic_pause(state)
            flush()
        try:
            await _close(client, state, router, flush)
        except TelegramError as exc:
            _defer_flood(router, state, exc, flush)
            logger.warning("topic_pause_close_pending", thread_id=tid, error=str(exc))
    elif state is None:
        router.set_topic_pause(
            TopicPause(uid, chat, tid, wid, window.topic_mark_owner, paused=False)
        )
        flush()
    elif state.paused:
        try:
            await _resume(client, state, router, backend, flush)
        except (TelegramError, OSError) as exc:
            _defer_flood(router, state, exc, flush)
            logger.warning("topic_resume_pending", thread_id=tid, error=str(exc))


def cleanup_candidates(
    user_id: int, chat_id: int, router: ThreadRouter = thread_router
) -> list[TopicPause]:
    return [
        state
        for state in router.topic_pauses.values()
        if state.user_id == user_id
        and state.chat_id == chat_id
        and state.thread_id != 1
        and state.paused
        and not state.missing
        and _current(router, state)
    ]


async def delete_paused_topics(
    client: TelegramClient,
    candidates: Iterable[TopicPause],
    *,
    router: ThreadRouter = thread_router,
    backend: Multiplexer = multiplexer,
    flush: Callable[[], None] = _flush,
) -> dict[str, int]:
    counts = {"deleted": 0, "protected": 0, "failed": 0}
    for state in candidates:
        async with _lock(state.key):
            try:
                window = await backend.find_window_by_id(state.window_id)
            except OSError, RuntimeError, ValueError:
                counts["protected"] += 1
                continue
            if (
                not enabled()
                or _chat_cooling(router, state.chat_id)
                or not _current(router, state)
                or not state.paused
                or state.thread_id == 1
                or not _verified(window, state.owner, state.window_id)
                or window is None
                or window.topic_marked is not False
                or router.has_target_provisioning(state.window_id)
                or router.has_topic_provisioning(state.chat_id, state.thread_id)
            ):
                counts["protected"] += 1
                continue
            if not router.begin_paused_topic_deletion(state):
                counts["protected"] += 1
                continue
            try:
                result = await client.delete_forum_topic(
                    state.chat_id,
                    state.thread_id,
                    rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS,
                )
                if result is False:
                    counts["failed"] += 1
                    continue
            except RetryAfter as exc:
                _defer_flood(router, state, exc, flush)
                counts["failed"] += 1
                break
            except TelegramError as exc:
                if not is_missing(exc):
                    counts["failed"] += 1
                    continue
            finally:
                router.end_topic_deletion(state)
            if _current(router, state):
                router.set_topic_pause(replace(state, missing=True, closed=True))
                flush()
            counts["deleted"] += 1
    return counts
