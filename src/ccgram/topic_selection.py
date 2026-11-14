"""Verified native writes and an owner-keyed journal of unfinished side effects."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import structlog

from .multiplexer.base import canonical_window_id
from .topic_mark_journal import TopicMarkOperation

if TYPE_CHECKING:
    from .multiplexer.base import Multiplexer, WindowRef
    from .thread_router import ThreadRouter

logger = structlog.get_logger()
_selection_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def _selection_lock(window_id: str, *, owner: str | None = None) -> asyncio.Lock:
    key = (
        f"owner:{owner}"
        if owner is not None
        else f"window:{canonical_window_id(window_id)}"
    )
    lock = _selection_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _selection_locks[key] = lock
    return lock


async def _read_window(window_id: str, mux: Multiplexer) -> WindowRef | None:
    try:
        return await mux.find_window_by_id(window_id)
    except Exception:
        logger.warning(
            "topic_selection_owner_unconfirmed", window_id=window_id, exc_info=True
        )
        return None


async def set_topic_selection(
    window: WindowRef, marked: bool, *, backend: Multiplexer
) -> bool:
    """Use the CLI's captured native witness without re-resolving its route."""
    if not window.topic_markable or not window.topic_mark_owner:
        return False
    return (
        await backend.set_topic_mark(
            window.window_id, marked, owner=window.topic_mark_owner
        )
        is True
    )


def _flush() -> None:
    # Lazy: native CLI imports must not initialize persistent bot state.
    from .session import session_manager

    session_manager.flush_state()


def _queue(window_id: str, owner: str | None, desired: bool) -> TopicMarkOperation:
    # Lazy: unfinished bot effects are independent of the native CLI.
    from .window_state_ports import lifecycle_state

    operation = lifecycle_state.queue_topic_mark(window_id, owner, desired)
    _flush()
    return operation


def _settle(operation: TopicMarkOperation) -> bool:
    # Lazy: the CLI reads immutable DTOs without initializing bot state.
    from .window_state_ports import lifecycle_state

    if not lifecycle_state.settle_topic_mark(operation):
        return False
    try:
        _flush()
    except BaseException:
        lifecycle_state.restore_topic_mark(operation)
        raise
    return True


def _same_selection_resource(
    window_id: str, owner: str | None, other_id: str, other_owner: str | None
) -> bool:
    if owner is not None and other_owner is not None:
        return owner == other_owner
    return canonical_window_id(window_id) == canonical_window_id(other_id)


def _has_remaining_binding(
    router: ThreadRouter, window_id: str, owner: str | None
) -> bool:
    # Lazy: native CLI operations do not inspect Telegram bindings.
    from .window_state_ports import lifecycle_state

    return any(
        _same_selection_resource(
            window_id, owner, wid, lifecycle_state.get_topic_mark_owner(wid)
        )
        for _, _, wid in router.iter_thread_bindings()
    )


def _has_remaining_claim(
    router: ThreadRouter, window_id: str, owner: str | None
) -> bool:
    # Lazy: creation claims retain ownership in the existing bot state.
    from .window_state_ports import lifecycle_state

    return any(
        target is not None
        and _same_selection_resource(
            window_id, owner, target, lifecycle_state.get_topic_mark_owner(target)
        )
        for claim in router.iter_topic_provisionings()
        for target in (claim.target_id, claim.previous_target_id)
    )


def _has_remaining_consumers(
    router: ThreadRouter, operation: TopicMarkOperation
) -> bool:
    return _has_remaining_binding(
        router, operation.window_id, operation.owner
    ) or _has_remaining_claim(router, operation.window_id, operation.owner)


async def _finish_topic_selection(
    operation: TopicMarkOperation,
    mux: Multiplexer,
    *,
    router: ThreadRouter | None = None,
) -> bool:
    # Lazy: native CLI reads journal records, but never owns their completion.
    from .window_state_ports import lifecycle_state

    if not lifecycle_state.topic_mark_operation_is_current(operation):
        return False
    if operation.owner is None:
        window = await _read_window(operation.window_id, mux)
        if (
            window is not None
            and not window.topic_markable
            and canonical_window_id(window.window_id)
            == canonical_window_id(operation.window_id)
        ):
            return _settle(operation)
        logger.warning(
            "topic_selection_owner_unconfirmed", window_id=operation.window_id
        )
        return False
    async with _selection_lock(operation.window_id, owner=operation.owner):
        if not lifecycle_state.topic_mark_operation_is_current(operation) or (
            operation.desired
            and lifecycle_state.is_topic_detach_pending(
                operation.window_id, owner=operation.owner
            )
        ):
            return False
        # Lazy: only clear completion checks binding and provisioning consumers.
        from .thread_router import thread_router

        consumer_router = thread_router if router is None else router
        if not operation.desired and _has_remaining_consumers(
            consumer_router, operation
        ):
            return _settle(operation)
        # A failed preparation checkpoint can leave a record only in memory.
        # Every retry must durably retain its exact intent before native I/O.
        _flush()
        try:
            verified = await mux.set_topic_mark(
                operation.window_id, operation.desired, owner=operation.owner
            )
        except Exception:
            logger.warning(
                "topic_selection_retry_pending",
                window_id=operation.window_id,
                owner=operation.owner,
                marked=operation.desired,
                exc_info=True,
            )
            return False
        if verified is not True:
            logger.warning(
                "native_topic_mark_unverified",
                window_id=operation.window_id,
                owner=operation.owner,
                marked=operation.desired,
            )
            return False
        return _settle(operation)


async def verify_topic_target(
    window_id: str,
    *,
    owner: str | None,
    require_marked: bool,
    mux: Multiplexer | None = None,
) -> WindowRef | None:
    """Return a verified current snapshot, never route through an owner handle."""
    # Lazy: the native CLI injects its own backend.
    from .multiplexer import multiplexer

    mux = multiplexer if mux is None else mux
    window = await _read_window(window_id, mux)
    if window is None or canonical_window_id(window.window_id) != canonical_window_id(
        window_id
    ):
        return None
    if (
        owner is not None
        and window.topic_mark_owner is not None
        and window.topic_mark_owner != owner
    ):
        return None
    if (
        require_marked
        and window.topic_markable
        and (window.topic_marked is not True or window.topic_mark_owner is None)
    ):
        return None
    return window


async def select_topic_target(
    window_id: str, *, mux: Multiplexer | None = None
) -> bool:
    """Checkpoint an observed owner and immutable mark-on intent before writing."""
    # Lazy: only bot selection uses the wired backend.
    from .multiplexer import multiplexer

    # Lazy: native CLI imports must not initialize lifecycle owners.
    from .window_state_ports import lifecycle_state

    mux = multiplexer if mux is None else mux
    async with _selection_lock(window_id):
        window = await _read_window(window_id, mux)
        owner = (
            window.topic_mark_owner
            if window is not None
            else lifecycle_state.get_topic_mark_owner(window_id)
        )
        if lifecycle_state.is_topic_detach_pending(window_id, owner=owner):
            return False
        if window is not None and not window.topic_markable:
            lifecycle_state.record_topic_mark_owner(window_id, None)
            _flush()
            return True
        if owner is not None:
            lifecycle_state.record_topic_mark_owner(window_id, owner)
        operation = _queue(window_id, owner, True)
        if window is None or owner is None:
            logger.warning("topic_selection_owner_unconfirmed", window_id=window_id)
            return False
        if not await _finish_topic_selection(operation, mux):
            return False
        current = await verify_topic_target(
            window_id, owner=owner, require_marked=True, mux=mux
        )
        return current is not None and not lifecycle_state.is_topic_detach_pending(
            window_id, owner=owner
        )


async def detach_topic_target(
    user_id: int,
    thread_id: int,
    window_id: str,
    *,
    chat_id: int | None = None,
    retirement_reason: str = "keep_remote",
    router: ThreadRouter | None = None,
    mux: Multiplexer | None = None,
) -> bool:
    """Observe while bound, then synchronously checkpoint detach before clear."""
    # Lazy: native CLI imports do not instantiate Config.
    from .config import config

    # Lazy: only lifecycle operations use the bot-wired backend.
    from .multiplexer import multiplexer

    # Lazy: Telegram routing is not native marker identity.
    from .thread_router import thread_router

    # Lazy: lifecycle metadata and journal share the bot's existing state owner.
    from .window_state_ports import legacy_state, lifecycle_state

    router = thread_router if router is None else router
    mux = multiplexer if mux is None else mux
    current = router.get_window_for_thread(user_id, thread_id, chat_id)
    if current is None or canonical_window_id(current) != canonical_window_id(
        window_id
    ):
        return True
    if config.topic_discovery_mode != "marked" or legacy_state.is_legacy_herdr(
        window_id
    ):
        router.unbind_thread(
            user_id, thread_id, chat_id=chat_id, retirement_reason=retirement_reason
        )
        return True
    window = await _read_window(window_id, mux)
    owner = (
        window.topic_mark_owner
        if window is not None
        else lifecycle_state.get_topic_mark_owner(window_id)
    )
    if owner is not None:
        lifecycle_state.observe_topic_mark_owner(window_id, owner)
    current = router.get_window_for_thread(user_id, thread_id, chat_id)
    if current is None or canonical_window_id(current) != canonical_window_id(
        window_id
    ):
        return True
    router.unbind_thread(
        user_id, thread_id, chat_id=chat_id, retirement_reason=retirement_reason
    )
    if _has_remaining_binding(router, window_id, owner):
        return True
    lifecycle_state.record_topic_detached(window_id, owner)
    split = window is not None and not window.topic_markable
    claimed = _has_remaining_claim(router, window_id, owner)
    if split or claimed:
        _flush()
        return True
    operation = _queue(window_id, owner, False)
    return await _finish_topic_selection(operation, mux, router=router)


async def recover_topic_selections(
    *, mux: Multiplexer | None = None, operation: TopicMarkOperation | None = None
) -> None:
    """Retry captured immutable operations without stale routing-ID fallback."""
    # Lazy: bot retries do not affect native CLI import requirements.
    from .multiplexer import multiplexer

    # Lazy: the owner journal is initialized only for bot lifecycle work.
    from .window_state_ports import lifecycle_state

    mux = multiplexer if mux is None else mux
    operations = (
        [operation]
        if operation is not None
        else lifecycle_state.pending_topic_selections()
    )
    for pending in operations:
        await _finish_topic_selection(pending, mux)
