"""Verified native selection and crash-safe terminal-owned lifecycle operations.

The native CLI seam imports no Config/session singleton. Routing IDs locate
conversations; opaque native owners authorize marker operations across re-keying.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import structlog

from .multiplexer.base import canonical_window_id

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
    window_id: str, marked: bool, *, backend: Multiplexer
) -> bool:
    """CLI-safe native operation; never infer or generate an owner."""
    window = await _read_window(window_id, backend)
    if window is None or not window.topic_markable or not window.topic_mark_owner:
        return False
    return (
        await backend.set_topic_mark(
            window.window_id, marked, owner=window.topic_mark_owner
        )
        is True
    )


def _checkpoint(
    window_id: str, *, detached: bool, pending: bool | None, owner: str | None
) -> None:
    # Lazy: native CLI imports must not initialize persistent bot state.
    from .session import session_manager

    # Lazy: lifecycle ports are needed only for bot-owned operations.
    from .window_state_ports import lifecycle_state

    lifecycle_state.set_topic_selection(
        window_id, detached=detached, pending=pending, owner=owner
    )
    session_manager.flush_state()


def _settle(window_id: str, desired: bool, owner: str | None) -> bool:
    # Lazy: native CLI imports must not initialize lifecycle owners.
    from .window_state_ports import lifecycle_state

    if (
        lifecycle_state.get_pending_topic_mark(window_id) is not desired
        or lifecycle_state.get_topic_mark_owner(window_id) != owner
    ):
        return False
    detached = lifecycle_state.is_topic_detached(window_id)
    try:
        _checkpoint(window_id, detached=detached, pending=None, owner=owner)
    except BaseException:
        lifecycle_state.set_topic_selection(
            window_id, detached=detached, pending=desired, owner=owner
        )
        raise
    return True


async def _finish_topic_selection(window_id: str, mux: Multiplexer) -> bool:
    # Lazy: retries belong to the bot lifecycle, not the native CLI.
    from .window_state_ports import lifecycle_state

    desired = lifecycle_state.get_pending_topic_mark(window_id)
    if desired is None:
        return not lifecycle_state.is_topic_detached(window_id)
    owner = lifecycle_state.get_topic_mark_owner(window_id)
    if owner is None:
        window = await _read_window(window_id, mux)
        if window is not None and not window.topic_markable:
            return _settle(window_id, desired, None)
        logger.warning("topic_selection_owner_unconfirmed", window_id=window_id)
        return False
    async with _selection_lock(window_id, owner=owner):
        if (
            lifecycle_state.get_pending_topic_mark(window_id) is not desired
            or lifecycle_state.get_topic_mark_owner(window_id) != owner
        ):
            return False
        try:
            verified = await mux.set_topic_mark(window_id, desired, owner=owner)
        except Exception:
            logger.warning(
                "topic_selection_retry_pending",
                window_id=window_id,
                owner=owner,
                marked=desired,
                exc_info=True,
            )
            return False
        if verified is not True:
            logger.warning(
                "native_topic_mark_unverified",
                window_id=window_id,
                owner=owner,
                marked=desired,
            )
            return False
        return _settle(window_id, desired, owner)


async def _selection_still_current(
    window_id: str, owner: str, mux: Multiplexer
) -> bool:
    # Lazy: only bot selection needs persistent pending-clear evidence.
    from .window_state_ports import lifecycle_state

    window = await _read_window(window_id, mux)
    return (
        window is not None
        and window.topic_markable
        and window.topic_mark_owner == owner
        and window.topic_marked is True
        and not lifecycle_state.is_topic_detach_pending(window_id, owner=owner)
    )


async def select_topic_target(
    window_id: str, *, mux: Multiplexer | None = None
) -> bool:
    """Capture a proven native owner before mark-on, retaining durable retries."""
    # Lazy: the CLI injects its backend and does not need bot wiring.
    from .multiplexer import multiplexer

    # Lazy: bot state must not initialize on native CLI import.
    from .window_state_ports import lifecycle_state

    mux = multiplexer if mux is None else mux
    async with _selection_lock(window_id):
        saved_owner = lifecycle_state.get_topic_mark_owner(window_id)
        if lifecycle_state.is_topic_detach_pending(window_id, owner=saved_owner):
            return False
        window = await _read_window(window_id, mux)
        owner = window.topic_mark_owner if window is not None else None
        if lifecycle_state.is_topic_detach_pending(window_id, owner=owner):
            return False
        if window is not None and not window.topic_markable:
            _checkpoint(window_id, detached=False, pending=None, owner=None)
            return True
        if (
            lifecycle_state.get_pending_topic_mark(window_id) is True
            and saved_owner is not None
            and owner is not None
            and owner != saved_owner
        ):
            return False
        _checkpoint(
            window_id,
            detached=False,
            pending=True,
            owner=owner if owner is not None else saved_owner,
        )
        if window is None or not owner:
            logger.warning("topic_selection_owner_unconfirmed", window_id=window_id)
            return False
        if not await _finish_topic_selection(window_id, mux):
            return False
        return await _selection_still_current(window_id, owner, mux)


def _has_remaining_binding(
    router: ThreadRouter, window_id: str, owner: str | None
) -> bool:
    # Lazy: only bot detachment reads binding ownership.
    from .window_state_ports import lifecycle_state

    return any(
        canonical_window_id(wid) == canonical_window_id(window_id)
        or (owner is not None and lifecycle_state.get_topic_mark_owner(wid) == owner)
        for _, _, wid in router.iter_thread_bindings()
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
    """Refresh while bound, then checkpoint detach before native clear awaits.

    A historical ownerless target which is no longer verifiable stays an
    unfinished operation. It cannot authorize mutation of another terminal.
    """
    # Lazy: native CLI operations must not instantiate Config.
    from .config import config

    # Lazy: only lifecycle operations use the bot-wired backend.
    from .multiplexer import multiplexer

    # Lazy: routing ownership is needed only for deliberate bot detach.
    from .thread_router import thread_router

    # Lazy: native CLI imports must not initialize lifecycle state.
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
    owner = lifecycle_state.get_topic_mark_owner(window_id)
    if lifecycle_state.get_pending_topic_mark(window_id) is None and window is not None:
        owner = window.topic_mark_owner or owner
    current = router.get_window_for_thread(user_id, thread_id, chat_id)
    if current is None or canonical_window_id(current) != canonical_window_id(
        window_id
    ):
        return True
    router.unbind_thread(
        user_id, thread_id, chat_id=chat_id, retirement_reason=retirement_reason
    )
    if _has_remaining_binding(router, window_id, owner):
        lifecycle_state.observe_topic_mark_owner(window_id, owner)
        return True
    split = window is not None and not window.topic_markable
    _checkpoint(window_id, detached=True, pending=None if split else False, owner=owner)
    if split:
        return True
    return await _finish_topic_selection(window_id, mux)


async def recover_topic_selections(*, mux: Multiplexer | None = None) -> None:
    """Retry by captured native owner, including bare-shell/re-keyed terminals."""
    # Lazy: retries run only in the bot, not the native CLI.
    from .multiplexer import multiplexer

    # Lazy: defer lifecycle owner initialization until bot recovery runs.
    from .window_state_ports import lifecycle_state

    mux = multiplexer if mux is None else mux
    for window_id in lifecycle_state.pending_topic_selections():
        await _finish_topic_selection(window_id, mux)
