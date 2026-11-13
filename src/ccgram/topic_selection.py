"""Native topic selection and crash-safe deliberate detach.

The low-level marker operation has no bot/config dependencies. Lifecycle
operations import their owners lazily so the terminal CLI can use that seam
without constructing Config or SessionManager.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import structlog

from .multiplexer.base import canonical_window_id

if TYPE_CHECKING:
    from .multiplexer.base import Multiplexer
    from .thread_router import ThreadRouter

logger = structlog.get_logger()

# Serialize native writes for the same terminal. Weak entries disappear after
# the last operation/waiter; durable retry intent lives in WindowState instead.
_selection_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def _selection_lock(window_id: str) -> asyncio.Lock:
    key = canonical_window_id(window_id)
    lock = _selection_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _selection_locks[key] = lock
    return lock


async def set_topic_selection(
    window_id: str, marked: bool, *, backend: Multiplexer
) -> bool:
    """Set a native mark, rejecting absent/unmarkable targets.

    Multiplexer.set_topic_mark itself verifies the native result. No binding,
    lifecycle, or Config state is read or changed by this CLI-safe operation.
    """
    window = await backend.find_window_by_id(window_id)
    if window is None or not window.topic_markable:
        return False
    return await backend.set_topic_mark(window.window_id, marked) is True


async def _finish_topic_selection(window_id: str, mux: Multiplexer) -> bool:
    """Apply durable desired mark while the caller holds the terminal lock."""
    # Lazy: native CLI imports this module without initializing bot state.
    from .session import session_manager

    # Lazy: port package initialization is needed only for bot lifecycle work.
    from .window_state_ports import lifecycle_state

    desired = lifecycle_state.get_pending_topic_mark(window_id)
    if desired is None:
        return True
    try:
        window = await mux.find_window_by_id(window_id)
        if window is None:
            return False
        # An agterm split is explicitly bindable, but has no native mark of its
        # own. Never delegate its selection to the primary session.
        if (
            window.topic_markable
            and await mux.set_topic_mark(window.window_id, desired) is not True
        ):
            logger.warning(
                "native_topic_mark_unverified", window_id=window_id, marked=desired
            )
            return False
    except Exception:
        logger.warning(
            "topic_selection_retry_pending",
            window_id=window_id,
            marked=desired,
            exc_info=True,
        )
        return False
    if lifecycle_state.get_pending_topic_mark(window_id) is not desired:
        # A deliberate detach can supersede an in-flight mark-on write before
        # waiting for this lock. Its persisted clear must remain unfinished.
        return False
    detached = lifecycle_state.is_topic_detached(window_id)
    lifecycle_state.set_topic_selection(window_id, detached=detached, pending=None)
    try:
        session_manager.flush_state()
    except BaseException:
        # Retain retry ownership in memory as well as on disk when settling
        # the verified native write cannot be checkpointed.
        lifecycle_state.set_topic_selection(
            window_id, detached=detached, pending=desired
        )
        raise
    return True


async def select_topic_target(
    window_id: str, *, mux: Multiplexer | None = None
) -> bool:
    """Select before explicit binding; False leaves a durable mark-on retry.

    Callers retain their provisioning/adoption guard through the subsequent
    synchronous bind. Auto mode may warn and bind after a failed mark attempt;
    marked mode must leave provisioning unfinished until verification succeeds.
    """
    # Lazy: the CLI injects its backend and does not need bot wiring.
    from .multiplexer import multiplexer

    # Lazy: native CLI imports must not initialize Config through session.
    from .session import session_manager

    # Lazy: initialize lifecycle owners only on the bot selection path.
    from .window_state_ports import lifecycle_state

    mux = multiplexer if mux is None else mux
    async with _selection_lock(window_id):
        if lifecycle_state.is_topic_detach_pending(window_id):
            return False
        lifecycle_state.set_topic_selection(window_id, detached=False, pending=True)
        session_manager.flush_state()
        return await _finish_topic_selection(window_id, mux)


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
    """Unbind intentionally; False means native clear remains unfinished.

    The last marked-mode binding is removed together with durable clear intent
    before any backend await. Cancellation/failure cannot lose that intent or
    turn the deliberately detached terminal into an abandoned failed create.
    """
    # Lazy: native CLI operations must not instantiate Config.
    from .config import config

    # Lazy: only lifecycle operations use the bot-wired backend.
    from .multiplexer import multiplexer

    # Lazy: native CLI imports must not initialize persistent bot state.
    from .session import session_manager

    # Lazy: binding ownership is needed only for deliberate bot detach.
    from .thread_router import thread_router

    # Lazy: feature ports belong to the bot lifecycle, not the native CLI.
    from .window_state_ports import legacy_state, lifecycle_state

    router = thread_router if router is None else router
    mux = multiplexer if mux is None else mux
    current = router.get_window_for_thread(user_id, thread_id, chat_id)
    if current is None or canonical_window_id(current) != canonical_window_id(
        window_id
    ):
        return True
    router.unbind_thread(
        user_id, thread_id, chat_id=chat_id, retirement_reason=retirement_reason
    )
    if (
        config.topic_discovery_mode != "marked"
        or router.has_window(window_id)
        or legacy_state.is_legacy_herdr(window_id)
    ):
        return True
    lifecycle_state.set_topic_selection(window_id, detached=True, pending=False)
    session_manager.flush_state()
    async with _selection_lock(window_id):
        return await _finish_topic_selection(window_id, mux)


async def recover_topic_selections(*, mux: Multiplexer | None = None) -> None:
    """Retry unfinished verified native mark writes, including after restart."""
    # Lazy: retries run only in the bot, not the native CLI.
    from .multiplexer import multiplexer

    # Lazy: defer lifecycle owner initialization until bot recovery runs.
    from .window_state_ports import lifecycle_state

    mux = multiplexer if mux is None else mux
    for window_id in lifecycle_state.pending_topic_selections():
        async with _selection_lock(window_id):
            await _finish_topic_selection(window_id, mux)
