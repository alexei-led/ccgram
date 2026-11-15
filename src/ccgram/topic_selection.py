"""Verified native writes and an owner-keyed journal of unfinished side effects."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import threading
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import structlog

from .multiplexer.base import canonical_window_id
from .topic_mark_journal import (
    TopicMarkOperation,
    has_topic_mark_conflict,
    operation_matches,
)
from .topic_mark_types import TopicMarkWriteWitness

if TYPE_CHECKING:
    from .multiplexer.base import Multiplexer, WindowRef
    from .thread_router import ThreadRouter
    from .window_state_store import WindowStateStore

logger = structlog.get_logger()
_selection_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_ENDPOINT_SCOPE_LENGTH = 64


def is_topic_mark_endpoint_scope(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _ENDPOINT_SCOPE_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


async def read_topic_mark_endpoint_scope(backend: Multiplexer) -> str:
    scope = await backend.topic_mark_endpoint_scope()
    if not isinstance(scope, str) or not is_topic_mark_endpoint_scope(scope):
        raise RuntimeError("Native marker endpoint is unconfirmed")
    return scope


async def require_topic_mark_endpoint_scope(
    backend: Multiplexer, expected: str
) -> None:
    if not is_topic_mark_endpoint_scope(expected) or (
        await read_topic_mark_endpoint_scope(backend) != expected
    ):
        raise RuntimeError(
            "Native marker endpoint changed or does not match this command"
        )


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


def _finish_attempt(
    store: WindowStateStore,
    flush: Callable[[], None],
    operation: TopicMarkOperation,
    dispatch_id: str,
) -> TopicMarkOperation:
    current = store.topic_mark_operations.get(operation.key)
    finished = store.finish_topic_mark_attempt(
        operation.window_id, owner=operation.owner, dispatch_id=dispatch_id
    )
    if finished is None or current is None:
        raise ValueError("Stale native marker completion")
    try:
        flush()
    except BaseException:
        store.restore_topic_mark(current)
        raise
    return finished


class _TopicMarkGuard:
    def __init__(
        self,
        operation: TopicMarkOperation,
        store: WindowStateStore,
        flush: Callable[[], None],
        backend_name: str,
        before_arm: Callable[[], None] | None = None,
        *,
        endpoint_scope: str | None = None,
        terminal_receipt: Callable[[str], None] | None = None,
    ) -> None:
        self.operation = operation
        self.store = store
        self.flush = flush
        self.backend_name = backend_name
        self.before_arm = before_arm
        self.endpoint_scope = endpoint_scope
        self.terminal_receipt = terminal_receipt
        self.finished: TopicMarkOperation | None = None
        self._thread = threading.get_ident()
        self._loop = asyncio.get_running_loop()
        self._dispatch_id: str | None = None

    def _check_context(self) -> None:
        if (
            threading.get_ident() != self._thread
            or asyncio.get_running_loop() is not self._loop
        ):
            raise RuntimeError("Marker journal callback is outside its state owner")

    def arm(self, witness: TopicMarkWriteWitness) -> None:
        self._check_context()
        if (
            self._dispatch_id is not None
            or witness.backend != self.backend_name
            or self.operation.owner is None
        ):
            raise ValueError("Duplicate or foreign native marker attempt")
        if (
            self.endpoint_scope is not None
            and witness.endpoint_scope != self.endpoint_scope
        ):
            raise RuntimeError("Native marker endpoint changed before dispatch")
        if has_topic_mark_conflict(
            self.store.topic_mark_operations.values(),
            self.operation.window_id,
            self.operation.owner,
            self.operation.desired,
        ):
            raise RuntimeError("Pending native marker work changed before dispatch")
        if self.before_arm is not None:
            self.before_arm()
        self.store.begin_topic_mark_attempt(self.operation, witness)
        self._dispatch_id = witness.dispatch_id
        self.flush()

    def terminal(self, dispatch_id: str) -> None:
        self._check_context()
        if self._dispatch_id != dispatch_id:
            raise ValueError("Stale native marker completion")
        if self.terminal_receipt is not None:
            self.terminal_receipt(dispatch_id)
        else:
            self.finished = _finish_attempt(
                self.store, self.flush, self.operation, dispatch_id
            )


def _settle_stored(
    operation: TopicMarkOperation, store: WindowStateStore, flush: Callable[[], None]
) -> bool:
    if not store.settle_topic_mark(operation):
        return False
    try:
        flush()
    except BaseException:
        store.restore_topic_mark(operation)
        raise
    return True


async def _write_topic_mark(
    operation: TopicMarkOperation,
    backend: Multiplexer,
    store: WindowStateStore,
    flush: Callable[[], None],
    before_arm: Callable[[], None] | None = None,
) -> bool:
    flush()
    guard = _TopicMarkGuard(
        operation, store, flush, backend.capabilities.name, before_arm
    )
    verified = await backend.set_topic_mark(
        operation.window_id,
        operation.desired,
        owner=operation.owner,
        write_guard=guard,
    )
    current = store.topic_mark_operations.get(operation.key)
    return (
        verified is True
        and current is not None
        and current.desired == operation.desired
        and current.attempt is None
        and _settle_stored(current, store, flush)
    )


def _manual_topic_operation(
    window: WindowRef, marked: bool, store: WindowStateStore
) -> tuple[TopicMarkOperation, TopicMarkOperation | None]:
    if has_topic_mark_conflict(
        store.topic_mark_operations.values(),
        window.window_id,
        window.topic_mark_owner,
        marked,
    ):
        raise RuntimeError("Pending native marker work owns this target")
    candidate = TopicMarkOperation(window.window_id, window.topic_mark_owner, marked)
    prior = store.topic_mark_operations.get(candidate.key)
    return (candidate if prior is None else prior), prior


async def set_topic_selection_locked(
    window: WindowRef,
    marked: bool,
    *,
    backend: Multiplexer,
    store: WindowStateStore,
    flush: Callable[[], None],
    endpoint_scope: str,
) -> bool:
    """Bind manual authority to its actual origin, never daemon configuration."""
    if not window.topic_markable or not window.topic_mark_owner:
        return False
    await require_topic_mark_endpoint_scope(backend, endpoint_scope)
    operation, prior = _manual_topic_operation(window, marked, store)

    def before_arm() -> None:
        if store.topic_mark_operations.get(operation.key) is not prior:
            raise RuntimeError("Native marker intent changed before dispatch")
        if prior is None:
            store.queue_topic_mark(window.window_id, window.topic_mark_owner, marked)

    receipt: str | None = None

    def terminal_receipt(dispatch_id: str) -> None:
        nonlocal receipt
        if receipt is not None:
            raise ValueError("Duplicate native marker completion")
        receipt = dispatch_id

    guard = _TopicMarkGuard(
        operation,
        store,
        flush,
        backend.capabilities.name,
        before_arm,
        endpoint_scope=endpoint_scope,
        terminal_receipt=terminal_receipt,
    )
    verified = await backend.set_topic_mark(
        operation.window_id, marked, owner=operation.owner, write_guard=guard
    )
    # Keep the durable fence until origin is confirmed; failed cleanup cannot
    # leave a new, unfenced request runnable against a replacement endpoint.
    await require_topic_mark_endpoint_scope(backend, endpoint_scope)
    if receipt is not None:
        guard.finished = _finish_attempt(store, flush, operation, receipt)
    if verified is not True or has_topic_mark_conflict(
        store.topic_mark_operations.values(),
        window.window_id,
        window.topic_mark_owner,
        marked,
    ):
        return False
    current = store.topic_mark_operations.get(operation.key)
    if current is prior or current is guard.finished:
        return current is None or _settle_stored(current, store, flush)
    return True


async def _recover_topic_mark_attempt(
    operation: TopicMarkOperation,
    backend: Multiplexer,
    store: WindowStateStore,
    flush: Callable[[], None],
) -> TopicMarkOperation | None:
    attempt = operation.attempt
    current = store.topic_mark_operations.get(operation.key)
    if (
        operation.owner is None
        or attempt is None
        or current is None
        or current.attempt != attempt
    ):
        return None
    try:
        retired = await backend.recover_topic_mark_write(attempt.witness)
    except Exception:
        logger.warning("topic_mark_retirement_unconfirmed", exc_info=True)
        return None
    if retired is not True:
        return None
    return _finish_attempt(store, flush, operation, attempt.witness.dispatch_id)


async def set_topic_selection(
    window: WindowRef,
    marked: bool,
    *,
    backend: Multiplexer,
    store: WindowStateStore,
    flush: Callable[[], None],
    endpoint_scope: str,
) -> bool:
    if not window.topic_markable or not window.topic_mark_owner:
        return False
    async with _selection_lock(window.window_id, owner=window.topic_mark_owner):
        await require_topic_mark_endpoint_scope(backend, endpoint_scope)
        for pending in tuple(store.topic_mark_operations.values()):
            if pending.attempt is not None and operation_matches(
                pending, window.window_id, window.topic_mark_owner
            ):
                await _recover_topic_mark_attempt(pending, backend, store, flush)
        return await set_topic_selection_locked(
            window,
            marked,
            backend=backend,
            store=store,
            flush=flush,
            endpoint_scope=endpoint_scope,
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
    # Lazy: only bot lifecycle operations use the installed state store.
    from .window_state_store import get_window_store

    return _settle_stored(operation, get_window_store(), _flush)


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
    return any(
        target is not None
        and _same_selection_resource(window_id, owner, target, claim.topic_mark_owner)
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
        return await _finish_topic_selection_locked(operation, mux, router=router)


async def _finish_topic_selection_locked(
    operation: TopicMarkOperation,
    mux: Multiplexer,
    *,
    router: ThreadRouter | None,
) -> bool:
    # Lazy: only the bot retry path accesses installed lifecycle/routing state.
    from .window_state_ports import lifecycle_state

    # Lazy: keep installed state initialization behind the daemon retry path.
    from .window_state_store import get_window_store

    # Lazy: initialize the routing singleton only after acquiring writer authority.
    from .thread_router import thread_router

    store = get_window_store()
    if not lifecycle_state.topic_mark_operation_is_current(operation):
        return False
    if operation.attempt is not None:
        recovered = await _recover_topic_mark_attempt(operation, mux, store, _flush)
        if recovered is None:
            return False
        operation = recovered
    if operation.desired and lifecycle_state.is_topic_detach_pending(
        operation.window_id, owner=operation.owner
    ):
        return False
    consumer_router = thread_router if router is None else router
    if not operation.desired and _has_remaining_consumers(consumer_router, operation):
        return _settle(operation)

    def before_arm() -> None:
        if not operation.desired and _has_remaining_consumers(
            consumer_router, operation
        ):
            raise ValueError("A binding or creation now protects this marker")

    try:
        return await _write_topic_mark(operation, mux, store, _flush, before_arm)
    except Exception:
        logger.warning(
            "topic_selection_retry_pending",
            window_id=operation.window_id,
            owner=operation.owner,
            marked=operation.desired,
            exc_info=True,
        )
        return False


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
    if owner is not None and window.topic_mark_owner != owner:
        return None
    if (
        require_marked
        and window.topic_markable
        and (window.topic_marked is not True or window.topic_mark_owner is None)
    ):
        return None
    return window


async def select_topic_target(
    window_id: str,
    *,
    mux: Multiplexer | None = None,
    claim_id: str | None = None,
    router: ThreadRouter | None = None,
    expected_owner: str | None = None,
) -> bool:
    """Checkpoint native intent without replacing a known caller or claim witness."""
    # Lazy: only bot selection uses the wired backend.
    from .multiplexer import multiplexer

    # Lazy: native CLI imports must not initialize lifecycle owners.
    from .window_state_ports import lifecycle_state

    mux = multiplexer if mux is None else mux
    async with _selection_lock(window_id):
        window = await _read_window(window_id, mux)
        if expected_owner is not None and (
            window is None or window.topic_mark_owner != expected_owner
        ):
            return False
        owner = (
            window.topic_mark_owner
            if window is not None
            else lifecycle_state.get_topic_mark_owner(window_id)
        )
        if claim_id is not None:
            owner = window.topic_mark_owner if window is not None else None
            # Lazy: only provisioning selection captures an immutable claim witness.
            from .thread_router import thread_router

            claim_router = thread_router if router is None else router
            claim = claim_router.get_topic_provisioning(claim_id)
            if (
                claim is None
                or claim.target_id is None
                or canonical_window_id(claim.target_id)
                != canonical_window_id(window_id)
            ):
                return False
            if claim.topic_mark_owner is not None and (
                window is None or owner != claim.topic_mark_owner
            ):
                return False
            claim_router.capture_provisioning_owner(claim_id, owner)
            _flush()
        if lifecycle_state.is_topic_detach_pending(
            window_id, owner=owner
        ) or lifecycle_state.is_topic_mark_quarantined(window_id, owner=owner):
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
        return (
            current is not None
            and not lifecycle_state.is_topic_detach_pending(window_id, owner=owner)
            and not lifecycle_state.is_topic_mark_quarantined(window_id, owner=owner)
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
    owner = lifecycle_state.get_topic_mark_owner(window_id)
    if owner is None and window is not None:
        owner = window.topic_mark_owner
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
