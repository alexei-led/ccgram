"""Operator CLI for native Telegram-topic marks, independent of bot startup."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from .multiplexer import get_multiplexer
from .multiplexer.base import Multiplexer, WindowRef
from .multiplexer.registry import detect_multiplexer_name
from .topic_discovery import (
    DEFAULT_TOPIC_DISCOVERY_MODE,
    TOPIC_DISCOVERY_ENV,
    parse_topic_discovery_mode,
)
from .topic_mark_journal import (
    TOPIC_MARK_OPERATIONS_KEY,
    operation_matches,
    read_topic_mark_operations,
)
from .utils import atomic_write_json, ccgram_dir, load_ccgram_env
from .state_writer_lock import StateWriterLease


def _select_targets(windows: list[WindowRef], target: str | None) -> list[WindowRef]:
    if target is None:
        return windows
    matches = [window for window in windows if window.matches(target)]
    if not matches:
        raise LookupError(
            f"Target not found: {target}. Use ccgram topic status to list targets."
        )
    if len(matches) != 1:
        raise LookupError(f"Target is ambiguous: {target}")
    return matches


async def _capture_topic_inventory(
    backend: Multiplexer,
) -> tuple[list[WindowRef] | None, str]:
    # Lazy: read-only status need not resolve mutation provenance.
    from .topic_selection import (
        read_topic_mark_endpoint_scope,
        require_topic_mark_endpoint_scope,
    )

    scope = await read_topic_mark_endpoint_scope(backend)
    windows = await backend.list_windows_for_reconciliation()
    await require_topic_mark_endpoint_scope(backend, scope)
    return windows, scope


async def _topic_action(
    backend: Multiplexer, action: str, target: str | None
) -> dict[str, object]:
    if action == "status":
        windows = await backend.list_windows_for_reconciliation()
        endpoint_scope = None
    else:
        if action not in {"mark", "unmark"} or target is None:
            raise ValueError("Mark and unmark require one explicit target.")
        windows, endpoint_scope = await _capture_topic_inventory(backend)
    if windows is None:
        raise RuntimeError("Backend listing is unconfirmed; no selection was changed.")
    selected = _select_targets(windows, target)
    try:
        operations = read_topic_mark_operations(ccgram_dir() / "state.json")
    except (OSError, ValueError, UnicodeError) as exc:
        raise RuntimeError(
            "Pending marker journal is unconfirmed; no selection was changed."
        ) from exc
    if action == "status":
        return {
            "targets": [
                {
                    "id": window.window_id,
                    "name": window.window_name,
                    "marked": window.topic_marked,
                    "markable": window.topic_markable,
                    "owner": window.topic_mark_owner,
                }
                for window in selected
            ],
            "pending_operations": [
                operation.to_dict()
                for operation in operations
                if target is None
                or operation_matches(
                    operation, selected[0].window_id, selected[0].topic_mark_owner
                )
            ],
        }
    if endpoint_scope is None:
        raise RuntimeError("Native marker endpoint is unconfirmed")
    window = selected[0]
    if not window.topic_markable:
        raise ValueError(
            "This target cannot be natively marked; use an explicit Telegram bind."
        )

    if not window.topic_mark_owner:
        raise RuntimeError("Native selection ownership is unconfirmed")
    marked = action == "mark"
    path = ccgram_dir() / "state.json"
    lease = StateWriterLease(path)
    if not lease.try_acquire():
        # Lazy: status never creates a socket, lease or marker operation.
        from .topic_mark_ipc import request_topic_mark

        return await request_topic_mark(
            path,
            backend.capabilities.name,
            window,
            marked,
            endpoint_scope=endpoint_scope,
        )
    try:
        return await _offline_topic_mark(
            lease, backend, window, marked, endpoint_scope=endpoint_scope
        )
    finally:
        lease.close()


async def _offline_topic_mark(
    lease: StateWriterLease,
    backend: Multiplexer,
    window: WindowRef,
    marked: bool,
    *,
    endpoint_scope: str,
) -> dict[str, object]:
    # Lazy: offline marker writes reuse the codec/store, never SessionManager.
    from .window_state_store import WindowStateStore

    # Lazy: acquire the offline writer lease before loading mutation orchestration.
    from .topic_selection import set_topic_selection

    if not lease.held:
        raise RuntimeError("Offline marker mutation needs the state writer lease")
    try:
        state = json.loads(lease.state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        state = {}
    if not isinstance(state, dict):
        raise ValueError("State must be an object")
    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _window_id: None
    )
    store.load_topic_mark_journal(state)

    def flush() -> None:
        snapshot = dict(state)
        snapshot.pop(TOPIC_MARK_OPERATIONS_KEY, None)
        snapshot.update(store.topic_mark_journal_to_dict())
        atomic_write_json(lease.state_path, snapshot)

    if not await set_topic_selection(
        window,
        marked,
        backend=backend,
        store=store,
        flush=flush,
        endpoint_scope=endpoint_scope,
    ):
        raise RuntimeError(
            "Marker update could not be confirmed. Inspect ccgram topic status."
        )
    return {"target": window.window_id, "marked": marked}


def topic_main(action: str, target: str | None) -> str:
    """Run one native selection command without validating Telegram credentials."""
    # Lazy: reuse logging setup without loading the bot; stdout is reserved for JSON.
    from .main import setup_logging

    setup_logging("WARNING", output=sys.stderr)
    load_ccgram_env()
    name = os.environ.get("CCGRAM_MULTIPLEXER", "auto")
    if name == "auto":
        name = detect_multiplexer_name(dict(os.environ))

    mode = parse_topic_discovery_mode(
        os.environ.get(TOPIC_DISCOVERY_ENV, DEFAULT_TOPIC_DISCOVERY_MODE)
    )
    result = asyncio.run(_topic_action(get_multiplexer(name), action, target))
    return json.dumps({"backend": name, "mode": mode, **result}, ensure_ascii=False)
