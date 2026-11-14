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
    has_topic_mark_conflict,
    operation_matches,
    read_topic_mark_operations,
)
from .utils import ccgram_dir, load_ccgram_env


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


async def _topic_action(
    backend: Multiplexer, action: str, target: str | None
) -> dict[str, object]:
    windows = await backend.list_windows_for_reconciliation()
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
    if action not in {"mark", "unmark"} or target is None:
        raise ValueError("Mark and unmark require one explicit target.")
    window = selected[0]
    if not window.topic_markable:
        raise ValueError(
            "This target cannot be natively marked; use an explicit Telegram bind."
        )

    # Lazy: read-only status and CLI help must not load lifecycle dependencies.
    from .topic_selection import set_topic_selection

    marked = action == "mark"
    if has_topic_mark_conflict(
        operations, window.window_id, window.topic_mark_owner, marked
    ):
        raise RuntimeError(
            "A conflicting pending marker operation owns this target; "
            "retry after recovery. No selection was changed."
        )
    if not await set_topic_selection(window, marked, backend=backend):
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
