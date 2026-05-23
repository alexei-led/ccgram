"""Read-only window state queries — free functions for handler use.

Provides the same window-state read accessors as ``SessionManager`` but as
module-level free functions.  Handler modules that only need to *read* window
state can import from here instead of ``session``, reducing their coupling
surface from the full ``SessionManager`` singleton to a set of narrow query
functions that depend only on ``window_state_store`` and ``config``.

Write operations (``set_window_provider``, ``set_window_approval_mode``, etc.)
remain on ``SessionManager`` — only modules that genuinely mutate state should
import it.
"""

from __future__ import annotations

from pathlib import Path

from .window_state_ports import tool_state as _tool_state
from .window_state_store import (
    APPROVAL_MODES,
    DEFAULT_APPROVAL_MODE,
    window_store,
)
from .window_view import WindowView


def view_window(window_id: str) -> WindowView | None:
    """Read-only snapshot of a window's state, or None if no state exists."""
    ws = window_store.window_states.get(window_id)
    if ws is None:
        return None
    return WindowView(
        window_id=window_id,
        cwd=ws.cwd or "",
        provider_name=ws.provider_name,
        approval_mode=ws.approval_mode,
        batch_mode=ws.batch_mode,
        tool_call_visibility=ws.tool_call_visibility,
        transcript_path=Path(ws.transcript_path) if ws.transcript_path else None,
        window_name=ws.window_name,
        session_id=ws.session_id,
        external=ws.external,
        origin=ws.origin,
    )


def get_window_provider(window_id: str) -> str | None:
    """Return the provider name for a window, or None if not set."""
    state = window_store.window_states.get(window_id)
    return state.provider_name if state else None


def get_approval_mode(window_id: str) -> str:
    """Get approval mode for a window (default: 'normal')."""
    state = window_store.window_states.get(window_id)
    mode = state.approval_mode if state else DEFAULT_APPROVAL_MODE
    return mode if mode in APPROVAL_MODES else DEFAULT_APPROVAL_MODE


def get_batch_mode(window_id: str) -> str:
    """Get batch mode for a window (delegates to ``tool_state`` port).

    Per-window value wins when the state row exists with a valid mode.
    Falls through to global config (ephemeral_tools) when no row exists
    or the stored mode is invalid.
    """
    return _tool_state.get_batch_mode(window_id)


def is_ephemeral_tools(window_id: str) -> bool:
    """Return True if the resolved batch mode for window_id is 'ephemeral'."""
    return _tool_state.is_ephemeral_tools(window_id)


def get_tool_call_visibility(window_id: str) -> str:
    """Get raw per-window tool-call visibility (default/shown/hidden)."""
    return _tool_state.get_tool_call_visibility(window_id)


def is_tool_calls_hidden(window_id: str) -> bool:
    """Resolved boolean: should tool_use/tool_result be suppressed for this window?

    Composes the per-window override with the global ``config.hide_tool_calls``
    default. Per-window ``shown``/``hidden`` always wins; ``default`` falls
    through to the global setting.
    """
    return _tool_state.is_tool_calls_hidden(window_id)


def get_session_id_for_window(window_id: str) -> str | None:
    """Look up session_id for a window from window_states."""
    return window_store.get_session_id_for_window(window_id)


def window_count() -> int:
    """Number of tracked windows."""
    return len(window_store.window_states)


def iter_window_ids() -> list[str]:
    """All tracked window IDs."""
    return list(window_store.window_states.keys())
