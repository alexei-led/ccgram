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

from .window_state_ports import identity_state as _identity_state
from .window_state_ports import lifecycle_state as _lifecycle_state
from .window_state_ports import tool_state as _tool_state
from .window_state_store import window_store
from .window_view import WindowView


def view_window(window_id: str) -> WindowView | None:
    """Read-only snapshot of a window's state, or None if no state exists."""
    identity = _identity_state.get_identity(window_id)
    if identity is None:
        return None
    # All three ports read from the same window_states dict, so once
    # identity is non-None the other projections are too. The explicit
    # check survives `python -O` (asserts stripped) and avoids an
    # AttributeError if the invariant ever breaks.
    lifecycle = _lifecycle_state.get_lifecycle(window_id)
    tools = _tool_state.get_tool_modes(window_id)
    if lifecycle is None or tools is None:
        return None
    return WindowView(
        window_id=window_id,
        cwd=identity.cwd,
        provider_name=identity.provider_name,
        approval_mode=identity.approval_mode,
        batch_mode=tools.batch_mode,
        tool_call_visibility=tools.tool_call_visibility,
        transcript_path=identity.transcript_path,
        window_name=identity.window_name,
        session_id=identity.session_id,
        external=lifecycle.external,
        origin=lifecycle.origin,
    )


def get_window_provider(window_id: str) -> str | None:
    """Return the provider name for a window, or None if not set."""
    return _identity_state.get_provider_name(window_id)


def get_approval_mode(window_id: str) -> str:
    """Get approval mode for a window (default: 'normal')."""
    return _identity_state.get_approval_mode(window_id)


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
