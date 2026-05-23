"""Tool-mode feature port — batch mode and tool-call visibility projection.

Reads preserve the existing global-config fallback semantics from
``window_query`` so callers can route through this port without
behavior change. Writes delegate to ``WindowStateStore`` (validation
and save scheduling already implemented there).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import config
from ..window_state_store import (
    BATCH_MODES,
    DEFAULT_BATCH_MODE,
    DEFAULT_TOOL_CALL_VISIBILITY,
    TOOL_CALL_VISIBILITY_MODES,
    window_store,
)


@dataclass(frozen=True, slots=True)
class ToolModeProjection:
    """Read-only snapshot of tool-mode settings for a window."""

    window_id: str
    batch_mode: str
    tool_call_visibility: str
    batch_mode_resolved: str
    tool_calls_hidden_resolved: bool


def _resolve_batch_mode(state_batch_mode: str | None) -> str:
    if state_batch_mode is not None and state_batch_mode in BATCH_MODES:
        return state_batch_mode
    return "ephemeral" if config.ephemeral_tools else DEFAULT_BATCH_MODE


def _resolve_tool_calls_hidden(visibility: str) -> bool:
    if visibility == "hidden":
        return True
    if visibility == "shown":
        return False
    return config.hide_tool_calls


def get_tool_modes(window_id: str) -> ToolModeProjection | None:
    """Tool-mode projection (None if no state tracked)."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return None
    visibility = (
        state.tool_call_visibility
        if state.tool_call_visibility in TOOL_CALL_VISIBILITY_MODES
        else DEFAULT_TOOL_CALL_VISIBILITY
    )
    batch_mode = (
        state.batch_mode if state.batch_mode in BATCH_MODES else DEFAULT_BATCH_MODE
    )
    return ToolModeProjection(
        window_id=window_id,
        batch_mode=batch_mode,
        tool_call_visibility=visibility,
        batch_mode_resolved=_resolve_batch_mode(state.batch_mode),
        tool_calls_hidden_resolved=_resolve_tool_calls_hidden(visibility),
    )


def get_batch_mode(window_id: str) -> str:
    """Batch mode with global-config fallback (matches window_query)."""
    state = window_store.window_states.get(window_id)
    return _resolve_batch_mode(state.batch_mode if state else None)


def is_ephemeral_tools(window_id: str) -> bool:
    """Resolved 'ephemeral' batch mode for a window."""
    return get_batch_mode(window_id) == "ephemeral"


def get_tool_call_visibility(window_id: str) -> str:
    """Raw per-window tool-call visibility (default/shown/hidden)."""
    state = window_store.window_states.get(window_id)
    mode = state.tool_call_visibility if state else DEFAULT_TOOL_CALL_VISIBILITY
    return mode if mode in TOOL_CALL_VISIBILITY_MODES else DEFAULT_TOOL_CALL_VISIBILITY


def is_tool_calls_hidden(window_id: str) -> bool:
    """Resolved hidden-tool-call decision composing per-window + global config."""
    return _resolve_tool_calls_hidden(get_tool_call_visibility(window_id))


def set_batch_mode(window_id: str, mode: str) -> None:
    """Set batch mode. Raises ValueError on unknown mode."""
    window_store.set_batch_mode(window_id, mode)


def cycle_batch_mode(window_id: str) -> str:
    """Cycle batch mode: batched → ephemeral → verbose → batched. Returns new mode."""
    return window_store.cycle_batch_mode(window_id)


def set_tool_call_visibility(window_id: str, mode: str) -> None:
    """Set tool-call visibility. Raises ValueError on unknown mode."""
    window_store.set_tool_call_visibility(window_id, mode)


def cycle_tool_call_visibility(window_id: str) -> str:
    """Cycle tool-call visibility: default → shown → hidden → default. Returns new mode."""
    return window_store.cycle_tool_call_visibility(window_id)


__all__ = [
    "ToolModeProjection",
    "cycle_batch_mode",
    "cycle_tool_call_visibility",
    "get_batch_mode",
    "get_tool_call_visibility",
    "get_tool_modes",
    "is_ephemeral_tools",
    "is_tool_calls_hidden",
    "set_batch_mode",
    "set_tool_call_visibility",
]
