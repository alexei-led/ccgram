"""Identity-state feature port — provider/session/cwd/transcript projection.

Read projections cover provider name, session id, cwd, transcript path,
window name, and approval mode. Provider writes are intentionally
*not* exposed — they require provider-capability resolution and stay on
``SessionManager.set_window_provider``. Approval mode is a simple
enum-validated setter and is exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..window_state_store import (
    APPROVAL_MODES,
    DEFAULT_APPROVAL_MODE,
    window_store,
)


@dataclass(frozen=True, slots=True)
class IdentityProjection:
    """Read-only snapshot of provider/session/cwd identity for a window."""

    window_id: str
    provider_name: str
    session_id: str
    cwd: str
    transcript_path: Path | None
    window_name: str
    approval_mode: str


def get_identity(window_id: str) -> IdentityProjection | None:
    """Frozen identity projection, or None if no state is tracked."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return None
    return IdentityProjection(
        window_id=window_id,
        provider_name=state.provider_name,
        session_id=state.session_id,
        cwd=state.cwd or "",
        transcript_path=(
            Path(state.transcript_path) if state.transcript_path else None
        ),
        window_name=state.window_name,
        approval_mode=(
            state.approval_mode
            if state.approval_mode in APPROVAL_MODES
            else DEFAULT_APPROVAL_MODE
        ),
    )


def get_provider_name(window_id: str) -> str | None:
    """Provider name for a window, or None if untracked."""
    state = window_store.window_states.get(window_id)
    return state.provider_name if state else None


def get_session_id(window_id: str) -> str | None:
    """Non-empty session id for a window, or None."""
    state = window_store.window_states.get(window_id)
    if state is None:
        return None
    sid = state.session_id
    return sid if sid else None


def get_cwd(window_id: str) -> str:
    """CWD for a window, or empty string when untracked."""
    state = window_store.window_states.get(window_id)
    return state.cwd if state else ""


def get_transcript_path(window_id: str) -> str:
    """Raw transcript path string for a window, or empty when untracked."""
    state = window_store.window_states.get(window_id)
    return state.transcript_path if state else ""


def get_window_name(window_id: str) -> str:
    """Display name for a window, or empty when untracked."""
    state = window_store.window_states.get(window_id)
    return state.window_name if state else ""


def get_approval_mode(window_id: str) -> str:
    """Approval mode for a window. Defaults to 'normal'."""
    state = window_store.window_states.get(window_id)
    mode = state.approval_mode if state else DEFAULT_APPROVAL_MODE
    return mode if mode in APPROVAL_MODES else DEFAULT_APPROVAL_MODE


def set_window_approval_mode(window_id: str, mode: str) -> None:
    """Set approval mode. Raises ValueError on unknown mode."""
    window_store.set_window_approval_mode(window_id, mode)


def is_provider_manually_overridden(window_id: str) -> bool:
    """True if the user explicitly chose this window's provider via /agent.

    Auto-detection (`_detect_and_apply_provider`) must skip overridden
    windows. Cleared by `/agent auto`.
    """
    state = window_store.window_states.get(window_id)
    # ``is True`` (not truthy) so a stand-in MagicMock attribute in tests
    # — which would be truthy as a MagicMock instance — doesn't accidentally
    # short-circuit auto-detection.
    return state is not None and state.provider_manual_override is True


def set_provider_manual_override(window_id: str, *, value: bool) -> None:
    """Mark or clear the provider manual-override flag."""
    window_store.set_provider_manual_override(window_id, value=value)


def clear_transcript_path(window_id: str) -> None:
    """Clear the persisted transcript path for a window.

    Used by provider-switch coordination when the new provider has a
    chat-first command path (shell-like) and the old transcript no
    longer applies. Schedules a save so the cleared field persists
    even when called outside a surrounding provider write.
    """
    window_store.clear_transcript_path(window_id)


__all__ = [
    "IdentityProjection",
    "clear_transcript_path",
    "get_approval_mode",
    "get_cwd",
    "get_identity",
    "get_provider_name",
    "get_session_id",
    "get_transcript_path",
    "get_window_name",
    "is_provider_manually_overridden",
    "set_provider_manual_override",
    "set_window_approval_mode",
]
