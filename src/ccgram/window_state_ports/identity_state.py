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


def clear_transcript_path(window_id: str) -> None:
    """Clear the persisted transcript path for a window.

    Used by provider-switch coordination when the new provider has a
    chat-first command path (shell-like) and the old transcript no
    longer applies. Matches prior behavior: in-memory mutation only;
    save scheduling rides along with the surrounding provider change.
    """
    state = window_store.window_states.get(window_id)
    if state is None or not state.transcript_path:
        return
    state.transcript_path = ""


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
    "set_window_approval_mode",
]
