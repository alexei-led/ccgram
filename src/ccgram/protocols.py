"""Narrow Protocol interfaces for SessionManager consumers.

Defines typed contracts so handler modules can depend on the specific
slice of SessionManager they actually use, rather than the full class.
Protocols are used at type-check time only — at runtime, handlers still
receive the concrete SessionManager singleton.

Protocols:
  - WindowStateStore: window state, display names, session IDs
  - UserPreferences: notification/approval/batch mode per window
  - SessionResolver: session resolution and message history
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .session import ClaudeSession, WindowState


class WindowStateStore(Protocol):
    """Read/write access to per-window persistent state."""

    def get_window_state(self, window_id: str) -> WindowState: ...
    def get_display_name(self, window_id: str) -> str: ...
    def get_session_id_for_window(self, window_id: str) -> str | None: ...
    def clear_window_session(self, window_id: str) -> None: ...


class UserPreferences(Protocol):
    """Per-window user preference management."""

    def get_notification_mode(self, window_id: str) -> str: ...
    def set_notification_mode(self, window_id: str, mode: str) -> None: ...
    def cycle_notification_mode(self, window_id: str) -> str: ...
    def get_approval_mode(self, window_id: str) -> str: ...
    def set_window_approval_mode(self, window_id: str, mode: str) -> None: ...
    def get_batch_mode(self, window_id: str) -> str: ...
    def cycle_batch_mode(self, window_id: str) -> str: ...


class SessionResolver(Protocol):
    """Window-to-session resolution and transcript access."""

    async def resolve_session_for_window(
        self, window_id: str
    ) -> ClaudeSession | None: ...

    async def get_recent_messages(
        self,
        window_id: str,
        *,
        start_byte: int = 0,
        end_byte: int | None = None,
    ) -> tuple[list[dict], int]: ...
