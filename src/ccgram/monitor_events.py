"""Event data types for the session monitor subsystem.

Dependency-free dataclasses shared between transcript_reader, session_monitor,
and handler modules. Keeping these in a dedicated module breaks the import
cycles that arise when transcript_reader and session_monitor each import the
other to access these types.

All three types are re-exported from session_monitor for backward-compatible
imports — external code should continue importing from session_monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SessionInfo:
    """Information about a Claude Code session file."""

    session_id: str
    file_path: Path


@dataclass
class NewMessage:
    """A new message detected by the monitor."""

    session_id: str
    text: str
    is_complete: bool
    content_type: str = "text"  # "text" or "thinking"
    phase: str | None = None
    tool_use_id: str | None = None
    role: str = "assistant"  # "user" or "assistant"
    tool_name: str | None = None  # For tool_use messages, the tool name
    provider_name: str = ""  # Transcript provider for queue provenance


@dataclass
class NewWindowEvent:
    """A new tmux window detected via session_map changes."""

    window_id: str
    session_id: str
    window_name: str
    cwd: str
