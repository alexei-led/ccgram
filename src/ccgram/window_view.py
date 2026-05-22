"""Read-only window projection — frozen snapshot for handler reads.

Many handlers only need to read a single field from ``WindowState`` (most
commonly ``cwd``). Importing the full ``SessionManager`` and reaching into
``window_states`` for one scalar couples those handlers to the entire
state shape, so renaming a field cascades through every call site.

``WindowView`` is the read-only contract: a frozen dataclass with the
fields that handlers actually consume. ``SessionManager.view_window``
builds a snapshot at call time. Mutation paths still go through
``window_store`` directly — this projection is for reads only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WindowView:
    """Read-only snapshot of a window's state."""

    window_id: str
    cwd: str
    provider_name: str
    approval_mode: str
    batch_mode: str
    tool_call_visibility: str
    transcript_path: Path | None
    window_name: str
    session_id: str
    external: bool
    origin: str
