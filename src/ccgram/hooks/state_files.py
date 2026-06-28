"""Versioned parse/serialize contracts for hook-written state files.

Owns the schema for ``events.jsonl`` (one JSONL line per hook event) and
``session_map.json`` (dict of window-key → session-info).  Both files cross
the boundary between the short-lived hook process and the long-lived bot
process, so their shapes must be stable and backward-compatible.

Design decisions
----------------
* Version is per-record (field ``schema_version``), not a top-level file
  field, so incremental append-only writes remain safe.
* Versionless records (no ``schema_version`` key) are accepted as legacy
  v1 — byte-identical to the previous unversioned writes.
* Records with an unknown future version are skipped with a logged reason
  (controlled degradation, not a crash).
* Records missing required fields are rejected with ``StateFileValidationError``.
* I/O (file locking, corrupt-file backup, aiofiles) stays in the callers
  (``hook.py``, ``session_map.py``, ``event_reader.py``); this module is
  pure parse/serialize.

Import constraints
------------------
This module is imported by ``hook.py``, which runs inside agent panes
without bot configuration.  Keep imports to stdlib only (no ``config``,
no ``providers``, no ``utils``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

EVENTS_SCHEMA_VERSION: int = 1
SESSION_MAP_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StateFileValidationError(ValueError):
    """Raised when a state-file record fails validation.

    Callers should log the message at debug/warning level, skip the record,
    and advance any byte offset so the invalid line is not re-read.
    """


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventLogRecord:
    """One parsed record from ``events.jsonl``."""

    schema_version: int
    ts: float
    event: str
    window_key: str
    session_id: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SessionMapEntry:
    """One parsed entry from ``session_map.json``."""

    schema_version: int
    session_id: str
    cwd: str
    window_name: str
    transcript_path: str
    provider_name: str


# ---------------------------------------------------------------------------
# Event-log helpers
# ---------------------------------------------------------------------------

_REQUIRED_EVENT_FIELDS: tuple[str, ...] = ("event", "window_key", "session_id")


def parse_event_record(raw: dict[str, Any]) -> EventLogRecord:
    """Parse one ``events.jsonl`` record dict into an ``EventLogRecord``.

    Accepts both legacy (versionless) records and explicit v1 records.
    Raises ``StateFileValidationError`` for:
    * non-dict input (e.g. a JSON array or scalar on the line)
    * missing required fields (``event``, ``window_key``, ``session_id``)
    * an unsupported schema_version (> ``EVENTS_SCHEMA_VERSION``)
    """
    if not isinstance(raw, dict):
        raise StateFileValidationError(
            f"Event record must be a JSON object, got {type(raw).__name__!r}"
        )
    version = raw.get("schema_version")
    if version is None:
        # Legacy record — treat as v1.
        version = 1
    elif not isinstance(version, int) or version > EVENTS_SCHEMA_VERSION:
        raise StateFileValidationError(
            f"Unsupported events schema_version {version!r}; "
            f"expected <= {EVENTS_SCHEMA_VERSION}"
        )

    missing = [f for f in _REQUIRED_EVENT_FIELDS if not raw.get(f)]
    if missing:
        raise StateFileValidationError(
            f"Event record missing required fields: {missing}"
        )

    return EventLogRecord(
        schema_version=version,
        ts=raw.get("ts") or 0.0,
        event=raw["event"],
        window_key=raw["window_key"],
        session_id=raw["session_id"],
        data=raw.get("data") or {},
    )


def serialize_event_record(
    event_type: str,
    session_id: str,
    window_key: str,
    data: dict[str, Any],
    *,
    ts: float | None = None,
) -> dict[str, Any]:
    """Build a v1 event record dict ready for ``json.dumps``.

    The ``separators=(",", ":")`` compact form must be applied by the caller
    (``hook._write_event``) to preserve the existing byte layout.
    """
    return {
        "schema_version": EVENTS_SCHEMA_VERSION,
        "ts": ts if ts is not None else time.time(),
        "event": event_type,
        "window_key": window_key,
        "session_id": session_id,
        "data": data,
    }


# ---------------------------------------------------------------------------
# Session-map helpers
# ---------------------------------------------------------------------------

_REQUIRED_SESSION_MAP_FIELDS: tuple[str, ...] = ("session_id",)


def parse_session_map_entry(raw: dict[str, Any]) -> SessionMapEntry:
    """Parse one ``session_map.json`` entry dict into a ``SessionMapEntry``.

    Accepts legacy (versionless) entries and explicit v1 entries.
    Raises ``StateFileValidationError`` for:
    * non-dict input
    * missing ``session_id``
    * an unsupported schema_version (> ``SESSION_MAP_SCHEMA_VERSION``)
    """
    if not isinstance(raw, dict):
        raise StateFileValidationError(
            f"Session map entry must be a JSON object, got {type(raw).__name__!r}"
        )
    version = raw.get("schema_version")
    if version is None:
        version = 1
    elif not isinstance(version, int) or version > SESSION_MAP_SCHEMA_VERSION:
        raise StateFileValidationError(
            f"Unsupported session_map schema_version {version!r}; "
            f"expected <= {SESSION_MAP_SCHEMA_VERSION}"
        )

    missing = [f for f in _REQUIRED_SESSION_MAP_FIELDS if not raw.get(f)]
    if missing:
        raise StateFileValidationError(
            f"Session map entry missing required fields: {missing}"
        )

    return SessionMapEntry(
        schema_version=version,
        session_id=raw["session_id"],
        cwd=raw.get("cwd", ""),
        window_name=raw.get("window_name", ""),
        transcript_path=raw.get("transcript_path", ""),
        provider_name=raw.get("provider_name", ""),
    )


def serialize_session_map_entry(
    session_id: str,
    cwd: str,
    window_name: str,
    transcript_path: str,
    provider_name: str,
) -> dict[str, Any]:
    """Build a v1 session-map entry dict ready for JSON serialization."""
    return {
        "schema_version": SESSION_MAP_SCHEMA_VERSION,
        "session_id": session_id,
        "cwd": cwd,
        "window_name": window_name,
        "transcript_path": transcript_path,
        "provider_name": provider_name,
    }
