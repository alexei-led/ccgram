"""Monitor state persistence — tracks byte offsets for each session.

Persists TrackedSession records (session_id, file_path, last_byte_offset)
to ~/.ccgram/monitor_state.json so the session monitor can resume
incremental reading after restarts without re-sending old messages.

Key classes: MonitorState, TrackedSession.
"""

import json
import structlog
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import atomic_write_json

logger = structlog.get_logger()


@dataclass
class TrackedSession:
    """State for a tracked Claude Code session.

    ``last_byte_offset`` is the *delivered* watermark (persisted): entries
    before it were handed to the message queue when the queue was fully
    drained, so a restart resumes from there and never re-sends them.
    ``parsed_offset`` is the in-memory parse position: entries between the
    watermark and it are parsed and queued but not yet confirmed delivered
    (TASK-5 at-least-once delivery).
    """

    session_id: str
    file_path: str  # Path to .jsonl file
    last_byte_offset: int = 0  # Delivered watermark (persisted)
    parsed_offset: int = -1  # In-memory parse position (-1 = follow watermark)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization (delivered watermark only)."""
        d = asdict(self)
        d.pop("parsed_offset", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackedSession":
        """Create from dict."""
        return cls(
            session_id=data.get("session_id", ""),
            file_path=data.get("file_path", ""),
            last_byte_offset=data.get("last_byte_offset", 0),
        )


@dataclass
class MonitorState:
    """Persistent state for the session monitor.

    Stores tracking information for all monitored sessions
    and the events.jsonl byte offset to prevent replaying
    historical hook events after restarts.
    """

    state_file: Path
    tracked_sessions: dict[str, TrackedSession] = field(default_factory=dict)
    events_offset: int = 0
    _dirty: bool = field(default=False, repr=False)

    def load(self) -> None:
        """Load state from file."""
        if not self.state_file.exists():
            logger.debug("State file does not exist: %s", self.state_file)
            return

        try:
            data = json.loads(self.state_file.read_text())
            sessions = data.get("tracked_sessions", {})
            self.tracked_sessions = {
                k: TrackedSession.from_dict(v) for k, v in sessions.items()
            }
            self.events_offset = data.get("events_offset", 0)
            logger.info(
                "Loaded %d tracked sessions from state", len(self.tracked_sessions)
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load state file: %s", e)
            self.tracked_sessions = {}

    def save(self) -> None:
        """Save state to file atomically."""
        data = {
            "tracked_sessions": {
                k: v.to_dict() for k, v in self.tracked_sessions.items()
            },
            "events_offset": self.events_offset,
        }

        try:
            atomic_write_json(self.state_file, data)
            self._dirty = False
        except OSError:
            logger.exception("Failed to save state file")

    def get_session(self, session_id: str) -> TrackedSession | None:
        """Get tracked session by ID."""
        return self.tracked_sessions.get(session_id)

    def update_session(self, session: TrackedSession) -> None:
        """Update or add a tracked session."""
        self.tracked_sessions[session.session_id] = session
        self._dirty = True

    def remove_session(self, session_id: str) -> None:
        """Remove a tracked session."""
        if session_id in self.tracked_sessions:
            del self.tracked_sessions[session_id]
            self._dirty = True

    def commit_parsed_offsets(self, session_ids: set[str] | None = None) -> bool:
        """Fold acknowledged in-memory parse positions into the watermark.

        ``session_ids`` is selected by the delivery boundary once every
        transcript-derived task for that session has acknowledged delivery or
        an explicit intentional drop. ``None`` retains the compatibility path
        for callers that have no per-session delivery work.
        """
        advanced = False
        for session in self.tracked_sessions.values():
            if session_ids is not None and session.session_id not in session_ids:
                continue
            if (
                session.parsed_offset >= 0
                and session.parsed_offset != session.last_byte_offset
            ):
                # Also LOWER: a replaced/shrunken transcript clamps the parse
                # position to EOF (no replay, 9c3297b); persisting the clamp
                # keeps the watermark from going stale-high.
                session.last_byte_offset = session.parsed_offset
                advanced = True
                self._dirty = True
        return advanced

    def save_if_dirty(self) -> None:
        """Save state only if it has been modified."""
        if self._dirty:
            self.save()
