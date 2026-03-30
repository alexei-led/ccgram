"""Polling strategy classes for terminal status monitoring.

Decomposes the polling subsystem state into focused, independently testable
strategy classes:
  - TerminalStatusStrategy: pyte screen buffer state, RC debounce, content-hash cache
  - InteractiveUIStrategy: pane alert hash state for deduplication
  - TopicLifecycleStrategy: autoclose timers, dead notification tracking, probe failures

Each strategy owns its state and state management methods. Domain-specific
async functions (which depend on tmux, Telegram, providers, etc.) remain in
polling_coordinator.py and use these strategies for state access. This separation
enables independent testing of state logic without mocking external deps.
"""

import structlog
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..providers.base import StatusUpdate

if TYPE_CHECKING:
    from ..screen_buffer import ScreenBuffer

logger = structlog.get_logger()

# ── Constants ───────────────────────────────────────────────────────────

# Transcript activity heuristic threshold (seconds).
_ACTIVITY_THRESHOLD = 10.0

# Startup timeout before transitioning to idle (seconds).
_STARTUP_TIMEOUT = 30.0

# RC debounce: require RC absent for this long before clearing badge.
_RC_DEBOUNCE_SECONDS = 3.0

# Consecutive topic probe failure threshold.
_MAX_PROBE_FAILURES = 3

# Typing indicator throttle interval (seconds).
_TYPING_INTERVAL = 4.0

# Pane count cache TTL for multi-pane scanning (seconds).
_PANE_COUNT_TTL = 5.0

# Shell commands indicating agent has exited.
SHELL_COMMANDS = frozenset({"bash", "zsh", "fish", "sh", "dash", "tcsh", "csh", "ksh"})


def is_shell_prompt(pane_current_command: str) -> bool:
    """Check if the pane is running a shell (agent has exited)."""
    cmd = pane_current_command.strip().rsplit("/", 1)[-1]
    return cmd in SHELL_COMMANDS


# ── Shared result dataclass ─────────────────────────────────────────────


@dataclass
class PollResult:
    """Shared result returned by polling strategies."""

    status_text: str | None = None
    emoji_state: str | None = None
    is_interactive: bool = False
    skip_status: bool = False


# ── Per-window / per-topic state ────────────────────────────────────────


@dataclass
class WindowPollState:
    """Per-window polling state, keyed by window_id."""

    has_seen_status: bool = False
    startup_time: float | None = None
    probe_failures: int = 0
    screen_buffer: "ScreenBuffer | None" = field(default=None, repr=False)
    pane_count_cache: tuple[int, float] | None = None
    unbound_timer: float | None = None
    last_pane_hash: int = 0
    last_pyte_result: StatusUpdate | None = field(default=None, repr=False)
    last_rendered_text: str | None = None
    rc_active: bool = False
    rc_off_since: float | None = None
    last_rc_detected: bool = False


@dataclass
class TopicPollState:
    """Per-topic polling state, keyed by (user_id, thread_id)."""

    autoclose: tuple[str, float] | None = None
    last_typing_sent: float | None = None


# ── TerminalStatusStrategy ──────────────────────────────────────────────


class TerminalStatusStrategy:
    """Pyte screen buffer state, RC debounce, content-hash cache.

    Owns WindowPollState instances keyed by window_id. Domain-specific parsing
    functions (parse_with_pyte, check_transcript_activity) live in polling_coordinator.py
    and access state through this strategy.
    """

    def __init__(self) -> None:
        self._states: dict[str, WindowPollState] = {}

    def get_state(self, window_id: str) -> WindowPollState:
        """Get or create WindowPollState for a window."""
        return self._states.setdefault(window_id, WindowPollState())

    def clear_state(self, window_id: str) -> None:
        """Remove all polling state for a window."""
        self._states.pop(window_id, None)

    def clear_screen_buffer(self, window_id: str) -> None:
        """Remove a window's ScreenBuffer, caches, and pyte results."""
        ws = self._states.get(window_id)
        if ws:
            ws.screen_buffer = None
            ws.pane_count_cache = None
            ws.last_pane_hash = 0
            ws.last_pyte_result = None
            ws.last_rendered_text = None

    def reset_screen_buffer_state(self) -> None:
        """Reset all ScreenBuffers and caches (for testing)."""
        for ws in self._states.values():
            ws.screen_buffer = None
            ws.pane_count_cache = None
            ws.last_pane_hash = 0
            ws.last_pyte_result = None
            ws.last_rendered_text = None
            ws.rc_active = False
            ws.rc_off_since = None

    def clear_unbound_timers(self, bound_ids: set[str], live_ids: set[str]) -> None:
        """Clear unbound timers for windows that are now bound or gone."""
        for wid, ws in list(self._states.items()):
            if ws.unbound_timer is not None and (
                wid in bound_ids or wid not in live_ids
            ):
                ws.unbound_timer = None

    def get_expired_unbound(self, now: float, timeout: float) -> list[str]:
        """Return window IDs whose unbound timer has expired."""
        return [
            wid
            for wid, ws in self._states.items()
            if ws.unbound_timer is not None and now - ws.unbound_timer >= timeout
        ]

    def get_orphaned_window_ids(
        self, live_ids: set[str], bound_ids: set[str]
    ) -> list[str]:
        """Return window IDs that are neither live nor bound."""
        return [
            wid for wid in self._states if wid not in live_ids and wid not in bound_ids
        ]

    def is_rc_active(self, window_id: str) -> bool:
        """Check whether Remote Control is currently active for a window."""
        ws = self._states.get(window_id)
        return ws.rc_active if ws else False

    def update_rc_state(self, ws: WindowPollState, rc_detected: bool) -> None:
        """Update Remote Control state with debounce on removal."""
        if rc_detected:
            ws.rc_active = True
            ws.rc_off_since = None
        elif ws.rc_active:
            now = time.monotonic()
            if ws.rc_off_since is None:
                ws.rc_off_since = now
            elif now - ws.rc_off_since >= _RC_DEBOUNCE_SECONDS:
                ws.rc_active = False
                ws.rc_off_since = None

    def get_screen_buffer(
        self, window_id: str, columns: int, rows: int
    ) -> "ScreenBuffer":
        """Get or create a ScreenBuffer for a window, resizing if needed."""
        from ..screen_buffer import ScreenBuffer

        ws = self.get_state(window_id)
        buf = ws.screen_buffer
        if buf is None or not isinstance(buf, ScreenBuffer):
            buf = ScreenBuffer(columns=columns, rows=rows)
            ws.screen_buffer = buf
        elif buf.columns != columns or buf.rows != rows:
            buf.resize(columns, rows)
        else:
            buf.reset()
        return buf

    def parse_with_pyte(
        self,
        window_id: str,
        pane_text: str,
        columns: int = 0,
        rows: int = 0,
    ) -> StatusUpdate | None:
        """Parse terminal via pyte screen buffer for status and interactive UI.

        Content-hash optimization: unchanged pane content returns cached result
        without re-parsing.
        """
        from ..terminal_parser import (
            detect_remote_control,
            format_status_display,
            parse_from_screen,
            parse_status_from_screen,
        )

        if (
            not isinstance(columns, int)
            or not isinstance(rows, int)
            or columns <= 0
            or rows <= 0
        ):
            columns, rows = 200, 50

        ws = self.get_state(window_id)
        content_hash = hash((pane_text, columns, rows))
        if (
            content_hash == ws.last_pane_hash
            and ws.last_pane_hash != 0
            and (ws.last_pyte_result is None or not ws.last_pyte_result.is_interactive)
        ):
            self.update_rc_state(ws, ws.last_rc_detected)
            return ws.last_pyte_result

        buf = self.get_screen_buffer(window_id, columns, rows)
        buf.feed(pane_text)
        ws.last_rendered_text = buf.rendered_text

        rc_detected = detect_remote_control(buf.display)
        ws.last_rc_detected = rc_detected
        self.update_rc_state(ws, rc_detected)

        interactive = parse_from_screen(buf)
        if interactive:
            result = StatusUpdate(
                raw_text=interactive.content,
                display_label=interactive.name,
                is_interactive=True,
                ui_type=interactive.name,
            )
            ws.last_pane_hash = content_hash
            ws.last_pyte_result = result
            return result

        raw_status = parse_status_from_screen(buf)
        if raw_status:
            result = StatusUpdate(
                raw_text=raw_status,
                display_label=format_status_display(raw_status),
            )
            ws.last_pane_hash = content_hash
            ws.last_pyte_result = result
            return result

        ws.last_pane_hash = content_hash
        ws.last_pyte_result = None
        return None


# ── InteractiveUIStrategy ───────────────────────────────────────────────


class InteractiveUIStrategy:
    """Pane alert hash state for multi-pane interactive prompt deduplication.

    Async scanning functions (scan_window_panes, check_interactive_only) remain
    in polling_coordinator.py and access state through this strategy.
    """

    def __init__(self, terminal: TerminalStatusStrategy) -> None:
        self._terminal = terminal
        self._pane_alert_hashes: dict[str, tuple[str, float, str]] = {}

    def has_pane_alert(self, pane_id: str) -> bool:
        """Check whether a pane currently has an active alert."""
        return pane_id in self._pane_alert_hashes

    def get_pane_alert(self, pane_id: str) -> tuple[str, float, str] | None:
        """Return pane alert tuple (hash, timestamp, window_id), or None."""
        return self._pane_alert_hashes.get(pane_id)

    def set_pane_alert(
        self, pane_id: str, content_hash: str, timestamp: float, window_id: str
    ) -> None:
        """Record a pane alert entry."""
        self._pane_alert_hashes[pane_id] = (content_hash, timestamp, window_id)

    def remove_pane_alert(self, pane_id: str) -> None:
        """Remove a single pane alert entry."""
        self._pane_alert_hashes.pop(pane_id, None)

    def prune_stale_pane_alerts(self, window_id: str, live_pane_ids: set[str]) -> None:
        """Remove alerts for panes of a window that no longer exist."""
        stale = [
            pid
            for pid, v in self._pane_alert_hashes.items()
            if v[2] == window_id and pid not in live_pane_ids
        ]
        for pid in stale:
            self._pane_alert_hashes.pop(pid, None)

    def clear_pane_alerts(self, window_id: str) -> None:
        """Remove pane alert state for a specific window only."""
        stale = [pid for pid, v in self._pane_alert_hashes.items() if v[2] == window_id]
        for pid in stale:
            self._pane_alert_hashes.pop(pid, None)

    def clear_all_alerts(self) -> None:
        """Clear all pane alert state (for testing)."""
        self._pane_alert_hashes.clear()


# ── TopicLifecycleStrategy ──────────────────────────────────────────────


class TopicLifecycleStrategy:
    """Autoclose timers, dead notification tracking, probe failure state.

    Async lifecycle functions (check_autoclose_timers, handle_dead_window_notification,
    probe_topic_existence, etc.) remain in polling_coordinator.py and access state through
    this strategy.
    """

    def __init__(self, terminal: TerminalStatusStrategy) -> None:
        self._terminal = terminal
        self._states: dict[tuple[int, int], TopicPollState] = {}
        self._dead_notified: set[tuple[int, int, str]] = set()

    def get_state(self, user_id: int, thread_id: int) -> TopicPollState:
        """Get or create TopicPollState for a topic."""
        return self._states.setdefault((user_id, thread_id), TopicPollState())

    def is_dead_notified(self, user_id: int, thread_id: int, window_id: str) -> bool:
        """Check if a dead notification was already sent for this topic/window."""
        return (user_id, thread_id, window_id) in self._dead_notified

    def mark_dead_notified(self, user_id: int, thread_id: int, window_id: str) -> None:
        """Record that a dead notification was sent."""
        self._dead_notified.add((user_id, thread_id, window_id))

    def iter_autoclose_timers(self) -> list[tuple[int, int, TopicPollState]]:
        """Return list of (user_id, thread_id, state) for topics with state."""
        return [(uid, tid, ts) for (uid, tid), ts in self._states.items()]

    def clear_state(self, user_id: int, thread_id: int) -> None:
        """Remove all polling state for a topic."""
        self._states.pop((user_id, thread_id), None)

    def start_autoclose_timer(
        self, user_id: int, thread_id: int, state: str, now: float
    ) -> None:
        """Start or maintain an autoclose timer for done/dead state."""
        ts = self.get_state(user_id, thread_id)
        existing = ts.autoclose
        if existing is None or existing[0] != state:
            ts.autoclose = (state, now)

    def clear_autoclose_timer(self, user_id: int, thread_id: int) -> None:
        """Clear autoclose timer for a topic (on cleanup or when active)."""
        ts = self._states.get((user_id, thread_id))
        if ts:
            ts.autoclose = None

    def reset_autoclose_state(self) -> None:
        """Reset all autoclose tracking (for testing)."""
        for ts in self._states.values():
            ts.autoclose = None
        for ws in self._terminal._states.values():
            ws.unbound_timer = None

    def clear_dead_notification(self, user_id: int, thread_id: int) -> None:
        """Remove dead notification tracking for a topic."""
        self._dead_notified.difference_update(
            {k for k in self._dead_notified if k[0] == user_id and k[1] == thread_id}
        )

    def reset_dead_notification_state(self) -> None:
        """Reset all dead notification tracking (for testing)."""
        self._dead_notified.clear()

    def clear_probe_failures(self, window_id: str) -> None:
        """Reset probe failure counter for a window."""
        ws = self._terminal._states.get(window_id)
        if ws:
            ws.probe_failures = 0

    def reset_probe_failures_state(self) -> None:
        """Reset all probe failure tracking (for testing)."""
        for ws in self._terminal._states.values():
            ws.probe_failures = 0

    def clear_typing_state(self, user_id: int, thread_id: int) -> None:
        """Clear typing indicator throttle for a topic."""
        ts = self._states.get((user_id, thread_id))
        if ts:
            ts.last_typing_sent = None

    def reset_typing_state(self) -> None:
        """Reset all typing indicator tracking (for testing)."""
        for ts in self._states.values():
            ts.last_typing_sent = None

    def clear_seen_status(self, window_id: str) -> None:
        """Clear startup status tracking for a window."""
        ws = self._terminal._states.get(window_id)
        if ws:
            ws.has_seen_status = False
            ws.startup_time = None

    def reset_seen_status_state(self) -> None:
        """Reset all startup status tracking (for testing)."""
        for ws in self._terminal._states.values():
            ws.has_seen_status = False
            ws.startup_time = None

    def record_probe_failure(self, window_id: str) -> int:
        """Increment probe failure counter; log once when threshold is reached."""
        ws = self._terminal.get_state(window_id)
        ws.probe_failures += 1
        count = ws.probe_failures
        if count == _MAX_PROBE_FAILURES:
            logger.info(
                "Suspending topic probe for %s after %d consecutive failures",
                window_id,
                count,
            )
        return count


# ── Module-level strategy singletons ────────────────────────────────────

terminal_strategy = TerminalStatusStrategy()
interactive_strategy = InteractiveUIStrategy(terminal_strategy)
lifecycle_strategy = TopicLifecycleStrategy(terminal_strategy)


# ── Module-level convenience functions ────────────────────────────────
# Thin delegates so consumers don't need to know about strategy internals.


def clear_window_poll_state(window_id: str) -> None:
    """Remove all polling state for a window."""
    terminal_strategy.clear_state(window_id)


def clear_screen_buffer(window_id: str) -> None:
    """Remove a window's ScreenBuffer, pane count cache, and pyte cache."""
    terminal_strategy.clear_screen_buffer(window_id)


def reset_screen_buffer_state() -> None:
    """Reset all ScreenBuffers and caches (for testing)."""
    terminal_strategy.reset_screen_buffer_state()
    interactive_strategy.clear_all_alerts()


def is_rc_active(window_id: str) -> bool:
    """Check whether Remote Control is currently active for a window."""
    return terminal_strategy.is_rc_active(window_id)


def clear_topic_poll_state(user_id: int, thread_id: int) -> None:
    """Remove all polling state for a topic."""
    lifecycle_strategy.clear_state(user_id, thread_id)


def clear_autoclose_timer(user_id: int, thread_id: int) -> None:
    """Remove autoclose timer for a topic (called on cleanup)."""
    lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)


def reset_autoclose_state() -> None:
    """Reset all autoclose tracking (for testing)."""
    lifecycle_strategy.reset_autoclose_state()


def clear_dead_notification(user_id: int, thread_id: int) -> None:
    """Remove dead notification tracking for a topic (called on cleanup)."""
    lifecycle_strategy.clear_dead_notification(user_id, thread_id)


def reset_dead_notification_state() -> None:
    """Reset all dead notification tracking (for testing)."""
    lifecycle_strategy.reset_dead_notification_state()


def clear_probe_failures(window_id: str) -> None:
    """Reset probe failure counter for a window (e.g. on user activity)."""
    lifecycle_strategy.clear_probe_failures(window_id)


def reset_probe_failures_state() -> None:
    """Reset all probe failure tracking (for testing)."""
    lifecycle_strategy.reset_probe_failures_state()


def clear_typing_state(user_id: int, thread_id: int) -> None:
    """Clear typing indicator throttle for a topic (called on cleanup)."""
    lifecycle_strategy.clear_typing_state(user_id, thread_id)


def clear_seen_status(window_id: str) -> None:
    """Clear startup status tracking for a window (called on cleanup)."""
    lifecycle_strategy.clear_seen_status(window_id)


def reset_seen_status_state() -> None:
    """Reset all startup status tracking (for testing)."""
    lifecycle_strategy.reset_seen_status_state()


def reset_typing_state() -> None:
    """Reset all typing indicator tracking (for testing)."""
    lifecycle_strategy.reset_typing_state()


def has_pane_alert(pane_id: str) -> bool:
    """Check whether a pane currently has an active alert."""
    return interactive_strategy.has_pane_alert(pane_id)


def clear_pane_alerts(window_id: str) -> None:
    """Remove pane alert state for a specific window only."""
    interactive_strategy.clear_pane_alerts(window_id)
