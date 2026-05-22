"""Pure types and constants for the polling subsystem — leaf-level, no I/O.

This module exists so callers (notably ``window_tick.decide``) can depend on
the polling contract without loading the stateful singletons in
``polling_state``. F4's pure-decision-kernel invariant is enforced at the
import level by ``tests/ccgram/handlers/polling/test_polling_types_purity.py``:
importing this module must NOT execute ``polling_state`` top-level code.

Imports are restricted to stdlib + ``ccgram.providers.base.StatusUpdate`` plus
``TYPE_CHECKING``-only references to ``telegram.Bot`` and
``ccgram.screen_buffer.ScreenBuffer``. Anything else is a regression.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ...providers.base import StatusUpdate

if TYPE_CHECKING:
    from telegram import Bot

    from ...screen_buffer import ScreenBuffer

# ── Constants ───────────────────────────────────────────────────────────

# Transcript activity heuristic threshold (seconds).
ACTIVITY_THRESHOLD = 10.0

# Startup timeout before transitioning to idle (seconds).
STARTUP_TIMEOUT = 30.0

# RC debounce: require RC absent for this long before clearing badge.
RC_DEBOUNCE_SECONDS = 3.0

# Consecutive topic probe failure threshold.
MAX_PROBE_FAILURES = 3

# Typing indicator throttle interval (seconds).
TYPING_INTERVAL = 4.0

# Pane count cache TTL for multi-pane scanning (seconds).
PANE_COUNT_TTL = 5.0

# Shell commands indicating agent has exited.
SHELL_COMMANDS = frozenset({"bash", "zsh", "fish", "sh", "dash", "tcsh", "csh", "ksh"})


def is_shell_prompt(pane_current_command: str) -> bool:
    """Check if the pane is running a shell (agent has exited)."""
    cmd = pane_current_command.strip().rsplit("/", 1)[-1]
    return cmd in SHELL_COMMANDS


# ── Per-window / per-topic state dataclasses ────────────────────────────


@dataclass
class WindowPollState:
    """Per-window polling state, keyed by window_id."""

    has_seen_status: bool = False
    startup_time: float | None = None
    probe_failures: int = 0
    screen_buffer: ScreenBuffer | None = field(default=None, repr=False)
    pane_count_cache: tuple[int, float] | None = None
    unbound_timer: float | None = None
    last_pane_hash: int | None = None
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


# ── Observe→Decide→Act types ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TickContext:
    """All inputs to the tick decision — pure data, no I/O.

    Coordinator computes all inputs (including those with side effects like
    is_recently_active) before constructing this context, then passes it to
    the pure decide_tick function.
    """

    window_id: str
    resolved_status_text: str | None  # output of build_status_line; None when no status
    is_shell_prompt: bool  # pane_current_command is a bare shell (agent exited)
    has_seen_status: bool  # at least one status was previously sent for this window
    is_recently_active: bool  # transcript activity within ACTIVITY_THRESHOLD seconds
    startup_time: float | None  # None if no startup grace period is running
    is_dead_window: bool  # tmux window no longer exists
    supports_hook: bool  # provider emits hook events (Claude)


@dataclass(frozen=True, slots=True)
class TickDecision:
    """Output of decide_tick — what effects to apply.

    All fields default to no-op so callers only need to set what they care about.
    """

    send_status: bool = False
    status_text: str | None = None
    transition: Literal["idle", "done", "active", "starting"] | None = None
    show_recovery: bool = False


# ── Pane state types ─────────────────────────────────────────────────────

PaneStateName = Literal["active", "idle", "blocked", "dead"]


@dataclass(frozen=True, slots=True)
class PaneTransition:
    """Per-pane state transition emitted during a scan."""

    pane_id: str
    prev_state: PaneStateName | None
    new_state: PaneStateName
    # Captured at transition time so a dead pane's name is preserved for
    # downstream notifications even after the PaneInfo entry is removed.
    name: str | None = None


# Surfaces an interactive prompt to the user. Wired by window_tick.
BlockedAlertCallback = Callable[["Bot", int, str, int, str], Awaitable[None]]

# Forwards subscribed pane output. Wired by window_tick when a pane is marked
# ``subscribed`` in WindowState.panes; arguments mirror BlockedAlertCallback
# with the freshly-captured pane text appended.
PaneOutputCallback = Callable[["Bot", int, str, int, str, str], Awaitable[None]]
