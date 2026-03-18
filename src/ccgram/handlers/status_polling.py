"""Terminal status line polling for thread-bound windows.

Provides background polling of terminal status lines for all active users:
  - Detects Claude Code status (working, waiting, done, etc.)
  - Detects interactive UIs (permission prompts) not triggered via JSONL
  - Updates status messages in Telegram
  - Polls thread_bindings (each topic = one window)
  - Detects Claude process exit (pane command reverts to shell)
  - Auto-closes stale topics after configurable timeout
  - Auto-kills unbound windows (topic closed, window kept alive) after TTL
  - Periodically probes topic existence via unpin_all_forum_topic_messages
    (silent no-op when no pins); cleans up deleted topics (kills tmux window
    + unbinds thread). Consecutive probe failures are tracked per window;
    after _MAX_PROBE_FAILURES timeouts, probing is suspended until user activity

Key components:
  - STATUS_POLL_INTERVAL: Polling frequency (1 second)
  - TOPIC_CHECK_INTERVAL: Topic existence probe frequency (60 seconds)
  - status_poll_loop: Background polling task
  - update_status_message: Poll and enqueue status updates
  - is_shell_prompt: Detect Claude exit (shell resumed in pane)
  - clear_dead_notification: Clear dead window notification tracking
  - Proactive recovery: sends recovery keyboard when a window dies
  - Auto-close: closes topics stuck in done/dead state
"""

import asyncio
import contextlib
import structlog
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from telegram import Bot

if TYPE_CHECKING:
    from ..screen_buffer import ScreenBuffer
    from ..tmux_manager import TmuxWindow
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError

from ..config import config
from ..providers import (
    detect_provider_from_command,
    detect_provider_from_runtime,
    get_provider_for_window,
    should_probe_pane_title_for_provider_detection,
)
from ..providers.base import StatusUpdate
from ..session import session_manager
from ..window_resolver import is_foreign_window
from ..session_monitor import get_active_monitor
from ..tmux_manager import tmux_manager
from ..utils import log_throttle_sweep, log_throttled
from .interactive_ui import (
    clear_interactive_msg,
    get_interactive_window,
    handle_interactive_ui,
)
from .cleanup import clear_topic_state
from .message_queue import (
    clear_tool_msg_ids_for_topic,
    enqueue_status_update,
    get_message_queue,
)
from .message_sender import rate_limit_send_message
from .recovery_callbacks import build_recovery_keyboard
from .topic_emoji import update_topic_emoji

# Top-level loop resilience: catch any error to keep polling alive
_LoopError = (TelegramError, OSError, RuntimeError, ValueError)

# Exponential backoff bounds for loop errors (seconds)
_BACKOFF_MIN = 2.0
_BACKOFF_MAX = 30.0

logger = structlog.get_logger()

# Status polling interval
STATUS_POLL_INTERVAL = 1.0  # seconds - faster response (rate limiting at send layer)

# Topic existence probe interval
TOPIC_CHECK_INTERVAL = 60.0  # seconds

# Shell commands indicating Claude has exited and the shell prompt is back
SHELL_COMMANDS = frozenset({"bash", "zsh", "fish", "sh", "dash", "tcsh", "csh", "ksh"})

# Consecutive topic probe failure threshold. After _MAX_PROBE_FAILURES
# consecutive timeouts, probing is suspended to stop log spam and useless API calls.
_MAX_PROBE_FAILURES = 3

# Typing indicator throttle interval.
# Telegram typing action expires after ~5s; we re-send every 4s.
_TYPING_INTERVAL = 4.0

# Transcript activity heuristic: if transcript was written to within this many
# seconds, treat the window as active even without a terminal status signal.
_ACTIVITY_THRESHOLD = 10.0

# Startup timeout: after this many seconds without any status or transcript
# activity, transition from "starting up" to idle instead of staying green forever.
_STARTUP_TIMEOUT = 30.0


# ── Consolidated per-window and per-topic polling state ────────────────


@dataclass
class WindowPollState:
    """Per-window polling state, keyed by window_id."""

    has_seen_status: bool = False
    startup_time: float | None = None
    probe_failures: int = 0
    screen_buffer: ScreenBuffer | None = field(default=None, repr=False)
    pane_count_cache: tuple[int, float] | None = None
    unbound_timer: float | None = None


@dataclass
class TopicPollState:
    """Per-topic polling state, keyed by (user_id, thread_id)."""

    autoclose: tuple[str, float] | None = None
    last_typing_sent: float | None = None


_window_poll_state: dict[str, WindowPollState] = {}
_topic_poll_state: dict[tuple[int, int], TopicPollState] = {}

# These stay as separate module-level state (different key patterns):
_dead_notified: set[tuple[int, int, str]] = set()
_pane_alert_hashes: dict[str, tuple[str, float, str]] = {}


def _get_window_state(window_id: str) -> WindowPollState:
    """Get or create WindowPollState for a window."""
    return _window_poll_state.setdefault(window_id, WindowPollState())


def _get_topic_state(user_id: int, thread_id: int) -> TopicPollState:
    """Get or create TopicPollState for a topic."""
    return _topic_poll_state.setdefault((user_id, thread_id), TopicPollState())


def _get_screen_buffer(window_id: str, columns: int, rows: int) -> ScreenBuffer:
    """Get or create a ScreenBuffer for a window, resizing if needed."""
    from ..screen_buffer import ScreenBuffer

    ws = _get_window_state(window_id)
    buf = ws.screen_buffer
    if (
        buf is None
        or not isinstance(buf, ScreenBuffer)
        or buf.columns != columns
        or buf.rows != rows
    ):
        buf = ScreenBuffer(columns=columns, rows=rows)
        ws.screen_buffer = buf
    else:
        buf.reset()
    return buf


def clear_screen_buffer(window_id: str) -> None:
    """Remove a window's ScreenBuffer and pane count cache (called on cleanup)."""
    ws = _window_poll_state.get(window_id)
    if ws:
        ws.screen_buffer = None
        ws.pane_count_cache = None


def clear_window_poll_state(window_id: str) -> None:
    """Remove all polling state for a window."""
    _window_poll_state.pop(window_id, None)


def clear_topic_poll_state(user_id: int, thread_id: int) -> None:
    """Remove all polling state for a topic."""
    _topic_poll_state.pop((user_id, thread_id), None)


def reset_screen_buffer_state() -> None:
    """Reset all ScreenBuffers and caches (for testing)."""
    for ws in _window_poll_state.values():
        ws.screen_buffer = None
        ws.pane_count_cache = None
    _pane_alert_hashes.clear()


def is_shell_prompt(pane_current_command: str) -> bool:
    """Check if the pane is running a shell (Claude has exited)."""
    cmd = pane_current_command.strip().rsplit("/", 1)[-1]
    return cmd in SHELL_COMMANDS


async def _send_typing_throttled(bot: Bot, user_id: int, thread_id: int | None) -> None:
    """Send typing indicator if enough time has elapsed since the last one."""
    if thread_id is None:
        return
    ts = _get_topic_state(user_id, thread_id)
    now = time.monotonic()
    if now - (ts.last_typing_sent or 0.0) < _TYPING_INTERVAL:
        return
    ts.last_typing_sent = now
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    with contextlib.suppress(TelegramError):
        await bot.send_chat_action(
            chat_id=chat_id,
            message_thread_id=thread_id,
            action=ChatAction.TYPING,
        )


def clear_autoclose_timer(user_id: int, thread_id: int) -> None:
    """Remove autoclose timer for a topic (called on cleanup)."""
    ts = _topic_poll_state.get((user_id, thread_id))
    if ts:
        ts.autoclose = None


def reset_autoclose_state() -> None:
    """Reset all autoclose tracking (for testing)."""
    for ts in _topic_poll_state.values():
        ts.autoclose = None
    for ws in _window_poll_state.values():
        ws.unbound_timer = None


def clear_dead_notification(user_id: int, thread_id: int) -> None:
    """Remove dead notification tracking for a topic (called on cleanup)."""
    _dead_notified.difference_update(
        {k for k in _dead_notified if k[0] == user_id and k[1] == thread_id}
    )


def reset_dead_notification_state() -> None:
    """Reset all dead notification tracking (for testing)."""
    _dead_notified.clear()


def clear_probe_failures(window_id: str) -> None:
    """Reset probe failure counter for a window (e.g. on user activity)."""
    ws = _window_poll_state.get(window_id)
    if ws:
        ws.probe_failures = 0


def reset_probe_failures_state() -> None:
    """Reset all probe failure tracking (for testing)."""
    for ws in _window_poll_state.values():
        ws.probe_failures = 0


def clear_typing_state(user_id: int, thread_id: int) -> None:
    """Clear typing indicator throttle for a topic (called on cleanup)."""
    ts = _topic_poll_state.get((user_id, thread_id))
    if ts:
        ts.last_typing_sent = None


def clear_seen_status(window_id: str) -> None:
    """Clear startup status tracking for a window (called on cleanup)."""
    ws = _window_poll_state.get(window_id)
    if ws:
        ws.has_seen_status = False
        ws.startup_time = None


def reset_seen_status_state() -> None:
    """Reset all startup status tracking (for testing)."""
    for ws in _window_poll_state.values():
        ws.has_seen_status = False
        ws.startup_time = None


def reset_typing_state() -> None:
    """Reset all typing indicator tracking (for testing)."""
    for ts in _topic_poll_state.values():
        ts.last_typing_sent = None


def _start_autoclose_timer(
    user_id: int, thread_id: int, state: str, now: float
) -> None:
    """Start or maintain an autoclose timer for a topic in done/dead state."""
    ts = _get_topic_state(user_id, thread_id)
    existing = ts.autoclose
    if existing is None or existing[0] != state:
        ts.autoclose = (state, now)


def _clear_autoclose_if_active(user_id: int, thread_id: int) -> None:
    """Clear autoclose timer when topic becomes active/idle (session alive)."""
    ts = _topic_poll_state.get((user_id, thread_id))
    if ts:
        ts.autoclose = None


async def _check_unbound_window_ttl(live_windows: list | None = None) -> None:
    """Kill unbound tmux windows whose TTL has expired.

    Unbound windows are live tmux windows not bound to any topic. They appear
    when a topic is closed (window kept alive for rebinding). After
    autoclose_done_minutes they are auto-killed.

    Args:
        live_windows: Pre-fetched tmux windows (avoids duplicate subprocess call).
            Falls back to fetching if None.
    """
    timeout = config.autoclose_done_minutes * 60
    if timeout <= 0:
        return

    # Build set of currently bound window IDs
    bound_ids: set[str] = set()
    for _, _, wid in session_manager.iter_thread_bindings():
        bound_ids.add(wid)

    # Get all live tmux windows (use pre-fetched if available)
    if live_windows is None:
        live_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in live_windows}

    # Remove timers for windows that got rebound or no longer exist
    for wid, ws in list(_window_poll_state.items()):
        if ws.unbound_timer is not None and (wid in bound_ids or wid not in live_ids):
            ws.unbound_timer = None

    # Track newly unbound windows
    now = time.monotonic()
    for w in live_windows:
        if w.window_id not in bound_ids:
            ws = _get_window_state(w.window_id)
            if ws.unbound_timer is None:
                ws.unbound_timer = now

    # Kill expired unbound windows
    expired = [
        wid
        for wid, ws in _window_poll_state.items()
        if ws.unbound_timer is not None and now - ws.unbound_timer >= timeout
    ]
    for wid in expired:
        ws = _window_poll_state.get(wid)
        if ws:
            ws.unbound_timer = None
        from ..tmux_manager import clear_vim_state

        clear_vim_state(wid)
        await tmux_manager.kill_window(wid)
        logger.info("Auto-killed unbound window %s (TTL expired)", wid)


async def _check_autoclose_timers(bot: Bot) -> None:
    """Close topics whose done/dead timers have expired."""
    if not _topic_poll_state:
        return

    now = time.monotonic()
    expired: list[tuple[int, int]] = []

    for (user_id, thread_id), ts in _topic_poll_state.items():
        if ts.autoclose is None:
            continue
        state, entered_at = ts.autoclose
        if state == "done":
            timeout = config.autoclose_done_minutes * 60
        elif state == "dead":
            timeout = config.autoclose_dead_minutes * 60
        else:
            continue

        if timeout <= 0:
            continue

        if now - entered_at >= timeout:
            expired.append((user_id, thread_id))

    for user_id, thread_id in expired:
        ts = _topic_poll_state.get((user_id, thread_id))
        if ts:
            ts.autoclose = None
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        try:
            await bot.close_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
            logger.info(
                "Auto-closed topic: chat=%d thread=%d (user=%d)",
                chat_id,
                thread_id,
                user_id,
            )
        except TelegramError as e:
            logger.debug("Failed to auto-close topic thread=%d: %s", thread_id, e)


def _check_transcript_activity(window_id: str, now: float) -> bool:
    """Check if recent transcript writes indicate an active agent.

    Returns True if transcript was written to within _ACTIVITY_THRESHOLD.
    Side-effect: marks window as "has seen status" and clears startup timer.
    """
    session_id = session_manager.get_session_id_for_window(window_id)
    if not session_id:
        return False

    mon = get_active_monitor()
    if not mon:
        return False
    last_activity = mon.get_last_activity(session_id)
    if last_activity and (now - last_activity) < _ACTIVITY_THRESHOLD:
        ws = _get_window_state(window_id)
        ws.has_seen_status = True
        ws.startup_time = None
        return True
    return False


async def _transition_to_idle(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int,
    chat_id: int,
    display: str,
    notif_mode: str,
) -> None:
    """Transition a window to idle state (emoji, autoclose, typing, status)."""
    _get_window_state(window_id).startup_time = None
    await update_topic_emoji(bot, chat_id, thread_id, "idle", display)
    _clear_autoclose_if_active(user_id, thread_id)
    _get_topic_state(user_id, thread_id).last_typing_sent = None
    if notif_mode not in ("muted", "errors_only"):
        from .callback_data import IDLE_STATUS_TEXT

        await enqueue_status_update(
            bot, user_id, window_id, IDLE_STATUS_TEXT, thread_id=thread_id
        )
    else:
        # Muted windows: clear any lingering status message
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)


async def _handle_no_status(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None,
    pane_current_command: str,
    notif_mode: str,
) -> None:
    """Handle a window with no provider-detected terminal status.

    Falls back to transcript activity heuristic, then shell/idle/startup detection.
    """
    now = time.monotonic()
    is_active = _check_transcript_activity(window_id, now)

    if is_active:
        await _send_typing_throttled(bot, user_id, thread_id)
        if thread_id is not None:
            chat_id = session_manager.resolve_chat_id(user_id, thread_id)
            display = session_manager.get_display_name(window_id)
            await update_topic_emoji(bot, chat_id, thread_id, "active", display)
            _clear_autoclose_if_active(user_id, thread_id)
        return

    if thread_id is None:
        return

    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    display = session_manager.get_display_name(window_id)
    ws = _get_window_state(window_id)

    if is_shell_prompt(pane_current_command):
        ws.startup_time = None
        # Hookless providers (Codex/Gemini) often sit at shell-like prompts while
        # still being an active topic. Keep idle controls visible instead of
        # clearing the status message.
        state = session_manager.get_window_state(window_id)
        raw_provider = getattr(state, "provider_name", "")
        provider_name = raw_provider.lower() if isinstance(raw_provider, str) else ""
        if provider_name in ("codex", "gemini"):
            ws.has_seen_status = True
            await _transition_to_idle(
                bot, user_id, window_id, thread_id, chat_id, display, notif_mode
            )
            return

        await update_topic_emoji(bot, chat_id, thread_id, "done", display)
        _start_autoclose_timer(user_id, thread_id, "done", now)
        _get_topic_state(user_id, thread_id).last_typing_sent = None
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)
    elif ws.has_seen_status:
        await _transition_to_idle(
            bot, user_id, window_id, thread_id, chat_id, display, notif_mode
        )
    elif ws.startup_time is None:
        # First poll without status — start grace period
        ws.startup_time = now
        await _send_typing_throttled(bot, user_id, thread_id)
        await update_topic_emoji(bot, chat_id, thread_id, "active", display)
        _clear_autoclose_if_active(user_id, thread_id)
    elif now - ws.startup_time >= _STARTUP_TIMEOUT:
        # Startup timed out — treat as idle
        ws.has_seen_status = True
        await _transition_to_idle(
            bot, user_id, window_id, thread_id, chat_id, display, notif_mode
        )
    else:
        # Still in startup grace period
        await _send_typing_throttled(bot, user_id, thread_id)
        await update_topic_emoji(bot, chat_id, thread_id, "active", display)
        _clear_autoclose_if_active(user_id, thread_id)


def _parse_with_pyte(window_id: str, pane_text: str) -> StatusUpdate | None:
    """Try pyte-based screen parsing for status and interactive UI detection.

    Feeds the plain pane text into a ScreenBuffer sized for a standard
    terminal, then uses the screen-based parsers. Returns a StatusUpdate
    or None if nothing detected.
    """
    from ..screen_buffer import ScreenBuffer
    from ..terminal_parser import (
        format_status_display,
        parse_from_screen,
        parse_status_from_screen,
    )

    # Use a standard terminal size; pyte needs dimensions to render
    columns, rows = 200, 50
    buf = _get_screen_buffer(window_id, columns, rows)
    if not isinstance(buf, ScreenBuffer):
        return None

    buf.feed(pane_text)

    # Check interactive UI first (takes precedence)
    interactive = parse_from_screen(buf)
    if interactive:
        return StatusUpdate(
            raw_text=interactive.content,
            display_label=interactive.name,
            is_interactive=True,
            ui_type=interactive.name,
        )

    # Check status line
    raw_status = parse_status_from_screen(buf)
    if raw_status:
        return StatusUpdate(
            raw_text=raw_status,
            display_label=format_status_display(raw_status),
        )

    return None


# ── Multi-pane scanning (agent teams) ─────────────────────────────────
# When a window has >1 pane (e.g. Claude Code agent teams in split-pane
# mode), non-active panes are scanned for interactive prompts and alerts
# are surfaced in the Telegram topic.


def has_pane_alert(pane_id: str) -> bool:
    """Check whether a pane currently has an active alert."""
    return pane_id in _pane_alert_hashes


def clear_pane_alerts(window_id: str) -> None:
    """Remove pane alert state for a specific window only."""
    stale = [pid for pid, v in _pane_alert_hashes.items() if v[2] == window_id]
    for pid in stale:
        _pane_alert_hashes.pop(pid, None)


_PANE_COUNT_TTL = 5.0  # seconds


async def _scan_window_panes(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int,
) -> None:
    """Scan non-active panes for interactive prompts and surface alerts.

    Fast path: uses cached pane count to skip single-pane windows without
    a subprocess call. Cache refreshes every 5 seconds.
    """
    now = time.monotonic()
    ws = _get_window_state(window_id)
    cached = ws.pane_count_cache
    if cached and cached[1] > now and cached[0] <= 1:
        return  # Cached single-pane — no subprocess needed

    panes = await tmux_manager.list_panes(window_id)
    ws.pane_count_cache = (len(panes), now + _PANE_COUNT_TTL)
    live_pane_ids = {p.pane_id for p in panes}

    # Clean up alerts for panes of THIS window that no longer exist
    # (must run before the early return so alerts clear when dropping to single-pane)
    stale = [
        pid
        for pid, v in _pane_alert_hashes.items()
        if v[2] == window_id and pid not in live_pane_ids
    ]
    for pid in stale:
        _pane_alert_hashes.pop(pid, None)

    if len(panes) <= 1:
        return

    now = time.monotonic()

    for pane in panes:
        if pane.active:
            continue  # Active pane handled by the normal status_polling path

        pane_text = await tmux_manager.capture_pane_by_id(
            pane.pane_id, window_id=window_id
        )
        if not pane_text:
            continue

        # Use provider-level parsing (same as active pane detection)
        provider = get_provider_for_window(window_id)
        status = provider.parse_terminal_status(pane_text, pane_title="")
        if status is None or not status.is_interactive:
            # No interactive UI — clear stale alert if any
            _pane_alert_hashes.pop(pane.pane_id, None)
            continue

        # Interactive UI detected — check if it's new or changed
        prompt_text = status.raw_text or ""

        existing = _pane_alert_hashes.get(pane.pane_id)
        if existing and existing[0] == prompt_text:
            # Same prompt, already notified — skip
            continue

        _pane_alert_hashes[pane.pane_id] = (prompt_text, now, window_id)
        logger.info(
            "Pane %s in window %s has interactive UI, surfacing alert",
            pane.pane_id,
            window_id,
        )
        await handle_interactive_ui(
            bot, user_id, window_id, thread_id, pane_id=pane.pane_id
        )


async def update_status_message(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    *,
    _window: TmuxWindow | None = None,
) -> None:
    """Poll terminal and enqueue status update for user's active window.

    Also detects permission prompt UIs (not triggered via JSONL) and enters
    interactive mode when found.

    Args:
        _window: Pre-fetched TmuxWindow (avoids duplicate list_windows call
            when called from the poll loop).
    """
    w = _window or await tmux_manager.find_window_by_id(window_id)
    if not w:
        # Window gone, enqueue clear
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)
        return

    pane_text = await tmux_manager.capture_pane(w.window_id)
    if not pane_text:
        # Transient capture failure - keep existing status message
        return

    # Passive vim INSERT mode tracking — feed the polling cache so that
    # _ensure_vim_insert_mode() has a warm cache for the common case.
    # Use tail-only check to avoid false positives from historical output.
    from ..tmux_manager import _has_insert_indicator, notify_vim_insert_seen

    if _has_insert_indicator(pane_text):
        notify_vim_insert_seen(w.window_id)

    interactive_window = get_interactive_window(user_id, thread_id)
    should_check_new_ui = True

    # Parse terminal status: try pyte-based parsing first, fall back to regex
    status = _parse_with_pyte(window_id, pane_text)

    if status is None:
        # pyte path returned nothing — fall back to provider regex parsing
        provider = get_provider_for_window(window_id)
        pane_title = ""
        if provider.capabilities.uses_pane_title:
            pane_title = await tmux_manager.get_pane_title(w.window_id)
        status = provider.parse_terminal_status(pane_text, pane_title=pane_title)

    if interactive_window == window_id:
        # User is in interactive mode for THIS window
        if status is not None and status.is_interactive:
            # Interactive UI still showing — skip status update (user is interacting)
            return
        # Interactive UI gone — clear interactive mode, fall through to status check.
        # Don't re-check for new UI this cycle (the old one just disappeared).
        await clear_interactive_msg(user_id, bot, thread_id)
        should_check_new_ui = False
    elif interactive_window is not None:
        # User is in interactive mode for a DIFFERENT window (window switched)
        # Clear stale interactive mode
        await clear_interactive_msg(user_id, bot, thread_id)

    # Check for permission prompt (interactive UI not triggered via JSONL)
    if should_check_new_ui and status is not None and status.is_interactive:
        await handle_interactive_ui(bot, user_id, window_id, thread_id)
        return

    # Normal status line check — use display_label for formatted text
    status_line = status.display_label if status and not status.is_interactive else None

    # Suppress status message updates for muted/errors_only windows,
    # but only AFTER interactive UI detection, rename sync, and emoji updates above.
    notif_mode = session_manager.get_notification_mode(window_id)

    if status_line:
        ws = _get_window_state(window_id)
        ws.has_seen_status = True
        ws.startup_time = None
        await _send_typing_throttled(bot, user_id, thread_id)
        if notif_mode not in ("muted", "errors_only"):
            # Append subagent count if any are active
            from .hook_events import get_subagent_count

            subagent_count = get_subagent_count(window_id)
            display_status = status_line
            if subagent_count:
                display_status = f"{status_line} ({subagent_count} subagent{'s' if subagent_count > 1 else ''})"
            await enqueue_status_update(
                bot,
                user_id,
                window_id,
                display_status,
                thread_id=thread_id,
            )
        # Update topic emoji to active (agent is working)
        if thread_id is not None:
            chat_id = session_manager.resolve_chat_id(user_id, thread_id)
            display = session_manager.get_display_name(window_id)
            await update_topic_emoji(bot, chat_id, thread_id, "active", display)
            _clear_autoclose_if_active(user_id, thread_id)
    else:
        await _handle_no_status(
            bot, user_id, window_id, thread_id, w.pane_current_command, notif_mode
        )


async def _handle_dead_window_notification(
    bot: Bot, user_id: int, thread_id: int, wid: str
) -> None:
    """Send proactive recovery notification for a dead window (once per death)."""
    dead_key = (user_id, thread_id, wid)
    if dead_key in _dead_notified:
        return
    _get_window_state(wid).has_seen_status = False

    # Clean up stale tool message IDs for this topic (window is dead,
    # no more tool_result edits will arrive).

    clear_tool_msg_ids_for_topic(user_id, thread_id)
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    display = session_manager.get_display_name(wid)
    await update_topic_emoji(bot, chat_id, thread_id, "dead", display)
    _start_autoclose_timer(user_id, thread_id, "dead", time.monotonic())

    window_state = session_manager.get_window_state(wid)
    cwd = window_state.cwd or ""
    try:
        dir_exists = bool(cwd) and await asyncio.to_thread(Path(cwd).is_dir)
    except OSError:
        dir_exists = False
    if dir_exists:
        keyboard = build_recovery_keyboard(wid)
        text = (
            f"\u26a0 Session `{display}` ended.\n"
            f"\U0001f4c2 `{cwd}`\n\n"
            "Tap a button or send a message to recover."
        )
    else:
        text = f"\u26a0 Session `{display}` ended."
        keyboard = None
    await rate_limit_send_message(
        bot,
        chat_id,
        text,
        message_thread_id=thread_id,
        reply_markup=keyboard,
    )
    _dead_notified.add(dead_key)


def _record_probe_failure(window_id: str) -> int:
    """Increment probe failure counter; log once when threshold is reached."""
    ws = _get_window_state(window_id)
    ws.probe_failures += 1
    count = ws.probe_failures
    if count == _MAX_PROBE_FAILURES:
        logger.info(
            "Suspending topic probe for %s after %d consecutive failures",
            window_id,
            count,
        )
    return count


async def _prune_stale_state(live_windows: list) -> None:
    """Sync display names and prune orphaned state entries.

    Called every TOPIC_CHECK_INTERVAL from the poll loop with pre-fetched
    live tmux windows to avoid duplicate subprocess calls.
    """
    live_ids = {w.window_id for w in live_windows}
    live_pairs = [(w.window_id, w.window_name) for w in live_windows]
    session_manager.sync_display_names(live_pairs)
    session_manager.prune_stale_state(live_ids)


async def _probe_topic_existence(bot: Bot) -> None:
    """Probe all bound topics via Telegram API; detect deleted topics."""
    for user_id, thread_id, wid in list(session_manager.iter_thread_bindings()):
        if _get_window_state(wid).probe_failures >= _MAX_PROBE_FAILURES:
            continue
        try:
            await bot.unpin_all_forum_topic_messages(
                chat_id=session_manager.resolve_chat_id(user_id, thread_id),
                message_thread_id=thread_id,
            )
            _get_window_state(wid).probe_failures = 0
        except TelegramError as e:
            if isinstance(e, BadRequest) and (
                "Topic_id_invalid" in e.message
                or "thread not found" in e.message.lower()
            ):
                # Topic deleted — kill window, unbind, and clean up state
                w = await tmux_manager.find_window_by_id(wid)
                if w:
                    await tmux_manager.kill_window(w.window_id)
                _get_window_state(wid).probe_failures = 0
                await clear_topic_state(user_id, thread_id, bot, window_id=wid)
                session_manager.unbind_thread(user_id, thread_id)
                logger.info(
                    "Topic deleted: killed window_id '%s' and "
                    "unbound thread %d for user %d",
                    wid,
                    thread_id,
                    user_id,
                )
            else:
                count = _record_probe_failure(wid)
                if count < _MAX_PROBE_FAILURES:
                    log_throttled(
                        logger,
                        f"topic-probe:{wid}",
                        "Topic probe error for %s: %s",
                        wid,
                        e,
                    )


async def _maybe_discover_transcript(
    window_id: str,
    *,
    _window: TmuxWindow | None = None,
) -> None:
    """Discover and register transcript for hookless providers (Codex, Gemini).

    Runs on each poll cycle for bound windows. For hookless providers, this
    allows transcript re-discovery when a new CLI session starts in the same
    tmux window. When a transcript is found, writes/updates a synthetic
    session_map entry so the session monitor tracks the current session.

    Provider resolution logic:
    - If ``state.provider_name`` is explicitly set AND provider has hooks,
      trust hook delivery and return early.
    - If ``state.provider_name`` is empty (auto-detection failed, e.g. Codex
      running under ``bun``), try all hookless providers' ``discover_transcript``
      to find a match.

    For externally-created windows, cwd may be empty (no hook to populate it).
    Falls back to the tmux window's pane_current_path.

    Args:
        _window: Pre-fetched TmuxWindow (avoids duplicate list_windows call
            when called from the poll loop).
    """
    from ..providers import registry

    state = session_manager.window_states.get(window_id)
    if not state:
        return

    w = _window or await tmux_manager.find_window_by_id(window_id)

    # Re-detect provider from the current pane to recover from stale mappings.
    if w and w.pane_current_command:
        detected = detect_provider_from_command(w.pane_current_command)
        if not detected and should_probe_pane_title_for_provider_detection(
            w.pane_current_command
        ):
            pane_title = await tmux_manager.get_pane_title(window_id)
            detected = detect_provider_from_runtime(
                w.pane_current_command,
                pane_title=pane_title,
            )
        if detected and detected != state.provider_name:
            session_manager.set_window_provider(window_id, detected, cwd=w.cwd or None)

    # If provider is explicitly set and supports hooks, trust hook delivery
    if state.provider_name:
        provider = get_provider_for_window(window_id)
        if provider.capabilities.supports_hook:
            return

    # Ensure cwd is available (fall back to tmux pane path)
    if not state.cwd:
        if not w or not w.cwd:
            return
        session_manager.set_window_provider(
            window_id, state.provider_name or "", cwd=w.cwd
        )

    # Determine which providers to try
    if state.provider_name:
        # Explicit hookless provider — try only that one
        provider = get_provider_for_window(window_id)
        providers_to_try = [(provider.capabilities.name, provider)]
    else:
        # Detection failed — check pane is alive (skip dead shells)
        if w and is_shell_prompt(w.pane_current_command):
            return
        # Try all hookless providers
        providers_to_try = [
            (name, registry.get(name))
            for name in registry.provider_names()
            if not registry.get(name).capabilities.supports_hook
        ]

    # Disable staleness check if pane process is alive
    pane_alive = w is not None and not is_shell_prompt(w.pane_current_command)

    # Foreign windows (emdash) are already fully qualified — no prefix needed
    if is_foreign_window(window_id):
        window_key = window_id
    else:
        window_key = f"{config.tmux_session_name}:{window_id}"
    for provider_name, provider in providers_to_try:
        # Active panes may have stale transcript mtimes (no recent writes yet);
        # bypass staleness checks for better hookless session recovery.
        max_age = 0 if pane_alive else None
        event = await asyncio.to_thread(
            provider.discover_transcript,
            state.cwd,
            window_key,
            max_age=max_age,
        )
        if event:
            if (
                state.session_id == event.session_id
                and state.transcript_path == event.transcript_path
                and state.provider_name == provider_name
            ):
                return
            session_manager.register_hookless_session(
                window_id=window_id,
                session_id=event.session_id,
                cwd=event.cwd,
                transcript_path=event.transcript_path,
                provider_name=provider_name,
            )
            await asyncio.to_thread(
                session_manager.write_hookless_session_map,
                window_id=window_id,
                session_id=event.session_id,
                cwd=event.cwd,
                transcript_path=event.transcript_path,
                provider_name=provider_name,
            )
            return


async def status_poll_loop(bot: Bot) -> None:
    """Background task to poll terminal status for all thread-bound windows."""
    logger.info("Status polling started (interval: %ss)", STATUS_POLL_INTERVAL)
    last_topic_check = 0.0
    _error_streak = 0
    while True:
        try:
            # Fetch all windows once per cycle — O(1) lookup replaces
            # per-binding find_window_by_id calls (O(N×M) → O(N+M)).
            all_windows = await tmux_manager.list_windows()
            emdash_windows = await tmux_manager.discover_emdash_sessions()
            all_windows.extend(emdash_windows)
            window_lookup: dict[str, TmuxWindow] = {w.window_id: w for w in all_windows}

            # Periodic topic existence probe + stale state cleanup
            now = time.monotonic()
            if now - last_topic_check >= TOPIC_CHECK_INTERVAL:
                last_topic_check = now
                await _prune_stale_state(all_windows)
                await _probe_topic_existence(bot)
                # Sweep stale log-throttle entries to prevent unbounded growth
                log_throttle_sweep()

            for user_id, thread_id, wid in list(session_manager.iter_thread_bindings()):
                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(window_id=wid)
                try:
                    # Already notified about this dead window — skip tmux check
                    if (user_id, thread_id, wid) in _dead_notified:
                        continue

                    w = window_lookup.get(wid)
                    if not w:
                        await _handle_dead_window_notification(
                            bot, user_id, thread_id, wid
                        )
                        continue

                    # Discover transcript for hookless providers (Codex, Gemini)
                    await _maybe_discover_transcript(wid, _window=w)

                    queue = get_message_queue(user_id)
                    if queue and not queue.empty():
                        continue
                    await update_status_message(
                        bot,
                        user_id,
                        wid,
                        thread_id=thread_id,
                        _window=w,
                    )
                    # Scan non-active panes for interactive prompts (agent teams)
                    await _scan_window_panes(bot, user_id, wid, thread_id)
                except (TelegramError, OSError) as e:
                    log_throttled(
                        logger,
                        f"status-update:{user_id}:{thread_id}",
                        "Status update error for user %s thread %s: %s",
                        user_id,
                        thread_id,
                        e,
                    )

            # Check timers at end of each poll cycle
            await _check_autoclose_timers(bot)
            await _check_unbound_window_ttl(all_windows)

        except _LoopError:
            logger.exception("Status poll loop error")
            backoff_delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**_error_streak))
            _error_streak += 1
            await asyncio.sleep(backoff_delay)
            continue

        _error_streak = 0
        await asyncio.sleep(STATUS_POLL_INTERVAL)
