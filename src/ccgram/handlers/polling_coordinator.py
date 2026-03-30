"""Polling coordinator for terminal status monitoring.

Orchestrates the polling cycle by iterating thread bindings and delegating
to strategy classes in polling_strategies.py. Contains all async orchestration
functions (update_status_message, interactive UI checks, transcript discovery,
shell relay, dead window handling) plus the main polling loop.

Key components:
  - status_poll_loop: Background polling task (entry point for bot.py)
  - update_status_message: Poll and enqueue status updates
  - STATUS_POLL_INTERVAL / TOPIC_CHECK_INTERVAL: Timing constants
"""

import asyncio
import contextlib
import structlog
import time
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
    detect_provider_from_pane,
    detect_provider_from_runtime,
    detect_provider_from_transcript_path,
    get_provider_for_window,
    should_probe_pane_title_for_provider_detection,
)
from ..providers.base import StatusUpdate
from ..session import session_manager
from ..session_monitor import get_active_monitor
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from ..utils import log_throttle_sweep, log_throttled
from ..window_resolver import is_foreign_window
from .cleanup import clear_topic_state
from .interactive_ui import (
    clear_interactive_mode,
    clear_interactive_msg,
    get_interactive_window,
    handle_interactive_ui,
    set_interactive_mode,
)
from .message_queue import (
    clear_tool_msg_ids_for_topic,
    enqueue_status_update,
    get_message_queue,
)
from .message_sender import rate_limit_send_message
from .polling_strategies import (
    _ACTIVITY_THRESHOLD,
    _MAX_PROBE_FAILURES,
    _PANE_COUNT_TTL,
    _STARTUP_TIMEOUT,
    _TYPING_INTERVAL,
    TopicPollState,
    WindowPollState,
    clear_window_poll_state,
    interactive_strategy,
    is_shell_prompt,
    lifecycle_strategy,
    terminal_strategy,
)
from .recovery_callbacks import build_recovery_keyboard
from .topic_emoji import update_topic_emoji

logger = structlog.get_logger()

# ── Timing constants ──────────────────────────────────────────────────────

STATUS_POLL_INTERVAL = 1.0  # seconds
TOPIC_CHECK_INTERVAL = 60.0  # seconds

# Exponential backoff bounds for loop errors (seconds)
_BACKOFF_MIN = 2.0
_BACKOFF_MAX = 30.0

# Top-level loop resilience: catch any error to keep polling alive
_LoopError = (TelegramError, OSError, RuntimeError, ValueError)


# ── State access helpers ─────────────────────────────────────────────────


def _get_window_state(window_id: str) -> WindowPollState:
    """Get or create WindowPollState for a window."""
    return terminal_strategy.get_state(window_id)


def _get_topic_state(user_id: int, thread_id: int) -> TopicPollState:
    """Get or create TopicPollState for a topic."""
    return lifecycle_strategy.get_state(user_id, thread_id)


def _get_screen_buffer(window_id: str, columns: int, rows: int) -> "ScreenBuffer":
    """Get or create a ScreenBuffer for a window, resizing if needed."""
    return terminal_strategy.get_screen_buffer(window_id, columns, rows)


# ── Typing throttle ─────────────────────────────────────────────────────


async def _send_typing_throttled(bot: Bot, user_id: int, thread_id: int | None) -> None:
    """Send typing indicator if enough time has elapsed since the last one."""
    if thread_id is None:
        return
    ts = _get_topic_state(user_id, thread_id)
    now = time.monotonic()
    if now - (ts.last_typing_sent or 0.0) < _TYPING_INTERVAL:
        return
    ts.last_typing_sent = now
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    with contextlib.suppress(TelegramError):
        await bot.send_chat_action(
            chat_id=chat_id,
            message_thread_id=thread_id,
            action=ChatAction.TYPING,
        )


# ── RC state / pyte parsing ─────────────────────────────────────────────


def _update_rc_state(ws: WindowPollState, rc_detected: bool) -> None:
    """Update Remote Control state with 3s debounce on removal."""
    terminal_strategy.update_rc_state(ws, rc_detected)


def _parse_with_pyte(
    window_id: str,
    pane_text: str,
    columns: int = 0,
    rows: int = 0,
) -> StatusUpdate | None:
    """Parse terminal via pyte screen buffer for status and interactive UI."""
    return terminal_strategy.parse_with_pyte(window_id, pane_text, columns, rows)


# ── Transcript activity check ───────────────────────────────────────────


def _check_transcript_activity(window_id: str, now: float) -> bool:
    """Check if recent transcript writes indicate an active agent."""
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


# ── Idle / no-status transitions ────────────────────────────────────────


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
    lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
    _get_topic_state(user_id, thread_id).last_typing_sent = None
    if notif_mode not in ("muted", "errors_only"):
        from .callback_data import IDLE_STATUS_TEXT

        await enqueue_status_update(
            bot, user_id, window_id, IDLE_STATUS_TEXT, thread_id=thread_id
        )
    else:
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)


async def _handle_no_status(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None,
    pane_current_command: str,
    notif_mode: str,
) -> None:
    """Handle a window with no provider-detected terminal status."""
    now = time.monotonic()
    is_active = _check_transcript_activity(window_id, now)

    if is_active:
        await _send_typing_throttled(bot, user_id, thread_id)
        if thread_id is not None:
            chat_id = thread_router.resolve_chat_id(user_id, thread_id)
            display = thread_router.get_display_name(window_id)
            await update_topic_emoji(bot, chat_id, thread_id, "active", display)
            lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
        return

    if thread_id is None:
        return

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    display = thread_router.get_display_name(window_id)
    ws = _get_window_state(window_id)

    if is_shell_prompt(pane_current_command):
        ws.startup_time = None
        state = session_manager.get_window_state(window_id)
        raw_provider = getattr(state, "provider_name", "")
        provider_name = raw_provider.lower() if isinstance(raw_provider, str) else ""
        if provider_name in ("codex", "gemini", "shell"):
            ws.has_seen_status = True
            await _transition_to_idle(
                bot, user_id, window_id, thread_id, chat_id, display, notif_mode
            )
            return

        await update_topic_emoji(bot, chat_id, thread_id, "done", display)
        lifecycle_strategy.start_autoclose_timer(user_id, thread_id, "done", now)
        _get_topic_state(user_id, thread_id).last_typing_sent = None
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)
    elif ws.has_seen_status:
        await _transition_to_idle(
            bot, user_id, window_id, thread_id, chat_id, display, notif_mode
        )
    elif ws.startup_time is None:
        ws.startup_time = now
        await _send_typing_throttled(bot, user_id, thread_id)
        await update_topic_emoji(bot, chat_id, thread_id, "active", display)
        lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
    elif now - ws.startup_time >= _STARTUP_TIMEOUT:
        ws.has_seen_status = True
        await _transition_to_idle(
            bot, user_id, window_id, thread_id, chat_id, display, notif_mode
        )
    else:
        await _send_typing_throttled(bot, user_id, thread_id)
        await update_topic_emoji(bot, chat_id, thread_id, "active", display)
        lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)


# ── Multi-pane scanning (agent teams) ─────────────────────────────────


async def _scan_window_panes(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int,
) -> None:
    """Scan non-active panes for interactive prompts and surface alerts."""
    now = time.monotonic()
    ws = _get_window_state(window_id)
    cached = ws.pane_count_cache
    if cached and cached[1] > now and cached[0] <= 1:
        return  # Cached single-pane — no subprocess needed

    panes = await tmux_manager.list_panes(window_id)
    ws.pane_count_cache = (len(panes), now + _PANE_COUNT_TTL)
    live_pane_ids = {p.pane_id for p in panes}

    interactive_strategy.prune_stale_pane_alerts(window_id, live_pane_ids)

    if len(panes) <= 1:
        return

    now = time.monotonic()

    for pane in panes:
        if pane.active:
            continue

        pane_text = await tmux_manager.capture_pane_by_id(
            pane.pane_id, window_id=window_id
        )
        if not pane_text:
            continue

        provider = get_provider_for_window(window_id)
        status = provider.parse_terminal_status(pane_text, pane_title="")
        if status is None or not status.is_interactive:
            interactive_strategy.remove_pane_alert(pane.pane_id)
            continue

        prompt_text = status.raw_text or ""

        existing = interactive_strategy.get_pane_alert(pane.pane_id)
        if existing and existing[0] == prompt_text:
            continue

        interactive_strategy.set_pane_alert(pane.pane_id, prompt_text, now, window_id)
        logger.info(
            "Pane %s in window %s has interactive UI, surfacing alert",
            pane.pane_id,
            window_id,
        )
        await handle_interactive_ui(
            bot, user_id, window_id, thread_id, pane_id=pane.pane_id
        )


# ── Interactive-only check ───────────────────────────────────────────────


async def _check_interactive_only(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int,
    *,
    _window: "TmuxWindow | None" = None,
) -> None:
    """Check for interactive UI without enqueuing status updates."""
    w = _window or await tmux_manager.find_window_by_id(window_id)
    if not w:
        return

    if get_interactive_window(user_id, thread_id) == window_id:
        return

    pane_text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not pane_text:
        return

    status = _parse_with_pyte(
        window_id, pane_text, columns=w.pane_width, rows=w.pane_height
    )

    if status is None:
        ws = _get_window_state(window_id)
        clean_text = (
            ws.last_rendered_text if ws.last_rendered_text is not None else pane_text
        )
        provider = get_provider_for_window(window_id)
        pane_title = ""
        if provider.capabilities.uses_pane_title:
            pane_title = await tmux_manager.get_pane_title(w.window_id)
        status = provider.parse_terminal_status(clean_text, pane_title=pane_title)

    if status is not None and status.is_interactive:
        set_interactive_mode(user_id, window_id, thread_id)
        handled = await handle_interactive_ui(bot, user_id, window_id, thread_id)
        if not handled:
            clear_interactive_mode(user_id, thread_id)


# ── Passive shell relay ──────────────────────────────────────────────────


async def _maybe_check_passive_shell(
    bot: Bot, user_id: int, window_id: str, thread_id: int
) -> None:
    """Relay shell output from direct tmux interaction to Telegram."""
    state = session_manager.get_window_state(window_id)
    if not state or state.provider_name != "shell":
        return
    ws = terminal_strategy.get_state(window_id)
    rendered = ws.last_rendered_text
    if rendered is None:
        raw = await tmux_manager.capture_pane(window_id)
        if not raw:
            return
        rendered = raw
    from .shell_capture import check_passive_shell_output

    await check_passive_shell_output(bot, user_id, thread_id, window_id, rendered)


# ── Transcript discovery ─────────────────────────────────────────────────


async def _maybe_discover_transcript(
    window_id: str,
    *,
    _window: "TmuxWindow | None" = None,
    bot: Bot | None = None,  # noqa: ARG001
    user_id: int = 0,  # noqa: ARG001
    thread_id: int = 0,  # noqa: ARG001
) -> None:
    """Discover and register transcript for hookless providers (Codex, Gemini)."""
    from ..providers import registry

    state = session_manager.window_states.get(window_id)
    if not state:
        return

    w = _window or await tmux_manager.find_window_by_id(window_id)

    if w and w.pane_current_command:
        detected = await detect_provider_from_pane(
            w.pane_current_command, pane_tty=w.pane_tty, window_id=window_id
        )
        if not detected and should_probe_pane_title_for_provider_detection(
            w.pane_current_command
        ):
            pane_title = await tmux_manager.get_pane_title(window_id)
            detected = detect_provider_from_runtime(
                w.pane_current_command,
                pane_title=pane_title,
            )
        if detected and detected != state.provider_name:
            old_provider = state.provider_name
            session_manager.set_window_provider(window_id, detected, cwd=w.cwd or None)
            if detected == "shell":
                state.transcript_path = ""  # shell has no transcripts
                from ..providers.shell import setup_shell_prompt

                await setup_shell_prompt(window_id, clear=False)
            elif old_provider == "shell":
                from .shell_capture import clear_shell_monitor_state

                clear_shell_monitor_state(window_id)
        elif not detected and state.transcript_path:
            inferred = detect_provider_from_transcript_path(state.transcript_path)
            if inferred and inferred != state.provider_name:
                session_manager.set_window_provider(
                    window_id,
                    inferred,
                    cwd=w.cwd or None,
                )

    if state.provider_name:
        provider = get_provider_for_window(window_id)
        if provider.capabilities.supports_hook:
            return

    if not state.cwd:
        if not w or not w.cwd:
            return
        session_manager.set_window_provider(
            window_id, state.provider_name or "", cwd=w.cwd
        )

    if state.provider_name:
        provider = get_provider_for_window(window_id)
        if provider.capabilities.name == "shell":
            return
        providers_to_try = [(provider.capabilities.name, provider)]
    else:
        if w and is_shell_prompt(w.pane_current_command):
            session_manager.set_window_provider(window_id, "shell")
            state.transcript_path = ""
            from ..providers.shell import setup_shell_prompt

            await setup_shell_prompt(window_id, clear=False)
            return
        providers_to_try = [
            (name, registry.get(name))
            for name in registry.provider_names()
            if not registry.get(name).capabilities.supports_hook and name != "shell"
        ]

    pane_alive = w is not None and not is_shell_prompt(w.pane_current_command)

    if is_foreign_window(window_id):
        window_key = window_id
    else:
        window_key = f"{config.tmux_session_name}:{window_id}"
    for provider_name, provider in providers_to_try:
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


# ── Dead window / probe helpers ─────────────────────────────────────────


async def _handle_dead_window_notification(
    bot: Bot, user_id: int, thread_id: int, wid: str
) -> None:
    """Send proactive recovery notification for a dead window (once per death)."""
    if lifecycle_strategy.is_dead_notified(user_id, thread_id, wid):
        return
    terminal_strategy.get_state(wid).has_seen_status = False

    clear_tool_msg_ids_for_topic(user_id, thread_id)
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    display = thread_router.get_display_name(wid)
    await update_topic_emoji(bot, chat_id, thread_id, "dead", display)
    lifecycle_strategy.start_autoclose_timer(
        user_id, thread_id, "dead", time.monotonic()
    )

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
    sent = await rate_limit_send_message(
        bot,
        chat_id,
        text,
        message_thread_id=thread_id,
        reply_markup=keyboard,
    )
    if sent is None:
        try:
            await bot.unpin_all_forum_topic_messages(
                chat_id=chat_id, message_thread_id=thread_id
            )
        except BadRequest as probe_err:
            if (
                "thread not found" in probe_err.message.lower()
                or "topic_id_invalid" in probe_err.message.lower()
            ):
                terminal_strategy.get_state(wid).probe_failures = 0
                await clear_topic_state(user_id, thread_id, bot, window_id=wid)
                thread_router.unbind_thread(user_id, thread_id)
                logger.info(
                    "Topic deleted: unbound window %s for thread %d, user %d",
                    wid,
                    thread_id,
                    user_id,
                )
        except TelegramError:
            pass
    lifecycle_strategy.mark_dead_notified(user_id, thread_id, wid)


# ── Autoclose timer management ────────────────────────────────────────────


async def _check_autoclose_timers(bot: Bot) -> None:
    """Close topics whose done/dead timers have expired."""
    all_topics = lifecycle_strategy.iter_autoclose_timers()
    if not all_topics:
        return

    now = time.monotonic()
    expired: list[tuple[int, int]] = []
    for user_id, thread_id, ts in all_topics:
        if ts.autoclose is None:
            continue
        state, entered_at = ts.autoclose
        if state == "done":
            timeout = config.autoclose_done_minutes * 60
        elif state == "dead":
            timeout = config.autoclose_dead_minutes * 60
        else:
            continue
        if timeout > 0 and now - entered_at >= timeout:
            expired.append((user_id, thread_id))

    for user_id, thread_id in expired:
        await _close_expired_topic(bot, user_id, thread_id)


async def _close_expired_topic(bot: Bot, user_id: int, thread_id: int) -> None:
    """Attempt to close/delete an expired topic and clean up state."""
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    window_id = thread_router.get_window_for_thread(user_id, thread_id)
    removed = False
    try:
        await bot.delete_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
        removed = True
    except TelegramError:
        try:
            await bot.close_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
            removed = True
        except TelegramError as e:
            logger.debug("Failed to auto-close topic thread=%d: %s", thread_id, e)
    if removed:
        lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
        logger.info(
            "Auto-removed topic: chat=%d thread=%d (user=%d)",
            chat_id,
            thread_id,
            user_id,
        )
        await clear_topic_state(user_id, thread_id, bot=bot, window_id=window_id)
        thread_router.unbind_thread(user_id, thread_id)


# ── Unbound window TTL ────────────────────────────────────────────────────


async def _check_unbound_window_ttl(live_windows: list | None = None) -> None:
    """Kill unbound tmux windows whose TTL has expired."""
    timeout = config.autoclose_done_minutes * 60
    if timeout <= 0:
        return

    bound_ids: set[str] = set()
    for _, _, wid in thread_router.iter_thread_bindings():
        bound_ids.add(wid)

    if live_windows is None:
        live_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in live_windows}

    terminal_strategy.clear_unbound_timers(bound_ids, live_ids)

    now = time.monotonic()
    for w in live_windows:
        if w.window_id not in bound_ids and not is_foreign_window(w.window_id):
            ws = terminal_strategy.get_state(w.window_id)
            if ws.unbound_timer is None:
                ws.unbound_timer = now

    await _kill_expired_unbound(now, timeout)
    _prune_orphaned_poll_state(live_ids, bound_ids)


async def _kill_expired_unbound(now: float, timeout: float) -> None:
    """Find and kill unbound windows past their TTL."""
    expired = terminal_strategy.get_expired_unbound(now, timeout)
    for wid in expired:
        from ..tmux_manager import clear_vim_state

        clear_vim_state(wid)
        await tmux_manager.kill_window(wid)
        clear_window_poll_state(wid)
        logger.info("Auto-killed unbound window %s (TTL expired)", wid)


def _prune_orphaned_poll_state(live_ids: set[str], bound_ids: set[str]) -> None:
    """Remove poll state for windows that are neither live nor bound."""
    for wid in terminal_strategy.get_orphaned_window_ids(live_ids, bound_ids):
        clear_window_poll_state(wid)


# ── Display name sync / state pruning ─────────────────────────────────────


async def _prune_stale_state(live_windows: list) -> None:
    """Sync display names and prune orphaned state entries."""
    live_ids = {w.window_id for w in live_windows}
    live_pairs = [(w.window_id, w.window_name) for w in live_windows]
    session_manager.sync_display_names(live_pairs)
    session_manager.prune_stale_state(live_ids)


# ── Topic existence probing ───────────────────────────────────────────────


async def _probe_topic_existence(bot: Bot) -> None:
    """Probe all bound topics via Telegram API; detect deleted topics."""
    for user_id, thread_id, wid in list(thread_router.iter_thread_bindings()):
        if terminal_strategy.get_state(wid).probe_failures >= _MAX_PROBE_FAILURES:
            continue
        try:
            await bot.unpin_all_forum_topic_messages(
                chat_id=thread_router.resolve_chat_id(user_id, thread_id),
                message_thread_id=thread_id,
            )
            terminal_strategy.get_state(wid).probe_failures = 0
        except TelegramError as e:
            if isinstance(e, BadRequest) and (
                "Topic_id_invalid" in e.message
                or "thread not found" in e.message.lower()
            ):
                w = await tmux_manager.find_window_by_id(wid)
                if w:
                    await tmux_manager.kill_window(w.window_id)
                terminal_strategy.get_state(wid).probe_failures = 0
                await clear_topic_state(user_id, thread_id, bot, window_id=wid)
                thread_router.unbind_thread(user_id, thread_id)
                logger.info(
                    "Topic deleted: killed window_id '%s' and "
                    "unbound thread %d for user %d",
                    wid,
                    thread_id,
                    user_id,
                )
            else:
                count = lifecycle_strategy.record_probe_failure(wid)
                if count < _MAX_PROBE_FAILURES:
                    log_throttled(
                        logger,
                        f"topic-probe:{wid}",
                        "Topic probe error for %s: %s",
                        wid,
                        e,
                    )


# ── Main orchestration ──────────────────────────────────────────────────


async def update_status_message(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    *,
    _window: "TmuxWindow | None" = None,
) -> None:
    """Poll terminal and enqueue status update for user's active window."""
    w = _window or await tmux_manager.find_window_by_id(window_id)
    if not w:
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)
        return

    pane_text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not pane_text:
        return

    interactive_window = get_interactive_window(user_id, thread_id)
    should_check_new_ui = True

    status = _parse_with_pyte(
        window_id, pane_text, columns=w.pane_width, rows=w.pane_height
    )

    # Passive vim INSERT mode tracking
    from ..tmux_manager import _has_insert_indicator, notify_vim_insert_seen

    ws = _get_window_state(window_id)
    vim_text = ws.last_rendered_text if ws.last_rendered_text is not None else pane_text
    if _has_insert_indicator(vim_text):
        notify_vim_insert_seen(w.window_id)

    if status is None:
        clean_text = (
            ws.last_rendered_text if ws.last_rendered_text is not None else pane_text
        )
        provider = get_provider_for_window(window_id)
        pane_title = ""
        if provider.capabilities.uses_pane_title:
            pane_title = await tmux_manager.get_pane_title(w.window_id)
        status = provider.parse_terminal_status(clean_text, pane_title=pane_title)

    if interactive_window == window_id:
        if status is not None and status.is_interactive:
            return
        await clear_interactive_msg(user_id, bot, thread_id)
        should_check_new_ui = False
    elif interactive_window is not None:
        await clear_interactive_msg(user_id, bot, thread_id)

    if should_check_new_ui and status is not None and status.is_interactive:
        await handle_interactive_ui(bot, user_id, window_id, thread_id)
        return

    status_line = status.display_label if status and not status.is_interactive else None

    notif_mode = session_manager.get_notification_mode(window_id)

    if status_line:
        ws = _get_window_state(window_id)
        ws.has_seen_status = True
        ws.startup_time = None
        await _send_typing_throttled(bot, user_id, thread_id)
        if notif_mode not in ("muted", "errors_only"):
            from .hook_events import build_subagent_label, get_subagent_names

            subagent_names = get_subagent_names(window_id)
            display_status = status_line
            if subagent_names:
                label = build_subagent_label(subagent_names)
                display_status = f"{status_line} ({label})"
            await enqueue_status_update(
                bot,
                user_id,
                window_id,
                display_status,
                thread_id=thread_id,
            )
        if thread_id is not None:
            chat_id = thread_router.resolve_chat_id(user_id, thread_id)
            display = thread_router.get_display_name(window_id)
            await update_topic_emoji(bot, chat_id, thread_id, "active", display)
            lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
    else:
        await _handle_no_status(
            bot, user_id, window_id, thread_id, w.pane_current_command, notif_mode
        )


# ── Main loop ─────────────────────────────────────────────────────────────


async def status_poll_loop(bot: Bot) -> None:
    """Background task to poll terminal status for all thread-bound windows."""
    logger.info("Status polling started (interval: %ss)", STATUS_POLL_INTERVAL)
    last_topic_check = 0.0
    _error_streak = 0
    while True:
        try:
            all_windows = await tmux_manager.list_windows()
            external_windows = await tmux_manager.discover_external_sessions()
            all_windows.extend(external_windows)
            window_lookup: dict[str, "TmuxWindow"] = {
                w.window_id: w for w in all_windows
            }

            now = time.monotonic()
            if now - last_topic_check >= TOPIC_CHECK_INTERVAL:
                last_topic_check = now
                await _prune_stale_state(all_windows)
                await _probe_topic_existence(bot)
                log_throttle_sweep()

            for user_id, thread_id, wid in list(thread_router.iter_thread_bindings()):
                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(window_id=wid)
                try:
                    if lifecycle_strategy.is_dead_notified(user_id, thread_id, wid):
                        continue

                    w = window_lookup.get(wid)
                    if not w:
                        await _handle_dead_window_notification(
                            bot, user_id, thread_id, wid
                        )
                        continue

                    await _maybe_discover_transcript(
                        wid,
                        _window=w,
                        bot=bot,
                        user_id=user_id,
                        thread_id=thread_id,
                    )

                    queue = get_message_queue(user_id)
                    if queue and not queue.empty():
                        await _check_interactive_only(
                            bot, user_id, wid, thread_id, _window=w
                        )
                        await _scan_window_panes(bot, user_id, wid, thread_id)
                        await _maybe_check_passive_shell(bot, user_id, wid, thread_id)
                        continue
                    await update_status_message(
                        bot,
                        user_id,
                        wid,
                        thread_id=thread_id,
                        _window=w,
                    )
                    await _scan_window_panes(bot, user_id, wid, thread_id)
                    await _maybe_check_passive_shell(bot, user_id, wid, thread_id)
                except (TelegramError, OSError) as e:
                    log_throttled(
                        logger,
                        f"status-update:{user_id}:{thread_id}",
                        "Status update error for user %s thread %s: %s",
                        user_id,
                        thread_id,
                        e,
                    )

            await _check_autoclose_timers(bot)
            await _check_unbound_window_ttl(all_windows)

        except _LoopError:
            logger.exception("Status poll loop error")
            backoff_delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**_error_streak))
            _error_streak += 1
            await asyncio.sleep(backoff_delay)
            continue
        except Exception:
            logger.exception("Unexpected error in status poll loop")
            backoff_delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**_error_streak))
            _error_streak += 1
            await asyncio.sleep(backoff_delay)
            continue

        _error_streak = 0
        await asyncio.sleep(STATUS_POLL_INTERVAL)
