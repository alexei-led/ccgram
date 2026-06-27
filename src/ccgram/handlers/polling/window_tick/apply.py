"""Side-effecting transition functions for window_tick.

All Telegram, tmux, and singleton mutations live here. Functions accept
the inputs gathered by ``observe`` and the decision returned by
``decide``, and apply the resulting effects: emoji updates, status
enqueuing, typing indicators, autoclose timers, dead-window
notifications, multi-pane scans, passive shell relay.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError

from .... import window_query
from ....claude_task_state import (
    build_subagent_label,
    claude_task_state,
    get_subagent_names,
)
from ....config import config
from ....providers import get_provider_for_window
from ....telegram_client import PTBTelegramClient
from ....thread_router import thread_router
from ....multiplexer import agent_status_cache
from ....multiplexer import multiplexer as tmux_manager
from ....window_state_ports.pane_state import (
    get_pane_lifecycle_notify,
    get_pane_projection,
)
from ...callback_data import IDLE_STATUS_TEXT
from ...cleanup import clear_topic_state
from ...interactive import (
    clear_interactive_mode,
    clear_interactive_msg,
    get_interactive_window,
    handle_interactive_ui,
    set_interactive_mode,
)
from ...messaging_pipeline.message_queue import (
    clear_tool_msg_ids_for_topic,
    enqueue_status_update,
)
from ...messaging_pipeline.message_sender import rate_limit_send_message, safe_send
from ...recovery.recovery_banner import RecoveryBanner, render_banner
from ...status.topic_emoji import update_topic_emoji
from ..polling_state import (
    lifecycle_strategy,
    pane_status_strategy,
    terminal_poll_state,
)
from ..polling_types import PaneTransition, TickDecision
from .decide import decide_tick
from .observe import _check_vim_insert, _resolve_status, build_context

if TYPE_CHECKING:
    from telegram import Bot

    from ....providers.base import AgentProvider
    from ....multiplexer.base import WindowRef as TmuxWindow

logger = structlog.get_logger()


def _get_provider(window_id: str) -> "AgentProvider":
    return get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    )


# ── Typing throttle ─────────────────────────────────────────────────────


async def _send_typing_throttled(
    bot: "Bot", user_id: int, thread_id: int | None
) -> None:
    if thread_id is None:
        return
    if lifecycle_strategy.is_typing_throttled(user_id, thread_id):
        return
    lifecycle_strategy.record_typing_sent(user_id, thread_id)
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    client = PTBTelegramClient(bot)
    with contextlib.suppress(TelegramError):
        await client.send_chat_action(
            chat_id=chat_id,
            message_thread_id=thread_id,
            action=ChatAction.TYPING,
        )


# ── Idle / no-status transitions ────────────────────────────────────────


async def _transition_to_idle(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int,
    chat_id: int,
    display: str,
) -> None:
    terminal_poll_state.cancel_startup_timer(window_id)
    client = PTBTelegramClient(bot)
    await update_topic_emoji(client, chat_id, thread_id, "idle", display)
    lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
    lifecycle_strategy.clear_typing_state(user_id, thread_id)
    await enqueue_status_update(
        client, user_id, window_id, IDLE_STATUS_TEXT, thread_id=thread_id
    )


# ── Multi-pane scanning (agent teams) ─────────────────────────────────


async def _surface_pane_alert(
    bot: "Bot", user_id: int, window_id: str, thread_id: int, pane_id: str
) -> None:
    await handle_interactive_ui(
        PTBTelegramClient(bot), user_id, window_id, thread_id, pane_id=pane_id
    )


_PANE_OUTPUT_PREVIEW_LINES = 12


async def _forward_pane_output(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int,
    pane_id: str,
    pane_text: str,
) -> None:
    """Forward a subscribed pane's freshly-captured text to its bound topic.

    Uses the screen buffer to strip ANSI, keeps the tail of the capture so
    the user sees the most-recent output, and labels the message with the
    pane's friendly name when one is set.
    """

    pane = get_pane_projection(window_id, pane_id)
    if pane is None or not pane.subscribed:
        return
    cleaned = pane_text.strip()
    if not cleaned:
        return
    lines = cleaned.splitlines()
    if len(lines) > _PANE_OUTPUT_PREVIEW_LINES:
        lines = lines[-_PANE_OUTPUT_PREVIEW_LINES:]
    label = f"{pane.name} ({pane_id})" if pane.name else pane_id
    body = "\n".join(lines)
    text = f"\U0001f4e1 {label}\n```\n{body}\n```"
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    try:
        await safe_send(
            PTBTelegramClient(bot), chat_id, text, message_thread_id=thread_id
        )
    except TelegramError as exc:
        logger.warning(
            "pane output forward failed",
            window_id=window_id,
            pane_id=pane_id,
            error=str(exc),
        )


async def _scan_window_panes(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int,
) -> None:
    """Delegate multi-pane scanning to ``PaneStatusStrategy``."""
    transitions = await pane_status_strategy.scan_window(
        bot,
        user_id,
        window_id,
        thread_id,
        on_blocked=_surface_pane_alert,
        on_pane_output=_forward_pane_output,
    )
    if transitions:
        await _notify_pane_lifecycle(bot, user_id, window_id, thread_id, transitions)


async def _notify_pane_lifecycle(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int,
    transitions: list[PaneTransition],
) -> None:
    """Emit one-line "pane created"/"pane closed" notifications when enabled."""
    enabled = get_pane_lifecycle_notify(window_id, config.pane_lifecycle_notify)
    if not enabled:
        return

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    for t in transitions:
        if t.prev_state is not None:
            continue
        if t.new_state == "dead":
            label = f"{t.name} ({t.pane_id})" if t.name else t.pane_id
            text = f"➖ pane {label} closed"
        else:
            pane = get_pane_projection(window_id, t.pane_id)
            label = f"{pane.name} ({t.pane_id})" if pane and pane.name else t.pane_id
            text = f"➕ pane {label} created"
        try:
            await safe_send(
                PTBTelegramClient(bot), chat_id, text, message_thread_id=thread_id
            )
        except TelegramError as exc:
            logger.warning(
                "pane lifecycle notify failed",
                window_id=window_id,
                pane_id=t.pane_id,
                error=str(exc),
            )


# ── Interactive-only check ───────────────────────────────────────────────


async def _check_interactive_only(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int,
    *,
    _window: "TmuxWindow | None" = None,
) -> None:
    w = _window or await tmux_manager.find_window_by_id(window_id)
    if not w:
        return

    if get_interactive_window(user_id, thread_id) == window_id:
        return

    pane_text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not pane_text:
        return

    status = await _resolve_status(window_id, pane_text, w)

    if status is not None and status.is_interactive:
        set_interactive_mode(user_id, window_id, thread_id)
        handled = await handle_interactive_ui(
            PTBTelegramClient(bot), user_id, window_id, thread_id
        )
        if not handled:
            clear_interactive_mode(user_id, thread_id)


# ── Passive shell relay ──────────────────────────────────────────────────


async def _maybe_check_passive_shell(
    bot: "Bot", user_id: int, window_id: str, thread_id: int
) -> None:
    if not _get_provider(window_id).capabilities.chat_first_command_path:
        return
    ws = terminal_poll_state.get_state(window_id)
    rendered = ws.last_rendered_text
    if rendered is None:
        raw = await tmux_manager.capture_pane(window_id)
        if not raw:
            return
        rendered = raw
    # Lazy: shell_capture is registered via callback_registry; importing
    # at top forms apply → shell_capture → polling cycle through the
    # shell prompt approval keyboard.
    # Lazy: shell.shell_capture imports apply indirectly through the broker
    from ...shell.shell_capture import check_passive_shell_output

    await check_passive_shell_output(
        PTBTelegramClient(bot), user_id, thread_id, window_id, rendered
    )


# ── Dead window notification ─────────────────────────────────────────────


async def _handle_dead_window_notification(
    bot: "Bot", user_id: int, thread_id: int, wid: str
) -> None:
    if lifecycle_strategy.is_dead_notified(user_id, thread_id, wid):
        return
    # Mark notified before the first await: the push (event-stream) and poll
    # paths both call this for the same window and could otherwise both pass the
    # guard above before either marks, sending two banners + two autoclose timers.
    lifecycle_strategy.mark_dead_notified(user_id, thread_id, wid)
    # Evict any push-cached status so a dead/replaced window (herdr reuses tab
    # ids across restart) can't serve a stale "working" to the status poll.
    agent_status_cache.clear(wid)
    terminal_poll_state.clear_seen_status(wid)

    clear_tool_msg_ids_for_topic(user_id, thread_id)
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    display = thread_router.get_display_name(wid)
    await update_topic_emoji(
        PTBTelegramClient(bot), chat_id, thread_id, "dead", display
    )
    lifecycle_strategy.start_autoclose_timer(
        user_id, thread_id, "dead", time.monotonic()
    )

    view = window_query.view_window(wid)
    cwd = view.cwd if view else ""
    try:
        dir_exists = bool(cwd) and await asyncio.to_thread(Path(cwd).is_dir)
    except OSError:
        dir_exists = False
    if dir_exists:
        banner = RecoveryBanner(
            chat_id=chat_id,
            thread_id=thread_id,
            window_id=wid,
            mode="dead",
            provider=window_query.get_window_provider(wid),
            display=display,
            cwd=cwd,
        )
        text, keyboard = render_banner(banner)
    else:
        text = f"⚠ Session `{display}` ended."
        keyboard = None
    sent = await rate_limit_send_message(
        PTBTelegramClient(bot),
        chat_id,
        text,
        message_thread_id=thread_id,
        reply_markup=keyboard,
    )
    if sent is None:
        client = PTBTelegramClient(bot)
        try:
            await client.unpin_all_forum_topic_messages(
                chat_id=chat_id, message_thread_id=thread_id
            )
        except BadRequest as probe_err:
            if (
                "thread not found" in probe_err.message.lower()
                or "topic_id_invalid" in probe_err.message.lower()
            ):
                terminal_poll_state.reset_probe_failures(wid)
                await clear_topic_state(
                    user_id,
                    thread_id,
                    client,
                    window_id=wid,
                    window_dead=True,
                )
                thread_router.unbind_thread(user_id, thread_id)
                logger.info(
                    "Topic deleted: unbound window %s for thread %d, user %d",
                    wid,
                    thread_id,
                    user_id,
                )
        except TelegramError:
            pass


# ── Decision-application transitions ───────────────────────────────────


async def _apply_active_transition(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None,
    decision: TickDecision,
) -> None:
    if decision.send_status:
        claude_task_state.clear_wait_header(window_id)
        claude_task_state.set_last_status(window_id, decision.status_text or "")
        terminal_poll_state.mark_seen_status(window_id)
        await _send_typing_throttled(bot, user_id, thread_id)
        subagent_names = get_subagent_names(window_id)
        display_status = decision.status_text or ""
        if subagent_names:
            label = build_subagent_label(subagent_names)
            display_status = f"{display_status} ({label})"
        await enqueue_status_update(
            PTBTelegramClient(bot),
            user_id,
            window_id,
            display_status,
            thread_id=thread_id,
        )
    else:
        claude_task_state.clear_wait_header(window_id)
        await _send_typing_throttled(bot, user_id, thread_id)
    if thread_id is not None:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        display = thread_router.get_display_name(window_id)
        await update_topic_emoji(
            PTBTelegramClient(bot), chat_id, thread_id, "active", display
        )
        lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)


async def _apply_done_transition(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None,
) -> None:
    if thread_id is None:
        return
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    display = thread_router.get_display_name(window_id)
    terminal_poll_state.cancel_startup_timer(window_id)
    client = PTBTelegramClient(bot)
    await update_topic_emoji(client, chat_id, thread_id, "done", display)
    lifecycle_strategy.start_autoclose_timer(
        user_id, thread_id, "done", time.monotonic()
    )
    lifecycle_strategy.clear_typing_state(user_id, thread_id)
    await enqueue_status_update(client, user_id, window_id, None, thread_id=thread_id)
    if not _get_provider(window_id).capabilities.supports_hook:
        terminal_poll_state.mark_seen_status(window_id)


async def _apply_starting_transition(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None,
) -> None:
    ws = terminal_poll_state.peek_state(window_id)
    if ws is None or ws.startup_time is None:
        terminal_poll_state.begin_startup_timer(window_id, time.monotonic())
    await _send_typing_throttled(bot, user_id, thread_id)
    if thread_id is not None:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        display = thread_router.get_display_name(window_id)
        await update_topic_emoji(
            PTBTelegramClient(bot), chat_id, thread_id, "active", display
        )
        lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)


async def _apply_tick_decision(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None,
    decision: TickDecision,
) -> None:
    """Apply the effects dictated by a ``TickDecision``. All I/O lives here."""
    if decision.show_recovery or decision.transition is None:
        return

    if decision.transition == "active":
        await _apply_active_transition(bot, user_id, window_id, thread_id, decision)
    elif decision.transition == "idle" and thread_id is not None:
        await _transition_to_idle(
            bot,
            user_id,
            window_id,
            thread_id,
            thread_router.resolve_chat_id(user_id, thread_id),
            thread_router.get_display_name(window_id),
        )
    elif decision.transition == "done":
        await _apply_done_transition(bot, user_id, window_id, thread_id)
    elif decision.transition == "starting":
        await _apply_starting_transition(bot, user_id, window_id, thread_id)


# ── Status-update orchestration ─────────────────────────────────────────


async def _update_status(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    *,
    _window: "TmuxWindow | None" = None,
) -> None:
    w = _window or await tmux_manager.find_window_by_id(window_id)
    if not w:
        await enqueue_status_update(
            PTBTelegramClient(bot), user_id, window_id, None, thread_id=thread_id
        )
        return

    pane_text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not pane_text:
        return

    _check_vim_insert(window_id, pane_text, w)
    status = await _resolve_status(window_id, pane_text, w)

    interactive_window = get_interactive_window(user_id, thread_id)
    should_check_new_ui = True

    client = PTBTelegramClient(bot)
    if interactive_window == window_id:
        if status is not None and status.is_interactive:
            return
        await clear_interactive_msg(user_id, client, thread_id)
        should_check_new_ui = False
    elif interactive_window is not None:
        await clear_interactive_msg(user_id, client, thread_id)

    if should_check_new_ui and status is not None and status.is_interactive:
        await handle_interactive_ui(client, user_id, window_id, thread_id)
        return

    ctx = build_context(window_id, w, status)
    decision = decide_tick(ctx)
    await _apply_tick_decision(
        bot,
        user_id,
        window_id,
        thread_id,
        decision,
    )


__all__ = [
    "_apply_active_transition",
    "_apply_done_transition",
    "_apply_starting_transition",
    "_apply_tick_decision",
    "_check_interactive_only",
    "_forward_pane_output",
    "_handle_dead_window_notification",
    "_maybe_check_passive_shell",
    "_notify_pane_lifecycle",
    "_scan_window_panes",
    "_send_typing_throttled",
    "_surface_pane_alert",
    "_transition_to_idle",
    "_update_status",
]
