"""Side-effecting transition functions for window_tick.

All Telegram, tmux, and singleton mutations live here. Functions accept
the inputs gathered by ``observe`` and the decision returned by
``decide``, and apply the resulting effects: emoji updates, status
enqueuing, typing indicators, dead-window
notifications, multi-pane scans, passive shell relay.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

import structlog
from telegram.constants import ChatAction
from telegram.error import TelegramError

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
from ....multiplexer.base import canonical_window_id
from ....multiplexer.reconciliation import window_presence
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
from ...messaging_pipeline.message_sender import safe_send
from ...status.topic_emoji import update_topic_emoji
from ...topics.topic_deletion import retire_topic_binding
from ...topics.topic_orchestration import is_pending_creation
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
    from ..polling_runtime import PollingRuntime

logger = structlog.get_logger()


def _get_provider(window_id: str) -> "AgentProvider":
    return get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    )


# ── Typing throttle ─────────────────────────────────────────────────────


async def _send_typing_throttled(
    bot: "Bot",
    user_id: int,
    thread_id: int | None,
    runtime: "PollingRuntime | None" = None,
) -> None:
    if thread_id is None:
        return
    lc = runtime.lifecycle if runtime is not None else lifecycle_strategy
    if lc.is_typing_throttled(user_id, thread_id):
        return
    lc.record_typing_sent(user_id, thread_id)
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
    runtime: "PollingRuntime | None" = None,
    *,
    send_status: bool = True,
) -> None:
    """Idle transition; ``send_status=False`` does the side effects only.

    Used for never-active windows on startup timeout: the window must be
    settled and its typing/emoji restored, but a fresh "Ready" bubble would
    be restart noise (issue #180).
    """
    ps = runtime.poll_state if runtime is not None else terminal_poll_state
    lc = runtime.lifecycle if runtime is not None else lifecycle_strategy
    # Settle the window, don't just stop its clock: clearing the startup
    # timestamp alone leaves it indistinguishable from one that never started,
    # so the very next tick decides "starting" again — green topic, typing
    # indicator, another 30s grace, idle, repeat. A quiet settlement also
    # cannot use has_seen_status: that flag means a genuine status was shown.
    if send_status:
        ps.mark_seen_status(window_id)
        ps.mark_idle_status_announced(window_id)
    else:
        ps.mark_startup_quietly_settled(window_id)
    client = PTBTelegramClient(bot)
    await update_topic_emoji(client, chat_id, thread_id, "idle", display)
    lc.clear_typing_state(user_id, thread_id)
    if not send_status:
        return
    await enqueue_status_update(
        client,
        user_id,
        window_id,
        IDLE_STATUS_TEXT,
        thread_id=thread_id,
        transient=True,
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
    runtime: "PollingRuntime | None" = None,
) -> None:
    """Delegate multi-pane scanning to ``PaneStatusStrategy``."""
    pss = runtime.pane_status if runtime is not None else pane_status_strategy
    transitions = await pss.scan_window(
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
    runtime: "PollingRuntime | None" = None,
) -> None:
    w = _window or await tmux_manager.find_window_by_id(window_id)
    if not w:
        return

    if get_interactive_window(user_id, thread_id) == window_id:
        return

    pane_text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not pane_text:
        return

    status = await _resolve_status(window_id, pane_text, w, runtime=runtime)

    if status is not None and status.is_interactive:
        set_interactive_mode(user_id, window_id, thread_id)
        handled = await handle_interactive_ui(
            PTBTelegramClient(bot), user_id, window_id, thread_id
        )
        if not handled:
            clear_interactive_mode(user_id, thread_id)


# ── Passive shell relay ──────────────────────────────────────────────────


async def _maybe_check_passive_shell(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int,
    runtime: "PollingRuntime | None" = None,
) -> None:
    if not _get_provider(window_id).capabilities.chat_first_command_path:
        return
    ps = runtime.poll_state if runtime is not None else terminal_poll_state
    ws = ps.get_state(window_id)
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
    bot: "Bot",
    user_id: int,
    thread_id: int,
    wid: str,
    runtime: "PollingRuntime | None" = None,
) -> None:
    lc = runtime.lifecycle if runtime is not None else lifecycle_strategy
    ps = runtime.poll_state if runtime is not None else terminal_poll_state
    if lc.is_dead_notified(user_id, thread_id, wid) or is_pending_creation(wid):
        return
    # Mark notified before the first await: the push (event-stream) and poll
    # paths both call this for the same window and could otherwise both pass the
    # guard above before either marks, sending two notifications.
    lc.mark_dead_notified(user_id, thread_id, wid)
    try:
        chat_ids = _exact_dead_topic_chat_ids(user_id, thread_id, wid)
        if chat_ids is None:
            return
        presence = await window_presence(wid, tmux_manager)
        if presence is not False or is_pending_creation(wid):
            return
        # Evict any push-cached status so a dead/replaced window (herdr reuses
        # tab ids across restart) can't serve stale status to a later bind.
        agent_status_cache.clear(wid)
        ps.clear_seen_status(wid)
        clear_tool_msg_ids_for_topic(user_id, thread_id)
        for chat_id in chat_ids:
            if is_pending_creation(wid):
                break
            outcome = await _delete_dead_topic_immediately(
                bot, user_id, thread_id, wid, chat_id, runtime=runtime
            )
            if outcome == "rate_limited":
                break
    finally:
        lc.clear_dead_notification(user_id, thread_id)


def _exact_dead_topic_chat_ids(
    user_id: int, thread_id: int, wid: str
) -> list[int] | None:
    """Return all known chat bindings for a dead window."""
    candidates = [
        chat_id
        for bound_user, chat_id, bound_thread, bound_wid in thread_router.iter_thread_bindings_with_chat()
        if bound_user == user_id
        and bound_thread == thread_id
        and canonical_window_id(bound_wid) == canonical_window_id(wid)
    ]
    if not candidates or any(chat_id is None for chat_id in candidates):
        return None
    return [chat_id for chat_id in candidates if chat_id is not None]


async def _delete_dead_topic_immediately(
    bot: "Bot",
    user_id: int,
    thread_id: int,
    wid: str,
    chat_id: int,
    *,
    runtime: "PollingRuntime | None" = None,
) -> str:
    """Retire and delete a confirmed-dead session topic without recovery UI."""
    lc = runtime.lifecycle if runtime is not None else lifecycle_strategy
    client = PTBTelegramClient(bot)

    async def clear_state_before_delete() -> None:
        await clear_topic_state(
            user_id,
            thread_id,
            client,
            window_id=wid,
            chat_id=chat_id,
            window_dead=True,
        )
        lc.mark_dead_notified(user_id, thread_id, wid)

    outcome = await retire_topic_binding(
        client,
        user_id,
        thread_id,
        wid,
        router=thread_router,
        chat_id=chat_id,
        before_delete=clear_state_before_delete,
    )
    logger.info(
        "dead_session_topic_cleanup",
        user_id=user_id,
        thread_id=thread_id,
        window_id=wid,
        outcome=outcome,
    )
    return outcome


# ── Decision-application transitions ───────────────────────────────────


async def _apply_active_transition(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None,
    decision: TickDecision,
    runtime: "PollingRuntime | None" = None,
) -> None:
    ps = runtime.poll_state if runtime is not None else terminal_poll_state
    if decision.send_status:
        claude_task_state.clear_wait_header(window_id)
        claude_task_state.set_last_status(window_id, decision.status_text or "")
        ps.mark_seen_status(window_id)
        await _send_typing_throttled(bot, user_id, thread_id, runtime=runtime)
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
            transient=True,
        )
    else:
        claude_task_state.clear_wait_header(window_id)
        await _send_typing_throttled(bot, user_id, thread_id, runtime=runtime)
    if thread_id is not None:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        display = thread_router.get_display_name(window_id)
        await update_topic_emoji(
            PTBTelegramClient(bot), chat_id, thread_id, "active", display
        )


async def _apply_done_transition(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None,
    runtime: "PollingRuntime | None" = None,
) -> None:
    if thread_id is None:
        return
    ps = runtime.poll_state if runtime is not None else terminal_poll_state
    lc = runtime.lifecycle if runtime is not None else lifecycle_strategy
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    display = thread_router.get_display_name(window_id)
    # Same reason as the idle transition: a window that reached done has
    # finished starting, whoever reports its completion. Leaving the flag
    # unset for hook-backed providers put exactly those windows — the ones
    # whose Stop hook makes done reliable — back into the startup grace on
    # the next tick, so a finished agent kept re-painting its topic green.
    ps.mark_seen_status(window_id)
    client = PTBTelegramClient(bot)
    await update_topic_emoji(client, chat_id, thread_id, "done", display)
    lc.clear_typing_state(user_id, thread_id)
    await enqueue_status_update(
        client,
        user_id,
        window_id,
        None,
        thread_id=thread_id,
        transient=True,
    )


async def _apply_starting_transition(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None,
    runtime: "PollingRuntime | None" = None,
) -> None:
    ps = runtime.poll_state if runtime is not None else terminal_poll_state
    ws = ps.peek_state(window_id)
    if ws is None or ws.startup_time is None:
        ps.begin_startup_timer(window_id, time.monotonic())
    await _send_typing_throttled(bot, user_id, thread_id, runtime=runtime)
    if thread_id is not None:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        display = thread_router.get_display_name(window_id)
        await update_topic_emoji(
            PTBTelegramClient(bot), chat_id, thread_id, "active", display
        )


async def _apply_tick_decision(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None,
    decision: TickDecision,
    runtime: "PollingRuntime | None" = None,
) -> None:
    """Apply the effects dictated by a ``TickDecision``. All I/O lives here."""
    if decision.show_recovery or decision.transition is None:
        return

    if decision.transition == "active":
        await _apply_active_transition(
            bot, user_id, window_id, thread_id, decision, runtime=runtime
        )
    elif decision.transition == "idle" and thread_id is not None:
        await _transition_to_idle(
            bot,
            user_id,
            window_id,
            thread_id,
            thread_router.resolve_chat_id(user_id, thread_id),
            thread_router.get_display_name(window_id),
            runtime=runtime,
            send_status=decision.send_status,
        )
    elif decision.transition == "done":
        await _apply_done_transition(
            bot, user_id, window_id, thread_id, runtime=runtime
        )
    elif decision.transition == "starting":
        await _apply_starting_transition(
            bot, user_id, window_id, thread_id, runtime=runtime
        )


# ── Status-update orchestration ─────────────────────────────────────────


async def _update_status(
    bot: "Bot",
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    *,
    _window: "TmuxWindow | None" = None,
    runtime: "PollingRuntime | None" = None,
) -> None:
    w = _window or await tmux_manager.find_window_by_id(window_id)
    if not w:
        await enqueue_status_update(
            PTBTelegramClient(bot),
            user_id,
            window_id,
            None,
            thread_id=thread_id,
            transient=True,
        )
        return

    pane_text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not pane_text:
        return

    _check_vim_insert(window_id, pane_text, w, runtime=runtime)
    status = await _resolve_status(window_id, pane_text, w, runtime=runtime)

    interactive_window = get_interactive_window(user_id, thread_id)
    should_check_new_ui = True

    client = PTBTelegramClient(bot)
    if interactive_window == window_id:
        if status is not None and status.is_interactive:
            return
        await clear_interactive_msg(user_id, client, thread_id)
        clear_interactive_mode(user_id, thread_id)
        should_check_new_ui = False
    elif interactive_window is not None:
        await clear_interactive_msg(user_id, client, thread_id)

    if should_check_new_ui and status is not None and status.is_interactive:
        await handle_interactive_ui(client, user_id, window_id, thread_id)
        return

    ctx = build_context(window_id, w, status, runtime=runtime)
    decision = decide_tick(ctx)
    await _apply_tick_decision(
        bot,
        user_id,
        window_id,
        thread_id,
        decision,
        runtime=runtime,
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
