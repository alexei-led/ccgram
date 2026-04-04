"""Hook event dispatcher — routes structured events to handlers.

Receives HookEvent objects from the session monitor's event reader and
dispatches them to the appropriate handler based on event type. This
provides instant, structured notification of agent state changes instead
of relying solely on terminal scraping.

Key function: dispatch_hook_event().
"""

from collections.abc import Sequence

import structlog

from telegram import Bot

from ..claude_task_state import claude_task_state, classify_wait_message
from ..providers.base import HookEvent
from ..session import session_manager
from ..thread_router import thread_router
from ..topic_state_registry import topic_state

logger = structlog.get_logger()

_WINDOW_KEY_PARTS = 2


def _resolve_users_for_window_key(
    window_key: str,
) -> list[tuple[int, int, str]]:
    """Resolve window_key to list of (user_id, thread_id, window_id).

    The window_key format is "tmux_session:window_id" (e.g. "ccgram:@0").
    We extract the window_id part and look up thread bindings.
    """
    # Extract window_id from key (e.g. "ccgram:@0" -> "@0")
    parts = window_key.rsplit(":", 1)
    if len(parts) < _WINDOW_KEY_PARTS:
        return []
    window_id = parts[1]

    results: list[tuple[int, int, str]] = []
    for user_id, thread_id, bound_wid in thread_router.iter_thread_bindings():
        if bound_wid == window_id:
            results.append((user_id, thread_id, window_id))
    return results


async def _handle_notification(event: HookEvent, bot: Bot) -> None:
    """Handle a Notification event — render interactive UI."""
    from .interactive_ui import (
        clear_interactive_mode,
        get_interactive_window,
        handle_interactive_ui,
        set_interactive_mode,
    )
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        logger.debug(
            "No users bound for notification event window_key=%s", event.window_key
        )
        return

    tool_name = event.data.get("tool_name", "")
    logger.debug(
        "Hook notification: tool_name=%s, window_key=%s",
        tool_name,
        event.window_key,
    )
    wait_header = classify_wait_message(event.data.get("message", ""))

    for user_id, thread_id, window_id in users:
        if wait_header:
            claude_task_state.set_wait_header(window_id, wait_header)
            await enqueue_status_update(
                bot, user_id, window_id, None, thread_id=thread_id
            )

        # Skip if already in interactive mode for this window
        existing = get_interactive_window(user_id, thread_id)
        if existing == window_id:
            logger.debug(
                "Interactive mode already set for user=%d window=%s, skipping",
                user_id,
                window_id,
            )
            continue

        # Set interactive mode before rendering to prevent racing with terminal scraping
        set_interactive_mode(user_id, window_id, thread_id)

        # Wait briefly for Claude Code to render the UI in the terminal
        import asyncio

        await asyncio.sleep(0.3)

        handled = await handle_interactive_ui(bot, user_id, window_id, thread_id)
        if not handled:
            clear_interactive_mode(user_id, thread_id)


async def _enhance_with_llm_summary(
    bot: Bot,
    users: Sequence[tuple[int, int | None, str]],
    window_id: str,
    transcript_path: str,
    num_turns: int,
) -> None:
    """Async enhancement: replace Ready header with LLM summary via status queue."""
    try:
        from ..llm.summarizer import summarize_completion

        summary = await summarize_completion(transcript_path)
        if not summary:
            return

        from .message_queue import enqueue_status_update

        enhanced = claude_task_state.format_completion_text(
            window_id, num_turns=num_turns
        )
        enhanced = enhanced.replace("\u2713 Ready", f"\u2713 Done \u2014 {summary}", 1)

        for user_id, thread_id, _window_id in users:
            notif_mode = session_manager.get_notification_mode(window_id)
            if notif_mode not in ("muted", "errors_only"):
                await enqueue_status_update(
                    bot, user_id, window_id, enhanced, thread_id=thread_id
                )
    except RuntimeError, OSError, ValueError:
        logger.debug("LLM summary enhancement failed", exc_info=True)


async def _handle_stop(event: HookEvent, bot: Bot) -> None:
    """Handle a Stop event — transition status directly to idle.

    Topic emoji remains poller-owned. Hook-driven idle flips can fight the
    transcript/activity heuristic and cause active/idle rename churn on quiet
    topics, so Stop only updates the status bubble and broker delivery state.
    Muted/errors_only windows get their status cleared instead.
    """
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    stop_reason = event.data.get("stop_reason", "")
    logger.debug(
        "Hook stop: window_key=%s, stop_reason=%s",
        event.window_key,
        stop_reason,
    )

    num_turns = event.data.get("num_turns", 0)
    for user_id, thread_id, window_id in users:
        claude_task_state.clear_wait_header(window_id)
        notif_mode = session_manager.get_notification_mode(window_id)
        if notif_mode in ("muted", "errors_only"):
            status_text = None
        else:
            status_text = claude_task_state.format_completion_text(
                window_id, num_turns=num_turns
            )
        await enqueue_status_update(
            bot, user_id, window_id, status_text, thread_id=thread_id
        )

    # Trigger immediate broker delivery for the idle window
    from .periodic_tasks import run_broker_cycle

    await run_broker_cycle(bot, idle_windows=frozenset({event.window_key}))

    # Fire async LLM summary enhancement (non-blocking)
    first_window_id = users[0][2]
    if first_window_id:
        transcript_path = session_manager.get_window_state(
            first_window_id
        ).transcript_path
        if transcript_path:
            import asyncio

            asyncio.create_task(
                _enhance_with_llm_summary(
                    bot, users, first_window_id, transcript_path, num_turns
                )
            )


# Track active subagents per window: window_id -> {subagent_id -> name}
_active_subagents: dict[str, dict[str, str]] = {}

_MAX_DISPLAYED_NAMES = 3


def get_subagent_names(window_id: str) -> list[str]:
    """Return names of active subagents for a window."""
    return list(_active_subagents.get(window_id, {}).values())


def build_subagent_label(names: list[str]) -> str | None:
    """Build a display label for active subagents.

    Returns None if no subagents are active.
    """
    if not names:
        return None
    if len(names) == 1:
        return f"\U0001f916 {names[0]}"
    joined = ", ".join(names[:_MAX_DISPLAYED_NAMES])
    return f"\U0001f916 {len(names)} subagents: {joined}"


@topic_state.register("window")
def clear_subagents(window_id: str) -> None:
    """Clear all subagent tracking for a window."""
    _active_subagents.pop(window_id, None)


async def _handle_subagent_start(event: HookEvent, bot: Bot) -> None:
    """Handle SubagentStart — track active subagent and notify."""
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    window_id = users[0][2]  # all users share the same window_id
    subagent_id = event.data.get("subagent_id", "")
    name = (
        (event.data.get("name") or "").strip()
        or (event.data.get("description") or "").strip()
        or subagent_id[:12]
        or "subagent"
    )

    _active_subagents.setdefault(window_id, {})[subagent_id] = name

    logger.debug(
        "Subagent started: window=%s, count=%d, name=%s",
        window_id,
        len(_active_subagents[window_id]),
        name,
    )

    for user_id, thread_id, _ in users:
        await enqueue_status_update(
            bot,
            user_id,
            window_id,
            f"\U0001f916 Subagent started: {name}",
            thread_id=thread_id,
        )


async def _handle_subagent_stop(event: HookEvent, bot: Bot) -> None:
    """Handle SubagentStop — remove subagent from tracking and notify."""
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    window_id = users[0][2]
    subagent_id = event.data.get("subagent_id", "")

    agents = _active_subagents.get(window_id)
    if not agents:
        return
    name = agents.pop(subagent_id, subagent_id[:12] or "subagent")
    if not agents:
        _active_subagents.pop(window_id, None)

    logger.debug(
        "Subagent stopped: window=%s, remaining=%d, name=%s",
        window_id,
        len(_active_subagents.get(window_id, {})),
        name,
    )

    for user_id, thread_id, _ in users:
        await enqueue_status_update(
            bot,
            user_id,
            window_id,
            f"\U0001f916 Subagent done: {name}",
            thread_id=thread_id,
        )


async def _handle_teammate_idle(event: HookEvent, bot: Bot) -> None:
    """Handle TeammateIdle — notify topic that a teammate went idle."""
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    teammate_name = event.data.get("teammate_name", "unknown")
    logger.info(
        "Teammate idle: window_key=%s, teammate=%s",
        event.window_key,
        teammate_name,
    )

    for user_id, thread_id, window_id in users:
        text = f"\U0001f4a4 Teammate '{teammate_name}' went idle"
        await enqueue_status_update(bot, user_id, window_id, text, thread_id=thread_id)


async def _handle_stop_failure(event: HookEvent, bot: Bot) -> None:
    """Handle a StopFailure event — alert on API error termination."""
    from .message_sender import rate_limit_send_message

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    error = event.data.get("error", "unknown")
    error_details = event.data.get("error_details", "")
    logger.warning(
        "Hook StopFailure: window_key=%s, error=%s, details=%s",
        event.window_key,
        error,
        error_details,
    )

    detail = f": {error_details}" if error_details else ""
    text = f"\u26a0 API error — {error}{detail}"

    for user_id, thread_id, _window_id in users:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        await rate_limit_send_message(bot, chat_id, text, message_thread_id=thread_id)


async def _handle_session_end(event: HookEvent, bot: Bot) -> None:
    """Handle a SessionEnd event — clean up session lifecycle."""
    from .message_queue import enqueue_status_update
    from .polling_strategies import clear_seen_status
    from .topic_emoji import update_topic_emoji

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    reason = event.data.get("reason", "")
    logger.info(
        "Hook SessionEnd: window_key=%s, reason=%s",
        event.window_key,
        reason,
    )

    # Clear session association and subagent tracking so next launch starts fresh
    if users:
        window_id = users[0][2]
        claude_task_state.clear_window(window_id)
        session_manager.clear_window_session(window_id)
        clear_subagents(window_id)

    for user_id, thread_id, window_id in users:
        clear_seen_status(window_id)
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        display = thread_router.get_display_name(window_id)
        await update_topic_emoji(bot, chat_id, thread_id, "done", display)
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)


async def _handle_task_completed(event: HookEvent, bot: Bot) -> None:
    """Handle TaskCompleted — notify topic that a task was completed."""
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    task_subject = event.data.get("task_subject", "")
    teammate_name = event.data.get("teammate_name", "")
    logger.info(
        "Task completed: window_key=%s, task=%s, by=%s",
        event.window_key,
        task_subject,
        teammate_name,
    )

    for user_id, thread_id, window_id in users:
        task_id = event.data.get("task_id", "")
        tracked = False
        if task_id:
            tracked = claude_task_state.mark_task_completed(
                window_id,
                event.session_id,
                task_id,
                subject=task_subject,
            )
        if tracked or claude_task_state.has_snapshot(window_id):
            await enqueue_status_update(
                bot, user_id, window_id, None, thread_id=thread_id
            )
            continue

        text = f"\u2705 Task completed: {task_subject}"
        if teammate_name:
            text += f" (by '{teammate_name}')"
        await enqueue_status_update(bot, user_id, window_id, text, thread_id=thread_id)


async def dispatch_hook_event(event: HookEvent, bot: Bot) -> None:
    """Route hook events to appropriate handlers."""
    match event.event_type:
        case "Notification":
            await _handle_notification(event, bot)
        case "Stop":
            await _handle_stop(event, bot)
        case "StopFailure":
            await _handle_stop_failure(event, bot)
        case "SessionEnd":
            await _handle_session_end(event, bot)
        case "SubagentStart":
            await _handle_subagent_start(event, bot)
        case "SubagentStop":
            await _handle_subagent_stop(event, bot)
        case "TeammateIdle":
            await _handle_teammate_idle(event, bot)
        case "TaskCompleted":
            await _handle_task_completed(event, bot)
        case (
            "SessionStart"
            | "UserPromptSubmit"
            | "PreToolUse"
            | "PostToolUse"
            | "PostToolUseFailure"
            | "PermissionRequest"
            | "ConfigChange"
            | "WorktreeCreate"
            | "WorktreeRemove"
            | "PreCompact"
        ):
            pass  # Not actionable for the bot — SessionStart handled via session_map.json
        case _:
            logger.debug("Ignoring unknown hook event type: %s", event.event_type)
