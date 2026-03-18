"""Hook event dispatcher — routes structured events to handlers.

Receives HookEvent objects from the session monitor's event reader and
dispatches them to the appropriate handler based on event type. This
provides instant, structured notification of agent state changes instead
of relying solely on terminal scraping.

Key function: dispatch_hook_event().
"""

import structlog
from dataclasses import dataclass
from typing import Any

from telegram import Bot

from ..session import session_manager

logger = structlog.get_logger()

_WINDOW_KEY_PARTS = 2


@dataclass
class HookEvent:
    """A structured event from the hook event log."""

    event_type: str  # "Notification", "Stop", etc.
    window_key: str  # "ccgram:@0"
    session_id: str
    data: dict[str, Any]
    timestamp: float


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
    for user_id, thread_id, bound_wid in session_manager.iter_thread_bindings():
        if bound_wid == window_id:
            results.append((user_id, thread_id, window_id))
    return results


async def _handle_notification(event: HookEvent, bot: Bot) -> None:
    """Handle a Notification event — render interactive UI."""
    from .interactive_ui import (
        get_interactive_window,
        handle_interactive_ui,
        set_interactive_mode,
    )

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

    for user_id, thread_id, window_id in users:
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
            from .interactive_ui import clear_interactive_mode

            clear_interactive_mode(user_id, thread_id)


async def _handle_stop(event: HookEvent, bot: Bot) -> None:
    """Handle a Stop event — instant done detection."""
    from .status_polling import (
        _start_autoclose_timer,
        clear_seen_status,
    )
    from .message_queue import enqueue_status_update
    from .topic_emoji import update_topic_emoji

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    import time

    now = time.monotonic()
    stop_reason = event.data.get("stop_reason", "")
    logger.debug(
        "Hook stop: window_key=%s, stop_reason=%s",
        event.window_key,
        stop_reason,
    )

    for user_id, thread_id, window_id in users:
        clear_seen_status(window_id)
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        display = session_manager.get_display_name(window_id)
        await update_topic_emoji(bot, chat_id, thread_id, "done", display)
        _start_autoclose_timer(user_id, thread_id, "done", now)
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)


# Track active subagents per window: window_id -> set of subagent_ids
_active_subagents: dict[str, set[str]] = {}


def get_subagent_count(window_id: str) -> int:
    """Return the number of active subagents for a window."""
    return len(_active_subagents.get(window_id, set()))


def clear_subagents(window_id: str) -> None:
    """Clear all subagent tracking for a window."""
    _active_subagents.pop(window_id, None)


async def _handle_subagent_start(event: HookEvent, bot: Bot) -> None:  # noqa: ARG001
    """Handle SubagentStart — track active subagent."""
    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    window_id = users[0][2]  # all users share the same window_id
    subagent_id = event.data.get("subagent_id", "")

    _active_subagents.setdefault(window_id, set()).add(subagent_id)

    count = len(_active_subagents[window_id])
    logger.debug(
        "Subagent started: window=%s, count=%d, name=%s",
        window_id,
        count,
        event.data.get("name", ""),
    )


async def _handle_subagent_stop(event: HookEvent, bot: Bot) -> None:  # noqa: ARG001
    """Handle SubagentStop — remove subagent from tracking."""
    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    window_id = users[0][2]
    subagent_id = event.data.get("subagent_id", "")

    ids = _active_subagents.get(window_id)
    if not ids:
        return
    ids.discard(subagent_id)
    if not ids:
        _active_subagents.pop(window_id, None)

    count = get_subagent_count(window_id)
    logger.debug(
        "Subagent stopped: window=%s, remaining=%d, id=%s",
        window_id,
        count,
        subagent_id,
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
            | "SessionEnd"
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
