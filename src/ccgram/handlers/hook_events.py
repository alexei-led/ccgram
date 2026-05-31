"""Hook event dispatcher — routes structured events to handlers.

Receives HookEvent objects from the session monitor's event reader and
dispatches them to the appropriate handler based on event type. This
provides instant, structured notification of agent state changes instead
of relying solely on terminal scraping.

Key function: dispatch_hook_event().
"""

import asyncio

import structlog

from ..claude_task_state import classify_wait_message, claude_task_state
from ..providers.base import HookEvent
from ..session_lifecycle import session_lifecycle
from ..telegram_client import TelegramClient
from ..thread_router import thread_router
from ..window_query import view_window
from .interactive import (
    clear_interactive_mode,
    get_interactive_window,
    handle_interactive_ui,
    set_interactive_mode,
)
from .messaging_pipeline.message_queue import enqueue_status_update
from .polling.polling_state import reset_window_polling_state
from .status.topic_emoji import update_topic_emoji


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


async def _handle_notification(event: HookEvent, client: TelegramClient) -> None:
    """Handle a Notification event — render interactive UI."""
    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        logger.debug(
            "No users bound for notification event window_key=%s", event.window_key
        )
        return

    tool_name = event.data.get("tool_name", "")
    provider_name = event.data.get("provider_name", "claude")
    logger.debug(
        "Hook notification: provider=%s, tool_name=%s, window_key=%s",
        provider_name,
        tool_name,
        event.window_key,
    )
    wait_header = classify_wait_message(event.data.get("message", ""))

    for user_id, thread_id, window_id in users:
        if wait_header:
            session_lifecycle.handle_notification_wait(window_id, wait_header)
            await enqueue_status_update(
                client, user_id, window_id, None, thread_id=thread_id
            )

        if provider_name != "claude":
            message = str(event.data.get("message", "") or "Agent notification")
            await enqueue_status_update(
                client, user_id, window_id, f"⚠ {message}", thread_id=thread_id
            )
            continue

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

        await asyncio.sleep(0.3)

        handled = await handle_interactive_ui(client, user_id, window_id, thread_id)
        if not handled:
            clear_interactive_mode(user_id, thread_id)


_LLM_SUMMARY_TIMEOUT = 3.0  # seconds to wait for LLM summary before falling back to the standard completion text


async def _get_llm_summary(transcript_path: str) -> str | None:
    """Try to get an LLM summary, returning None on failure."""
    try:
        # Lazy: llm package wires httpx + provider configs; only loaded
        # when a Stop event actually wants a summary.
        from ..llm.summarizer import summarize_completion

        return await summarize_completion(transcript_path)
    except RuntimeError, OSError, ValueError:
        logger.debug("LLM summary failed", exc_info=True)
        return None


async def _handle_stop(event: HookEvent, client: TelegramClient) -> None:
    """Handle a Stop event — transition status directly to idle.

    Topic emoji remains poller-owned. Hook-driven idle flips can fight the
    transcript/activity heuristic and cause active/idle rename churn on quiet
    topics, so Stop only updates the status bubble.
    """

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    stop_reason = event.data.get("stop_reason", "")
    provider_name = event.data.get("provider_name", "claude")
    logger.debug(
        "Hook stop: provider=%s, window_key=%s, stop_reason=%s",
        provider_name,
        event.window_key,
        stop_reason,
    )

    num_turns = event.data.get("num_turns", 0)

    # Try LLM summary with timeout to avoid flicker (send once, not twice).
    # If LLM is available and responds within timeout, include summary in the
    # initial status message. Otherwise fall back to plain Ready.
    # All users share the same window_id — fetch view once, reuse in loop.
    first_window_id = users[0][2]
    view = view_window(first_window_id)
    summary: str | None = None
    if view and view.transcript_path:
        try:
            summary = await asyncio.wait_for(
                _get_llm_summary(str(view.transcript_path)),
                timeout=_LLM_SUMMARY_TIMEOUT,
            )
        except TimeoutError:
            logger.debug("LLM summary timed out after %ss", _LLM_SUMMARY_TIMEOUT)

    for user_id, thread_id, window_id in users:
        session_lifecycle.handle_stop_task_state(window_id)
        if provider_name == "claude":
            status_text = claude_task_state.format_completion_text(
                window_id, num_turns=num_turns
            )
        else:
            status_text = "✓ Ready"
        if summary and status_text:
            status_text = status_text.replace("✓ Ready", f"✓ Done — {summary}", 1)
        await enqueue_status_update(
            client, user_id, window_id, status_text, thread_id=thread_id
        )


async def _handle_subagent_start(event: HookEvent, _client: TelegramClient) -> None:
    """Handle SubagentStart — track active subagent count and name."""
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

    count = session_lifecycle.handle_subagent_start(window_id, subagent_id, name)

    logger.debug(
        "Subagent started: window=%s, count=%d, name=%s",
        window_id,
        count,
        name,
    )

    # No immediate status update — the polling loop (1s) already appends
    # subagent count/names to the status bubble via get_subagent_names().


async def _handle_subagent_stop(event: HookEvent, _client: TelegramClient) -> None:
    """Handle SubagentStop — remove subagent from tracking."""
    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    window_id = users[0][2]
    subagent_id = event.data.get("subagent_id", "")

    name, remaining = session_lifecycle.handle_subagent_stop(window_id, subagent_id)

    logger.debug(
        "Subagent stopped: window=%s, remaining=%d, name=%s",
        window_id,
        remaining,
        name,
    )

    # No immediate status update — polling loop shows updated count within 1s.


async def _handle_teammate_idle(event: HookEvent, client: TelegramClient) -> None:
    """Handle TeammateIdle — notify topic that a teammate went idle."""

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
        await enqueue_status_update(
            client, user_id, window_id, text, thread_id=thread_id
        )


async def _handle_stop_failure(event: HookEvent, client: TelegramClient) -> None:
    """Handle a StopFailure event — alert on API error termination."""
    # Lazy: messaging_pipeline.message_sender pulls in safe_reply wiring
    # which transitively reaches hook_events for completion summaries.
    # Lazy: messaging_pipeline imports hook_events indirectly via the dispatch chain
    from .messaging_pipeline.message_sender import rate_limit_send_message

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
    text = f"⚠ API error — {error}{detail}"

    for user_id, thread_id, _window_id in users:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        await rate_limit_send_message(
            client, chat_id, text, message_thread_id=thread_id
        )


async def _handle_session_end(event: HookEvent, client: TelegramClient) -> None:
    """Handle a SessionEnd event — clean up session lifecycle."""

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    reason = event.data.get("reason", "")
    logger.info(
        "Hook SessionEnd: window_key=%s, reason=%s",
        event.window_key,
        reason,
    )

    if users:
        window_id = users[0][2]
        session_lifecycle.handle_session_end(window_id)

    for user_id, thread_id, window_id in users:
        reset_window_polling_state(window_id)
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        display = thread_router.get_display_name(window_id)
        await update_topic_emoji(client, chat_id, thread_id, "done", display)
        await enqueue_status_update(
            client, user_id, window_id, None, thread_id=thread_id
        )


async def _handle_task_completed(event: HookEvent, client: TelegramClient) -> None:
    """Handle TaskCompleted — notify topic that a task was completed."""

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
            tracked = session_lifecycle.handle_task_completed(
                window_id,
                event.session_id,
                task_id,
                subject=task_subject,
            )
        if tracked or claude_task_state.has_snapshot(window_id):
            await enqueue_status_update(
                client, user_id, window_id, None, thread_id=thread_id
            )
            continue

        text = f"✅ Task completed: {task_subject}"
        if teammate_name:
            text += f" (by '{teammate_name}')"
        await enqueue_status_update(
            client, user_id, window_id, text, thread_id=thread_id
        )


async def dispatch_hook_event(event: HookEvent, client: TelegramClient) -> None:
    """Route hook events to appropriate handlers."""
    match event.event_type:
        case "Notification":
            await _handle_notification(event, client)
        case "Stop":
            await _handle_stop(event, client)
        case "StopFailure":
            await _handle_stop_failure(event, client)
        case "SessionEnd":
            await _handle_session_end(event, client)
        case "SubagentStart":
            await _handle_subagent_start(event, client)
        case "SubagentStop":
            await _handle_subagent_stop(event, client)
        case "TeammateIdle":
            await _handle_teammate_idle(event, client)
        case "TaskCompleted":
            await _handle_task_completed(event, client)
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
            | "PostCompact"
        ):
            pass  # Not actionable for the bot — SessionStart handled via session_map.json
        case _:
            logger.debug("Ignoring unknown hook event type: %s", event.event_type)
