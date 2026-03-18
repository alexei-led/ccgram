"""Unified cleanup API for topic state.

Provides centralized cleanup functions that coordinate state cleanup across
all modules, preventing memory leaks when topics are deleted.

Functions:
  - clear_topic_state: Clean up all memory state for a specific topic
  - clear_dead_notification (delegated): Clear dead window notification tracking
"""

from typing import Any

from telegram import Bot

from ..utils import log_throttle_reset
from .interactive_ui import clear_interactive_msg
from .message_queue import (
    clear_status_msg_info,
    clear_tool_msg_ids_for_topic,
    enqueue_status_update,
)
from .topic_emoji import clear_topic_emoji_state
from .user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT


async def clear_topic_state(
    user_id: int,
    thread_id: int,
    bot: Bot | None = None,
    user_data: dict[str, Any] | None = None,
    window_id: str | None = None,
) -> None:
    """Clear all memory state associated with a topic.

    This should be called when:
      - A topic is closed or deleted
      - A thread binding becomes stale (window deleted externally)

    Removes full dict entries from _topic_poll_state / _window_poll_state
    (not just field resets) to prevent orphaned state accumulation.
    Also cleans up status messages, tool tracking, interactive UI, emoji,
    command history, and user_data pending state.
    """
    # Clear status message from Telegram (if bot available) or just tracking
    if bot is not None:
        await enqueue_status_update(
            bot, user_id, window_id or "", None, thread_id=thread_id
        )
    else:
        clear_status_msg_info(user_id, thread_id)

    # Clear tool message ID tracking
    clear_tool_msg_ids_for_topic(user_id, thread_id)

    # Clear poll state (lazy import to avoid circular dep)
    from .status_polling import (
        clear_dead_notification,
        clear_pane_alerts,
        clear_topic_poll_state,
        clear_window_poll_state,
    )

    clear_dead_notification(user_id, thread_id)
    clear_topic_poll_state(user_id, thread_id)
    if window_id:
        from ..tmux_manager import clear_vim_state

        clear_vim_state(window_id)
        clear_window_poll_state(window_id)
        clear_pane_alerts(window_id)
        log_throttle_reset(f"topic-probe:{window_id}")
        log_throttle_reset(f"status-update:{user_id}:{thread_id}")
        from .hook_events import clear_subagents

        clear_subagents(window_id)

    # Clear interactive UI state (also deletes message from chat)
    await clear_interactive_msg(user_id, bot, thread_id)

    # Clear topic emoji tracking (needs chat_id; use 0 as fallback)
    from ..session import session_manager

    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    clear_topic_emoji_state(chat_id, thread_id)

    # Clear command history for this topic
    from .command_history import clear_history

    clear_history(user_id, thread_id)

    # Clear pending thread state from user_data
    if user_data is not None and user_data.get(PENDING_THREAD_ID) == thread_id:
        user_data.pop(PENDING_THREAD_ID, None)
        user_data.pop(PENDING_THREAD_TEXT, None)
