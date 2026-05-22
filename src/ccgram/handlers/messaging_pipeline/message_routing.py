"""Inbound message routing — handles new assistant messages from SessionMonitor.

Routes messages from the session monitor to Telegram topics: thinking-block
gating, interactive-tool detection, offset tracking, and content queue
management.
"""

import asyncio
from pathlib import Path

import structlog

from ... import session_query
from ...session_monitor import NewMessage
from ...telegram_client import TelegramClient
from ...user_preferences import user_preferences
from ..interactive import (
    INTERACTIVE_TOOL_NAMES,
    clear_interactive_mode,
    clear_interactive_msg,
    get_interactive_msg_id,
    handle_interactive_ui,
    set_interactive_mode,
)
from ..response_builder import build_response_parts
from .message_queue import enqueue_content_message, get_message_queue

logger = structlog.get_logger()

_MIN_THINKING_LENGTH = 20


async def handle_new_message(msg: NewMessage, client: TelegramClient) -> None:  # noqa: C901, PLR0912
    """Handle a new assistant message — enqueue for sequential processing.

    Messages are queued per-user to ensure status messages always appear last.
    Routes via thread_bindings to deliver to the correct topic.
    """
    status = "complete" if msg.is_complete else "streaming"
    logger.debug(
        "handle_new_message [%s]: session=%s, text_len=%d",
        status,
        msg.session_id,
        len(msg.text),
    )

    active_users = session_query.find_users_for_session(msg.session_id)

    if not active_users:
        logger.debug("No active users for session %s", msg.session_id)
        return

    for user_id, window_id, thread_id in active_users:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            window_id=window_id, session_id=msg.session_id
        )

        if msg.content_type == "thinking":
            stripped = (msg.text or "").strip()
            if len(stripped) < _MIN_THINKING_LENGTH:
                continue

        if msg.tool_name in INTERACTIVE_TOOL_NAMES and msg.content_type == "tool_use":
            set_interactive_mode(user_id, window_id, thread_id)
            queue = get_message_queue(user_id)
            if queue:
                await queue.join()
            await asyncio.sleep(0.3)
            handled = await handle_interactive_ui(client, user_id, window_id, thread_id)
            if handled:
                session = await session_query.resolve_session_for_window(window_id)
                if session and session.file_path:
                    try:
                        file_size = Path(session.file_path).stat().st_size
                        user_preferences.update_user_window_offset(
                            user_id, window_id, file_size
                        )
                    except OSError:
                        pass
                continue
            else:
                clear_interactive_mode(user_id, thread_id)

        if get_interactive_msg_id(user_id, thread_id):
            await clear_interactive_msg(user_id, client, thread_id)

        parts = build_response_parts(
            msg.text,
            msg.is_complete,
            msg.content_type,
            msg.role,
        )

        if msg.is_complete:
            await enqueue_content_message(
                client=client,
                user_id=user_id,
                window_id=window_id,
                parts=parts,
                tool_use_id=msg.tool_use_id,
                tool_name=msg.tool_name,
                content_type=msg.content_type,  # type: ignore[arg-type]  # NewMessage.content_type is str, narrows at runtime
                role=msg.role,  # type: ignore[arg-type]  # NewMessage.role is str, narrows at runtime
                thread_id=thread_id,
            )

            session = await session_query.resolve_session_for_window(window_id)
            if session and session.file_path:
                try:
                    file_size = Path(session.file_path).stat().st_size
                    user_preferences.update_user_window_offset(
                        user_id, window_id, file_size
                    )
                except OSError:
                    pass
