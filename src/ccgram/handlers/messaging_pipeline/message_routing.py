"""Inbound message routing — handles new assistant messages from SessionMonitor.

Routes messages from the session monitor to Telegram topics: thinking-block
gating, interactive-tool detection, offset tracking, and content queue
management.
"""

import asyncio
import contextlib
from pathlib import Path

import structlog

from ... import session_query
from ...session_monitor import NewMessage
from ...telegram_client import TelegramClient, unwrap_bot
from ...telegram_draft import DRAFT_UNSET, DraftStream
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
from ..telegram_origin import consume_telegram_injection
from .message_queue import enqueue_content_message, get_message_queue

logger = structlog.get_logger()

_MIN_THINKING_LENGTH = 20

# One draft per session/topic. Provider updates are cumulative snapshots, not deltas.
_DRAFT_TTL_SECONDS = 25.0
_active_drafts: dict[tuple[int, str, int | None, int], DraftStream] = {}
_draft_expiry_tasks: dict[tuple[int, str, int | None, int], asyncio.Task[None]] = {}


async def _update_window_offset(user_id: int, window_id: str) -> None:
    """Advance transcript offset after a complete message is handled."""
    session = await session_query.resolve_session_for_window(window_id)
    if not session or not session.file_path:
        return
    try:
        file_size = Path(session.file_path).stat().st_size
        user_preferences.update_user_window_offset(user_id, window_id, file_size)
    except OSError:
        return


def _arm_draft_expiry(
    key: tuple[int, str, int | None, int], draft: DraftStream
) -> None:
    """Drop stalled native drafts before Telegram expires their preview."""
    previous = _draft_expiry_tasks.pop(key, None)
    if previous is not None:
        previous.cancel()
    _draft_expiry_tasks[key] = asyncio.create_task(
        _expire_draft(key, draft),
        name=f"assistant-draft-expiry:{key[0]}:{key[3]}",
    )


async def _expire_draft(
    key: tuple[int, str, int | None, int], draft: DraftStream
) -> None:
    try:
        await asyncio.sleep(_DRAFT_TTL_SECONDS)
        if _active_drafts.get(key) is draft:
            _active_drafts.pop(key, None)
            await draft.abort()
    except asyncio.CancelledError:
        raise
    finally:
        current = _draft_expiry_tasks.get(key)
        if current is asyncio.current_task():
            _draft_expiry_tasks.pop(key, None)


async def _cancel_draft_expiry(key: tuple[int, str, int | None, int]) -> None:
    task = _draft_expiry_tasks.pop(key, None)
    if task is None or task is asyncio.current_task():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _handle_assistant_stream(
    msg: NewMessage,
    client: TelegramClient,
    user_id: int,
    thread_id: int | None,
    chat_id: int | None,
) -> bool:
    """Deliver an assistant text snapshot to its draft, if applicable."""
    if msg.role != "assistant" or msg.content_type != "text" or chat_id is None:
        return False

    key = (user_id, msg.session_id, thread_id, chat_id)
    draft = _active_drafts.get(key)

    if msg.is_complete:
        if draft is not None:
            _active_drafts.pop(key, None)
            await _cancel_draft_expiry(key)
            await draft.abort()
        return False

    if draft is None:
        draft = DraftStream(
            unwrap_bot(client),
            chat_id,
            message_thread_id=thread_id,
        )
        await draft.start(msg.text)
        if draft.mode == DRAFT_UNSET:
            return True
        _active_drafts[key] = draft
    else:
        await draft.replace(msg.text)
    _arm_draft_expiry(key, draft)
    return True


async def enqueue_backlog_skip_notice(client: TelegramClient, intent: object) -> None:
    """Queue a visible, receipt-tracked notice for a confirmed skipped range."""
    snapshot = int(getattr(intent, "snapshot_offset"))
    start = int(getattr(intent, "range_start"))
    count = int(getattr(intent, "skipped_count"))
    enqueued = await enqueue_content_message(
        client=client,
        user_id=int(getattr(intent, "user_id")),
        window_id=str(getattr(intent, "window_id")),
        parts=[
            f"⏭ Skipped {count} queued transcript item(s) for live view "
            f"(bytes {start}–{snapshot}). Raw transcript retained."
        ],
        thread_id=getattr(intent, "thread_id"),
        chat_id=int(getattr(intent, "chat_id")),
        is_backlog_notice=True,
    )
    if not enqueued:
        raise RuntimeError("backlog skip notice could not enter the delivery queue")


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

    for user_id, window_id, thread_id, chat_id in active_users:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            window_id=window_id, session_id=msg.session_id
        )

        if (
            msg.is_complete
            and msg.role == "user"
            and consume_telegram_injection(
                user_id, window_id, thread_id, msg.text, chat_id
            )
        ):
            logger.debug("Suppressed Telegram-originated user transcript message")
            continue

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
                await _update_window_offset(user_id, window_id)
                continue
            else:
                clear_interactive_mode(user_id, thread_id)

        if get_interactive_msg_id(user_id, thread_id):
            await clear_interactive_msg(user_id, client, thread_id)

        if await _handle_assistant_stream(
            msg,
            client,
            user_id,
            thread_id,
            chat_id,
        ):
            continue

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
                chat_id=chat_id,
                source_session_id=msg.session_id,
            )

            await _update_window_offset(user_id, window_id)
