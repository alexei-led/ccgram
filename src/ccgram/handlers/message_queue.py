"""Per-user message queue management for ordered message delivery.

Provides a queue-based message processing system that ensures:
  - Messages are sent in receive order (FIFO)
  - Status messages always follow content messages
  - Consecutive content messages can be merged for efficiency
  - Rate limiting is respected
  - Thread-aware sending: each MessageTask carries an optional thread_id
    for Telegram topic support

Key components:
  - MessageTask: Dataclass representing a queued message task (with thread_id)
  - get_or_create_queue: Get or create queue and worker for a user
  - Message queue worker: Background task processing user's queue
  - Content task processing with tool_use/tool_result handling
  - Status message tracking and conversion (keyed by (user_id, thread_id))
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from typing import Literal

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import RetryAfter, TelegramError

from ..markdown_v2 import convert_markdown
from ..session import session_manager
from ..providers import get_provider_for_window
from ..tmux_manager import tmux_manager
from ..utils import task_done_callback
from .callback_data import (
    CB_STATUS_ESC,
    CB_STATUS_NOTIFY,
    CB_STATUS_RECALL,
    CB_STATUS_REMOTE,
    CB_STATUS_SCREENSHOT,
    NOTIFY_MODE_ICONS,
)
from .message_sender import NO_LINK_PREVIEW, rate_limit_send_message
import contextlib

# Top-level loop resilience: catch any error to keep the worker alive
_LoopError = (TelegramError, OSError, RuntimeError, ValueError)

logger = structlog.get_logger()

# Merge limit for content messages
MERGE_MAX_LENGTH = 3800  # Leave room for markdown conversion overhead

# Batch limits for tool call chains
# Keep conservative: header + entries + result text + separators + MarkdownV2 escaping
# must fit 4096 chars. Worst case: 10 * (250 + 85 + 6) + 20 ≈ 3430 chars.
BATCH_MAX_LENGTH = 2800
BATCH_MAX_ENTRIES = 10


@dataclass
class ToolBatchEntry:
    """A single tool call entry within a batch."""

    tool_use_id: str | None
    tool_use_text: str  # Formatted summary from build_response_parts
    tool_result_text: str | None = None  # None until result arrives


@dataclass
class ToolBatch:
    """Accumulator for consecutive tool calls to batch into one Telegram message."""

    window_id: str
    thread_id: int  # thread_id_or_0
    entries: list[ToolBatchEntry] = field(default_factory=list)
    telegram_msg_id: int | None = None
    total_length: int = 0


def _is_batch_eligible(task: MessageTask) -> bool:
    """Check if a task is eligible for tool call batching."""
    return task.task_type == "content" and task.content_type in (
        "tool_use",
        "tool_result",
    )


def format_batch_message(entries: list[ToolBatchEntry]) -> str:
    """Render a batch of tool calls as a single compact message.

    Format:
        ⚡ 3 tool calls
        📖 Read  src/foo.py       ⎿  42 lines
        ✏️ Edit  src/foo.py       ⎿  +3 −1
        ⚡ Bash  make test        ⏳
    """
    count = len(entries)
    label = "tool call" if count == 1 else "tool calls"
    lines = [f"\u26a1 {count} {label}"]

    for entry in entries:
        line = entry.tool_use_text
        if entry.tool_result_text is not None:
            line = f"{line}  \u23bf  {entry.tool_result_text}"
        else:
            line = f"{line}  \u23f3"
        lines.append(line)

    return "\n".join(lines)


def build_status_keyboard(
    window_id: str, history: list[str] | None = None
) -> InlineKeyboardMarkup:
    """Build inline keyboard for status messages: [↑ cmd] row + [Esc] [Screenshot] [Bell] [RC]."""
    from .command_history import truncate_for_display
    from .status_polling import is_rc_active

    rows: list[list[InlineKeyboardButton]] = []

    # History recall row (up to 2 buttons)
    if history:
        hist_row: list[InlineKeyboardButton] = []
        for idx, cmd in enumerate(history[:2]):
            label = truncate_for_display(cmd, 20)
            hist_row.append(
                InlineKeyboardButton(
                    f"\u2191 {label}",
                    callback_data=f"{CB_STATUS_RECALL}{window_id}:{idx}"[:64],
                )
            )
        rows.append(hist_row)

    # Control row
    mode = session_manager.get_notification_mode(window_id)
    bell = NOTIFY_MODE_ICONS.get(mode, "\U0001f514")
    rc_label = "\U0001f4e1\u2713" if is_rc_active(window_id) else "\U0001f4e1"
    rows.append(
        [
            InlineKeyboardButton(
                "\u238b Esc",
                callback_data=f"{CB_STATUS_ESC}{window_id}"[:64],
            ),
            InlineKeyboardButton(
                "\U0001f4f8",
                callback_data=f"{CB_STATUS_SCREENSHOT}{window_id}"[:64],
            ),
            InlineKeyboardButton(
                bell,
                callback_data=f"{CB_STATUS_NOTIFY}{window_id}"[:64],
            ),
            InlineKeyboardButton(
                rc_label,
                callback_data=f"{CB_STATUS_REMOTE}{window_id}"[:64],
            ),
        ]
    )
    return InlineKeyboardMarkup(rows)


@dataclass
class MessageTask:
    """Message task for queue processing."""

    task_type: Literal["content", "status_update", "status_clear"]
    text: str | None = None
    window_id: str | None = None
    # content type fields
    parts: list[str] = field(default_factory=list)
    tool_use_id: str | None = None
    content_type: str = "text"
    thread_id: int | None = None  # Telegram topic thread_id for targeted send


# Per-user message queues and worker tasks
_message_queues: dict[int, asyncio.Queue[MessageTask]] = {}
_queue_workers: dict[int, asyncio.Task[None]] = {}
_queue_locks: dict[int, asyncio.Lock] = {}  # Protect drain/refill operations

# Map (tool_use_id, user_id, thread_id_or_0) -> telegram message_id
# for editing tool_use messages with results
_tool_msg_ids: dict[tuple[str, int, int], int] = {}

# Status message tracking: (user_id, thread_id_or_0) -> (message_id, window_id, last_text)
_status_msg_info: dict[tuple[int, int], tuple[int, str, str]] = {}

# Active tool batches: (user_id, thread_id_or_0) -> ToolBatch
_active_batches: dict[tuple[int, int], ToolBatch] = {}


def get_message_queue(user_id: int) -> asyncio.Queue[MessageTask] | None:
    """Get the message queue for a user (if exists)."""
    return _message_queues.get(user_id)


def get_or_create_queue(bot: Bot, user_id: int) -> asyncio.Queue[MessageTask]:
    """Get or create message queue and worker for a user."""
    if user_id not in _message_queues:
        _message_queues[user_id] = asyncio.Queue()
        _queue_locks[user_id] = asyncio.Lock()
        # Start worker task for this user
        task = asyncio.create_task(_message_queue_worker(bot, user_id))
        task.add_done_callback(task_done_callback)
        _queue_workers[user_id] = task
    return _message_queues[user_id]


def _inspect_queue(queue: asyncio.Queue[MessageTask]) -> list[MessageTask]:
    """Non-destructively inspect all items in queue.

    Drains the queue and returns all items. Caller must refill.
    """
    items: list[MessageTask] = []
    while not queue.empty():
        try:
            item = queue.get_nowait()
            items.append(item)
        except asyncio.QueueEmpty:
            break
    return items


def _can_merge_tasks(base: MessageTask, candidate: MessageTask) -> bool:
    """Check if two content tasks can be merged."""
    if base.window_id != candidate.window_id:
        return False
    if candidate.task_type != "content":
        return False
    # tool_use/tool_result break merge chain
    # - tool_use: will be edited later by tool_result
    # - tool_result: edits previous message, merging would cause order issues
    if base.content_type in ("tool_use", "tool_result"):
        return False
    return candidate.content_type not in ("tool_use", "tool_result")


async def _merge_content_tasks(
    queue: asyncio.Queue[MessageTask],
    first: MessageTask,
    lock: asyncio.Lock,
) -> tuple[MessageTask, int]:
    """Merge consecutive content tasks from queue.

    Returns: (merged_task, merge_count) where merge_count is the number of
    additional tasks merged (0 if no merging occurred).

    Note on queue counter management:
        When we put items back, we call task_done() to compensate for the
        internal counter increment caused by put_nowait(). This is necessary
        because the items were already counted when originally enqueued.
        Without this compensation, queue.join() would wait indefinitely.
    """
    merged_parts = list(first.parts)
    current_length = sum(len(p) for p in merged_parts)
    merge_count = 0

    async with lock:
        items = _inspect_queue(queue)
        remaining: list[MessageTask] = []

        for i, task in enumerate(items):
            if not _can_merge_tasks(first, task):
                # Can't merge, keep this and all remaining items
                remaining = items[i:]
                break

            # Check length before merging
            task_length = sum(len(p) for p in task.parts)
            if current_length + task_length > MERGE_MAX_LENGTH:
                # Too long, stop merging
                remaining = items[i:]
                break

            merged_parts.extend(task.parts)
            current_length += task_length
            merge_count += 1

        # Put remaining items back into the queue
        for item in remaining:
            queue.put_nowait(item)
            # Compensate: this item was already counted when first enqueued,
            # put_nowait adds a duplicate count that must be removed
            queue.task_done()

    if merge_count == 0:
        return first, 0

    return (
        MessageTask(
            task_type="content",
            window_id=first.window_id,
            parts=merged_parts,
            tool_use_id=first.tool_use_id,
            content_type=first.content_type,
            thread_id=first.thread_id,
        ),
        merge_count,
    )


async def _coalesce_status_updates(
    queue: asyncio.Queue[MessageTask],
    first: MessageTask,
    lock: asyncio.Lock,
) -> tuple[MessageTask, int]:
    """Keep only the latest pending status_update for the same topic/window.

    Returns: (selected_task, dropped_count) where dropped_count is the number
    of queued tasks removed and already accounted for.
    """
    if first.task_type != "status_update":
        return first, 0

    selected = first
    dropped = 0
    key = (first.thread_id or 0, first.window_id or "")

    async with lock:
        items = _inspect_queue(queue)
        remaining: list[MessageTask] = []

        for task in items:
            if task.task_type != "status_update":
                remaining.append(task)
                continue
            task_key = (task.thread_id or 0, task.window_id or "")
            if task_key == key:
                # Same topic/window status update; keep latest only.
                selected = task
                dropped += 1
            else:
                remaining.append(task)

        for item in remaining:
            queue.put_nowait(item)
            queue.task_done()

    return selected, dropped


def _should_batch(window_id: str) -> bool:
    """Check if batching is enabled for a window."""
    return session_manager.get_batch_mode(window_id) == "batched"


async def _process_batch_task(bot: Bot, user_id: int, task: MessageTask) -> None:
    """Add a tool_use or tool_result to the active batch, send/edit the batch message."""
    window_id = task.window_id or ""
    thread_id = task.thread_id or 0
    bkey = (user_id, thread_id)
    chat_id = session_manager.resolve_chat_id(user_id, task.thread_id)

    batch = _active_batches.get(bkey)

    if task.content_type == "tool_result":
        if not task.tool_use_id or not batch:
            # No batch or no tool_use_id — process as standalone message
            await _process_content_task(bot, user_id, task)
            return
        # Find matching entry and update with result text
        for entry in batch.entries:
            if entry.tool_use_id == task.tool_use_id:
                result_text = task.text or ""
                first_line = result_text.split("\n", 1)[0][:80]
                entry.tool_result_text = first_line
                break
        else:
            # No matching entry — flush batch, send result standalone
            await _flush_batch(bot, user_id, thread_id)
            await _process_content_task(bot, user_id, task)
            return
    elif task.content_type == "tool_use":
        if not batch or batch.window_id != window_id:
            if batch:
                await _flush_batch(bot, user_id, thread_id)
            batch = ToolBatch(window_id=window_id, thread_id=thread_id)
            _active_batches[bkey] = batch

        entry_text = task.text or "\n".join(task.parts) or "tool call"
        entry = ToolBatchEntry(
            tool_use_id=task.tool_use_id,
            tool_use_text=entry_text,
        )
        batch.entries.append(entry)
        batch.total_length += len(entry_text)

        # Check if batch exceeds limits — flush and start new
        if (
            len(batch.entries) >= BATCH_MAX_ENTRIES
            or batch.total_length > BATCH_MAX_LENGTH
        ):
            overflow_entry = batch.entries.pop()
            batch.total_length -= len(entry_text)
            await _flush_batch(bot, user_id, thread_id)
            batch = ToolBatch(window_id=window_id, thread_id=thread_id)
            batch.entries.append(overflow_entry)
            batch.total_length = len(entry_text)
            _active_batches[bkey] = batch
    else:
        # Defensive: route unexpected content_type to normal processing
        await _process_content_task(bot, user_id, task)
        return

    # Send or edit batch message
    batch_text = format_batch_message(batch.entries)

    if batch.telegram_msg_id is None:
        # Clear status message first, then send new batch message
        await _do_clear_status_message(bot, user_id, thread_id)
        sent = await rate_limit_send_message(
            bot,
            chat_id,
            batch_text,
            **_send_kwargs(task.thread_id),  # type: ignore[arg-type]
        )
        if sent:
            batch.telegram_msg_id = sent.message_id
    else:
        # Edit existing batch message (MarkdownV2 with plain text fallback)
        md_text = convert_markdown(batch_text)
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=batch.telegram_msg_id,
                text=md_text,
                parse_mode="MarkdownV2",
                link_preview_options=NO_LINK_PREVIEW,
            )
        except RetryAfter:
            raise
        except TelegramError:
            with contextlib.suppress(TelegramError):
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=batch.telegram_msg_id,
                    text=batch_text,
                    link_preview_options=NO_LINK_PREVIEW,
                )


async def _flush_batch(bot: Bot, user_id: int, thread_id_or_0: int) -> None:
    """Finalize the active batch: do a final edit and clear state."""
    bkey = (user_id, thread_id_or_0)
    batch = _active_batches.pop(bkey, None)
    if not batch or not batch.entries:
        return

    thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    batch_text = format_batch_message(batch.entries)

    if batch.telegram_msg_id is None:
        # First send failed earlier — attempt one send before dropping
        await rate_limit_send_message(
            bot,
            chat_id,
            batch_text,
            **_send_kwargs(thread_id),  # type: ignore[arg-type]
        )
        return

    # Final edit with all results resolved
    md_text = convert_markdown(batch_text)
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=batch.telegram_msg_id,
            text=md_text,
            parse_mode="MarkdownV2",
            link_preview_options=NO_LINK_PREVIEW,
        )
    except RetryAfter:
        raise
    except TelegramError:
        with contextlib.suppress(TelegramError):
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=batch.telegram_msg_id,
                text=batch_text,
                link_preview_options=NO_LINK_PREVIEW,
            )


async def _handle_content_task(
    bot: Bot,
    user_id: int,
    task: MessageTask,
    queue: asyncio.Queue[MessageTask],
    lock: asyncio.Lock,
) -> int:
    """Route a content task through batching or normal processing.

    Returns the number of additional merged tasks (caller must call task_done for each).
    """
    # Batch-eligible tool tasks with batching enabled
    if _is_batch_eligible(task) and task.window_id and _should_batch(task.window_id):
        await _process_batch_task(bot, user_id, task)
        return 0

    # Non-tool content: flush any active batch first
    thread_id = task.thread_id or 0
    bkey = (user_id, thread_id)
    if bkey in _active_batches:
        await _flush_batch(bot, user_id, thread_id)

    # Try to merge consecutive content tasks
    merged_task, merge_count = await _merge_content_tasks(queue, task, lock)
    if merge_count > 0:
        logger.debug("Merged %d tasks for user %s", merge_count, user_id)
    await _process_content_task(bot, user_id, merged_task)
    return merge_count


async def _message_queue_worker(bot: Bot, user_id: int) -> None:
    """Process message tasks for a user sequentially."""
    queue = _message_queues[user_id]
    lock = _queue_locks[user_id]
    logger.debug("Message queue worker started for user %s", user_id)

    while True:
        try:
            task = await queue.get()
            try:
                if task.task_type == "content":
                    extra = await _handle_content_task(bot, user_id, task, queue, lock)
                    for _ in range(extra):
                        queue.task_done()
                elif task.task_type == "status_update":
                    # Flush batch before status
                    thread_id = task.thread_id or 0
                    bkey = (user_id, thread_id)
                    if bkey in _active_batches:
                        await _flush_batch(bot, user_id, thread_id)
                    collapsed_task, dropped = await _coalesce_status_updates(
                        queue, task, lock
                    )
                    if dropped > 0:
                        for _ in range(dropped):
                            queue.task_done()
                    await _process_status_update_task(bot, user_id, collapsed_task)
                elif task.task_type == "status_clear":
                    thread_id = task.thread_id or 0
                    bkey = (user_id, thread_id)
                    if bkey in _active_batches:
                        await _flush_batch(bot, user_id, thread_id)
                    await _do_clear_status_message(bot, user_id, thread_id)
            except RetryAfter as e:
                retry_secs = min(
                    60,
                    (
                        e.retry_after
                        if isinstance(e.retry_after, int)
                        else int(e.retry_after.total_seconds())
                    ),
                )
                logger.warning(
                    "Flood control for user %s, pausing %ss", user_id, retry_secs
                )
                await asyncio.sleep(retry_secs)
            except (TelegramError, OSError):  # fmt: skip
                logger.exception(
                    "Error processing message task for user %s (thread %s)",
                    user_id,
                    task.thread_id,
                )
            finally:
                queue.task_done()
        except asyncio.CancelledError:
            logger.debug("Message queue worker cancelled for user %s", user_id)
            break
        except _LoopError:
            logger.exception(
                "Unexpected error in queue worker for user %s",
                user_id,
            )


def _send_kwargs(thread_id: int | None) -> dict[str, int]:
    """Build message_thread_id kwargs for bot.send_message()."""
    if thread_id is not None:
        return {"message_thread_id": thread_id}
    return {}


async def _process_content_task(bot: Bot, user_id: int, task: MessageTask) -> None:
    """Process a content message task."""
    window_id = task.window_id or ""
    thread_id = task.thread_id or 0
    chat_id = session_manager.resolve_chat_id(user_id, task.thread_id)

    # 1. Handle tool_result editing (merged parts are edited together)
    if task.content_type == "tool_result" and task.tool_use_id:
        _tkey = (task.tool_use_id, user_id, thread_id)
        edit_msg_id = _tool_msg_ids.pop(_tkey, None)
        if edit_msg_id is not None:
            # Clear status message first
            await _do_clear_status_message(bot, user_id, thread_id)
            # Join all parts for editing (merged content goes together)
            full_text = "\n\n".join(task.parts)
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=edit_msg_id,
                    text=full_text,
                    parse_mode="MarkdownV2",
                    link_preview_options=NO_LINK_PREVIEW,
                )
                await _check_and_send_status(bot, user_id, window_id, task.thread_id)
                return
            except RetryAfter:
                raise
            except TelegramError:
                try:
                    # Fallback: strip markdown
                    plain_text = task.text or full_text
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=edit_msg_id,
                        text=plain_text,
                        link_preview_options=NO_LINK_PREVIEW,
                    )
                    await _check_and_send_status(
                        bot, user_id, window_id, task.thread_id
                    )
                    return
                except RetryAfter:
                    raise
                except TelegramError:
                    logger.debug("Failed to edit tool msg %s, sending new", edit_msg_id)
                    # Fall through to send as new message

    # 2. Send content messages, converting status message to first content part
    first_part = True
    last_msg_id: int | None = None
    for part in task.parts:
        sent = None

        # For first part, try to convert status message to content (edit instead of delete)
        if first_part:
            first_part = False
            converted_msg_id = await _convert_status_to_content(
                bot,
                user_id,
                thread_id,
                window_id,
                part,
            )
            if converted_msg_id is not None:
                last_msg_id = converted_msg_id
                continue

        sent = await rate_limit_send_message(
            bot,
            chat_id,
            part,
            **_send_kwargs(task.thread_id),  # type: ignore[arg-type]
        )

        if sent:
            last_msg_id = sent.message_id

    # 3. Record tool_use message ID for later editing
    if last_msg_id and task.tool_use_id and task.content_type == "tool_use":
        _tool_msg_ids[(task.tool_use_id, user_id, thread_id)] = last_msg_id

    # 4. After content, check and send status
    await _check_and_send_status(bot, user_id, window_id, task.thread_id)


async def _convert_status_to_content(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
    window_id: str,
    content_text: str,
) -> int | None:
    """Convert status message to content message by editing it.

    Returns the message_id if converted successfully, None otherwise.
    """
    skey = (user_id, thread_id_or_0)
    info = _status_msg_info.pop(skey, None)
    if not info:
        return None

    thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)

    msg_id, stored_wid, _ = info
    if stored_wid != window_id:
        # Different window, just delete the old status
        with contextlib.suppress(TelegramError):
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        return None

    # Edit status message to show content (remove status buttons)
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text=content_text,
            parse_mode="MarkdownV2",
            reply_markup=None,
            link_preview_options=NO_LINK_PREVIEW,
        )
        return msg_id
    except RetryAfter:
        raise
    except TelegramError:
        try:
            # Fallback to plain text
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=content_text,
                reply_markup=None,
                link_preview_options=NO_LINK_PREVIEW,
            )
            return msg_id
        except RetryAfter:
            raise
        except TelegramError as e:
            logger.debug("Failed to convert status to content: %s", e)
            # Message might be deleted or too old, caller will send new message
            return None


def _get_idle_history(
    user_id: int, thread_id_or_0: int, status_text: str
) -> list[str] | None:
    """Return history list if the status is idle, else None."""
    from .callback_data import IDLE_STATUS_TEXT
    from .command_history import get_history

    if status_text != IDLE_STATUS_TEXT:
        return None
    return get_history(user_id, thread_id_or_0, limit=2) or None


async def _process_status_update_task(
    bot: Bot, user_id: int, task: MessageTask
) -> None:
    """Process a status update task."""
    window_id = task.window_id or ""
    thread_id = task.thread_id or 0
    chat_id = session_manager.resolve_chat_id(user_id, task.thread_id)
    skey = (user_id, thread_id)
    # task.text must be pre-formatted (display_label from StatusUpdate, not raw terminal text)
    status_text = task.text or ""

    if not status_text:
        # No status text means clear status
        await _do_clear_status_message(bot, user_id, thread_id)
        return

    current_info = _status_msg_info.get(skey)

    if current_info:
        msg_id, stored_wid, last_text = current_info

        if stored_wid != window_id:
            # Window changed - delete old and send new
            await _do_clear_status_message(bot, user_id, thread_id)
            await _do_send_status_message(
                bot, user_id, thread_id, window_id, status_text
            )
        elif status_text == last_text:
            # Same content, skip edit
            pass
        else:
            # Same window, text changed - edit in place
            history = _get_idle_history(user_id, thread_id, status_text)
            keyboard = build_status_keyboard(window_id, history=history)
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=convert_markdown(status_text),
                    parse_mode="MarkdownV2",
                    reply_markup=keyboard,
                    link_preview_options=NO_LINK_PREVIEW,
                )
                _status_msg_info[skey] = (msg_id, window_id, status_text)
            except RetryAfter:
                raise
            except TelegramError:
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text=status_text,
                        reply_markup=keyboard,
                        link_preview_options=NO_LINK_PREVIEW,
                    )
                    _status_msg_info[skey] = (msg_id, window_id, status_text)
                except RetryAfter:
                    raise
                except TelegramError as e:
                    logger.debug("Failed to edit status message: %s", e)
                    _status_msg_info.pop(skey, None)
                    await _do_send_status_message(
                        bot, user_id, thread_id, window_id, status_text
                    )
    else:
        # No existing status message, send new
        await _do_send_status_message(bot, user_id, thread_id, window_id, status_text)


async def _do_send_status_message(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
    window_id: str,
    text: str,
) -> None:
    """Send a new status message with action buttons and track it."""
    skey = (user_id, thread_id_or_0)
    thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    history = _get_idle_history(user_id, thread_id_or_0, text)
    keyboard = build_status_keyboard(window_id, history=history)
    sent = await rate_limit_send_message(
        bot,
        chat_id,
        text,
        reply_markup=keyboard,
        **_send_kwargs(thread_id),  # type: ignore[arg-type]
    )
    if sent:
        _status_msg_info[skey] = (sent.message_id, window_id, text)


async def _do_clear_status_message(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int = 0,
) -> None:
    """Delete the status message for a user (internal, called from worker)."""
    skey = (user_id, thread_id_or_0)
    info = _status_msg_info.pop(skey, None)
    if info:
        msg_id = info[0]
        thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except TelegramError as e:
            logger.debug("Failed to delete status message %s: %s", msg_id, e)


async def _check_and_send_status(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
) -> None:
    """Check terminal for status line and send status message if present."""
    # Skip if there are more messages pending in the queue
    queue = _message_queues.get(user_id)
    if queue and not queue.empty():
        return

    # Flush any active batch before showing status
    thread_id_or_0 = thread_id or 0
    bkey = (user_id, thread_id_or_0)
    if bkey in _active_batches:
        await _flush_batch(bot, user_id, thread_id_or_0)
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        return

    pane_text = await tmux_manager.capture_pane(w.window_id)
    if not pane_text:
        return

    # Suppress status for muted/errors_only windows (mirrors update_status_message)
    notif_mode = session_manager.get_notification_mode(window_id)
    if notif_mode in ("muted", "errors_only"):
        return

    status = get_provider_for_window(window_id).parse_terminal_status(pane_text)
    if status and not status.is_interactive:
        await _do_send_status_message(
            bot, user_id, thread_id_or_0, window_id, status.display_label
        )


async def enqueue_content_message(
    bot: Bot,
    user_id: int,
    window_id: str,
    parts: list[str],
    tool_use_id: str | None = None,
    content_type: str = "text",
    text: str | None = None,
    thread_id: int | None = None,
) -> None:
    """Enqueue a content message task."""
    queue = get_or_create_queue(bot, user_id)

    task = MessageTask(
        task_type="content",
        text=text,
        window_id=window_id,
        parts=parts,
        tool_use_id=tool_use_id,
        content_type=content_type,
        thread_id=thread_id,
    )
    queue.put_nowait(task)


async def enqueue_status_update(
    bot: Bot,
    user_id: int,
    window_id: str,
    status_text: str | None,
    thread_id: int | None = None,
) -> None:
    """Enqueue status update."""
    queue = get_or_create_queue(bot, user_id)

    if status_text:
        task = MessageTask(
            task_type="status_update",
            text=status_text,
            window_id=window_id,
            thread_id=thread_id,
        )
    else:
        task = MessageTask(task_type="status_clear", thread_id=thread_id)

    queue.put_nowait(task)


def clear_status_msg_info(user_id: int, thread_id: int | None = None) -> None:
    """Clear status message tracking for a user (and optionally a specific thread)."""
    skey = (user_id, thread_id or 0)
    _status_msg_info.pop(skey, None)


def clear_batch_for_topic(user_id: int, thread_id: int | None = None) -> None:
    """Clear active batch for a specific topic (called on topic cleanup)."""
    _active_batches.pop((user_id, thread_id or 0), None)


def clear_tool_msg_ids_for_topic(user_id: int, thread_id: int | None = None) -> None:
    """Clear tool message ID tracking for a specific topic.

    Removes all entries in _tool_msg_ids that match the given user and thread.
    """
    thread_id_or_0 = thread_id or 0
    # Find and remove all matching keys
    keys_to_remove = [
        key for key in _tool_msg_ids if key[1] == user_id and key[2] == thread_id_or_0
    ]
    for key in keys_to_remove:
        _tool_msg_ids.pop(key, None)


async def shutdown_workers() -> None:
    """Stop all queue workers (called during bot shutdown)."""
    for _user_id, worker in list(_queue_workers.items()):
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
    _queue_workers.clear()
    _message_queues.clear()
    _queue_locks.clear()
    _active_batches.clear()
    logger.info("Message queue workers stopped")
