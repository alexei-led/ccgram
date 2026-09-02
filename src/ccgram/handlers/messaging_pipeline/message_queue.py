"""Per-user message queue management for ordered message delivery.

Queue primitives (FIFO ordering, merging, coalescing) and the worker loop
that dispatches tasks to ``tool_batch`` and ``status_bubble``.  Status I/O,
task-list formatting, and keyboard rendering live in ``status_bubble``;
tool-use batching lives in ``tool_batch``.
"""

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass
from io import BytesIO
from typing import assert_never

import structlog
from telegram import MessageEntity
from telegram.error import RetryAfter, TelegramError
from telegramify_markdown import utf16_len

from ...config import config
from ...entity_formatting import convert_to_entities
from ...delivery_contract import (
    DeliveryOutcome,
    DeliveryReceipt,
    activate_delivery_receipt,
    deactivate_delivery_receipt,
    delivery_receipts_ready,
    get_active_delivery_receipt,
    new_delivery_receipt,
)
from ...telegram_client import TelegramClient
from ...telegram_rate_limiter import retry_after_seconds
from ...thread_router import thread_router
from ...topic_state_registry import topic_state
from ...window_resolver import resolve_window_alias
from ...multiplexer.window_liveness import is_window_live, reset_window_liveness
from ...utils import task_done_callback
from ...tts import TtsSynthesisError, get_synthesizer, prepare_tts_text
from ...window_query import is_tool_calls_hidden
from ..status.status_bubble import (
    clear_status_message,
    convert_status_to_content,
    process_status_clear,
    process_status_update,
)
from .backlog import BacklogSnapshot, register_snapshot_provider
from .message_sender import (
    edit_with_fallback,
    rate_limit_send,
    rate_limit_send_formatted_message,
    rate_limit_send_message,
    send_kwargs,
)
from .message_task import (
    ContentTask,
    ContentType,
    MessageRole,
    MessageTask,
    StatusClearTask,
    StatusUpdateTask,
    thread_key,
)
from .tool_batch import (
    clear_all_batches,
    flush_batch,
    flush_if_active,
    has_active_batch,
    has_ephemeral_active_batch,
    is_batch_eligible,
    process_tool_event,
    ToolEventResult,
)

logger = structlog.get_logger()

# Compatibility exports for internal callers; core code imports the neutral
# contract directly and does not depend on this handler implementation.
__all__ = [
    "DeliveryOutcome",
    "DeliveryReceipt",
    "activate_delivery_receipt",
    "deactivate_delivery_receipt",
    "delivery_receipts_ready",
    "new_delivery_receipt",
]

MERGE_MAX_LENGTH = 3800  # Leave room within Telegram's 4096 char message limit
_TEXT_BATCH_SEPARATOR = "\n\n"
_QUEUE_RETRY_BACKOFF_BASE_SECONDS = 2.0
_QUEUE_RETRY_BACKOFF_MAX_SECONDS = 60.0
_QUEUE_RETRY_JITTER_MAX_SECONDS = 1.0
_QUEUE_RETRY_BUDGET_SECONDS = 300.0
_QUEUE_RATE_LIMIT_LOG_COOLDOWN_SECONDS = 30.0

# Per-user message queues and worker tasks
_message_queues: dict[int, asyncio.Queue[MessageTask]] = {}
_queue_workers: dict[int, asyncio.Task[None]] = {}
_queue_locks: dict[int, asyncio.Lock] = {}  # Protect drain/refill operations

# In-flight sends: incremented around each task a worker is actively
# processing. "Queue empty" alone does not mean delivered.
_inflight_count = 0
_inflight_tasks: dict[int, ContentTask] = {}

# Per source/topic delivery lag, updated only when a content task reaches the
# Telegram boundary.  It is deliberately in-memory telemetry, not state.
_delivery_lags: dict[tuple[int, str, int], float] = {}
_rate_limit_log_last_at: dict[int, float] = {}


def _get_backlog_snapshot(
    user_id: int, window_id: str, thread_id: int | None
) -> BacklogSnapshot:
    """Return pending count, oldest age, and last delivery lag for one topic."""
    tkey = thread_key(thread_id)
    now = time.monotonic()
    tasks: list[ContentTask] = []
    queue = _message_queues.get(user_id)
    if queue is not None:
        tasks.extend(
            task
            for task in getattr(queue, "_queue", ())
            if isinstance(task, ContentTask)
            and task.window_id == window_id
            and thread_key(task.thread_id) == tkey
            and task.source_session_id is not None
        )
    inflight = _inflight_tasks.get(user_id)
    if (
        inflight is not None
        and inflight.window_id == window_id
        and thread_key(inflight.thread_id) == tkey
        and inflight.source_session_id is not None
    ):
        tasks.append(inflight)
    oldest = min((task.enqueued_monotonic for task in tasks), default=now)
    return BacklogSnapshot(
        pending_count=len(tasks),
        oldest_age_seconds=max(0.0, now - oldest) if tasks else 0.0,
        delivery_lag_seconds=_delivery_lags.get((user_id, window_id, tkey)),
    )


register_snapshot_provider(_get_backlog_snapshot)


async def purge_source_tasks(
    user_id: int,
    window_id: str,
    thread_id: int | None,
    source_session_id: str,
    snapshot_offset: int,
    chat_id: int | None = None,
) -> int | None:
    """Retire queued (never in-flight) source tasks through a frozen offset.

    Callers persist their skip barrier before this operation.  Each removed
    receipt is intentionally dropped so it cannot block the later notice
    receipt, while an in-flight send is left alone rather than cancelled.
    """
    queue = _message_queues.get(user_id)
    lock = _queue_locks.get(user_id)
    if queue is None or lock is None:
        return 0
    tkey = thread_key(thread_id)
    removed: list[ContentTask] = []
    async with lock:
        if (
            chat_id is not None
            and thread_router.resolve_window_for_thread(user_id, thread_id, chat_id)
            != window_id
        ):
            logger.warning(
                "Skipping stale backlog purge for rebound topic",
                user_id=user_id,
                window_id=window_id,
                thread_id=thread_id,
            )
            return None
        retained: list[MessageTask] = []
        for task in _drain_queue(queue):
            if (
                isinstance(task, ContentTask)
                and task.window_id == window_id
                and thread_key(task.thread_id) == tkey
                and task.source_session_id == source_session_id
                and task.source_checkpoint is not None
                and task.source_checkpoint <= snapshot_offset
            ):
                removed.append(task)
                queue.task_done()
            else:
                retained.append(task)
        for task in retained:
            queue.put_nowait(task)
            queue.task_done()
    for task in removed:
        for receipt in task.delivery_receipts:
            receipt.settle(DeliveryOutcome.INTENTIONALLY_DROPPED)
    return len(removed)


class DispatchResult(int):
    """Task-done count plus an explicit delivery outcome.

    It remains an ``int`` so the mature queue merge tests and callers retain
    their count contract while workers gain acknowledgement information.
    """

    outcome: DeliveryOutcome

    def __new__(cls, extra_task_done: int, outcome: DeliveryOutcome):
        value = int.__new__(cls, extra_task_done)
        value.outcome = outcome
        return value


@dataclass
class DispatchState:
    """Accounting populated before a merged dispatch reaches an await."""

    extra_task_done: int = 0
    merged_receipts: tuple[DeliveryReceipt, ...] = ()
    retry_task: ContentTask | None = None


@dataclass
class RetryDispatchState:
    """Caller-visible task state preserved when retry dispatch raises."""

    task: MessageTask
    merged_receipts: tuple[DeliveryReceipt, ...] = ()


def queues_idle() -> bool:
    """True when no queue has pending items and no worker is mid-send.

    Used by the session monitor to decide when parsed transcript entries
    count as delivered (committed watermark, issue #179).
    """
    if not _message_queues:
        return True
    return all(q.empty() for q in _message_queues.values()) and _inflight_count == 0


# Map (tool_use_id, user_id, thread_key) -> telegram message_id
# for editing tool_use messages with results
_tool_msg_ids: dict[tuple[str, int, int], int] = {}

_CAPTION_MAX_LENGTH = 1024  # Telegram Bot API caption limit


def _truncate_caption(text: str) -> str:
    """Truncate at last whitespace boundary under the Telegram caption limit."""
    if len(text) <= _CAPTION_MAX_LENGTH:
        return text
    truncated = text[: _CAPTION_MAX_LENGTH - 1]
    last_ws = truncated.rfind(" ")
    if last_ws > 0:
        truncated = truncated[:last_ws]
    return truncated + "…"


def _should_send_tts(task: ContentTask) -> bool:
    if not config.tts_provider:
        return False
    if task.content_type != "text":
        return False
    return task.role == "assistant"


async def _send_tts_voice(
    client: TelegramClient,
    chat_id: int,
    thread_id: int | None,
    text: str,
    *,
    window_id: str,
) -> bool:
    try:
        synthesizer = get_synthesizer()
    except (ValueError, ImportError) as exc:
        logger.warning("TTS not available for %s: %s", window_id, exc)
        return False
    if synthesizer is None:
        return False
    try:
        audio = await synthesizer.synthesize(text)
    except TtsSynthesisError as exc:
        logger.warning("TTS synthesis failed for %s: %s", window_id, exc)
        return False

    voice_file = BytesIO(audio.data)
    voice_file.name = audio.filename
    caption = _truncate_caption(text)
    await rate_limit_send(chat_id)
    try:
        await client.send_voice(
            chat_id=chat_id,
            voice=voice_file,
            caption=caption,
            **send_kwargs(thread_id),
        )
    except TelegramError as exc:
        logger.warning("Failed to send TTS voice for %s: %s", window_id, exc)
        return False
    return True


def get_message_queue(user_id: int) -> asyncio.Queue[MessageTask] | None:
    """Get the message queue for a user (if exists)."""
    return _message_queues.get(user_id)


def get_or_create_queue(
    client: TelegramClient, user_id: int
) -> asyncio.Queue[MessageTask]:
    """Get or create message queue and worker for a user.

    Also detects dead workers and respawns them so messages are not lost.
    """
    if user_id not in _message_queues:
        _message_queues[user_id] = asyncio.Queue()
        _queue_locks[user_id] = asyncio.Lock()

    # Respawn dead workers (can happen if an uncaught exception killed the task)
    existing = _queue_workers.get(user_id)
    if existing is None or existing.done():
        if existing is not None:
            logger.warning("Respawning dead queue worker for user %s", user_id)
        task = asyncio.create_task(_message_queue_worker(client, user_id))
        task.add_done_callback(task_done_callback)
        _queue_workers[user_id] = task
    return _message_queues[user_id]


def _drain_queue(queue: asyncio.Queue[MessageTask]) -> list[MessageTask]:
    """Drain all items from the queue and return them as a list.

    Destructive: the queue is empty after this call. Caller is responsible
    for re-enqueueing any items that should not be discarded.
    """
    items: list[MessageTask] = []
    while not queue.empty():
        try:
            item = queue.get_nowait()
            items.append(item)
        except asyncio.QueueEmpty:
            break
    return items


def _can_merge_tasks(base: ContentTask, candidate: MessageTask) -> bool:
    """Return whether two tasks can safely share one Telegram text message.

    Batching is deliberately narrower than generic content merging.  Only
    one-part, non-empty ``text`` tasks qualify, so tool edits, status updates,
    paginated messages, and TTS voice attachments retain their message
    boundaries.
    """
    if not isinstance(candidate, ContentTask):
        return False
    if (
        base.window_id != candidate.window_id
        or base.thread_id != candidate.thread_id
        or base.chat_id != candidate.chat_id
        or base.role != candidate.role
        # A missing chat ID follows the legacy status-conversion path, which
        # may edit an existing bubble instead of sending a message.
        or base.chat_id is None
    ):
        return False
    # A skip notice retains its own delivery boundary. Transcript tasks may
    # have different checkpoints, but only tasks from the same source can be
    # acknowledged together.
    if (
        base.is_backlog_notice
        or candidate.is_backlog_notice
        or base.source_session_id != candidate.source_session_id
        or base.content_type != "text"
        or candidate.content_type != "text"
        or base.tool_use_id is not None
        or candidate.tool_use_id is not None
        or base.tool_name is not None
        or candidate.tool_name is not None
        or base.is_text_batch
        or candidate.is_text_batch
        or len(base.parts) != 1
        or len(candidate.parts) != 1
        or not base.parts[0].strip()
        or not candidate.parts[0].strip()
    ):
        return False
    # A text task can produce a voice message when TTS is configured.  Keep
    # those media deliveries one-for-one with their source task.
    return not _should_send_tts(base)


async def _merge_content_tasks(
    queue: asyncio.Queue[MessageTask],
    first: ContentTask,
    lock: asyncio.Lock,
) -> tuple[ContentTask, int]:
    """Merge consecutive content tasks from queue.

    Returns: (merged_task, merge_count) where merge_count is the number of
    additional tasks merged (0 if no merging occurred).

    Note on queue counter management:
        put_nowait() on re-enqueued items increments the internal task counter
        again; task_done() compensates so the net count stays correct.
        Without this compensation, queue.join() would wait indefinitely.
    """
    merged_parts = list(first.parts)
    merged_receipts = list(first.delivery_receipts)
    current_length = sum(utf16_len(part) for part in merged_parts)
    oldest_enqueued_monotonic = first.enqueued_monotonic
    latest_source_checkpoint = first.source_checkpoint
    merge_count = 0

    async with lock:
        items = _drain_queue(queue)
        remaining: list[MessageTask] = []

        for i, task in enumerate(items):
            if not _can_merge_tasks(first, task):
                remaining = items[i:]
                break

            assert isinstance(task, ContentTask)
            task_length = sum(utf16_len(part) for part in task.parts)
            if (
                current_length + utf16_len(_TEXT_BATCH_SEPARATOR) + task_length
                > MERGE_MAX_LENGTH
            ):
                remaining = items[i:]
                break

            merged_parts.extend(task.parts)
            merged_receipts.extend(task.delivery_receipts)
            current_length += utf16_len(_TEXT_BATCH_SEPARATOR) + task_length
            oldest_enqueued_monotonic = min(
                oldest_enqueued_monotonic, task.enqueued_monotonic
            )
            if task.source_checkpoint is not None:
                latest_source_checkpoint = max(
                    latest_source_checkpoint or task.source_checkpoint,
                    task.source_checkpoint,
                )
            merge_count += 1

        for item in remaining:
            queue.put_nowait(item)
            queue.task_done()

    if merge_count == 0:
        return first, 0

    return (
        ContentTask(
            window_id=first.window_id,
            parts=tuple(merged_parts),
            tool_use_id=first.tool_use_id,
            content_type=first.content_type,
            role=first.role,
            thread_id=first.thread_id,
            chat_id=first.chat_id,
            delivery_receipts=tuple(merged_receipts),
            is_text_batch=True,
            source_session_id=first.source_session_id,
            source_checkpoint=latest_source_checkpoint,
            enqueued_monotonic=oldest_enqueued_monotonic,
        ),
        merge_count,
    )


async def _coalesce_status_updates(
    queue: asyncio.Queue[MessageTask],
    first: StatusUpdateTask,
    lock: asyncio.Lock,
) -> tuple[StatusUpdateTask, int]:
    """Keep only the latest pending status_update for the same topic/window.

    Returns: (selected_task, dropped_count) where dropped_count is the number
    of queued tasks removed and already accounted for.
    """
    selected = first
    dropped = 0
    key = (thread_key(first.thread_id), first.window_id)

    async with lock:
        items = _drain_queue(queue)
        remaining: list[MessageTask] = []

        for task in items:
            if not isinstance(task, StatusUpdateTask):
                remaining.append(task)
                continue
            task_key = (thread_key(task.thread_id), task.window_id)
            if task_key == key:
                selected = task
                dropped += 1
            else:
                remaining.append(task)

        for item in remaining:
            queue.put_nowait(item)
            queue.task_done()

    return selected, dropped


async def _handle_content_task(
    client: TelegramClient,
    user_id: int,
    task: ContentTask,
    queue: asyncio.Queue[MessageTask],
    lock: asyncio.Lock,
    dispatch_state: DispatchState | None = None,
) -> DispatchResult:
    """Route a content task through batching or normal processing."""
    if dispatch_state is None:
        dispatch_state = DispatchState()
    if task.content_type == "thinking" and config.hide_thinking:
        return DispatchResult(0, DeliveryOutcome.INTENTIONALLY_DROPPED)
    if task.content_type in ("tool_use", "tool_result") and is_tool_calls_hidden(
        task.window_id
    ):
        return DispatchResult(0, DeliveryOutcome.INTENTIONALLY_DROPPED)

    if is_batch_eligible(task):
        batch_result = await process_tool_event(client, user_id, task)
        if isinstance(batch_result, ToolEventResult):
            if batch_result.followup is not None:
                outcome = await _process_content_task(
                    client, user_id, batch_result.followup
                )
                return DispatchResult(0, outcome)
            outcome = DeliveryOutcome(batch_result.outcome.value)
            return DispatchResult(0, outcome)

        # Compatibility for callers that still return the former followup-only
        # contract. Production batching always returns ToolEventResult above.
        if batch_result is not None:
            outcome = await _process_content_task(client, user_id, batch_result)
            return DispatchResult(0, outcome)
        return DispatchResult(0, DeliveryOutcome.DELIVERED)

    await flush_if_active(client, user_id, task)

    merged_task, merge_count = await _merge_content_tasks(queue, task, lock)
    if merge_count > 0:
        logger.debug("Merged %d tasks for user %s", merge_count, user_id)
    original_receipts = len(task.delivery_receipts)
    dispatch_state.extra_task_done = merge_count
    dispatch_state.merged_receipts = merged_task.delivery_receipts[original_receipts:]
    dispatch_state.retry_task = merged_task
    outcome = await _process_content_task(client, user_id, merged_task)
    return DispatchResult(merge_count, outcome)


def _is_ghost_window_task_at_enqueue(window_id: str) -> bool:
    """Return True if the window is no longer bound to any topic."""
    if window_id:
        canonical_id = resolve_window_alias(window_id)
        if not thread_router.has_window(canonical_id) and not thread_router.has_window(
            window_id
        ):
            logger.debug("Skipping enqueue for unbound window %s", window_id)
            return True
    return False


def _is_stale_task(user_id: int, task: MessageTask) -> bool:
    """Return True when a queued task targets a confirmed-dead session."""
    if isinstance(task, StatusClearTask) or not task.window_id:
        return False
    if not is_window_live(task.window_id):
        logger.info(
            "Dropping queued message for closed multiplexer session",
            user_id=user_id,
            window_id=task.window_id,
            thread_id=getattr(task, "thread_id", None),
        )
        return True
    return False


async def _flush_batch_for_task(
    user_id: int, task: MessageTask, client: TelegramClient
) -> None:
    """Flush any active batch for the topic that owns this task."""
    tkey = thread_key(task.thread_id)
    if has_active_batch(user_id, tkey):
        await flush_batch(client, user_id, tkey)


async def _dispatch(
    client: TelegramClient,
    user_id: int,
    task: MessageTask,
    queue: asyncio.Queue[MessageTask],
    lock: asyncio.Lock,
    dispatch_state: DispatchState | None = None,
) -> DispatchResult:
    """Dispatch a task and report its explicit delivery outcome."""
    match task:
        case ContentTask() as ct:
            return await _handle_content_task(
                client, user_id, ct, queue, lock, dispatch_state
            )
        case StatusUpdateTask() as st:
            # Suppress status polls while an ephemeral tool batch owns the
            # bubble — the batch itself is the activity indicator. Flushing
            # to insert a status bubble causes a visible flicker (formatted
            # tool calls vanish, plain status appears, then the assistant
            # text replaces that).
            if has_ephemeral_active_batch(user_id, thread_key(st.thread_id)):
                # Drop any siblings the coalescer would have consumed so
                # the next poll cycle sees a clean queue.
                _, dropped = await _coalesce_status_updates(queue, st, lock)
                for _ in range(dropped):
                    queue.task_done()
                return DispatchResult(0, DeliveryOutcome.INTENTIONALLY_DROPPED)
            await _flush_batch_for_task(user_id, st, client)
            collapsed_task, dropped = await _coalesce_status_updates(queue, st, lock)
            if dropped > 0:
                for _ in range(dropped):
                    queue.task_done()
            await process_status_update(client, user_id, collapsed_task)
            return DispatchResult(0, DeliveryOutcome.DELIVERED)
        case StatusClearTask() as cl:
            await _flush_batch_for_task(user_id, cl, client)
            await process_status_clear(client, user_id, cl)
            return DispatchResult(0, DeliveryOutcome.DELIVERED)
        case _ as unreachable:
            assert_never(unreachable)


def _retry_task_for_state(state: DispatchState, task: MessageTask) -> MessageTask:
    return state.retry_task or task


def _delivery_receipts_for_settlement(
    task: MessageTask,
    merged_receipts: tuple[DeliveryReceipt, ...],
) -> list[DeliveryReceipt]:
    receipts = task.delivery_receipts if isinstance(task, ContentTask) else ()
    unique: dict[int, DeliveryReceipt] = {}
    for receipt in (*receipts, *merged_receipts):
        unique[id(receipt)] = receipt
    return list(unique.values())


def _mark_task_inflight(user_id: int, task: MessageTask) -> None:
    """Expose a content task to source-scoped backlog telemetry."""
    if isinstance(task, ContentTask):
        _inflight_tasks[user_id] = task


def _record_content_delivery(user_id: int, task: MessageTask) -> None:
    """Settle in-memory delivery-lag telemetry after a worker attempt."""
    if isinstance(task, ContentTask):
        _inflight_tasks.pop(user_id, None)
        _delivery_lags[(user_id, task.window_id, thread_key(task.thread_id))] = max(
            0.0, time.monotonic() - task.enqueued_monotonic
        )


def _log_rate_limit_queue_state(
    user_id: int,
    task: MessageTask,
    retry: int,
    retry_after: float,
    retry_in: float,
    blocked_for: float,
) -> None:
    """Periodically report queue pressure while Telegram flood control blocks it."""
    now = time.monotonic()
    last_at = _rate_limit_log_last_at.get(user_id)
    if last_at is not None and now - last_at < _QUEUE_RATE_LIMIT_LOG_COOLDOWN_SECONDS:
        return
    _rate_limit_log_last_at[user_id] = now
    queue = _message_queues.get(user_id)
    logger.debug(
        "Telegram rate limit is blocking message queue",
        user_id=user_id,
        queued_tasks=queue.qsize() if queue is not None else 0,
        current_task_in_flight=True,
        retry=retry,
        blocked_for_seconds=round(blocked_for, 1),
        telegram_retry_after_seconds=round(retry_after, 1),
        retry_in_seconds=round(retry_in, 1),
        window_id=getattr(task, "window_id", ""),
        thread_id=getattr(task, "thread_id", None),
    )


async def _dispatch_with_retry(
    client: TelegramClient,
    user_id: int,
    state: RetryDispatchState,
    queue: asyncio.Queue[MessageTask],
    lock: asyncio.Lock,
) -> DeliveryOutcome:
    """Dispatch one task, dropping dead targets and bounding flood retries."""
    rate_limit_retry = 0
    retry_started = time.monotonic()
    while True:
        dispatch_state = DispatchState()
        if _is_stale_task(user_id, state.task):
            return DeliveryOutcome.INTENTIONALLY_DROPPED
        outcome: DeliveryOutcome | None = None
        structlog.contextvars.clear_contextvars()
        with structlog.contextvars.bound_contextvars(
            window_id=getattr(state.task, "window_id", "")
        ):
            try:
                result = await _dispatch(
                    client,
                    user_id,
                    state.task,
                    queue,
                    lock,
                    dispatch_state,
                )
                outcome = getattr(result, "outcome", DeliveryOutcome.DELIVERED)
            except RetryAfter as exc:
                state.task = _retry_task_for_state(dispatch_state, state.task)
                rate_limit_retry += 1
                elapsed = time.monotonic() - retry_started
                if elapsed >= _QUEUE_RETRY_BUDGET_SECONDS:
                    logger.error(
                        "Telegram flood control retry budget exhausted",
                        user_id=user_id,
                        retry=rate_limit_retry,
                        retry_budget_seconds=_QUEUE_RETRY_BUDGET_SECONDS,
                    )
                    outcome = DeliveryOutcome.FAILED
                else:
                    telegram_delay = retry_after_seconds(exc)
                    exponent = min(rate_limit_retry - 1, 5)
                    backoff = min(
                        _QUEUE_RETRY_BACKOFF_MAX_SECONDS,
                        _QUEUE_RETRY_BACKOFF_BASE_SECONDS * (2**exponent),
                    )
                    jitter = random.uniform(0, _QUEUE_RETRY_JITTER_MAX_SECONDS)
                    retry_in = max(telegram_delay, backoff) + jitter
                    remaining = _QUEUE_RETRY_BUDGET_SECONDS - elapsed
                    if remaining <= 0:
                        outcome = DeliveryOutcome.FAILED
                    else:
                        retry_in = min(retry_in, remaining)
                        _log_rate_limit_queue_state(
                            user_id,
                            state.task,
                            rate_limit_retry,
                            telegram_delay,
                            retry_in,
                            elapsed,
                        )
                        logger.warning(
                            "Telegram flood control; retrying queued message",
                            user_id=user_id,
                            retry=rate_limit_retry,
                            telegram_retry_after_seconds=telegram_delay,
                            backoff_seconds=backoff,
                            jitter_seconds=jitter,
                            retry_in_seconds=retry_in,
                        )
                        await asyncio.sleep(retry_in)
            finally:
                structlog.contextvars.clear_contextvars()
                for _ in range(dispatch_state.extra_task_done):
                    queue.task_done()
                state.merged_receipts = dispatch_state.merged_receipts
        if outcome is not None:
            return outcome


async def _message_queue_worker(client: TelegramClient, user_id: int) -> None:
    global _inflight_count
    """Process message tasks for a user sequentially."""
    queue = _message_queues[user_id]
    lock = _queue_locks[user_id]
    logger.debug("Message queue worker started for user %s", user_id)

    while True:
        try:
            task = await queue.get()
            _inflight_count += 1
            _mark_task_inflight(user_id, task)
            outcome = DeliveryOutcome.DELIVERED
            retry_state = RetryDispatchState(task)
            try:
                outcome = await _dispatch_with_retry(
                    client, user_id, retry_state, queue, lock
                )
            except asyncio.CancelledError:
                # A bounded shutdown may cancel an in-flight send. Do not
                # acknowledge bytes whose task was interrupted; restart replays.
                outcome = DeliveryOutcome.FAILED
                raise
            except Exception:  # noqa: BLE001 — delivery failures must not kill workers
                outcome = DeliveryOutcome.FAILED
                logger.exception(
                    "Error processing message task for user %s (thread %s)",
                    user_id,
                    getattr(task, "thread_id", None),
                )
            finally:
                for receipt in _delivery_receipts_for_settlement(
                    retry_state.task, retry_state.merged_receipts
                ):
                    receipt.settle(outcome)
                _record_content_delivery(user_id, retry_state.task)
                _inflight_count -= 1
                queue.task_done()
        except asyncio.CancelledError:
            logger.debug("Message queue worker cancelled for user %s", user_id)
            break
        except Exception:
            logger.exception(
                "Unexpected error in queue worker for user %s",
                user_id,
            )


def _render_text_batch(parts: tuple[str, ...]) -> tuple[str, list[MessageEntity]]:
    """Render text tasks independently, then combine their Telegram payloads.

    Rendering raw markdown only after concatenation lets syntax opened in one
    task affect another task.  Converting each source task first preserves the
    exact entities it would have received alone, while shifted UTF-16 offsets
    make the combined API call valid.
    """
    plain_parts: list[str] = []
    entities: list[MessageEntity] = []
    offset = 0
    for raw_part in parts:
        plain, part_entities = convert_to_entities(raw_part)
        if plain_parts:
            plain_parts.append(_TEXT_BATCH_SEPARATOR)
            offset += utf16_len(_TEXT_BATCH_SEPARATOR)
        plain_parts.append(plain)
        for entity in part_entities:
            entities.append(
                MessageEntity(
                    type=entity.type,
                    offset=entity.offset + offset,
                    length=entity.length,
                    url=entity.url,
                    language=entity.language,
                    custom_emoji_id=entity.custom_emoji_id,
                )
            )
        offset += utf16_len(plain)
    return "".join(plain_parts), entities


async def _process_text_batch(
    client: TelegramClient, chat_id: int, task: ContentTask
) -> DeliveryOutcome:
    """Deliver a rendered text batch as one Telegram message."""
    plain_text, entities = _render_text_batch(task.parts)
    sent = await rate_limit_send_formatted_message(
        client,
        chat_id,
        plain_text,
        entities,
        **send_kwargs(task.thread_id),
    )
    return DeliveryOutcome.DELIVERED if sent else DeliveryOutcome.FAILED


async def _try_edit_tool_result(
    client: TelegramClient,
    user_id: int,
    chat_id: int,
    tkey: int,
    task: ContentTask,
) -> bool:
    """Edit a prior tool-use message when this task supplies its result."""
    if task.content_type != "tool_result" or not task.tool_use_id:
        return False
    key = (task.tool_use_id, user_id, tkey)
    edit_msg_id = _tool_msg_ids.get(key)
    if edit_msg_id is None:
        return False
    try:
        await clear_status_message(client, user_id, tkey)
        success = await edit_with_fallback(
            client,
            chat_id,
            edit_msg_id,
            "\n\n".join(task.parts),
        )
    except RetryAfter:
        # Keep the ID so the queue retry edits the original tool-use message.
        raise
    except Exception:
        _tool_msg_ids.pop(key, None)
        raise
    _tool_msg_ids.pop(key, None)
    if not success:
        logger.debug("Failed to edit tool msg %s, sending new", edit_msg_id)
    return success


async def _process_content_task(
    client: TelegramClient, user_id: int, task: ContentTask
) -> DeliveryOutcome:
    """Process a content message task and report whether it reached Telegram."""
    if task.is_backlog_notice and (
        task.chat_id is None
        or thread_router.resolve_window_for_thread(
            user_id, task.thread_id, task.chat_id
        )
        != task.window_id
    ):
        logger.warning(
            "Dropping stale backlog skip notice",
            user_id=user_id,
            window_id=task.window_id,
            thread_id=task.thread_id,
        )
        return DeliveryOutcome.FAILED

    tkey = thread_key(task.thread_id)
    chat_id = task.chat_id or thread_router.resolve_chat_id(user_id, task.thread_id)

    if task.is_text_batch:
        return await _process_text_batch(client, chat_id, task)

    if await _try_edit_tool_result(client, user_id, chat_id, tkey, task):
        return DeliveryOutcome.DELIVERED

    first_part = True
    last_msg_id: int | None = None
    for part in task.parts:
        sent = None

        if first_part and task.chat_id is None:
            first_part = False
            converted_msg_id = await convert_status_to_content(
                client,
                user_id,
                tkey,
                task.window_id,
                part,
            )
            if converted_msg_id is not None:
                last_msg_id = converted_msg_id
                continue
        else:
            first_part = False

        sent = await rate_limit_send_message(
            client, chat_id, part, **send_kwargs(task.thread_id)
        )

        if sent:
            last_msg_id = sent.message_id
        else:
            # The sender exhausted its entity/plain fallback without raising.
            # A transcript watermark must treat that as a terminal failure.
            return DeliveryOutcome.FAILED

    if _should_send_tts(task) and (tts_text := prepare_tts_text(task.parts)):
        await _send_tts_voice(
            client,
            chat_id,
            task.thread_id,
            tts_text,
            window_id=task.window_id,
        )

    if last_msg_id and task.tool_use_id and task.content_type == "tool_use":
        _tool_msg_ids[(task.tool_use_id, user_id, tkey)] = last_msg_id
    return DeliveryOutcome.DELIVERED


async def enqueue_content_message(
    client: TelegramClient,
    user_id: int,
    window_id: str,
    parts: list[str],
    tool_use_id: str | None = None,
    tool_name: str | None = None,
    content_type: ContentType = "text",
    role: MessageRole = "assistant",
    thread_id: int | None = None,
    chat_id: int | None = None,
    source_session_id: str | None = None,
    source_checkpoint: int | None = None,
    is_backlog_notice: bool = False,
) -> bool:
    """Enqueue a content message task and report whether it entered the queue."""
    if _is_ghost_window_task_at_enqueue(window_id):
        return False
    if not is_window_live(window_id):
        logger.info(
            "Skipping enqueue for closed multiplexer session", window_id=window_id
        )
        return False
    queue = get_or_create_queue(client, user_id)

    receipt = get_active_delivery_receipt()
    if receipt is not None:
        receipt.track()
    task = ContentTask(
        window_id=window_id,
        parts=tuple(parts),
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        content_type=content_type,
        role=role,
        thread_id=thread_id,
        chat_id=chat_id,
        delivery_receipts=(receipt,) if receipt is not None else (),
        source_session_id=source_session_id,
        source_checkpoint=(
            source_checkpoint
            if source_checkpoint is not None
            else (receipt.checkpoint if receipt is not None else None)
        ),
        is_backlog_notice=is_backlog_notice,
    )
    queue.put_nowait(task)
    return True


async def enqueue_status_update(
    client: TelegramClient,
    user_id: int,
    window_id: str,
    status_text: str | None,
    thread_id: int | None = None,
) -> None:
    """Enqueue status update or clear."""
    if status_text is not None and not is_window_live(window_id):
        logger.info(
            "Skipping status update for closed multiplexer session", window_id=window_id
        )
        return
    queue = get_or_create_queue(client, user_id)

    if status_text is not None:
        task: MessageTask = StatusUpdateTask(
            window_id=window_id,
            text=status_text,
            thread_id=thread_id,
        )
    else:
        task = StatusClearTask(
            window_id=window_id,
            thread_id=thread_id,
        )

    queue.put_nowait(task)


@topic_state.register("topic")
def clear_tool_msg_ids_for_topic(user_id: int, thread_id: int | None = None) -> None:
    """Clear tool message ID tracking for a specific topic.

    Removes all entries in _tool_msg_ids that match the given user and thread.
    """
    tkey = thread_key(thread_id)
    keys_to_remove = [
        key for key in _tool_msg_ids if key[1] == user_id and key[2] == tkey
    ]
    for key in keys_to_remove:
        _tool_msg_ids.pop(key, None)


async def shutdown_workers(drain_timeout: float = 10.0) -> None:
    """Stop all queue workers (called during client shutdown).

    The monitor parses transcript entries before the queue delivers them
    to Telegram; on this PR's delivered-watermark model anything the drain
    cannot finish is replayed on the next start. Draining while the HTTP
    transport is still alive (post_stop) bounds how much gets replayed.
    Callers must already have stopped the monitor so no new work arrives
    during the drain.
    """
    joins = [queue.join() for queue in _message_queues.values()]
    if joins:
        try:
            # Queue.join() accounts for both queued and in-flight work. Its
            # timeout must be external: asyncio.Queue has no timed join.
            await asyncio.wait_for(asyncio.gather(*joins), timeout=drain_timeout)
        except asyncio.TimeoutError:
            pending = sum(q.qsize() for q in _message_queues.values())
            logger.warning("Shutdown drain timeout: %d queued task(s) remain", pending)
    for _, worker in list(_queue_workers.items()):
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
    _queue_workers.clear()
    _message_queues.clear()
    _queue_locks.clear()
    _inflight_tasks.clear()
    _delivery_lags.clear()
    _rate_limit_log_last_at.clear()
    reset_window_liveness()
    clear_all_batches()
    logger.info("Message queue workers stopped")
