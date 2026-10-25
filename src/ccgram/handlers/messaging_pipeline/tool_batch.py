"""Claude tool-use batching — state machine, formatting, edit-in-place delivery.

Accumulates consecutive tool_use / tool_result messages into compact batch
messages displayed as a single Telegram message that is edited in place as
results arrive.  Overflow (entry count or character budget) triggers a flush
and a new batch.

Key components:
  - ToolBatchEntry / ToolBatch: batch state dataclasses
  - process_tool_event: state-machine entry point (add tool_use or tool_result)
  - flush_batch: finalize and send the last edit for a batch
  - is_batch_eligible: predicate combining task eligibility and window mode
  - format_batch_message: render batch entries as compact text
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import structlog

from ...delivery_contract import DeliveryReceipt
from ...telegram_client import TelegramClient, unwrap_bot
from ...telegram_draft import DRAFT_UNSET, DraftStream
from ...thread_router import thread_router
from ...topic_state_registry import topic_state
from ...window_state_ports.tool_state import get_batch_mode, is_ephemeral_tools
from .message_task import ContentTask, thread_key

logger = structlog.get_logger()

BATCH_MAX_LENGTH = 2800
BATCH_MAX_ENTRIES = 9


@dataclass
class ToolBatchEntry:
    """A single tool call entry within a batch."""

    tool_use_id: str | None
    tool_use_text: str  # Formatted summary from build_response_parts
    tool_result_text: str | None = None  # None until result arrives
    tool_name: str | None = None
    delivery_receipts: tuple[DeliveryReceipt, ...] = ()


@dataclass
class ToolBatch:
    """Accumulator for consecutive tool calls to batch into one Telegram message."""

    window_id: str
    thread_id: int  # thread_id_or_0
    entries: list[ToolBatchEntry] = field(default_factory=list)
    telegram_msg_id: int | None = None
    total_length: int = 0
    draft: DraftStream | None = None
    last_sent_text: str | None = None


class ToolEventOutcome(Enum):
    """Terminal delivery outcome reported by the batching boundary."""

    DELIVERED = "delivered"
    INTENTIONALLY_DROPPED = "intentionally_dropped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ToolEventResult:
    """Batch routing result with an explicit delivery acknowledgement."""

    followup: ContentTask | None = None
    outcome: ToolEventOutcome = ToolEventOutcome.DELIVERED


# Active tool batches: (user_id, thread_id_or_0) -> ToolBatch
_active_batches: dict[tuple[int, int], ToolBatch] = {}

_MARKDOWN_TOOL_PREFIX_RE = re.compile(r"^\*\*([^*]+)\*\*(.*)$")
_PLAIN_TASK_CREATE_RE = re.compile(r"^TaskCreate\s+(.+)$")
_TASK_TOOL_NAMES = frozenset({"TaskCreate", "TaskUpdate", "TaskList"})
_MIN_BACKTICK_WRAPPED_LENGTH = 2

# ---------------------------------------------------------------------------
# Public predicates
# ---------------------------------------------------------------------------


def is_batch_eligible(task: ContentTask) -> bool:
    """Check if a task should go through the batching pipeline."""
    return task.content_type in ("tool_use", "tool_result") and get_batch_mode(
        task.window_id
    ) in ("batched", "ephemeral")


def _is_admitted(admit: Callable[[], bool] | None) -> bool:
    """Default to the existing behavior for unscoped tool events."""
    return admit is None or admit()


def _batch_has_revoked_receipt(batch: ToolBatch) -> bool:
    """Whether any cached tool entry belongs to withdrawn producer work."""
    return any(
        receipt.revoked
        for entry in batch.entries
        for receipt in entry.delivery_receipts
    )


def _retire_revoked_batch(bkey: tuple[int, int], batch: ToolBatch) -> None:
    """Seal stale batch state without editing or deleting its visible history."""
    if _active_batches.get(bkey) is batch:
        _active_batches.pop(bkey, None)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_batch_message(
    entries: list[ToolBatchEntry], subagent_label: str | None = None
) -> str:
    """Render a batch of tool calls as a single compact message.

    Format:
        ⚡ 3 tool calls [🤖 write-tests]
        📖 Read  src/foo.py       ⎿  42 lines
        ✏️ Edit  src/foo.py       ⎿  +3 −1
        ⚡ Bash  make test        ⏳
    """
    task_create_message = _format_task_create_batch(entries, subagent_label)
    if task_create_message is not None:
        return task_create_message

    lines: list[str] = []
    if subagent_label:
        lines.append(subagent_label)
    lines.extend(_format_mixed_batch_lines(entries))

    return "\n".join(lines)


def _format_task_create_batch(
    entries: list[ToolBatchEntry], subagent_label: str | None
) -> str | None:
    """Render TaskCreate bursts as a numbered task list."""
    if not entries or any(entry.tool_name != "TaskCreate" for entry in entries):
        return None

    titles = [_extract_task_create_title(entry) for entry in entries]
    if any(not title for title in titles):
        return None

    action = (
        "Created"
        if all(entry.tool_result_text is not None for entry in entries)
        else "Creating"
    )
    task_label = "task" if len(entries) == 1 else "tasks"
    lines: list[str] = []
    if subagent_label:
        lines.append(subagent_label)
    if action == "Creating":
        lines.append(f"{action} {len(entries)} {task_label}\u2026")
    else:
        lines.append(f"{action} {len(entries)} {task_label}")
    lines.extend(f"{index}. {title}" for index, title in enumerate(titles, start=1))
    return "\n".join(lines)


def _collect_standard_run(
    entries: list[ToolBatchEntry], start: int
) -> tuple[list[ToolBatchEntry], int]:
    """Collect a consecutive run of standard (non-Task*) entries sharing tool_use_text and result bucket."""
    first = entries[start]
    run_text = first.tool_use_text
    run_bucket = first.tool_result_text
    run: list[ToolBatchEntry] = [first]
    index = start + 1
    while index < len(entries) and entries[index].tool_name not in _TASK_TOOL_NAMES:
        nxt = entries[index]
        if nxt.tool_use_text != run_text or nxt.tool_result_text != run_bucket:
            break
        run.append(nxt)
        index += 1
    return run, index


def _format_mixed_batch_lines(entries: list[ToolBatchEntry]) -> list[str]:
    """Render batch body lines, grouping task-tool runs into task sections."""
    lines: list[str] = []
    index = 0

    while index < len(entries):
        entry = entries[index]
        if entry.tool_name == "TaskCreate":
            task_entries: list[ToolBatchEntry] = []
            while index < len(entries) and entries[index].tool_name == "TaskCreate":
                task_entries.append(entries[index])
                index += 1
            section = _format_task_create_section(task_entries)
            if section:
                lines.extend(section)
            else:
                lines.extend(_format_batch_entry(task) for task in task_entries)
            continue
        if entry.tool_name == "TaskUpdate":
            update_entries: list[ToolBatchEntry] = []
            while index < len(entries) and entries[index].tool_name == "TaskUpdate":
                update_entries.append(entries[index])
                index += 1
            section = _format_task_update_section(update_entries)
            if section:
                lines.extend(section)
            else:
                lines.extend(_format_batch_entry(task) for task in update_entries)
            continue
        if entry.tool_name == "TaskList":
            lines.extend(_format_task_list_section(entry))
            index += 1
            continue

        run, index = _collect_standard_run(entries, index)
        lines.append(_format_batch_entry(run[-1], count=len(run)))

    return lines


def _format_task_create_section(entries: list[ToolBatchEntry]) -> list[str]:
    """Render a contiguous TaskCreate run inside a mixed batch."""
    if not entries:
        return []

    titles = [_extract_task_create_title(entry) for entry in entries]
    if any(not title for title in titles):
        return []

    action = (
        "Created"
        if all(entry.tool_result_text is not None for entry in entries)
        else "Creating"
    )
    task_label = "task" if len(entries) == 1 else "tasks"
    heading = (
        f"{action} {len(entries)} {task_label}\u2026"
        if action == "Creating"
        else f"{action} {len(entries)} {task_label}"
    )
    return [
        heading,
        *(f"{index}. {title}" for index, title in enumerate(titles, start=1)),
    ]


def _format_task_update_section(entries: list[ToolBatchEntry]) -> list[str]:
    """Render a contiguous TaskUpdate run inside a mixed batch."""
    if not entries:
        return []

    labels = [_extract_task_tool_suffix(entry) for entry in entries]
    if any(not label for label in labels):
        return []

    action = (
        "Updated"
        if all(entry.tool_result_text is not None for entry in entries)
        else "Updating"
    )
    task_label = "task" if len(entries) == 1 else "tasks"
    heading = (
        f"{action} {len(entries)} {task_label}\u2026"
        if action == "Updating"
        else f"{action} {len(entries)} {task_label}"
    )
    return [heading, *(f"- {label}" for label in labels)]


def _format_task_list_section(entry: ToolBatchEntry) -> list[str]:
    """Render TaskList as task-list sync progress."""
    summary = _extract_task_tool_suffix(entry)
    heading = (
        "Synced task list"
        if entry.tool_result_text is not None
        else "Refreshing task list\u2026"
    )
    if summary and summary != "refresh":
        heading = f"{heading} ({summary})"
    return [heading]


def _format_batch_entry(entry: ToolBatchEntry, count: int = 1) -> str:
    """Render one standard batch row \u2014 name + summary only, no status glyph."""
    line = entry.tool_use_text
    if count > 1:
        line = f"{line} \u00d7{count}"
    return line


def _extract_task_create_title(entry: ToolBatchEntry) -> str:
    """Extract the visible title from a TaskCreate summary."""
    return _extract_task_tool_suffix(entry)


def _extract_task_tool_suffix(entry: ToolBatchEntry) -> str:
    """Extract the summary text after a tool-call prefix.

    Handles the current ``{emoji} {name}: {summary}`` shape plus two legacy
    formats (``**Name** `summary`` and bare ``TaskCreate Title``) for back-
    compat with anything still sitting in old batches.
    """
    text = entry.tool_use_text.strip()
    if not text:
        return ""

    # Current shape: "📋 taskcreate: `TITLE`" (inline-mono summary).
    if ": " in text:
        _, _, suffix = text.partition(": ")
        suffix = suffix.strip()
        if (
            suffix.startswith("`")
            and suffix.endswith("`")
            and len(suffix) >= _MIN_BACKTICK_WRAPPED_LENGTH
        ):
            suffix = suffix[1:-1].strip()
        if suffix:
            return suffix

    markdown_match = _MARKDOWN_TOOL_PREFIX_RE.match(text)
    if markdown_match:
        _tool_name, suffix = markdown_match.groups()
        stripped = suffix.strip()
        if (
            stripped.startswith("`")
            and stripped.endswith("`")
            and len(stripped) >= _MIN_BACKTICK_WRAPPED_LENGTH
        ):
            stripped = stripped[1:-1].strip()
        return stripped

    plain_match = _PLAIN_TASK_CREATE_RE.match(text)
    if plain_match:
        return plain_match.group(1).strip()

    return text


# ---------------------------------------------------------------------------
# State machine — process_tool_event / flush_batch
# ---------------------------------------------------------------------------


def _prepare_unsent_batch(batch: ToolBatch) -> bool:
    """Remove withdrawn, never-published entries without rewriting seen history."""
    if not _batch_has_revoked_receipt(batch):
        return True
    if (
        batch.telegram_msg_id is not None
        or batch.last_sent_text is not None
        or (batch.draft is not None and batch.draft.mode != DRAFT_UNSET)
    ):
        return False
    batch.entries = [
        entry
        for entry in batch.entries
        if not any(receipt.revoked for receipt in entry.delivery_receipts)
    ]
    batch.total_length = sum(len(entry.tool_use_text) for entry in batch.entries)
    return bool(batch.entries)


def _batch_admission(
    batch: ToolBatch, admit: Callable[[], bool] | None
) -> Callable[[], bool]:
    """Keep cached entries and the current task valid through sender awaits."""
    return lambda: _is_admitted(admit) and not _batch_has_revoked_receipt(batch)


async def _send_ephemeral_batch(
    client: TelegramClient,
    user_id: int,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    thread_id_or_0: int,
    text: str,
    admit: Callable[[], bool],
) -> bool:
    # Lazy: status and sender helpers import the messaging pipeline.
    from ..status.status_bubble import clear_status_message

    # Lazy: message_sender is a sibling in this package's import cycle.
    from .message_sender import edit_with_fallback, safe_send

    if batch.telegram_msg_id is not None:
        return await edit_with_fallback(
            client, chat_id, batch.telegram_msg_id, text, admit=admit
        )
    await clear_status_message(client, user_id, thread_id_or_0)
    if not admit():
        return False
    await _rate_limit_chat(chat_id)
    if not admit():
        return False
    message = await safe_send(
        client, chat_id, text, message_thread_id=raw_thread_id, admit=admit
    )
    if message is None:
        return False
    batch.telegram_msg_id = message.message_id
    return True


async def _send_persistent_batch(
    client: TelegramClient,
    user_id: int,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    thread_id_or_0: int,
    text: str,
    admit: Callable[[], bool],
) -> bool:
    # Lazy: status imports the messaging pipeline during bootstrap.
    from ..status.status_bubble import clear_status_message

    if batch.draft is not None:
        return await batch.draft.replace_confirmed(text, admit=admit)
    await clear_status_message(client, user_id, thread_id_or_0)
    if not admit():
        return False
    await _rate_limit_chat(chat_id)
    if not admit():
        return False
    # A transcript receipt needs persistent delivery, not an ephemeral preview.
    batch.draft = DraftStream(
        unwrap_bot(client),
        chat_id,
        message_thread_id=raw_thread_id,
        force_legacy=True,
        admit=admit,
    )
    message_id = await batch.draft.start(text)
    if message_id is None:
        batch.draft = None
        return False
    batch.telegram_msg_id = message_id
    return True


async def _send_or_edit_batch(
    client: TelegramClient,
    user_id: int,
    batch: ToolBatch,
    chat_id: int,
    raw_thread_id: int | None,
    thread_id_or_0: int,
    *,
    admit: Callable[[], bool] | None = None,
) -> bool:
    """Send current batch content without editing withdrawn visible history."""
    if not _is_admitted(admit) or not _prepare_unsent_batch(batch):
        return False
    # Lazy: provider task tracking imports session readers during bootstrap.
    from ...claude_task_state import build_subagent_label, get_subagent_names

    label = build_subagent_label(get_subagent_names(batch.window_id))
    text = format_batch_message(batch.entries, subagent_label=label)
    if batch.last_sent_text == text and (
        batch.telegram_msg_id is not None or batch.draft is not None
    ):
        return True
    sender = (
        _send_ephemeral_batch
        if is_ephemeral_tools(batch.window_id)
        else _send_persistent_batch
    )
    delivered = await sender(
        client,
        user_id,
        batch,
        chat_id,
        raw_thread_id,
        thread_id_or_0,
        text,
        _batch_admission(batch, admit),
    )
    if delivered:
        batch.last_sent_text = text
    return delivered


async def _rate_limit_chat(chat_id: int) -> None:
    """Acquire the per-chat rate-limit slot before opening a new draft."""
    # Lazy: sibling import — message_sender ↔ tool_batch via
    # messaging_pipeline/__init__.
    # Lazy: avoids tool_batch ↔ message_sender cycle through the queue worker
    from .message_sender import rate_limit_send

    await rate_limit_send(chat_id)


async def _handle_tool_result(
    client: TelegramClient,
    user_id: int,
    task: ContentTask,
    batch: ToolBatch | None,
    thread_id_or_0: int,
    admit: Callable[[], bool] | None = None,
) -> tuple[ToolBatch | None, ContentTask | None]:
    """Process a tool_result event, updating the matching batch entry.

    Returns (updated_batch, followup) — followup is non-None when the result
    could not be absorbed into the batch and should be delivered as content.
    In ephemeral mode, orphan results (tool_use was FIFO-dropped) are dropped
    silently: returns (None, None) instead of flushing + delivering as content.
    """
    if not _is_admitted(admit):
        return None, None
    if not task.tool_use_id or not batch:
        return None, task
    for entry in batch.entries:
        if entry.tool_use_id == task.tool_use_id:
            text = "\n".join(task.parts) if task.parts else ""
            first_line = text.split("\n", 1)[0][:72]
            entry.tool_result_text = first_line
            known = {id(receipt) for receipt in entry.delivery_receipts}
            entry.delivery_receipts = (
                *entry.delivery_receipts,
                *(
                    receipt
                    for receipt in task.delivery_receipts
                    if id(receipt) not in known
                ),
            )
            return batch, None
    if is_ephemeral_tools(batch.window_id):
        return None, None
    await flush_batch(client, user_id, thread_id_or_0, admit=admit)
    if not _is_admitted(admit):
        return None, None
    return None, task


def _add_tool_use_entry(
    task: ContentTask,
    batch: ToolBatch,
    ephemeral: bool = False,
) -> bool:
    """Append a tool_use entry to the batch. Returns True if overflow occurred.

    In ephemeral mode, overflow drops oldest entries from the front (FIFO eviction)
    until the new entry fits, and always returns False (no flush-to-new-batch).
    """
    entry_text = "\n".join(task.parts) if task.parts else "tool call"
    if ephemeral:
        if len(entry_text) > BATCH_MAX_LENGTH:
            entry_text = entry_text[: BATCH_MAX_LENGTH - 1] + "…"
        while batch.entries and (
            len(batch.entries) >= BATCH_MAX_ENTRIES
            or batch.total_length + len(entry_text) > BATCH_MAX_LENGTH
        ):
            dropped = batch.entries.pop(0)
            batch.total_length -= len(dropped.tool_use_text)
        entry = ToolBatchEntry(
            tool_use_id=task.tool_use_id,
            tool_use_text=entry_text,
            tool_name=task.tool_name,
            delivery_receipts=task.delivery_receipts,
        )
        batch.entries.append(entry)
        batch.total_length += len(entry_text)
        return False
    if (
        len(batch.entries) >= BATCH_MAX_ENTRIES
        or batch.total_length + len(entry_text) > BATCH_MAX_LENGTH
    ):
        return True
    entry = ToolBatchEntry(
        tool_use_id=task.tool_use_id,
        tool_use_text=entry_text,
        tool_name=task.tool_name,
        delivery_receipts=task.delivery_receipts,
    )
    batch.entries.append(entry)
    batch.total_length += len(entry_text)
    return False


async def _deliver_tool_batch_event(
    client: TelegramClient,
    user_id: int,
    task: ContentTask,
    batch: ToolBatch,
    chat_id: int,
    thread_id_or_0: int,
    bkey: tuple[int, int],
    admit: Callable[[], bool] | None,
) -> ToolEventResult:
    delivered = await _send_or_edit_batch(
        client,
        user_id,
        batch,
        chat_id,
        task.thread_id,
        thread_id_or_0,
        admit=admit,
    )
    retired = _batch_has_revoked_receipt(batch)
    if retired:
        _retire_revoked_batch(bkey, batch)
    if not _is_admitted(admit):
        return ToolEventResult(outcome=ToolEventOutcome.INTENTIONALLY_DROPPED)
    if retired and not delivered:
        return ToolEventResult(followup=task)
    return ToolEventResult(
        outcome=ToolEventOutcome.DELIVERED if delivered else ToolEventOutcome.FAILED
    )


async def process_tool_event(
    client: TelegramClient,
    user_id: int,
    task: ContentTask,
    *,
    admit: Callable[[], bool] | None = None,
) -> ToolEventResult:
    """Route a tool event and explicitly acknowledge its Telegram delivery.

    ``followup`` is set when the queue must deliver the event as regular content.
    Otherwise ``outcome`` states whether batching delivered, intentionally dropped,
    or failed to deliver the event.
    """
    if not _is_admitted(admit):
        return ToolEventResult(outcome=ToolEventOutcome.INTENTIONALLY_DROPPED)

    window_id = task.window_id
    thread_id_or_0 = thread_key(task.thread_id)
    bkey = (user_id, thread_id_or_0)
    chat_id = thread_router.resolve_chat_id(user_id, task.thread_id)
    batch = _active_batches.get(bkey)
    if batch is not None and _batch_has_revoked_receipt(batch):
        _retire_revoked_batch(bkey, batch)
        batch = None

    if task.content_type == "tool_result":
        batch, followup = await _handle_tool_result(
            client, user_id, task, batch, thread_id_or_0, admit
        )
        if not _is_admitted(admit):
            return ToolEventResult(outcome=ToolEventOutcome.INTENTIONALLY_DROPPED)
        if batch is None:
            outcome = (
                ToolEventOutcome.INTENTIONALLY_DROPPED
                if followup is None
                else ToolEventOutcome.DELIVERED
            )
            return ToolEventResult(followup=followup, outcome=outcome)
    elif task.content_type == "tool_use":
        result = await _handle_tool_use_event(
            client, user_id, task, batch, window_id, thread_id_or_0, bkey, admit
        )
        if not _is_admitted(admit):
            return ToolEventResult(outcome=ToolEventOutcome.INTENTIONALLY_DROPPED)
        if isinstance(result, ContentTask):
            return ToolEventResult(followup=result)
        if result is None:
            return ToolEventResult(outcome=ToolEventOutcome.FAILED)
        batch = result
    else:
        return ToolEventResult(followup=task)

    return await _deliver_tool_batch_event(
        client, user_id, task, batch, chat_id, thread_id_or_0, bkey, admit
    )


async def _handle_tool_use_event(
    client: TelegramClient,
    user_id: int,
    task: ContentTask,
    batch: ToolBatch | None,
    window_id: str,
    thread_id_or_0: int,
    bkey: tuple[int, int],
    admit: Callable[[], bool] | None = None,
) -> ToolBatch | ContentTask | None:
    """Process a tool_use event, creating/flushing batches as needed.

    Returns a ToolBatch to continue with send/edit, a ContentTask if the caller
    should deliver it as regular content (double-overflow), or None on error.
    """
    if not _is_admitted(admit):
        return None
    if batch and batch.window_id != window_id:
        await flush_batch(client, user_id, thread_id_or_0, admit=admit)
        if not _is_admitted(admit):
            return None
        batch = None

    if not batch:
        batch = ToolBatch(window_id=window_id, thread_id=thread_id_or_0)
        _active_batches[bkey] = batch

    ephemeral = is_ephemeral_tools(window_id)
    overflow = _add_tool_use_entry(task, batch, ephemeral=ephemeral)
    if overflow:
        await flush_batch(client, user_id, thread_id_or_0, admit=admit)
        if not _is_admitted(admit):
            return None
        batch = ToolBatch(window_id=window_id, thread_id=thread_id_or_0)
        still_overflow = _add_tool_use_entry(task, batch)
        _active_batches[bkey] = batch
        if still_overflow:
            _active_batches.pop(bkey, None)
            return task

    return batch


async def flush_if_active(
    client: TelegramClient,
    user_id: int,
    task: ContentTask,
    *,
    admit: Callable[[], bool] | None = None,
) -> None:
    """Flush any active batch for the same topic before delivering non-batchable content."""
    thread_id_or_0 = thread_key(task.thread_id)
    if _is_admitted(admit) and has_active_batch(user_id, thread_id_or_0):
        await flush_batch(client, user_id, thread_id_or_0, admit=admit)


async def _finalize_open_draft(
    batch: ToolBatch,
    text: str,
    chat_id: int,
    admit: Callable[[], bool],
) -> bool:
    # Lazy: the API error is needed only while finalizing an open draft.
    from telegram.error import TelegramError

    assert batch.draft is not None
    try:
        if batch.draft.mode == DRAFT_UNSET:
            await _rate_limit_chat(chat_id)
            if not admit():
                return False
            await batch.draft.start(text, admit=admit)
            if batch.draft.mode == DRAFT_UNSET:
                logger.warning("flush_batch could not open a Telegram draft")
                return False
        await batch.draft.finalize(text, admit=admit)
    except TelegramError as exc:
        logger.warning("flush_batch finalize failed: %s", exc)
        return False
    return admit()


async def flush_batch(  # noqa: C901, PLR0911
    client: TelegramClient,
    user_id: int,
    thread_id_or_0: int,
    *,
    admit: Callable[[], bool] | None = None,
) -> None:
    """Finalize the active batch: do a final edit and clear state.

    In ephemeral mode, deletes the tool message instead of finalizing it.
    """
    # Lazy: only used inside the API error branch
    from telegram.error import TelegramError

    bkey = (user_id, thread_id_or_0)
    batch = _active_batches.get(bkey)
    if batch is None or not _is_admitted(admit):
        return
    if _batch_has_revoked_receipt(batch) or not batch.entries:
        _retire_revoked_batch(bkey, batch)
        return
    admit = _batch_admission(batch, admit)

    thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)

    if is_ephemeral_tools(batch.window_id):
        if batch.telegram_msg_id is not None and _is_admitted(admit):
            try:
                await client.delete_message(
                    chat_id=chat_id, message_id=batch.telegram_msg_id
                )
            except TelegramError as exc:
                logger.warning("flush_batch ephemeral delete failed: %s", exc)
        _active_batches.pop(bkey, None)
        return

    # Lazy: claude_task_state imports session readers; deferring keeps
    # this module's cold path tied only to the queue.
    # Lazy: claude_task_state import is provider-specific; resolved per-call
    from ...claude_task_state import build_subagent_label, get_subagent_names

    subagent_label = build_subagent_label(get_subagent_names(batch.window_id))
    batch_text = format_batch_message(batch.entries, subagent_label=subagent_label)

    if batch.draft is not None and not batch.draft.closed:
        finalized = await _finalize_open_draft(batch, batch_text, chat_id, admit)
        if finalized:
            _active_batches.pop(bkey, None)
        return

    if batch.telegram_msg_id is not None:
        if not _is_admitted(admit):
            return
        # Existing message but no active draft (e.g. batch built before
        # adoption, or draft already closed).  Edit the message in place.
        try:
            await client.edit_message_text(
                chat_id=chat_id,
                message_id=batch.telegram_msg_id,
                text=batch_text,
            )
        except TelegramError as exc:
            logger.warning("flush_batch edit failed: %s", exc)
        _active_batches.pop(bkey, None)
        return

    # No prior message at all — open a fresh draft and finalize immediately.
    await _rate_limit_chat(chat_id)
    if not _is_admitted(admit):
        return
    draft = DraftStream(
        unwrap_bot(client),
        chat_id,
        message_thread_id=thread_id,
        admit=admit,
    )
    batch.draft = draft
    try:
        await draft.start(batch_text)
        if draft.mode == DRAFT_UNSET:
            logger.warning("flush_batch could not open a Telegram draft")
            return
        await draft.finalize(admit=admit)
    except (RuntimeError, TelegramError) as exc:
        logger.warning("flush_batch start+finalize failed: %s", exc)
        return
    _active_batches.pop(bkey, None)


def has_active_batch(user_id: int, thread_id_or_0: int) -> bool:
    """Check if there is an active batch for a (user, thread) pair."""
    return (user_id, thread_id_or_0) in _active_batches


def has_ephemeral_active_batch(user_id: int, thread_id_or_0: int) -> bool:
    """Return True if an active batch exists for this (user, thread) and the
    batch's window is in ephemeral mode.

    Used by the queue dispatcher to suppress status updates while an
    ephemeral tool batch owns the bubble — the batch itself signals
    activity, and replacing it with a status bubble causes a visible
    flicker (formatted tool calls vanish, plain status appears, then the
    assistant text replaces that).
    """
    batch = _active_batches.get((user_id, thread_id_or_0))
    return batch is not None and is_ephemeral_tools(batch.window_id)


def _schedule_batch_abort(batch: ToolBatch) -> None:
    if batch.draft is None or batch.draft.closed:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(batch.draft.abort())


@topic_state.register("topic")
def clear_batch_for_topic(user_id: int, thread_id: int | None = None) -> None:
    """Clear active batch for a specific topic (called on topic cleanup)."""
    batch = _active_batches.pop((user_id, thread_key(thread_id)), None)
    if batch:
        _schedule_batch_abort(batch)


def clear_all_batches() -> None:
    """Clear all active batches (called on shutdown)."""
    batches = list(_active_batches.values())
    _active_batches.clear()
    for batch in batches:
        _schedule_batch_abort(batch)
