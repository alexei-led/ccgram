"""Message task sum type — pure data contract for the message queue.

Three frozen dataclasses replace the monolithic ``MessageTask`` that lived in
``message_queue``.  The module imports nothing from ``ccgram.handlers``,
keeping the dependency graph acyclic.
"""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

ContentType: TypeAlias = Literal["text", "thinking", "tool_use", "tool_result"]
MessageRole: TypeAlias = Literal["assistant", "user"]


@dataclass(frozen=True, slots=True)
class ContentTask:
    """A Telegram message to deliver."""

    window_id: str
    parts: tuple[str, ...]
    content_type: ContentType = "text"
    role: MessageRole = "assistant"
    tool_use_id: str | None = None
    tool_name: str | None = None
    thread_id: int | None = None
    chat_id: int | None = None
    # Opaque delivery-boundary receipts. Transcript producers set this when a
    # task belongs to a watermark cycle; task routing does not inspect it.
    delivery_receipts: tuple[object, ...] = field(compare=False, hash=False, default=())


@dataclass(frozen=True, slots=True)
class StatusUpdateTask:
    """An update to the status bubble (edit-in-place or send if missing)."""

    window_id: str
    text: str | None
    thread_id: int | None = None


@dataclass(frozen=True, slots=True)
class StatusClearTask:
    """A request to clear the status bubble for a topic."""

    window_id: str | None
    thread_id: int | None = None


MessageTask: TypeAlias = ContentTask | StatusUpdateTask | StatusClearTask


def thread_key(thread_id: int | None) -> int:
    """Normalise an optional thread_id to a dict key (None -> 0)."""
    return thread_id or 0
