"""Desk-mode digest: one quiet Telegram message per topic, refreshed in place.

While the user is at the desk (``presence.is_quiet``), transcript content is
not delivered message by message. It is counted here instead, and every
``config.desk_digest_minutes`` each topic with news gets one message:

    🟡 working · 12 messages, 8 tool calls since last update
    Latest reply:
    <expandable quote>
    updated 14:32 · desk mode

The first send of a desk period is silent (``disable_notification``); later
refreshes edit the same message, which never notifies. When the user goes
away the digest is brought up to date once and forgotten, and real-time
delivery takes over.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from telegram.error import RetryAfter, TelegramError

from ...config import config
from ...expandable_quote import format_expandable_quote
from ...telegram_client import TelegramClient
from ...topic_state_registry import topic_state
from ..messaging_pipeline.message_sender import (
    edit_with_fallback,
    rate_limit_send_message,
)
from ..status.topic_emoji import topic_state_badge

logger = structlog.get_logger()

_REPLY_MAX_CHARS = 3000
_MAX_ERROR_LINES = 3
_STATE_LABELS = {
    "active": "working",
    "idle": "idle",
    "waiting": "waiting for your choice",
    "done": "done",
    "dead": "closed",
}
# States that mean "the session needs nothing more from the agent's side".
_SETTLED_STATES = frozenset({"idle", "waiting", "done", "dead"})

# (user_id, chat_id, thread_id)
_Key = tuple[int, int, int]


@dataclass(slots=True)
class _Digest:
    window_id: str
    started_at: float
    messages: int = 0
    tool_calls: int = 0
    latest_reply: str = ""
    errors: list[str] = field(default_factory=list)
    message_id: int | None = None
    last_flush_at: float | None = None
    last_state: str | None = None
    dirty: bool = False


_digests: dict[_Key, _Digest] = {}
_lock = asyncio.Lock()


def _clock() -> float:
    return time.monotonic()


def _wall_time() -> str:
    return datetime.now().strftime("%H:%M")


def _digest(user_id: int, chat_id: int, thread_id: int, window_id: str) -> _Digest:
    key = (user_id, chat_id, thread_id)
    entry = _digests.get(key)
    if entry is None:
        entry = _Digest(window_id=window_id, started_at=_clock())
        _digests[key] = entry
    entry.window_id = window_id
    return entry


def record_content(
    user_id: int,
    chat_id: int | None,
    thread_id: int | None,
    window_id: str,
    *,
    role: str,
    content_type: str,
    text: str,
) -> None:
    """Count one complete transcript item held back in desk mode."""
    if chat_id is None or thread_id is None:
        return
    if role == "assistant" and content_type == "text" and text.strip():
        entry = _digest(user_id, chat_id, thread_id, window_id)
        entry.messages += 1
        entry.latest_reply = text.strip()
    elif content_type == "tool_use":
        entry = _digest(user_id, chat_id, thread_id, window_id)
        entry.tool_calls += 1
    else:
        return
    entry.dirty = True


def record_error(
    user_id: int, chat_id: int, thread_id: int, window_id: str, text: str
) -> None:
    """Keep an error line (e.g. an API error) for the next digest."""
    entry = _digest(user_id, chat_id, thread_id, window_id)
    entry.errors = [*entry.errors, text][-_MAX_ERROR_LINES:]
    entry.dirty = True


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _state_of(chat_id: int, thread_id: int) -> tuple[str, str]:
    badge = topic_state_badge(chat_id, thread_id)
    return badge if badge is not None else ("active", "")


def render(entry: _Digest, chat_id: int, thread_id: int) -> str:
    """The digest text (markdown; converted to entities at send time)."""
    state, emoji = _state_of(chat_id, thread_id)
    label = _STATE_LABELS.get(state, state)
    head = f"{emoji} {label}" if emoji else label
    counts = (
        f"{_plural(entry.messages, 'message')}, "
        f"{_plural(entry.tool_calls, 'tool call')} since last update"
    )
    lines = [f"{head} · {counts}"]
    if entry.latest_reply:
        reply = entry.latest_reply
        if len(reply) > _REPLY_MAX_CHARS:
            reply = reply[:_REPLY_MAX_CHARS] + "\n\n… /last for the full reply"
        lines.append("Latest reply:")
        lines.append(format_expandable_quote(reply))
    lines.extend(f"⚠ {line}" for line in entry.errors)
    lines.append(f"updated {_wall_time()} · desk mode")
    return "\n".join(lines)


async def _publish(
    client: TelegramClient,
    key: _Key,
    entry: _Digest,
    *,
    notify: bool = False,
    fresh: bool = False,
) -> None:
    _user_id, chat_id, thread_id = key
    text = render(entry, chat_id, thread_id)
    sent = False
    try:
        if entry.message_id is not None and not fresh:
            sent = await edit_with_fallback(client, chat_id, entry.message_id, text)
            if not sent:
                logger.info(
                    "Digest edit failed; sending a new one", thread_id=thread_id
                )
        if not sent:
            message = await rate_limit_send_message(
                client,
                chat_id,
                text,
                message_thread_id=thread_id,
                disable_notification=not notify,
            )
            if message is not None:
                entry.message_id = message.message_id
                sent = True
    except RetryAfter:
        logger.info("Digest send rate-limited; retrying later", thread_id=thread_id)
        return
    except TelegramError as exc:
        logger.warning("Digest send failed", thread_id=thread_id, error=str(exc))
        return
    if not sent:
        return
    entry.messages = 0
    entry.tool_calls = 0
    entry.errors = []
    entry.dirty = False
    entry.last_flush_at = _clock()
    entry.last_state = _state_of(chat_id, thread_id)[0]


async def flush_due(client: TelegramClient) -> None:
    """Send or refresh every digest whose interval has elapsed."""
    interval = config.desk_digest_minutes * 60
    now = _clock()
    async with _lock:
        for key, entry in list(_digests.items()):
            due_from = (
                entry.last_flush_at
                if entry.last_flush_at is not None
                else entry.started_at
            )
            if now - due_from < interval:
                continue
            state_changed = (
                entry.message_id is not None
                and _state_of(key[1], key[2])[0] != entry.last_state
            )
            if entry.dirty or state_changed:
                await _publish(client, key, entry)


async def close_out(client: TelegramClient, user_id: int, *, reason: str) -> None:
    """Bring a user's digests up to date once and forget them (desk -> away).

    After an idle switch the user has walked away, so a settled topic (done,
    idle, waiting for a choice) gets a fresh notifying message: an edit would
    be silent.
    """
    async with _lock:
        keys = [key for key in _digests if key[0] == user_id]
        for key in keys:
            entry = _digests.pop(key)
            has_news = entry.dirty or entry.message_id is not None
            if not has_news:
                continue
            state = _state_of(key[1], key[2])[0]
            if reason == "idle" and state in _SETTLED_STATES:
                await _publish(client, key, entry, notify=True, fresh=True)
            elif entry.dirty:
                await _publish(client, key, entry)


def has_digest(user_id: int, chat_id: int, thread_id: int) -> bool:
    return (user_id, chat_id, thread_id) in _digests


@topic_state.register("chat")
def clear_topic_digest(chat_id: int, thread_id: int) -> None:
    """Drop digest state when a topic is cleaned up."""
    for key in [k for k in _digests if k[1] == chat_id and k[2] == thread_id]:
        _digests.pop(key, None)


def _reset_digest_for_testing() -> None:
    _digests.clear()
