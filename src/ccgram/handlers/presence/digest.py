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

With ``CCGRAM_DESK_BRIEF=dm`` the timing is per user instead of per topic:
one interval after the first news of a desk period, ``desk_brief`` sends a
single notifying direct message covering every topic with news, and the topic
digests are refreshed silently at the same moment. Going away after an idle
spell sends one last brief; the topics then never ring on their own.
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
from ...session_query import iter_bound_topics, resolve_chat_id
from ...telegram_client import TelegramClient
from ...topic_state_registry import topic_state
from ...window_query import view_window
from ..messaging_pipeline.message_sender import (
    edit_with_fallback,
    rate_limit_send_message,
)
from ..provider_display import strip_provider_prefix
from ..status.topic_emoji import topic_state_badge
from . import desk_brief

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
# Desk-brief mode: user_id -> when the current brief period's first news came.
_brief_since: dict[int, float] = {}
_lock = asyncio.Lock()


def _clock() -> float:
    return time.monotonic()


def _wall_time() -> str:
    return datetime.now().strftime("%H:%M")


def _brief_enabled() -> bool:
    return config.desk_brief == "dm"


def _mark_news(user_id: int, entry: _Digest) -> None:
    entry.dirty = True
    if _brief_enabled():
        _brief_since.setdefault(user_id, _clock())


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
    _mark_news(user_id, entry)


def record_error(
    user_id: int, chat_id: int, thread_id: int, window_id: str, text: str
) -> None:
    """Keep an error line (e.g. an API error) for the next digest."""
    entry = _digest(user_id, chat_id, thread_id, window_id)
    entry.errors = [*entry.errors, text][-_MAX_ERROR_LINES:]
    _mark_news(user_id, entry)


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


def _has_news(key: _Key, entry: _Digest) -> bool:
    if entry.dirty:
        return True
    return (
        entry.last_state is not None
        and _state_of(key[1], key[2])[0] != entry.last_state
    )


def _topic_title(window_id: str) -> str:
    view = view_window(window_id)
    name = view.window_name if view is not None and view.window_name else ""
    if not name:
        return window_id
    return strip_provider_prefix(name, legacy=True) or name


def _brief_section(key: _Key, entry: _Digest) -> desk_brief.BriefSection:
    _user_id, chat_id, thread_id = key
    state, emoji = _state_of(chat_id, thread_id)
    return desk_brief.BriefSection(
        title=_topic_title(entry.window_id),
        state=state,
        emoji=emoji,
        label=_STATE_LABELS.get(state, state),
        messages=entry.messages,
        tool_calls=entry.tool_calls,
        latest_reply=entry.latest_reply if entry.dirty else "",
        errors=list(entry.errors),
        link=desk_brief.topic_link(chat_id, thread_id),
    )


def _quiet_count(user_id: int, listed: list[_Key]) -> int:
    """Bound topics of the user that the brief does not list."""
    bound = {
        (chat_id if chat_id is not None else resolve_chat_id(uid, thread_id), thread_id)
        for uid, chat_id, thread_id, _window_id in iter_bound_topics()
        if uid == user_id
    }
    return len(bound - {(key[1], key[2]) for key in listed})


def _render_brief(
    user_id: int, entries: list[tuple[_Key, _Digest]], *, away: bool
) -> str:
    return desk_brief.render_brief(
        [_brief_section(key, entry) for key, entry in entries],
        quiet_count=_quiet_count(user_id, [key for key, _entry in entries]),
        wall_time=_wall_time(),
        away=away,
    )


async def _flush_brief(client: TelegramClient, user_id: int) -> None:
    """Send one user's periodic brief, then refresh their topics silently."""
    news = [
        (key, entry)
        for key, entry in _digests.items()
        if key[0] == user_id and _has_news(key, entry)
    ]
    if not news:
        _brief_since.pop(user_id, None)
        return
    text = _render_brief(user_id, news, away=False)
    if await desk_brief.send_brief(client, user_id, text) == "retry":
        return
    _brief_since.pop(user_id, None)
    for key, entry in news:
        await _publish(client, key, entry)


async def _flush_briefs_due(client: TelegramClient) -> None:
    interval = config.desk_digest_minutes * 60
    now = _clock()
    async with _lock:
        for user_id, since in list(_brief_since.items()):
            if now - since >= interval:
                await _flush_brief(client, user_id)


async def flush_due(client: TelegramClient) -> None:
    """Send or refresh every digest (or desk brief) whose interval has elapsed."""
    if _brief_enabled():
        await _flush_briefs_due(client)
        return
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
    be silent. In desk-brief mode one notifying brief replaces those messages;
    they come back only if the brief cannot be sent.
    """
    async with _lock:
        entries = [
            (key, _digests.pop(key)) for key in list(_digests) if key[0] == user_id
        ]
        brief_sent = False
        if _brief_enabled():
            _brief_since.pop(user_id, None)
            if reason == "idle" and entries:
                text = _render_brief(user_id, entries, away=True)
                outcome = await desk_brief.send_brief(client, user_id, text)
                brief_sent = outcome == "sent"
        for key, entry in entries:
            has_news = entry.dirty or entry.message_id is not None
            if not has_news:
                continue
            state = _state_of(key[1], key[2])[0]
            if reason == "idle" and state in _SETTLED_STATES and not brief_sent:
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
    _brief_since.clear()
