"""Desk brief: one notifying direct message summarizing every session.

With ``CCGRAM_DESK_BRIEF=dm`` the per-topic digests stay silent and this brief
is the only desk-mode message that rings. ``digest`` decides when to send it
and feeds it one ``BriefSection`` per topic with news:

    🗂 Desk brief · 2 sessions with news
    🔴 **Sessions ▸ chore** · waiting for your choice
    2 messages, 5 tool calls
    💬 start of the latest reply, about 150 characters…
    [Open topic](https://t.me/c/4477516122/123)
    …
    3 other sessions quiet
    updated 14:32 · desk mode

This module renders and sends; it keeps no state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import structlog
from telegram.error import RetryAfter

from ...telegram_client import TelegramClient
from ..messaging_pipeline.message_sender import rate_limit_send_message

logger = structlog.get_logger()

EXCERPT_CHARS = 150

# Lower sorts first: what needs the user comes before what is still running.
_URGENCY = {"waiting": 0, "done": 2, "idle": 2, "dead": 3, "active": 4}
_ERROR_URGENCY = 1
_MARKDOWN_SPECIAL = re.compile(r"([\\*_`\[\]])")
_SUPERGROUP_PREFIX = "-100"
# Markdown that only decorates: emphasis/code markers and heading hashes.
_DECORATION = re.compile(r"\*\*|__|`+|(?:^|(?<=\s))#{1,6}\s")

BriefOutcome = Literal["sent", "retry", "failed"]


@dataclass(slots=True)
class BriefSection:
    """One topic's news since the last brief."""

    title: str
    state: str
    emoji: str
    label: str
    messages: int = 0
    tool_calls: int = 0
    latest_reply: str = ""
    errors: list[str] = field(default_factory=list)
    link: str | None = None


@dataclass(slots=True)
class GuestLine:
    """Traffic in one guest chat topic since the last brief (never held back)."""

    label: str
    messages: int
    link: str | None = None


def topic_link(chat_id: int, thread_id: int) -> str | None:
    """A t.me link into a supergroup topic; None for private-chat topics."""
    text = str(chat_id)
    if not text.startswith(_SUPERGROUP_PREFIX) or len(text) <= len(_SUPERGROUP_PREFIX):
        return None
    return f"https://t.me/c/{text[len(_SUPERGROUP_PREFIX) :]}/{thread_id}"


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """The start of a reply on one line, cut at ``limit`` characters.

    Markdown decoration is dropped, since the line is shown as plain text.
    """
    flat = " ".join(_DECORATION.sub("", text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rstrip() + "…"


def _escape(text: str) -> str:
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def _urgency(section: BriefSection) -> int:
    base = _URGENCY.get(section.state, len(_URGENCY))
    if section.errors and section.state != "waiting":
        return _ERROR_URGENCY
    return base


def sort_sections(sections: list[BriefSection]) -> list[BriefSection]:
    """Most urgent first; ties keep their order."""
    return sorted(sections, key=_urgency)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _render_section(section: BriefSection) -> list[str]:
    title = f"**{_escape(section.title)}** · {section.label}"
    lines = [f"{section.emoji} {title}" if section.emoji else title]
    if section.messages or section.tool_calls:
        lines.append(
            f"{_plural(section.messages, 'message')}, "
            f"{_plural(section.tool_calls, 'tool call')}"
        )
    else:
        lines.append("no new messages since the last brief")
    lines.extend(f"⚠ {_escape(line)}" for line in section.errors)
    if section.latest_reply:
        # A plain line: an expandable quote would eat the blank line that
        # separates sections, and the 💬 keeps list/heading syntax inert.
        lines.append(f"💬 {_escape(excerpt(section.latest_reply))}")
    if section.link:
        lines.append(f"[Open topic]({section.link})")
    return lines


def _render_guest_line(line: GuestLine) -> str:
    text = f"👥 **{_escape(line.label)}** · {_plural(line.messages, 'message')} (guest chat)"
    if line.link:
        text += f" · [Open]({line.link})"
    return text


def render_brief(
    sections: list[BriefSection],
    *,
    quiet_count: int,
    wall_time: str,
    away: bool = False,
    guests: list[GuestLine] | None = None,
) -> str:
    """The brief text (markdown; converted to entities at send time)."""
    guests = guests or []
    count = _plural(len(sections) + len(guests), "session")
    if away:
        lines = [f"🚶 You stepped away · {count} active while you were here"]
    else:
        lines = [f"🗂 Desk brief · {count} with news"]
    for section in sort_sections(sections):
        lines.append("")
        lines.extend(_render_section(section))
    if guests:
        lines.append("")
        lines.extend(_render_guest_line(line) for line in guests)
    lines.append("")
    if quiet_count > 0:
        noun = "session" if quiet_count == 1 else "sessions"
        lines.append(f"{quiet_count} other {noun} quiet")
    lines.append(f"updated {wall_time} · desk mode")
    return "\n".join(lines)


async def send_brief(client: TelegramClient, user_id: int, text: str) -> BriefOutcome:
    """Send the brief to the user's private chat with the bot, notifying."""
    try:
        message = await rate_limit_send_message(client, user_id, text)
    except RetryAfter:
        logger.info("Desk brief rate-limited; retrying later", user_id=user_id)
        return "retry"
    if message is None:
        logger.warning("Desk brief could not be sent", user_id=user_id)
        return "failed"
    logger.info("Desk brief sent", user_id=user_id, chars=len(text))
    return "sent"
