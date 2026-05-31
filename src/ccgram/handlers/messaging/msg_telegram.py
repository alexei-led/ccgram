"""Telegram notifications for inter-agent messaging.

Shows silent notifications in Telegram topics when agents send messages
to each other. Handles sent/delivered/reply/shell-pending notifications
and loop detection alerts with inline keyboard controls.

Key functions:
  - notify_message_sent: compact line in sender's topic
  - notify_messages_delivered: grouped notification for multiple messages
  - notify_reply_received: reply notification in original sender's topic
  - notify_pending_shell: pending message display in shell topic
  - notify_loop_detected: alert with [Pause] [Allow] keyboard
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)

from ...telegram_client import TelegramClient
from ...thread_router import thread_router
from ...topic_state_registry import topic_state
from ...utils import tmux_session_name
from ..callback_registry import register
from .msg_delivery import delivery_strategy
from ..messaging_pipeline.message_sender import (
    REACT_INBOX,
    rate_limit_send_message,
    react,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

    from ...mailbox import Message

logger = structlog.get_logger()

# Callback data prefixes for loop alert buttons
CB_MSG_LOOP_PAUSE = "ml:p:"
CB_MSG_LOOP_ALLOW = "ml:a:"

_WINDOW_KEY_PARTS = 2

_SUBJECT_MAX_LEN = 40
_BODY_PREVIEW_LEN = 100
_MAX_LOOP_ALERT_PAIRS = 100

# Map short hash → (window_a, window_b) for loop alert callback data.
# Telegram limits callback_data to 64 bytes; qualified IDs can be long.
_loop_alert_pairs: dict[str, tuple[str, str]] = {}


@topic_state.register("qualified")
def clear_loop_alerts(qualified_id: str) -> None:
    """Remove loop alert pairs involving this window."""
    stale = [
        k
        for k, (a, b) in _loop_alert_pairs.items()
        if a == qualified_id or b == qualified_id
    ]
    for k in stale:
        del _loop_alert_pairs[k]


def _extract_window_id(qualified_id: str) -> str:
    """Extract bare window ID from qualified ID (e.g. 'ccgram:@0' -> '@0')."""
    parts = qualified_id.rsplit(":", 1)
    if len(parts) < _WINDOW_KEY_PARTS:
        return qualified_id
    return parts[1]


def _is_local_qualified(qualified_id: str) -> bool:
    """Check if a qualified ID belongs to the local tmux session.

    Bare IDs (no colon) are considered local.  Qualified IDs are local
    only when their session prefix matches ``tmux_session_name()``.
    """
    if ":" not in qualified_id:
        return True
    session_prefix = qualified_id.rsplit(":", 1)[0]
    return session_prefix == tmux_session_name()


def resolve_topic(
    qualified_id: str,
) -> tuple[int, int, int, str] | None:
    """Find the Telegram topic for a qualified window ID.

    Tries the full qualified ID first, then falls back to the bare window
    ID (for local windows stored as ``@N``).  The bare-ID fallback is
    restricted to the local tmux session.

    Returns (user_id, thread_id, chat_id, window_id) or None.
    """
    bare_id = _extract_window_id(qualified_id)
    # First pass: exact qualified ID match
    for user_id, thread_id, bound_wid in thread_router.iter_thread_bindings():
        if bound_wid == qualified_id:
            chat_id = thread_router.resolve_chat_id(user_id, thread_id)
            return user_id, thread_id, chat_id, bound_wid
    # Second pass: bare window ID fallback (only for local session windows)
    if bare_id != qualified_id and _is_local_qualified(qualified_id):
        for user_id, thread_id, bound_wid in thread_router.iter_thread_bindings():
            if bound_wid == bare_id:
                chat_id = thread_router.resolve_chat_id(user_id, thread_id)
                return user_id, thread_id, chat_id, bound_wid
    return None


def _display_name(qualified_id: str) -> str:
    """Get display name for a qualified window ID."""
    # Try full qualified ID first
    name = thread_router.get_display_name(qualified_id)
    if name != qualified_id:
        return name
    bare_id = _extract_window_id(qualified_id)
    if bare_id == qualified_id:
        return qualified_id
    if not _is_local_qualified(qualified_id):
        return qualified_id
    return thread_router.get_display_name(bare_id)


def _format_subject(subject: str) -> str:
    """Truncate subject for inline display."""
    if not subject:
        return ""
    if len(subject) > _SUBJECT_MAX_LEN:
        return subject[: _SUBJECT_MAX_LEN - 3] + "..."
    return subject


async def notify_message_sent(
    client: TelegramClient,
    from_window: str,
    to_window: str,
    message: Message,
) -> None:
    """Send a compact notification in the sender's Telegram topic.

    Format: -> @5 (api-gateway) [request] API contract query
    """
    topic = resolve_topic(from_window)
    if topic is None:
        return

    _, thread_id, chat_id, _ = topic
    to_name = _display_name(to_window)
    to_wid = _extract_window_id(to_window)
    subj = _format_subject(message.subject)
    subj_part = f" {subj}" if subj else ""

    text = f"→ {to_wid} ({to_name}) [{message.type}]{subj_part}"

    await rate_limit_send_message(
        client,
        chat_id,
        text,
        message_thread_id=thread_id,
        disable_notification=True,
    )


async def notify_messages_delivered(
    client: TelegramClient,
    to_window: str,
    messages: list[Message],
) -> None:
    """Send a grouped notification for multiple delivered messages.

    Merges multiple messages into a single Telegram notification.
    """
    if not messages:
        return

    topic = resolve_topic(to_window)
    if topic is None:
        return

    _, thread_id, chat_id, _ = topic

    if len(messages) == 1:
        msg = messages[0]
        from_name = _display_name(msg.from_id)
        from_wid = _extract_window_id(msg.from_id)
        subj = _format_subject(msg.subject)
        subj_part = f" {subj}" if subj else ""
        text = f"← {from_wid} ({from_name}) [{msg.type}]{subj_part}"
    else:
        lines = [f"← {len(messages)} messages delivered:"]
        for msg in messages:
            from_name = _display_name(msg.from_id)
            from_wid = _extract_window_id(msg.from_id)
            subj = _format_subject(msg.subject)
            subj_part = f" {subj}" if subj else ""
            lines.append(f"  {from_wid} ({from_name}) [{msg.type}]{subj_part}")
        text = "\n".join(lines)

    sent = await rate_limit_send_message(
        client,
        chat_id,
        text,
        message_thread_id=thread_id,
        disable_notification=True,
    )
    # 📬 ack so the inter-agent delivery is visible in the topic without
    # competing with the agent's own status updates.
    if sent is not None:
        await react(client, chat_id, sent.message_id, REACT_INBOX)


async def notify_reply_received(
    client: TelegramClient,
    original_msg: Message,
    reply_msg: Message,
) -> None:
    """Notify the original sender's topic that a reply was received.

    Format: Reply received from @5 (api-gateway) for: API contract query
    """
    topic = resolve_topic(original_msg.from_id)
    if topic is None:
        return

    _, thread_id, chat_id, _ = topic
    from_name = _display_name(reply_msg.from_id)
    from_wid = _extract_window_id(reply_msg.from_id)
    subj = _format_subject(original_msg.subject)
    subj_part = f" for: {subj}" if subj else ""

    text = f"✓ Reply received from {from_wid} ({from_name}){subj_part}"

    await rate_limit_send_message(
        client,
        chat_id,
        text,
        message_thread_id=thread_id,
        disable_notification=True,
    )


async def notify_pending_shell(
    client: TelegramClient,
    window_id: str,
    message: Message,
) -> None:
    """Show a pending message in a shell topic (send_keys is skipped).

    Format: Pending message from @0 (payment-svc) [request]: ...body preview
    """
    topic = resolve_topic(window_id)
    if topic is None:
        return

    _, thread_id, chat_id, _ = topic
    from_name = _display_name(message.from_id)
    from_wid = _extract_window_id(message.from_id)
    subj = _format_subject(message.subject)
    subj_part = f" {subj}:" if subj else ":"

    body_preview = message.body[:_BODY_PREVIEW_LEN]
    if len(message.body) > _BODY_PREVIEW_LEN:
        body_preview += "..."

    text = (
        f"✉ Pending from {from_wid} ({from_name})"
        f" [{message.type}]{subj_part} {body_preview}"
    )

    await rate_limit_send_message(
        client,
        chat_id,
        text,
        message_thread_id=thread_id,
        disable_notification=True,
    )


async def notify_loop_detected(
    client: TelegramClient,
    window_a: str,
    window_b: str,
) -> None:
    """Alert that a messaging loop was detected, with control buttons.

    Posts in the topic of window_a with [Pause Messaging] [Allow 5 more].
    Uses a short hash in callback_data to stay within Telegram's 64-byte limit.
    """
    # Lazy: only used by the message-id helper
    import hashlib

    topic = resolve_topic(window_a)
    if topic is None:
        return

    _, thread_id, chat_id, _ = topic
    name_a = _display_name(window_a)
    name_b = _display_name(window_b)
    wid_a = _extract_window_id(window_a)
    wid_b = _extract_window_id(window_b)

    # Hash the pair to fit within 64-byte callback_data limit. Non-cryptographic
    # use (dedup key for an in-memory dict), so usedforsecurity=False is the
    # right signal — sha256 also avoids the bandit/MD5 noise.
    pair_full = f"{window_a}|{window_b}"
    pair_hash = hashlib.sha256(pair_full.encode(), usedforsecurity=False).hexdigest()[
        :16
    ]
    if len(_loop_alert_pairs) >= _MAX_LOOP_ALERT_PAIRS:
        oldest_key = next(iter(_loop_alert_pairs))
        del _loop_alert_pairs[oldest_key]
    _loop_alert_pairs[pair_hash] = (window_a, window_b)

    text = (
        f"⚠ Messaging loop detected between {wid_a} ({name_a}) and {wid_b} ({name_b})"
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Pause Messaging",
                    callback_data=f"{CB_MSG_LOOP_PAUSE}{pair_hash}",
                ),
                InlineKeyboardButton(
                    "Allow 5 more",
                    callback_data=f"{CB_MSG_LOOP_ALLOW}{pair_hash}",
                ),
            ]
        ]
    )

    await rate_limit_send_message(
        client,
        chat_id,
        text,
        message_thread_id=thread_id,
        disable_notification=True,
        reply_markup=keyboard,
    )


# ── Callback handlers for loop alert buttons ─────────────────────────


@register(CB_MSG_LOOP_PAUSE, CB_MSG_LOOP_ALLOW)
async def _handle_loop_alert(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle [Pause Messaging] / [Allow 5 more] button presses."""

    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()

    data = query.data
    if data.startswith(CB_MSG_LOOP_PAUSE):
        pair_hash = data[len(CB_MSG_LOOP_PAUSE) :]
        pair = _loop_alert_pairs.get(pair_hash)
        if pair:
            delivery_strategy.pause_peer(pair[0], pair[1])
            delivery_strategy.pause_peer(pair[1], pair[0])
        await _safe_edit_text(query, "⏸ Messaging paused between these windows")
    elif data.startswith(CB_MSG_LOOP_ALLOW):
        pair_hash = data[len(CB_MSG_LOOP_ALLOW) :]
        pair = _loop_alert_pairs.get(pair_hash)
        if pair:
            delivery_strategy.allow_more(pair[0], pair[1])
        await _safe_edit_text(query, "▶ Allowing 5 more exchanges")


async def _safe_edit_text(query: CallbackQuery, text: str) -> None:
    """Edit callback query message text, ignoring errors."""
    # Lazy: only used inside the suppression branch
    import contextlib

    # Lazy: only used inside the BadRequest handler
    from telegram.error import TelegramError

    with contextlib.suppress(TelegramError):
        await query.edit_message_text(text)
