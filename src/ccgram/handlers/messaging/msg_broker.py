"""Broker delivery cycle for inter-agent messaging.

Detects idle agent windows, injects pending messages via send_keys,
handles crash recovery, and delegates to msg_delivery for state tracking.

Key components:
  - broker_delivery_cycle: async delivery cycle called from poll loop
  - format_injection_text: message formatting for send_keys
"""

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram.error import TelegramError

from ...telegram_client import TelegramClient
from .msg_delivery import delivery_strategy
from .msg_telegram import (
    notify_loop_detected,
    notify_message_sent,
    notify_messages_delivered,
    notify_pending_shell,
    notify_reply_received,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...mailbox import Mailbox, Message
    from ...tmux_manager import TmuxManager

logger = structlog.get_logger()

# ── Constants ──────────────────────────────────────────────────────────

# Injection text hard cap (chars).
_INJECTION_CHAR_LIMIT = 500

# Broker delivery cycle interval (seconds).
BROKER_CYCLE_INTERVAL = 2.0

# Mailbox sweep interval (seconds) — runs inside poll loop.
SWEEP_INTERVAL = 300.0


def format_injection_text(
    msg_id: str,
    from_id: str,
    from_name: str,
    branch: str,
    subject: str,
    body: str,
    msg_type: str,
) -> str:
    """Format a message for send_keys injection.

    Returns a single-line string capped at _INJECTION_CHAR_LIMIT chars.
    Newlines are replaced with spaces, paragraphs with |.
    """
    context_parts = [from_name]
    if branch:
        context_parts.append(branch)
    context_str = ", ".join(context_parts)

    header = f"[MSG {msg_id} from {from_id} ({context_str})]"
    subj = f" {subject}:" if subject else ""

    cleaned_body = body.replace("\n\n", " | ").replace("\n", " ")

    if msg_type == "request":
        reply_hint = f' REPLY WITH: ccgram msg reply {msg_id} "your answer"'
    else:
        reply_hint = ""

    text = f"{header}{subj} {cleaned_body}{reply_hint}"

    if len(text) > _INJECTION_CHAR_LIMIT:
        text = text[: _INJECTION_CHAR_LIMIT - 3] + "..."

    return text


def format_file_reference(msg_id: str, file_path: str) -> str:
    """Format a file reference for long messages."""
    return f"[MSG {msg_id}] See: {file_path}"


_MERGED_CHAR_LIMIT = 1500


def merge_injection_texts(texts: list[str]) -> str:
    """Merge multiple injection texts into a single block."""
    merged = " --- ".join(texts)
    if len(merged) > _MERGED_CHAR_LIMIT:
        merged = merged[: _MERGED_CHAR_LIMIT - 3] + "..."
    return merged


def write_delivery_file(
    mailbox: "Mailbox", window_id: str, msg_id: str, body: str
) -> Path:
    """Write full message body to a delivery file for long messages."""
    path = mailbox.delivery_path(window_id, msg_id)
    path.write_text(body, encoding="utf-8")
    return path


def _collect_eligible(
    mailbox: "Mailbox", qualified_id: str, msg_rate_limit: int
) -> tuple[list["Message"], list[tuple[str, str]]]:
    """Collect eligible pending messages for a window.

    Filters out broadcasts, paused peers, and applies rate limiting
    and loop detection.

    Returns (eligible_messages, detected_loop_pairs).
    """
    pending = mailbox.inbox(qualified_id)
    if not pending:
        return [], []

    eligible = [
        m
        for m in pending
        if m.type != "broadcast"
        and m.status == "pending"
        and not delivery_strategy.is_paused(qualified_id, m.from_id)
    ]
    if not eligible:
        return [], []

    if not delivery_strategy.check_rate_limit(qualified_id, msg_rate_limit):
        logger.debug("Rate limit reached for window", window_id=qualified_id)
        return [], []

    loops: list[tuple[str, str]] = []
    seen_loops: set[tuple[str, str]] = set()
    for msg in eligible:
        if delivery_strategy.check_loop(qualified_id, msg.from_id):
            delivery_strategy.pause_peer(qualified_id, msg.from_id)
            delivery_strategy.pause_peer(msg.from_id, qualified_id)
            pair = (qualified_id, msg.from_id)
            if pair not in seen_loops:
                seen_loops.add(pair)
                loops.append(pair)
                logger.warning(
                    "Loop detected, pausing delivery",
                    window_a=qualified_id,
                    window_b=msg.from_id,
                )

    filtered = [
        m for m in eligible if not delivery_strategy.is_paused(qualified_id, m.from_id)
    ]
    return filtered, loops


def _format_for_delivery(msg: "Message", mailbox: "Mailbox", qualified_id: str) -> str:
    """Format a single message for send_keys injection."""
    body = msg.body
    if len(body) > _INJECTION_CHAR_LIMIT:
        delivery_path = write_delivery_file(mailbox, qualified_id, msg.id, body)
        return format_file_reference(msg.id, str(delivery_path))
    return format_injection_text(
        msg_id=msg.id,
        from_id=msg.from_id,
        from_name=msg.context.get("window_name", ""),
        branch=msg.context.get("branch", ""),
        subject=msg.subject,
        body=body,
        msg_type=msg.type,
    )


def _recover_stale_pending(mailbox: "Mailbox") -> None:
    """Re-inject stale pending messages on the first broker cycle after restart.

    Handles crash recovery: messages that were pending before the last shutdown
    are logged and allowed to flow through the normal delivery cycle. Re-injection
    may cause duplicate delivery in the rare case where the bot crashed after
    send_keys but before mark_delivered; that is preferable to silent message loss.
    """
    if delivery_strategy.is_crash_recovery_done():
        return
    delivery_strategy.mark_crash_recovery_done()
    stale = mailbox.pending_undelivered(min_age_seconds=5.0)
    for msg in stale:
        logger.info(
            "Crash recovery: re-injecting stale pending message",
            msg_id=msg.id,
            to_id=msg.to_id,
        )


async def broker_delivery_cycle(
    mailbox: "Mailbox",
    tmux_mgr: "TmuxManager",
    window_ids: "Iterable[str]",
    tmux_session: str,
    msg_rate_limit: int,
    client: "TelegramClient | None" = None,
    idle_windows: frozenset[str] = frozenset(),
) -> int:
    """Run one broker delivery cycle.

    Scans all inboxes for pending messages, checks idle windows,
    and injects via send_keys. Returns the number of messages delivered.

    When *client* is provided, Telegram notifications are sent for
    delivered messages, shell-pending messages, and loop detection.
    """
    # Lazy: msg_broker is registered as a periodic-task coroutine; these
    # cross-package imports are kept at call site so the registry import
    # path stays free of provider / window_resolver weight.
    # Lazy: providers / window_query proxies wired by SessionManager constructor
    from ...providers import get_provider_for_window

    # Lazy: providers / window_query proxies wired by SessionManager constructor
    from ...window_query import get_window_provider

    _recover_stale_pending(mailbox)

    delivered_count = 0

    for window_id in list(window_ids):
        qualified_id = f"{tmux_session}:{window_id}"

        provider = get_provider_for_window(
            window_id, provider_name=get_window_provider(window_id)
        )
        if not provider.capabilities.supports_mailbox_delivery:
            await _deliver_to_shell_topic(client, mailbox, qualified_id)
            continue

        # Hook-enabled providers get delivery via Stop event (hook_events.py).
        # Only deliver when explicitly marked idle; skip in periodic poll.
        if provider.capabilities.supports_hook and qualified_id not in idle_windows:
            continue

        to_deliver, loops = _collect_eligible(mailbox, qualified_id, msg_rate_limit)

        # Notify Telegram about detected loops
        for window_a, window_b in loops:
            await _notify_loop(client, window_a, window_b)

        if not to_deliver:
            continue

        texts = [_format_for_delivery(m, mailbox, qualified_id) for m in to_deliver]
        merged = merge_injection_texts(texts)
        success = await tmux_mgr.send_keys(window_id, merged, literal=True)

        if success:
            for msg in to_deliver:
                mailbox.mark_delivered(msg.id, qualified_id)
                delivery_strategy.record_exchange(qualified_id, msg.from_id)
            delivery_strategy.record_delivery(qualified_id)
            delivered_count += len(to_deliver)
            logger.info(
                "Broker delivered messages",
                window_id=qualified_id,
                count=len(to_deliver),
            )
            await _notify_delivered(client, qualified_id, to_deliver, mailbox)
            await _notify_senders(client, qualified_id, to_deliver)

    return delivered_count


async def _notify_delivered(
    client: "TelegramClient | None",
    to_window: str,
    messages: list["Message"],
    mailbox: "Mailbox | None" = None,
) -> None:
    """Send Telegram notification for delivered messages (if client available)."""
    if client is None:
        return

    try:
        await notify_messages_delivered(client, to_window, messages)
    except OSError, TelegramError:
        logger.debug("Failed to send delivery notification", window=to_window)

    if mailbox is not None:
        for msg in messages:
            if msg.type == "reply" and msg.reply_to:
                try:
                    original = mailbox.get(msg.reply_to, msg.from_id)
                    if original is not None:
                        await notify_reply_received(client, original, msg)
                except OSError, TelegramError:
                    logger.debug("Failed to send reply notification", msg_id=msg.id)


async def _notify_senders(
    client: "TelegramClient | None",
    to_window: str,
    messages: list["Message"],
) -> None:
    """Notify each sender's Telegram topic that their message was delivered."""
    if client is None:
        return

    for msg in messages:
        try:
            await notify_message_sent(client, msg.from_id, to_window, msg)
        except OSError, TelegramError:
            logger.debug("Failed to send sender notification", from_id=msg.from_id)


async def _notify_loop(
    client: "TelegramClient | None", window_a: str, window_b: str
) -> None:
    """Send Telegram loop detection alert (if client available)."""
    if client is None:
        return

    try:
        await notify_loop_detected(client, window_a, window_b)
    except OSError, TelegramError:
        logger.debug("Failed to send loop alert", window_a=window_a, window_b=window_b)


async def _deliver_to_shell_topic(
    client: "TelegramClient | None", mailbox: "Mailbox", qualified_id: str
) -> None:
    """Deliver messages to shell topics via Telegram notification.

    For shell windows there is no agent to receive send_keys, so the
    Telegram notification IS the delivery. Marks messages as delivered
    after notification to prevent repeated deliveries every broker cycle.
    """
    if client is None:
        return

    state = delivery_strategy.get_state(qualified_id)
    pending = mailbox.inbox(qualified_id)
    for msg in pending:
        if msg.status == "pending" and msg.id not in state.notified_shell_ids:
            try:
                await notify_pending_shell(client, qualified_id, msg)
                state.notified_shell_ids.add(msg.id)
                mailbox.mark_delivered(msg.id, qualified_id)
            except OSError, TelegramError:
                logger.debug(
                    "Failed to send shell pending notification", window=qualified_id
                )
