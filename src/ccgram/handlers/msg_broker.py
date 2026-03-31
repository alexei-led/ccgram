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

from .msg_delivery import delivery_strategy

if TYPE_CHECKING:
    from telegram import Bot

    from ..mailbox import Mailbox, Message
    from ..tmux_manager import TmuxManager

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
    mailbox_dir: Path, window_id: str, msg_id: str, body: str
) -> Path:
    """Write full message body to a delivery file for long messages."""
    from ..mailbox import sanitize_dir_name, validate_no_traversal

    validate_no_traversal(msg_id, "message ID")
    inbox_dir = mailbox_dir / sanitize_dir_name(window_id)
    tmp_dir = inbox_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    delivery_path = tmp_dir / f"deliver-{msg_id}.txt"
    delivery_path.write_text(body, encoding="utf-8")
    return delivery_path


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


def _format_for_delivery(msg: "Message", mailbox_dir: Path, qualified_id: str) -> str:
    """Format a single message for send_keys injection."""
    body = msg.body
    if len(body) > _INJECTION_CHAR_LIMIT:
        delivery_path = write_delivery_file(mailbox_dir, qualified_id, msg.id, body)
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
    """Mark stale pending messages as delivered on first broker cycle.

    Handles crash recovery: if the bot crashed after send_keys but before
    mark_delivered, these messages would otherwise be injected again.
    """
    if delivery_strategy.is_crash_recovery_done():
        return
    delivery_strategy.mark_crash_recovery_done()
    stale = mailbox.pending_undelivered(min_age_seconds=5.0)
    for msg in stale:
        mailbox.mark_delivered(msg.id, msg.to_id)
        logger.info(
            "Crash recovery: marked stale pending message as delivered",
            msg_id=msg.id,
            to_id=msg.to_id,
        )


async def broker_delivery_cycle(
    mailbox: "Mailbox",
    tmux_mgr: "TmuxManager",
    window_states: dict,
    tmux_session: str,
    msg_rate_limit: int,
    mailbox_dir: Path,
    bot: "Bot | None" = None,
    idle_windows: frozenset[str] = frozenset(),
) -> int:
    """Run one broker delivery cycle.

    Scans all inboxes for pending messages, checks idle windows,
    and injects via send_keys. Returns the number of messages delivered.

    When *bot* is provided, Telegram notifications are sent for
    delivered messages, shell-pending messages, and loop detection.
    """
    from ..providers import get_provider_for_window
    from ..window_resolver import is_foreign_window

    _recover_stale_pending(mailbox)

    delivered_count = 0

    for window_id in list(window_states):
        # Foreign windows (emdash) are already fully qualified
        if is_foreign_window(window_id):
            qualified_id = window_id
        else:
            qualified_id = f"{tmux_session}:{window_id}"

        provider = get_provider_for_window(window_id)
        if provider.capabilities.name == "shell":
            await _notify_shell_pending(bot, mailbox, qualified_id)
            continue

        # Hook-enabled providers get delivery via Stop event (hook_events.py).
        # Only deliver when explicitly marked idle; skip in periodic poll.
        if provider.capabilities.supports_hook and qualified_id not in idle_windows:
            continue

        to_deliver, loops = _collect_eligible(mailbox, qualified_id, msg_rate_limit)

        # Notify Telegram about detected loops
        for window_a, window_b in loops:
            await _notify_loop(bot, window_a, window_b)

        if not to_deliver:
            continue

        texts = [_format_for_delivery(m, mailbox_dir, qualified_id) for m in to_deliver]
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
            await _notify_delivered(bot, qualified_id, to_deliver, mailbox)
            await _notify_senders(bot, qualified_id, to_deliver)

    # Process pending spawn requests
    if bot is not None:
        await _process_spawn_requests(bot)

    return delivered_count


async def _notify_delivered(
    bot: "Bot | None",
    to_window: str,
    messages: list["Message"],
    mailbox: "Mailbox | None" = None,
) -> None:
    """Send Telegram notification for delivered messages (if bot available)."""
    if bot is None:
        return
    from .msg_telegram import notify_messages_delivered, notify_reply_received

    try:
        await notify_messages_delivered(bot, to_window, messages)
    except OSError, TelegramError:
        logger.debug("Failed to send delivery notification", window=to_window)

    if mailbox is not None:
        for msg in messages:
            if msg.type == "reply" and msg.reply_to:
                try:
                    original = mailbox.get(msg.reply_to, msg.from_id)
                    if original is not None:
                        await notify_reply_received(bot, original, msg)
                except OSError, TelegramError:
                    logger.debug("Failed to send reply notification", msg_id=msg.id)


async def _notify_senders(
    bot: "Bot | None",
    to_window: str,
    messages: list["Message"],
) -> None:
    """Notify each sender's Telegram topic that their message was delivered."""
    if bot is None:
        return
    from .msg_telegram import notify_message_sent

    for msg in messages:
        try:
            await notify_message_sent(bot, msg.from_id, to_window, msg)
        except OSError, TelegramError:
            logger.debug("Failed to send sender notification", from_id=msg.from_id)


async def _notify_loop(bot: "Bot | None", window_a: str, window_b: str) -> None:
    """Send Telegram loop detection alert (if bot available)."""
    if bot is None:
        return
    from .msg_telegram import notify_loop_detected

    try:
        await notify_loop_detected(bot, window_a, window_b)
    except OSError, TelegramError:
        logger.debug("Failed to send loop alert", window_a=window_a, window_b=window_b)


async def _notify_shell_pending(
    bot: "Bot | None", mailbox: "Mailbox", qualified_id: str
) -> None:
    """Notify shell topics about pending messages (if bot available).

    Marks messages as delivered after first notification to prevent
    repeated notifications every broker cycle.
    """
    if bot is None:
        return
    from .msg_telegram import notify_pending_shell

    state = delivery_strategy.get_state(qualified_id)
    pending = mailbox.inbox(qualified_id)
    for msg in pending:
        if msg.status == "pending" and msg.id not in state.notified_shell_ids:
            try:
                await notify_pending_shell(bot, qualified_id, msg)
                state.notified_shell_ids.add(msg.id)
                mailbox.mark_delivered(msg.id, qualified_id)
            except OSError, TelegramError:
                logger.debug(
                    "Failed to send shell pending notification", window=qualified_id
                )


async def _process_spawn_requests(bot: "Bot") -> None:
    """Scan for file-based spawn requests and post approval keyboards or auto-approve."""
    from ..spawn_request import pop_pending, scan_spawn_requests
    from .msg_spawn import (
        handle_spawn_approval,
        post_spawn_approval_keyboard,
    )

    from ..config import config

    new_requests = scan_spawn_requests(spawn_timeout=config.msg_spawn_timeout)
    for req in new_requests:
        try:
            if req.auto or config.msg_auto_spawn:
                await handle_spawn_approval(
                    req.id, bot, spawn_timeout=config.msg_spawn_timeout
                )
            else:
                posted = await post_spawn_approval_keyboard(
                    bot, req.requester_window, req
                )
                if not posted:
                    pop_pending(req.id)
        except OSError, TelegramError:
            pop_pending(req.id)
            logger.debug("Failed to process spawn request", request_id=req.id)
