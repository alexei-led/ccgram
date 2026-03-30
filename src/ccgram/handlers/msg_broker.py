"""Broker delivery strategy for inter-agent messaging.

Detects idle agent windows, injects pending messages via send_keys,
handles rate limiting, loop detection, and crash recovery. Follows
the TerminalStatusStrategy pattern from polling_strategies.py:
state-owning class with module-level singleton.

Key components:
  - MessageDeliveryStrategy: per-window delivery state
  - broker_delivery_cycle: async delivery cycle called from poll loop
  - format_injection_text: message formatting for send_keys
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram.error import TelegramError

if TYPE_CHECKING:
    from telegram import Bot

    from ..mailbox import Mailbox, Message
    from ..tmux_manager import TmuxManager

logger = structlog.get_logger()

# ── Constants ──────────────────────────────────────────────────────────

# Injection text hard cap (chars).
_INJECTION_CHAR_LIMIT = 500

# Rate limiting: max messages per window per window (5 min).
_RATE_WINDOW_SECONDS = 300.0

# Loop detection: max exchanges between a pair before pausing.
_LOOP_THRESHOLD = 5

# Loop detection window (seconds).
_LOOP_WINDOW_SECONDS = 600.0

# Broker delivery cycle interval (seconds).
BROKER_CYCLE_INTERVAL = 2.0

# Mailbox sweep interval (seconds) — runs inside poll loop.
SWEEP_INTERVAL = 300.0


# ── Per-window delivery state ──────────────────────────────────────────


@dataclass
class DeliveryState:
    """Per-window delivery tracking state."""

    delivery_timestamps: list[float] = field(default_factory=list)
    loop_counts: dict[str, list[float]] = field(default_factory=dict)
    paused_peers: set[str] = field(default_factory=set)
    notified_shell_ids: set[str] = field(default_factory=set)


class MessageDeliveryStrategy:
    """Owns per-window delivery state for broker message injection.

    Follows the TerminalStatusStrategy pattern: state dict keyed by
    qualified window ID, get_state/clear_state, module-level singleton.
    """

    def __init__(self) -> None:
        self._states: dict[str, DeliveryState] = {}

    def get_state(self, window_id: str) -> DeliveryState:
        return self._states.setdefault(window_id, DeliveryState())

    def clear_state(self, window_id: str) -> None:
        self._states.pop(window_id, None)

    def reset_all_state(self) -> None:
        self._states.clear()

    def check_rate_limit(self, window_id: str, max_rate: int) -> bool:
        """Return True if the window is within rate limits."""
        state = self.get_state(window_id)
        now = time.monotonic()
        cutoff = now - _RATE_WINDOW_SECONDS
        state.delivery_timestamps = [t for t in state.delivery_timestamps if t > cutoff]
        return len(state.delivery_timestamps) < max_rate

    def record_delivery(self, window_id: str) -> None:
        """Record a delivery timestamp for rate limiting."""
        state = self.get_state(window_id)
        state.delivery_timestamps.append(time.monotonic())

    def check_loop(self, window_a: str, window_b: str) -> bool:
        """Return True if a messaging loop is detected between two windows.

        A loop is detected when there are _LOOP_THRESHOLD or more exchanges
        between the same pair within _LOOP_WINDOW_SECONDS.
        """
        pair_key = _pair_key(window_a, window_b)
        state_a = self.get_state(window_a)
        now = time.monotonic()
        cutoff = now - _LOOP_WINDOW_SECONDS

        timestamps = state_a.loop_counts.get(pair_key, [])
        timestamps = [t for t in timestamps if t > cutoff]
        state_a.loop_counts[pair_key] = timestamps

        return len(timestamps) >= _LOOP_THRESHOLD

    def record_exchange(self, window_a: str, window_b: str) -> None:
        """Record a message exchange between two windows for loop detection.

        Records on both sides so check_loop works regardless of argument order.
        """
        pair_key = _pair_key(window_a, window_b)
        now = time.monotonic()
        for wid in (window_a, window_b):
            self.get_state(wid).loop_counts.setdefault(pair_key, []).append(now)

    def is_paused(self, window_id: str, peer_id: str) -> bool:
        """Check if delivery from peer_id to window_id is paused."""
        return peer_id in self.get_state(window_id).paused_peers

    def pause_peer(self, window_id: str, peer_id: str) -> None:
        """Pause delivery from peer_id to window_id."""
        self.get_state(window_id).paused_peers.add(peer_id)

    def unpause_peer(self, window_id: str, peer_id: str) -> None:
        """Resume delivery from peer_id to window_id."""
        self.get_state(window_id).paused_peers.discard(peer_id)

    def allow_more(self, window_a: str, window_b: str) -> None:
        """Clear loop counts and unpause to allow more exchanges."""
        pair_key = _pair_key(window_a, window_b)
        for wid in (window_a, window_b):
            state = self.get_state(wid)
            state.loop_counts.pop(pair_key, None)
            state.paused_peers.discard(window_a)
            state.paused_peers.discard(window_b)


# ── Module-level singleton ─────────────────────────────────────────────

delivery_strategy = MessageDeliveryStrategy()


def clear_delivery_state(window_id: str) -> None:
    delivery_strategy.clear_state(window_id)


def reset_delivery_state() -> None:
    delivery_strategy.reset_all_state()


# ── Helpers ────────────────────────────────────────────────────────────


def _pair_key(a: str, b: str) -> str:
    """Canonical key for a window pair (order-independent)."""
    return f"{min(a, b)}|{max(a, b)}"


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


def merge_injection_texts(texts: list[str]) -> str:
    """Merge multiple injection texts into a single block."""
    return " --- ".join(texts)


def write_delivery_file(
    mailbox_dir: Path, window_id: str, msg_id: str, body: str
) -> Path:
    """Write full message body to a delivery file for long messages."""
    from ..mailbox import _sanitize_dir_name

    inbox_dir = mailbox_dir / _sanitize_dir_name(window_id)
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
    for msg in eligible:
        if delivery_strategy.check_loop(qualified_id, msg.from_id):
            delivery_strategy.pause_peer(qualified_id, msg.from_id)
            loops.append((qualified_id, msg.from_id))
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
            await _notify_delivered(bot, qualified_id, to_deliver)

    # Process pending spawn requests
    if bot is not None:
        await _process_spawn_requests(bot)

    return delivered_count


async def _notify_delivered(
    bot: "Bot | None", to_window: str, messages: list["Message"]
) -> None:
    """Send Telegram notification for delivered messages (if bot available)."""
    if bot is None:
        return
    from .msg_telegram import notify_messages_delivered

    try:
        await notify_messages_delivered(bot, to_window, messages)
    except OSError, TelegramError:
        logger.debug("Failed to send delivery notification", window=to_window)


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
    from .msg_spawn import (
        handle_spawn_approval,
        post_spawn_approval_keyboard,
        scan_spawn_requests,
    )

    from ..config import config

    new_requests = scan_spawn_requests()
    for req in new_requests:
        try:
            if req.auto or config.msg_auto_spawn:
                await handle_spawn_approval(req.id, bot)
            else:
                await post_spawn_approval_keyboard(bot, req.requester_window, req)
        except OSError, TelegramError:
            logger.debug("Failed to process spawn request", request_id=req.id)
