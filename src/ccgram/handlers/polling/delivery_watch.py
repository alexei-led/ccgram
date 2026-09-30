"""Silent delivery-wedge early warning.

The 2026-09-26 incident (TASK-43) froze delivery for seven hours with
zero log lines; the operator noticed only because replies stopped
arriving. This watch turns that failure mode into a five-minute alarm:
it compares each bound window's durable delivery watermark (the
monitor's settled-receipt boundary) against its transcript size, and
when a large gap stops advancing it warns in the journal and posts ONE
notice in the topic. The notice is sent through the direct interactive
path, never the delivery queue: a wedged queue must not swallow its
own alarm.

The watermark is the correct signal, not the user read offset: the
read offset stamps at enqueue time and on /history paging, so it keeps
moving while delivery is dead (message_routing.py routes and stamps
even when the worker is frozen). The watermark freezes precisely
because receipts stop settling.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from ...session_query import iter_bound_topics
from ...session_state_ports import get_delivery_watermark
from ...telegram_client import TelegramClient
from ..messaging_pipeline.message_sender import safe_send

logger = structlog.get_logger()

# Transcript bytes that must be undelivered before the gap counts as a
# wedge candidate; below this a stalled watermark is normal quiet time.
# Generous on purpose: a long streaming thinking block grows the
# transcript without any complete message settling a receipt, and that
# benign burst must not page the operator. CCGRAM_DELIVERY_WATCH_GAP_KB=0
# disables the watch (same knob shape as CCGRAM_REPLAY_CAP_MB).
GAP_THRESHOLD_BYTES = max(
    0, int(float(os.getenv("CCGRAM_DELIVERY_WATCH_GAP_KB", "256") or 0) * 1024)
)
# How long a qualifying gap must stay frozen before the alarm fires.
# The check rides the 60s periodic gate, so this is five to six
# observations, not a timer.
STUCK_GRACE_S = 300.0

ALERT_TEXT = (
    "⚠️ Delivery stall detected: {gap_kb:.0f} KB of output has not"
    " been delivered for over {stuck:.0f}s. If replies look missing, restart"
    " the bridge; a restart clears the stall without losing messages."
)


@dataclass
class _WindowWatch:
    last_offset: int | None = None
    stuck_since: float | None = None
    alerted: bool = False
    transcript_path: str = ""
    size_at_stuck: int = 0


@dataclass
class DeliveryGapWatch:
    """Pure decision core: turns watermark/size observations into alarms.

    observe() is called once per periodic check per bound window. An
    alert fires at most once per incident; the incident ends when the
    watermark advances or the gap shrinks below the threshold, either of
    which re-arms the window.
    """

    gap_threshold: int = GAP_THRESHOLD_BYTES
    stuck_grace_s: float = STUCK_GRACE_S
    _windows: dict[str, _WindowWatch] = field(default_factory=dict)

    def observe(
        self,
        window_id: str,
        offset: int,
        size: int,
        now: float,
        *,
        transcript_path: str = "",
    ) -> bool:
        """Record one observation; True when the alarm should fire now."""
        state = self._windows.setdefault(
            window_id,
            _WindowWatch(last_offset=offset, transcript_path=transcript_path),
        )
        if transcript_path and state.transcript_path != transcript_path:
            # Same window, new session transcript: the old incident's
            # timers and latch describe a different file.
            state = _WindowWatch(last_offset=offset, transcript_path=transcript_path)
            self._windows[window_id] = state

        gap = max(size - offset, 0)
        if gap < self.gap_threshold or offset != state.last_offset:
            # Healthy movement, or nothing worth alarming about: the
            # previous incident (if any) is over.
            state.last_offset = offset
            state.stuck_since = None
            state.size_at_stuck = size
            state.alerted = False
            return False

        if state.stuck_since is None:
            state.stuck_since = now
            state.size_at_stuck = size
            return False
        stuck = now - state.stuck_since
        # The incident signature is a FROZEN watermark under a GROWING
        # transcript; a static transcript is quiet time, not a stall.
        grew = size > state.size_at_stuck
        if stuck >= self.stuck_grace_s and grew and not state.alerted:
            state.alerted = True
            return True
        return False

    def forget(self, bound: set[str]) -> None:
        """Drop state for windows that are no longer bound."""
        for window_id in list(self._windows):
            if window_id not in bound:
                del self._windows[window_id]

    def disarm(self, window_id: str) -> None:
        """Re-arm a window whose one-shot alert could not go out."""
        state = self._windows.get(window_id)
        if state is not None:
            state.alerted = False

    def pause(self, window_id: str) -> None:
        """Stop the stall clock without ending any incident."""
        state = self._windows.get(window_id)
        if state is not None:
            state.stuck_since = None


_watch = DeliveryGapWatch()
# Alert sends are fire-and-forget so a flood-retried Telegram call can
# never stall the poll cycle; the set only holds strong references.
_alert_tasks: set[asyncio.Task[None]] = set()


def reset_for_testing() -> None:
    """Drop all watch state and pending alerts (test isolation)."""
    global _watch
    _watch = DeliveryGapWatch()
    _alert_tasks.clear()


async def check_delivery_wedges(client: TelegramClient) -> None:
    """One periodic pass: measure every bound window, alarm on stalls."""
    if _watch.gap_threshold <= 0:
        return
    now = time.monotonic()
    topics = iter_bound_topics()
    _watch.forget({window_id for _, _, _, window_id in topics})
    for user_id, chat_id, thread_id, window_id in topics:
        delivery = get_delivery_watermark(window_id)
        if delivery is None or delivery.watermark < 0:
            # Not yet tracked: the monitor has not parsed anything to
            # settle a receipt for.
            continue
        if delivery.fenced:
            # Backlog-skip barrier or pending-tools fence: the monitor
            # freezes commits on purpose; that is not a wedge, and fence
            # time must not count toward the stall grace.
            _watch.pause(window_id)
            continue
        try:
            size = Path(delivery.transcript_path).stat().st_size
        except OSError:
            continue
        if not _watch.observe(
            window_id,
            delivery.watermark,
            size,
            now,
            transcript_path=delivery.transcript_path,
        ):
            continue
        gap = max(size - delivery.watermark, 0)
        logger.warning(
            "Delivery wedge suspected: transcript growing, watermark frozen",
            user_id=user_id,
            window_id=window_id,
            gap_bytes=gap,
            stuck_grace_s=_watch.stuck_grace_s,
        )
        if chat_id is None:
            # The one-shot must not be consumed by an unresolved chat
            # mapping; retry once the router knows the topic's chat.
            _watch.disarm(window_id)
            continue
        # Fire-and-forget send, decoupled from the poll loop: the cycle
        # must not wait on Telegram.
        task = asyncio.create_task(
            _send_alert(client, chat_id, thread_id, gap, window_id)
        )
        _alert_tasks.add(task)
        task.add_done_callback(_alert_tasks.discard)


async def _send_alert(
    client: TelegramClient,
    chat_id: int,
    thread_id: int,
    gap: int,
    window_id: str,
) -> None:
    try:
        # Direct send, never the delivery queue: the queue is the thing
        # under suspicion and must not carry its own alarm. safe_send
        # returns None on non-rate-limit Telegram failures (only
        # RetryAfter raises), so a None answer is also a failed alert.
        sent = await safe_send(
            client,
            chat_id,
            ALERT_TEXT.format(gap_kb=gap / 1024, stuck=_watch.stuck_grace_s),
            message_thread_id=thread_id,
        )
        if sent is None:
            raise RuntimeError("alert send returned None")
    except Exception:
        # RetryAfter propagates from safe_send once the limiter's budget
        # is spent, and a wedged system often has Telegram degraded too;
        # the topic notice must retry on a later pass instead of dying
        # as an unretrieved task exception.
        logger.exception(
            "Delivery-wedge alert send failed; will retry next pass",
            window_id=window_id,
        )
        _watch.disarm(window_id)
