"""Remote Control outcome probe — classify Claude /remote-control result.

After Claude's ``/remote-control`` is fired (status-bubble button or a
forwarded slash command) there is no built-in signal of the outcome:
silent on success, silent on "feature unavailable", silent on failure.
This module arms a short-lived background probe that captures the pane,
classifies the result with a pure regex classifier, and posts a single
status reply in the bound topic.

Both trigger paths call ``arm_rc_probe(window_id, client)``. The
per-window ``rc_probe_state`` field on ``WindowState`` de-dupes
double-taps; it is in-memory only (never serialized — see
``window_state_store``) and is safe to drop on restart.

Capability-gated to the Claude provider — other providers return early
and keep their existing "not supported by <provider>" behaviour.

Key components: ``RCOutcome`` result type, the pure
``classify_rc_output``, ``arm_rc_probe`` (double-tap guard + capability
gate), the ``_classify_loop`` polling coroutine, ``_send_outcome_reply``.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from ... import window_query
from ...providers import get_provider_for_window
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...utils import task_done_callback
from ...window_state_store import get_window_store
from ..messaging_pipeline.message_sender import safe_send

if TYPE_CHECKING:
    from ...telegram_client import TelegramClient

logger = structlog.get_logger()

_FIRST_CAPTURE_DELAY = 1.5
_RETRY_INTERVAL = 1.5
_TOTAL_TIMEOUT = 10.0
_SCAN_LINES = 30

_URL_RE = re.compile(r"https://claude\.ai/\S+|https?://\S*remote\S*", re.IGNORECASE)
_UNAVAILABLE_RE = re.compile(
    r"(?i)\b(?:not available|requires|upgrade|permission denied|unknown command)\b"
)
_FAILED_RE = re.compile(r"(?i)\b(?:error|failed)\b")
# The /remote-control (or /rc) echo we just sent. "unavailable"/"failed"
# words are only the command's outcome when they appear at or after this
# line — pre-RC scrollback routinely contains "error"/"requires"/etc.
_RC_ANCHOR_RE = re.compile(r"(?i)/remote-control\b|/rc\b")


class RCOutcomeKind(Enum):
    """Classification of the pane state after /remote-control was sent."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class RCOutcome:
    """Result of classifying captured pane output.

    ``detail`` carries the matched URL for ``SUCCESS`` and the offending
    line for ``UNAVAILABLE`` / ``FAILED``; empty for ``PENDING`` and for
    a success with no URL in the captured text.
    """

    kind: RCOutcomeKind
    detail: str = ""


def classify_rc_output(text: str) -> RCOutcome:
    """Classify the last lines of captured pane text.

    Priority: a Remote Control URL means RC actually started (success)
    and wins everywhere. Otherwise "unavailable" / generic error lines
    count only at or after the most recent ``/remote-control`` (``/rc``)
    echo — pre-RC scrollback routinely contains "error"/"failed"/
    "requires" and must not be misread as the command's outcome. No
    anchor and no URL → pending (keep polling).
    """
    if not text:
        return RCOutcome(RCOutcomeKind.PENDING)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tail = lines[-_SCAN_LINES:]
    blob = "\n".join(tail)

    url_match = _URL_RE.search(blob)
    if url_match:
        return RCOutcome(RCOutcomeKind.SUCCESS, url_match.group(0))

    anchor = next(
        (i for i in range(len(tail) - 1, -1, -1) if _RC_ANCHOR_RE.search(tail[i])),
        None,
    )
    if anchor is None:
        return RCOutcome(RCOutcomeKind.PENDING)
    scoped = tail[anchor:]

    for line in scoped:
        if _UNAVAILABLE_RE.search(line):
            return RCOutcome(RCOutcomeKind.UNAVAILABLE, line)

    for line in scoped:
        if _FAILED_RE.search(line):
            return RCOutcome(RCOutcomeKind.FAILED, line)

    return RCOutcome(RCOutcomeKind.PENDING)


def _format_reply(outcome: RCOutcome) -> str:
    if outcome.kind is RCOutcomeKind.SUCCESS:
        if outcome.detail:
            return f"\U0001f4e1 Remote Control active — `{outcome.detail}`"
        return "\U0001f4e1 Remote Control active."
    if outcome.kind is RCOutcomeKind.UNAVAILABLE:
        return f"\U0001f4e1 Remote Control unavailable — {outcome.detail}."
    if outcome.kind is RCOutcomeKind.FAILED:
        return f"\U0001f4e1 Remote Control failed — {outcome.detail}."
    return "\U0001f4e1 No response from /remote-control — check the pane."


async def _send_outcome_reply(
    client: TelegramClient, window_id: str, outcome: RCOutcome
) -> None:
    """Post the classified outcome in every topic bound to this window."""
    text = _format_reply(outcome)
    for user_id, thread_id, bound_wid in thread_router.iter_thread_bindings():
        if bound_wid != window_id:
            continue
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        await safe_send(client, chat_id, text, message_thread_id=thread_id)


async def _classify_loop(window_id: str, client: TelegramClient) -> None:
    """Capture + classify the pane until a verdict or the timeout.

    Resets ``rc_probe_state`` on every exit path (verdict, timeout,
    exception, cancellation) so a tracebacked probe never leaves the
    window stuck in ``armed``.
    """
    try:
        await asyncio.sleep(_FIRST_CAPTURE_DELAY)
        start = time.monotonic()
        while time.monotonic() - start < _TOTAL_TIMEOUT:
            pane = await tmux_manager.capture_pane(window_id)
            outcome = classify_rc_output(pane or "")
            if outcome.kind is not RCOutcomeKind.PENDING:
                await _send_outcome_reply(client, window_id, outcome)
                return
            # Lazy: polling_state import cycle (status ↔ polling) — same
            # reason status_bar_actions defers terminal_screen_buffer.
            from ..polling.polling_state import terminal_screen_buffer

            if terminal_screen_buffer.is_rc_active(window_id):
                await _send_outcome_reply(
                    client, window_id, RCOutcome(RCOutcomeKind.SUCCESS)
                )
                return
            await asyncio.sleep(_RETRY_INTERVAL)
        await _send_outcome_reply(client, window_id, RCOutcome(RCOutcomeKind.PENDING))
    finally:
        get_window_store().get_window_state(window_id).rc_probe_state = "classified"


def arm_rc_probe(window_id: str, client: TelegramClient) -> None:
    """Arm the RC outcome probe for a window (Claude provider only).

    No-op when the window's provider is not Claude or when a probe is
    already ``armed`` for the window (double-tap guard).
    """
    provider = get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    )
    if provider.capabilities.name != "claude":
        return

    state = get_window_store().get_window_state(window_id)
    if state.rc_probe_state == "armed":
        return
    state.rc_probe_state = "armed"
    state.rc_armed_at = time.monotonic()

    task = asyncio.create_task(_classify_loop(window_id, client))
    task.add_done_callback(task_done_callback)
