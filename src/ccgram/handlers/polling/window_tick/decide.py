"""Pure decision kernel for window_tick — no I/O, no side effects.

All inputs flow in via ``TickContext``; output is a ``TickDecision``.
This module imports nothing that touches tmux, Telegram, or singletons,
so its functions are deterministic and trivially unit-testable.
"""

from __future__ import annotations

import time

from ....providers.base import StatusUpdate
from ....terminal_parser import status_emoji_prefix
from ..polling_types import (
    STARTUP_TIMEOUT,
    TickContext,
    TickDecision,
    is_shell_prompt,
)


def build_status_line(status: StatusUpdate | None) -> str | None:
    if not status or status.is_interactive:
        return None
    if "\n" in status.raw_text:
        return status.raw_text
    return f"{status_emoji_prefix(status.raw_text)} {status.raw_text}"


def decide_tick(ctx: TickContext) -> TickDecision:
    """Pure status/idle transition decision — no I/O, no side effects.

    All mutable state reads (``has_seen_status``, ``is_recently_active``,
    ``startup_time``) must be computed by the coordinator before building
    ``TickContext``. The ``is_recently_active`` flag is special: its
    computation in the coordinator may mark_seen_status as a side effect,
    so it must not be re-derived here.
    """
    if ctx.is_dead_window:
        return TickDecision(show_recovery=True)

    if ctx.resolved_status_text:
        return TickDecision(
            send_status=True,
            status_text=ctx.resolved_status_text,
            transition="active",
        )

    if ctx.is_recently_active:
        return TickDecision(transition="active")

    if ctx.is_shell_prompt:
        if ctx.supports_hook:
            return TickDecision(transition="done")
        # Hookless provider back at its prompt: genuine end of turn, the
        # Ready bubble is wanted here.
        return TickDecision(transition="idle", send_status=True)

    if ctx.has_seen_status:
        return TickDecision(transition="idle", send_status=True)

    startup_expired = (
        ctx.startup_time is not None
        and (time.monotonic() - ctx.startup_time) >= STARTUP_TIMEOUT
    )
    # Quiet settle: a window that never showed a status line in this run
    # (agent not running, dormant tab, or already quietly settled) must
    # not announce "Ready" (one fresh bubble per topic per restart is
    # noise). The idle transition with send_status=False runs the side
    # effects (settle, typing cleared, emoji restored) minus the bubble,
    # and its quiet latch keeps later ticks silent until real status.
    if ctx.quiet_settled or startup_expired:
        return TickDecision(transition="idle", send_status=False)

    return TickDecision(transition="starting")


__all__ = ["build_status_line", "decide_tick", "is_shell_prompt"]
