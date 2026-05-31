"""Per-window poll cycle — one tick for one thread-bound tmux window.

Splits the per-window cycle into three layers:

* ``decide`` — pure decision kernel (no I/O, no side effects). Takes a
  ``TickContext``, returns a ``TickDecision``.
* ``observe`` — I/O readers that build the ``TickContext`` from tmux,
  the screen buffer, and the session monitor.
* ``apply`` — every Telegram, tmux, and singleton mutation lives here:
  emoji updates, status enqueuing, dead-window notifications, multi-pane
  scans, passive shell relay.

The package's public entry point is ``tick_window``; the polling
coordinator imports nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....telegram_client import PTBTelegramClient
from ...messaging_pipeline.message_queue import get_message_queue
from ...recovery.transcript_discovery import discover_and_register_transcript
from ..polling_state import (
    lifecycle_strategy,
    pane_status_strategy,
    terminal_poll_state,
    terminal_screen_buffer,
)
from ..polling_types import TickContext, TickDecision, is_shell_prompt
from .apply import (
    _apply_active_transition,
    _apply_done_transition,
    _apply_starting_transition,
    _apply_tick_decision,
    _check_interactive_only,
    _forward_pane_output,
    _handle_dead_window_notification,
    _maybe_check_passive_shell,
    _notify_pane_lifecycle,
    _scan_window_panes,
    _send_typing_throttled,
    _surface_pane_alert,
    _transition_to_idle,
    _update_status,
)
from .decide import build_status_line, decide_tick
from .observe import (
    _check_vim_insert,
    _get_last_activity_ts,
    _get_provider,
    _parse_with_pyte,
    _resolve_status,
    build_context,
)

if TYPE_CHECKING:
    from telegram import Bot

    from ....tmux_manager import TmuxWindow


async def tick_window(
    bot: "Bot",
    user_id: int,
    thread_id: int,
    window_id: str,
    window: "TmuxWindow | None",
) -> None:
    """Run one poll cycle for one window."""
    if lifecycle_strategy.is_dead_notified(user_id, thread_id, window_id):
        return

    if window is None:
        await _handle_dead_window_notification(bot, user_id, thread_id, window_id)
        return

    await discover_and_register_transcript(
        window_id,
        _window=window,
        client=PTBTelegramClient(bot),
        user_id=user_id,
        thread_id=thread_id,
    )

    queue = get_message_queue(user_id)
    if queue and not queue.empty():
        await _check_interactive_only(
            bot, user_id, window_id, thread_id, _window=window
        )
        await _scan_window_panes(bot, user_id, window_id, thread_id)
        await _maybe_check_passive_shell(bot, user_id, window_id, thread_id)
        return

    await _update_status(bot, user_id, window_id, thread_id=thread_id, _window=window)
    await _scan_window_panes(bot, user_id, window_id, thread_id)
    await _maybe_check_passive_shell(bot, user_id, window_id, thread_id)


__all__ = [
    "TickContext",
    "TickDecision",
    "_apply_active_transition",
    "_apply_done_transition",
    "_apply_starting_transition",
    "_apply_tick_decision",
    "_check_interactive_only",
    "_check_vim_insert",
    "_forward_pane_output",
    "_get_last_activity_ts",
    "_get_provider",
    "_handle_dead_window_notification",
    "_maybe_check_passive_shell",
    "_notify_pane_lifecycle",
    "_parse_with_pyte",
    "_resolve_status",
    "_scan_window_panes",
    "_send_typing_throttled",
    "_surface_pane_alert",
    "_transition_to_idle",
    "_update_status",
    "build_context",
    "build_status_line",
    "decide_tick",
    "is_shell_prompt",
    "lifecycle_strategy",
    "pane_status_strategy",
    "terminal_poll_state",
    "terminal_screen_buffer",
    "tick_window",
]
