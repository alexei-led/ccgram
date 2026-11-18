"""Telegram entrypoint for window-death facts from application adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...session_query import iter_bound_topics
from .window_tick.apply import _handle_dead_window_notification

if TYPE_CHECKING:
    from telegram import Bot

__all__ = ["handle_window_death"]


async def handle_window_death(bot: Bot, window_id: str) -> None:
    """Route death to current recipients; the shared policy rechecks each binding."""
    # The query snapshots bindings because deletion unbinds topics during await.
    for user_id, _chat_id, thread_id, bound_wid in iter_bound_topics():
        if bound_wid == window_id:
            await _handle_dead_window_notification(bot, user_id, thread_id, window_id)
