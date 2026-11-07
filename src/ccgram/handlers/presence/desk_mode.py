"""Desk-mode wiring: Telegram activity handler and the periodic tick."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from ...config import config
from ...telegram_client import PTBTelegramClient, TelegramClient
from . import digest, presence

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_background: set[asyncio.Task[None]] = set()


def _spawn_close_out(client: TelegramClient, user_id: int, reason: str) -> None:
    task = asyncio.create_task(
        digest.close_out(client, user_id, reason=reason),
        name=f"desk-close-out:{user_id}",
    )
    _background.add(task)
    task.add_done_callback(_background.discard)


async def telegram_activity_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Any update from an allowed user means they are on Telegram: away.

    Runs in its own early handler group, so the mode has switched before the
    update reaches the handlers that act on it. The digest close-out runs in
    the background so it never delays that update.
    """
    if not presence.enabled():
        return
    user = update.effective_user
    if user is None or not config.is_user_allowed(user.id):
        return
    if presence.note_telegram_activity(user.id):
        _spawn_close_out(PTBTelegramClient(context.bot), user.id, "telegram")


async def run_desk_mode_tick(client: TelegramClient) -> None:
    """Idle check (desk -> away) and digest refresh; called every few seconds."""
    if not presence.enabled():
        return
    for user_id in await presence.refresh_idle():
        await digest.close_out(client, user_id, reason="idle")
    await digest.flush_due(client)
