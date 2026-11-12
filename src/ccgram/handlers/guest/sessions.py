"""Guests starting sessions: ``/sessions new`` (or ➕) in a guest chat.

The new session is created through the session core in the chat's workspace
(the workspace routed to this chat by ``CCGRAM_TOPIC_CHAT_ROUTES``), running
``CCGRAM_GUEST_LAUNCH_COMMAND`` with the chat's label, so it starts with the
guest's restricted settings and never with the owner's. Discovery then adopts
the window and creates its topic in this chat, as for any session started on
the computer.

A chat holds at most ``CCGRAM_GUEST_MAX_SESSIONS`` sessions. Sessions still
starting (created, not yet adopted) count too, and starts in one chat are
serialised, so quick repeated taps cannot exceed the limit.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ...config import config
from ...multiplexer import multiplexer
from ...multiplexer.base import canonical_window_id
from ...multiplexer.reconciliation import list_windows_for_reconciliation
from ...session_protocol.models import ProtocolError
from ..topics.topic_routes import routed_workspace
from .scope import windows_in_chat

if TYPE_CHECKING:
    from ...session_protocol.telegram import TelegramSessionAdapter

logger = structlog.get_logger()

# The guest launcher starts Claude Code; the provider is recorded for the
# new window until discovery confirms it.
GUEST_PROVIDER = "claude"
# A created window not adopted within this long no longer holds a slot.
PENDING_TTL_S = 120.0

DISABLED = "Starting sessions from this chat is not enabled."
UNREACHABLE = "Could not reach the computer's terminals. Try again in a moment."
NO_WORKSPACE = "This chat has no workspace to start sessions in. Ask the owner."
FAILED = "Could not start a session. The owner can see why in the log."
LIMIT = (
    "This chat already has {count} sessions (the most is {max}). "
    "End one first: /sessions, then 🗑."
)
STARTED = (
    "Starting a new session ({count} of {max}). "
    "It appears here as a new topic in a few seconds."
)

_pending: dict[int, dict[str, float]] = {}
_locks: dict[int, asyncio.Lock] = {}


def _clock() -> float:
    return time.monotonic()


def _lock(chat_id: int) -> asyncio.Lock:
    return _locks.setdefault(chat_id, asyncio.Lock())


async def count_sessions(chat_id: int) -> int | None:
    """Live sessions bound in *chat_id* plus ones still starting; None if unknown."""
    windows = await list_windows_for_reconciliation(multiplexer)
    if windows is None:
        return None
    live = {canonical_window_id(w.window_id) for w in windows}
    bound = {canonical_window_id(w) for w in windows_in_chat(chat_id)}
    now = _clock()
    pending = _pending.get(chat_id, {})
    for window_id, started in list(pending.items()):
        key = canonical_window_id(window_id)
        # Adopted (now bound), gone, or never adopted: the slot is free again.
        if key in bound or key not in live or now - started > PENDING_TTL_S:
            del pending[window_id]
    return len(bound & live) + len(pending)


async def _seed_trust(folder: str) -> None:
    """Pre-accept the agent's folder-trust prompt, as /new does."""
    # Lazy: agent_startup imports the provider registry, which reaches back
    # into the handlers package.
    from ...agent_startup import preseed_trust

    await asyncio.to_thread(preseed_trust, GUEST_PROVIDER, Path(folder))


async def start_guest_session(chat_id: int, adapter: TelegramSessionAdapter) -> str:
    """Start one session for a guest chat; returns the reply for the chat."""
    label = config.guest_chats.get(chat_id)
    if label is None or not config.guest_launch_command:
        return DISABLED
    async with _lock(chat_id):
        count = await count_sessions(chat_id)
        if count is None:
            return UNREACHABLE
        limit = config.guest_max_sessions
        if count >= limit:
            logger.info("Guest session limit reached", chat_id=chat_id, count=count)
            return LIMIT.format(count=count, max=limit)
        workspace = await routed_workspace(chat_id)
        if workspace is None or not workspace.cwd:
            logger.warning("Guest chat has no routed workspace", chat_id=chat_id)
            return NO_WORKSPACE
        command = config.guest_launch_command.replace("{label}", shlex.quote(label))
        data = {
            "provider": GUEST_PROVIDER,
            "directory": workspace.cwd,
            "workspaceId": workspace.workspace_id,
        }
        try:
            await _seed_trust(workspace.cwd)
            created = await adapter.create(data, launch_command=command)
        except (ProtocolError, OSError) as exc:
            logger.warning(
                "Guest session start failed", chat_id=chat_id, error=str(exc)
            )
            return FAILED
        _pending.setdefault(chat_id, {})[created.target_id] = _clock()
        logger.info(
            "Guest session started",
            chat_id=chat_id,
            window_id=created.target_id,
            workspace=workspace.label,
        )
        return STARTED.format(count=count + 1, max=limit)


def _reset_guest_sessions_for_testing() -> None:
    _pending.clear()
    _locks.clear()
