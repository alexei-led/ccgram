"""Backend-neutral window send helpers.

``send_to_window`` / ``send_followup_to_window`` are convenience wrappers over
the active ``multiplexer`` backend (looked up via the module-level proxy) plus
the thread router for display-name logging.  They live here — not in a concrete
backend — so callers depend only on the ``multiplexer`` proxy (F1 boundary).
"""

from __future__ import annotations

import asyncio

import structlog

from ..thread_router import thread_router
from . import multiplexer

logger = structlog.get_logger(__name__)


async def send_to_window(
    window_id: str, text: str, *, raw: bool = False
) -> tuple[bool, str]:
    """Send text to a window by ID.

    Returns (success, message). Looks up the display name for logging, then
    delegates to ``multiplexer.find_window_by_id`` + ``send_keys``.
    """
    display = thread_router.get_display_name(window_id)
    logger.debug(
        "send_to_window: window_id=%s (%s), text_len=%d",
        window_id,
        display,
        len(text),
    )
    window = await multiplexer.find_window_by_id(window_id)
    if not window:
        return False, "Window not found (may have been closed)"
    success = await multiplexer.send_keys(window.window_id, text, raw=raw)
    if success:
        return True, f"Sent to {display}"
    return False, "Failed to send keys"


async def send_followup_to_window(window_id: str, text: str) -> tuple[bool, str]:
    """Send text to a Pi window as an Alt+Enter follow-up message."""
    display = thread_router.get_display_name(window_id)
    logger.debug(
        "send_followup_to_window: window_id=%s (%s), text_len=%d",
        window_id,
        display,
        len(text),
    )
    window = await multiplexer.find_window_by_id(window_id)
    if not window:
        return False, "Window not found (may have been closed)"
    if not await multiplexer.send_keys(
        window.window_id, text, enter=False, literal=True
    ):
        return False, "Failed to send follow-up text"
    await asyncio.sleep(0.5)
    if await multiplexer.send_keys(
        window.window_id, "M-Enter", enter=False, literal=False
    ):
        return True, f"Follow-up queued for {display}"
    return False, "Failed to send follow-up key"
