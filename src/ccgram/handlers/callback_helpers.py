"""Shared helpers for callback handler modules.

Provides utility functions used by multiple callback handler modules:
  - user_owns_window: Check if a user has any thread binding to a window
  - get_thread_id: Extract thread_id from a Telegram update
"""

from telegram import Update

from ..thread_router import thread_router
from .callback_data import CB_PANE_DELIMITER


def user_owns_window(user_id: int, window_id: str, chat_id: int | None = None) -> bool:
    """Check ownership in the callback's chat when available."""
    if chat_id is not None:
        return any(
            uid == user_id and bound_chat == chat_id and wid == window_id
            for uid, bound_chat, _tid, wid in thread_router.iter_thread_bindings_with_chat()
        )
    return window_id in thread_router.get_all_thread_windows(user_id).values()


def parse_target(target: str) -> tuple[str, str | None]:
    """Parse window_id and optional pane_id from callback target string.

    Target format: ``@0`` (window only) or ``@0|%3`` (tmux window + pane),
    with guarded opaque Herdr session targets in the window position.
    Raw Herdr tab/pane locators are not valid callback identities.

    The delimiter is ``CB_PANE_DELIMITER`` (``|``), not a colon, so opaque
    target data and tmux pane IDs round-trip without ambiguity.
    """
    if CB_PANE_DELIMITER in target:
        idx = target.index(CB_PANE_DELIMITER)
        return target[:idx], target[idx + 1 :]
    return target, None


def get_thread_id(update: Update) -> int | None:
    """Extract thread_id from an update, returning None if not in a named topic."""
    msg = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if msg is None:
        return None
    tid = getattr(msg, "message_thread_id", None)
    if tid is None or tid == 1:
        return None
    return tid
