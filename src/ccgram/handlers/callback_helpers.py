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


GENERAL_TOPIC_ID = 1
"""Telegram's General topic: the explicit control lane, never a session."""


def private_topic_id(message: object) -> int | None:
    """Return an observed private-chat topic ID, if present.

    Private bot topics use the regular topic-message shape: a private chat,
    ``is_topic_message``, and an integer ``message_thread_id``. Ordinary
    private DMs deliberately keep their legacy ``None`` thread identity.
    """
    chat = getattr(message, "chat", None)
    thread_id = getattr(message, "message_thread_id", None)
    if (
        getattr(chat, "type", None) == "private"
        and getattr(message, "is_topic_message", None) is True
        and isinstance(thread_id, int)
    ):
        return thread_id
    return None


def get_thread_id(update: Update) -> int | None:
    """Extract a non-General topic ID from a forum or private-topic update.

    Topic 1 is an explicit General/control invariant in both topic-capable
    surfaces. It must not become a session binding or be replaced by ``/new``.
    """
    msg = update.message or (
        update.callback_query.message if update.callback_query else None
    )
    if msg is None:
        return None

    private_thread_id = private_topic_id(msg)
    if private_thread_id is not None:
        chat_id = getattr(getattr(msg, "chat", None), "id", None)
        if isinstance(chat_id, int):
            thread_router.mark_private_topic_chat(chat_id)
        return private_thread_id if private_thread_id != GENERAL_TOPIC_ID else None

    # Forum topic IDs are valid only for group/supergroup forum messages.
    # Keep ordinary private DMs on their pre-threaded legacy path even if a
    # future update happens to expose a thread-like field without topic
    # metadata. Test doubles and compatible update objects can omit chat.type;
    # preserve their prior thread-ID behavior.
    chat_type = getattr(getattr(msg, "chat", None), "type", None)
    tid = getattr(msg, "message_thread_id", None)
    if not isinstance(chat_type, str):
        return tid if isinstance(tid, int) and tid != GENERAL_TOPIC_ID else None
    if chat_type not in ("group", "supergroup"):
        return None
    if not isinstance(tid, int) or tid == GENERAL_TOPIC_ID:
        return None
    return tid
