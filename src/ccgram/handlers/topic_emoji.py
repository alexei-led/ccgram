"""Topic emoji status updates via editForumTopic.

Updates topic names with status emoji prefixes to reflect session state:
  - Active (working): topic name prefixed with working emoji
  - Idle (waiting): topic name prefixed with idle emoji
  - Done (Claude exited): topic name prefixed with done emoji
  - Dead (window gone): topic name prefixed with dead emoji

Tracks per-topic state to avoid redundant API calls. Debounces transitions
to prevent rapid active/idle toggling from flooding the chat with rename
messages. Gracefully degrades when the bot lacks editForumTopic permission.

Key functions:
  - update_topic_emoji: Update emoji for a specific topic (debounced)
  - clear_topic_emoji_state: Clean up tracking for a topic
"""

import time

import structlog
from telegram import Bot
from telegram.error import BadRequest, TelegramError

logger = structlog.get_logger()

# Emoji prefixes for session states
EMOJI_ACTIVE = "\U0001f7e2"  # Green circle
EMOJI_IDLE = "\U0001f4a4"  # Zzz / sleeping
EMOJI_DONE = "\u2705"  # Check mark (Claude exited normally)
EMOJI_DEAD = "\u274c"  # Cross mark
EMOJI_YOLO = "\U0001f680"  # Rocket (positive YOLO indicator)
_EMOJI_DEAD_OLD = "\u26ab"  # Legacy dead emoji (black circle, pre-2026-02)

# Debounce: state must be stable for this many seconds before updating topic name.
# Prevents rapid active↔idle toggling from flooding chat with rename messages.
DEBOUNCE_SECONDS = 5.0

# Topic state tracking: (chat_id, thread_id) -> (state, approval_mode)
_topic_states: dict[tuple[int, int], tuple[str, str]] = {}

# Pending transitions: (chat_id, thread_id) -> (desired_state, first_seen_monotonic)
_pending_transitions: dict[tuple[int, int], tuple[str, float]] = {}

# Topic display names: (chat_id, thread_id) -> clean name (without emoji prefix).
# Updated when the incoming display name changes (write-through cache) so that
# tmux window renames and Telegram topic renames propagate correctly.
_topic_names: dict[tuple[int, int], str] = {}

# Chats where editForumTopic is disabled due to permission errors
_disabled_chats: set[int] = set()


def _resolve_topic_name(key: tuple[int, int], display_name: str) -> str:
    """Return the clean topic name, updating the cache when the name changes.

    On first call, strips emoji and stores the clean name. On subsequent calls,
    if the incoming display_name (stripped) differs from the stored name,
    overwrites the cache so tmux renames propagate to Telegram.
    """
    clean = strip_emoji_prefix(display_name)
    cached = _topic_names.get(key)
    if cached is None:
        _topic_names[key] = clean
        return clean
    if cached != clean:
        _topic_names[key] = clean
        # Invalidate state so next update_topic_emoji re-applies emoji with new name
        _topic_states.pop(key, None)
    return _topic_names[key]


def _resolve_approval_mode(chat_id: int, thread_id: int) -> str:
    """Resolve approval mode for a topic via session bindings."""
    from ..session import DEFAULT_APPROVAL_MODE, session_manager

    window_id = session_manager.get_window_for_chat_thread(chat_id, thread_id)
    if not window_id:
        return DEFAULT_APPROVAL_MODE
    return session_manager.get_approval_mode(window_id)


def format_topic_name_for_mode(display_name: str, approval_mode: str) -> str:
    """Format a topic display name with a positive mode badge."""
    clean_name = strip_emoji_prefix(display_name)
    if approval_mode == "yolo":
        return f"{EMOJI_YOLO} {clean_name}"
    return clean_name


async def update_topic_emoji(
    bot: Bot,
    chat_id: int,
    thread_id: int,
    state: str,
    display_name: str,
) -> None:
    """Update topic name with emoji prefix reflecting session state.

    Debounces transitions: the new state must be requested consistently for
    DEBOUNCE_SECONDS before the API call is made. This prevents rapid
    active/idle flickering from generating lots of "topic renamed" messages.

    Args:
        bot: Telegram Bot instance
        chat_id: Group chat ID
        thread_id: Forum topic thread ID
        state: One of "active", "idle", "done", "dead"
        display_name: Base topic name (without emoji prefix)
    """
    if chat_id in _disabled_chats:
        return

    key = (chat_id, thread_id)

    approval_mode = _resolve_approval_mode(chat_id, thread_id)
    state_token = (state, approval_mode)

    # Already in this state/mode — no transition needed
    if _topic_states.get(key) == state_token:
        _pending_transitions.pop(key, None)
        return

    emoji = {
        "active": EMOJI_ACTIVE,
        "idle": EMOJI_IDLE,
        "done": EMOJI_DONE,
        "dead": EMOJI_DEAD,
    }.get(state, "")

    if not emoji:
        return

    # Debounce: require the new state to be stable before applying
    now = time.monotonic()
    pending = _pending_transitions.get(key)
    if pending is None or pending[0] != state:
        # New or changed desired state — start debounce timer
        _pending_transitions[key] = (state, now)
        return

    if now - pending[1] < DEBOUNCE_SECONDS:
        # Not stable long enough yet
        return

    # Debounce passed — execute the transition
    _pending_transitions.pop(key, None)

    clean_name = _resolve_topic_name(key, display_name)
    mode_prefix = f"{EMOJI_YOLO} " if approval_mode == "yolo" else ""
    new_name = f"{emoji} {mode_prefix}{clean_name}"

    try:
        await bot.edit_forum_topic(
            chat_id=chat_id,
            message_thread_id=thread_id,
            name=new_name,
        )
        _topic_states[key] = state_token
        logger.debug(
            "Updated topic emoji: chat=%d thread=%d state=%s name='%s'",
            chat_id,
            thread_id,
            state,
            new_name,
        )
    except BadRequest as e:
        if "Not enough rights" in e.message:
            _disabled_chats.add(chat_id)
            logger.info(
                "Topic emoji disabled for chat %d: insufficient permissions",
                chat_id,
            )
        elif (
            "topic_not_modified" in e.message.lower() or "Topic_id_invalid" in e.message
        ):
            # Expected no-ops: already correct name or invalid topic
            _topic_states[key] = state_token
        else:
            logger.debug("Failed to update topic emoji: %s", e)
    except TelegramError:
        pass


def strip_emoji_prefix(name: str) -> str:
    """Remove known emoji prefix from a topic name."""
    for emoji in (EMOJI_ACTIVE, EMOJI_IDLE, EMOJI_DONE, EMOJI_DEAD, _EMOJI_DEAD_OLD):
        prefix = f"{emoji} "
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    yolo_prefix = f"{EMOJI_YOLO} "
    if name.startswith(yolo_prefix):
        return name[len(yolo_prefix) :]
    return name


def update_stored_topic_name(chat_id: int, thread_id: int, new_clean_name: str) -> None:
    """Overwrite the stored clean name for a topic.

    Called from FORUM_TOPIC_EDITED handler. Does not invalidate _topic_states
    since the Telegram topic already has the correct name — the next emoji
    cycle will naturally use the updated base name.
    """
    _topic_names[(chat_id, thread_id)] = new_clean_name


def clear_topic_emoji_state(chat_id: int, thread_id: int) -> None:
    """Clear emoji tracking for a topic (called on topic cleanup)."""
    key = (chat_id, thread_id)
    _topic_states.pop(key, None)
    _pending_transitions.pop(key, None)
    _topic_names.pop(key, None)


def reset_all_state() -> None:
    """Reset all tracking state (for testing)."""
    _topic_states.clear()
    _pending_transitions.clear()
    _disabled_chats.clear()
    _topic_names.clear()
