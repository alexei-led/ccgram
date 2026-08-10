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
from telegram.error import BadRequest, TelegramError

from ...config import config
from ...telegram_client import TelegramClient
from ...thread_router import thread_router
from ...topic_state_registry import topic_state
from ...window_query import get_approval_mode

logger = structlog.get_logger()

# Color circles used for the active/idle state prefix.
# Which color maps to which state depends on ``config.status_mode`` (see
# ``_state_emoji_map``). The constants are color-named (not state-named) so
# that ``strip_emoji_prefix`` and tests work regardless of mode.
EMOJI_GREEN_CIRCLE = "\U0001f7e2"
EMOJI_YELLOW_CIRCLE = "\U0001f7e1"
EMOJI_DONE = "\u2705"  # Check mark (agent exited normally)
EMOJI_DEAD = "\U0001f4a5"  # Collision / crash
EMOJI_YOLO = "\U0001f3b2"  # Dice (risk/gamble — auto-approve mode)
EMOJI_RC = "\U0001f4e1"  # Satellite dish (Remote Control active)
_EMOJI_DEAD_OLD = (
    "\u26ab",
    "\u274c",
)  # Legacy dead emoji (black circle pre-2026-02, cross mark pre-2026-03)

# Backward-compatible aliases — original (system-mode) defaults.
EMOJI_ACTIVE = EMOJI_GREEN_CIRCLE
EMOJI_IDLE = EMOJI_YELLOW_CIRCLE

# State → emoji mapping.
#   system mode (default): green=active (working), yellow=idle (paused).
#   user mode:             green=idle (waiting for me), yellow=active (busy).
_STATE_EMOJI_SYSTEM: dict[str, str] = {
    "active": EMOJI_GREEN_CIRCLE,
    "idle": EMOJI_YELLOW_CIRCLE,
    "done": EMOJI_DONE,
    "dead": EMOJI_DEAD,
}
_STATE_EMOJI_USER: dict[str, str] = {
    "active": EMOJI_YELLOW_CIRCLE,
    "idle": EMOJI_GREEN_CIRCLE,
    "done": EMOJI_DONE,
    "dead": EMOJI_DEAD,
}


def _state_emoji_map() -> dict[str, str]:
    """Return the active state→emoji table for the configured status mode."""
    return _STATE_EMOJI_USER if config.status_mode == "user" else _STATE_EMOJI_SYSTEM


# Debounce: state must be stable for this many seconds before updating topic name.
# Prevents rapid active↔idle toggling from flooding chat with rename messages.
#
# Asymmetric by design:
#   → active (5s):  fast feedback — user sees "agent is working" quickly
#   → idle (30s):   slow transition — brief pauses during work don't cause flicker
#   → done/dead (5s): meaningful lifecycle events, fire fast
DEBOUNCE_TO_ACTIVE_SECONDS = 5.0
DEBOUNCE_TO_IDLE_SECONDS = 30.0
DEBOUNCE_TERMINAL_SECONDS = 5.0  # done, dead

_DEBOUNCE_BY_STATE: dict[str, float] = {
    "active": DEBOUNCE_TO_ACTIVE_SECONDS,
    "idle": DEBOUNCE_TO_IDLE_SECONDS,
    "done": DEBOUNCE_TERMINAL_SECONDS,
    "dead": DEBOUNCE_TERMINAL_SECONDS,
}

# Topic state tracking: (chat_id, thread_id) -> (state, approval_mode, rc_active)
_topic_states: dict[tuple[int, int], tuple[str, str, bool]] = {}

# Pending transitions: (chat_id, thread_id) -> (desired_state, first_seen_monotonic)
_pending_transitions: dict[tuple[int, int], tuple[str, float]] = {}

# Topics inherited from a previous run, seeded at startup. Their first observed
# state is applied without debounce — the emoji currently on screen was painted
# by a process that is gone, so there is nothing to flicker against and every
# tick of delay is a topic showing a stale colour.
_awaiting_first_paint: set[tuple[int, int]] = set()

# Topic display names: (chat_id, thread_id) -> clean name (without emoji prefix).
# Updated when the incoming display name changes (write-through cache) so that
# tmux window renames and Telegram topic renames propagate correctly.
_topic_names: dict[tuple[int, int], str] = {}

# Chats where editForumTopic is disabled due to permission errors
_disabled_chats: set[int] = set()


def _resolve_topic_name(key: tuple[int, int], display_name: str) -> tuple[str, bool]:
    """Return the clean topic name and whether it changed.

    On first call, strips emoji and stores the clean name. On subsequent calls,
    if the incoming display_name (stripped) differs from the stored name,
    overwrites the cache so tmux renames propagate to Telegram.
    """
    clean = strip_emoji_prefix(display_name)
    cached = _topic_names.get(key)
    if cached is None:
        _topic_names[key] = clean
        return clean, True
    if cached != clean:
        _topic_names[key] = clean
        return clean, True
    return cached, False


def _should_apply_update(
    key: tuple[int, int],
    state: str,
    state_token: tuple[str, str, bool],
    *,
    name_changed: bool,
    now: float,
) -> bool:
    """Return True when the topic rename should be sent to Telegram."""
    if _topic_states.get(key) == state_token:
        _pending_transitions.pop(key, None)
        return name_changed

    if key in _awaiting_first_paint:
        # A topic inherited from a previous run: its emoji on screen is
        # whatever the last process left there, and the debounce damps
        # flicker *between* observed states, not the first sighting of one.
        # Making this wait leaves every restarted topic showing a stale
        # colour for the full debounce.
        _awaiting_first_paint.discard(key)
        _pending_transitions.pop(key, None)
        return True

    pending = _pending_transitions.get(key)
    if pending is None or pending[0] != state:
        _pending_transitions[key] = (state, now)
        return False

    debounce = _DEBOUNCE_BY_STATE.get(state, DEBOUNCE_TO_IDLE_SECONDS)
    if now - pending[1] < debounce:
        return False

    _pending_transitions.pop(key, None)
    return True


def _compose_topic_name(
    clean_name: str,
    *,
    state: str = "",
    approval_mode: str = "normal",
    rc_active: bool = False,
) -> str:
    """Build the full Telegram topic title from state badges and clean name."""
    parts: list[str] = []
    emoji = _state_emoji_map().get(state, "")
    if emoji:
        parts.append(emoji)
    if rc_active:
        parts.append(EMOJI_RC)
    if approval_mode == "yolo":
        parts.append(EMOJI_YOLO)
    parts.append(clean_name)
    return " ".join(parts)


async def _edit_topic_name(
    client: TelegramClient,
    chat_id: int,
    thread_id: int,
    key: tuple[int, int],
    new_name: str,
    *,
    state_token: tuple[str, str, bool] | None = None,
    state: str = "",
) -> None:
    """Apply a topic name update with shared Telegram error handling."""
    try:
        await client.edit_forum_topic(
            chat_id=chat_id,
            message_thread_id=thread_id,
            name=new_name,
        )
        if state_token is not None:
            _topic_states[key] = state_token
        logger.debug(
            "Updated topic emoji: chat=%d thread=%d state=%s name='%s'",
            chat_id,
            thread_id,
            state or "sync",
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
            if state_token is not None:
                _topic_states[key] = state_token
        else:
            logger.debug("Failed to update topic emoji: %s", e)
    except TelegramError:
        pass


def _resolve_approval_mode(chat_id: int, thread_id: int) -> str:
    """Resolve approval mode for a topic via session bindings."""
    window_id = thread_router.get_window_for_chat_thread(chat_id, thread_id)
    if not window_id:
        return "normal"
    return get_approval_mode(window_id)


def _resolve_rc_mode(chat_id: int, thread_id: int) -> bool:
    """Resolve Remote Control active state for a topic via session bindings."""
    window_id = thread_router.get_window_for_chat_thread(chat_id, thread_id)
    if not window_id:
        return False
    # Lazy: polling_state cycle — same as status_bar_actions sites.
    from ..polling.polling_state import terminal_screen_buffer

    return terminal_screen_buffer.is_rc_active(window_id)


def format_topic_name_for_mode(display_name: str, approval_mode: str) -> str:
    """Format a topic display name with a positive mode badge."""
    clean_name = strip_emoji_prefix(display_name)
    return _compose_topic_name(clean_name, approval_mode=approval_mode)


async def sync_topic_name(
    client: TelegramClient,
    chat_id: int,
    thread_id: int,
    display_name: str,
) -> None:
    """Force a topic title refresh to the current clean tmux name.

    Preserves the last known lifecycle emoji when it is cached so `/sync`
    can repair stale titles without waiting for a later state transition.
    """
    if chat_id in _disabled_chats:
        return

    key = (chat_id, thread_id)
    clean_name, _ = _resolve_topic_name(key, display_name)
    approval_mode = _resolve_approval_mode(chat_id, thread_id)
    rc_active = _resolve_rc_mode(chat_id, thread_id)
    cached = _topic_states.get(key)
    state = cached[0] if cached else ""
    state_token = (state, approval_mode, rc_active) if cached else None
    new_name = _compose_topic_name(
        clean_name,
        state=state,
        approval_mode=approval_mode,
        rc_active=rc_active,
    )
    await _edit_topic_name(
        client,
        chat_id,
        thread_id,
        key,
        new_name,
        state_token=state_token,
        state=state,
    )


async def update_topic_emoji(
    client: TelegramClient,
    chat_id: int,
    thread_id: int,
    state: str,
    display_name: str,
) -> None:
    """Update topic name with emoji prefix reflecting session state.

    Debounces transitions: the new state must be requested consistently for
    the debounce period before the API call is made. This prevents rapid
    active/idle flickering from generating lots of "topic renamed" messages.

    Args:
        client: Telegram client (TelegramClient Protocol)
        chat_id: Group chat ID
        thread_id: Forum topic thread ID
        state: One of "active", "idle", "done", "dead"
        display_name: Base topic name (without emoji prefix)
    """
    if chat_id in _disabled_chats:
        return

    key = (chat_id, thread_id)
    clean_name, name_changed = _resolve_topic_name(key, display_name)

    approval_mode = _resolve_approval_mode(chat_id, thread_id)
    rc_active = _resolve_rc_mode(chat_id, thread_id)
    state_token = (state, approval_mode, rc_active)

    emoji = _state_emoji_map().get(state, "")
    if not emoji:
        return

    now = time.monotonic()
    if not _should_apply_update(
        key,
        state,
        state_token,
        name_changed=name_changed,
        now=now,
    ):
        return

    new_name = _compose_topic_name(
        clean_name,
        state=state,
        approval_mode=approval_mode,
        rc_active=rc_active,
    )
    await _edit_topic_name(
        client,
        chat_id,
        thread_id,
        key,
        new_name,
        state_token=state_token,
        state=state,
    )


def strip_emoji_prefix(name: str) -> str:
    """Remove known emoji prefix from a topic name."""
    for emoji in (EMOJI_ACTIVE, EMOJI_IDLE, EMOJI_DONE, EMOJI_DEAD, *_EMOJI_DEAD_OLD):
        prefix = f"{emoji} "
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    # Strip badge emojis (RC, YOLO) — order: RC before YOLO matches composition order
    for badge in (EMOJI_RC, EMOJI_YOLO):
        badge_prefix = f"{badge} "
        if name.startswith(badge_prefix):
            name = name[len(badge_prefix) :]
    return name


def update_stored_topic_name(chat_id: int, thread_id: int, new_clean_name: str) -> None:
    """Overwrite the stored clean name for a topic.

    Called from FORUM_TOPIC_EDITED handler. Does not invalidate _topic_states
    since the Telegram topic already has the correct name — the next emoji
    cycle will naturally use the updated base name.
    """
    _topic_names[(chat_id, thread_id)] = new_clean_name


@topic_state.register("chat")
def clear_topic_emoji_state(chat_id: int, thread_id: int) -> None:
    """Clear emoji tracking for a topic (called on topic cleanup)."""
    key = (chat_id, thread_id)
    _topic_states.pop(key, None)
    _pending_transitions.pop(key, None)
    _topic_names.pop(key, None)
    _awaiting_first_paint.discard(key)


def mark_awaiting_first_paint(chat_id: int, thread_id: int) -> None:
    """Seed a topic inherited from a previous run (called once at startup).

    Its next observed state is applied immediately instead of waiting out
    the debounce, so a restarted ccgram repaints stale emoji on the first
    poll rather than 30s later.
    """
    _awaiting_first_paint.add((chat_id, thread_id))


_MAX_DISABLED_CHATS = 1000


@topic_state.register("chat")
def clear_disabled_chat(chat_id: int, _thread_id: int = 0) -> None:
    """Remove a chat from the disabled set (called on topic cleanup)."""
    _disabled_chats.discard(chat_id)
    if len(_disabled_chats) > _MAX_DISABLED_CHATS:
        _disabled_chats.clear()


def reset_all_state() -> None:
    """Reset all tracking state (for testing)."""
    _topic_states.clear()
    _pending_transitions.clear()
    _disabled_chats.clear()
    _topic_names.clear()
    _awaiting_first_paint.clear()
