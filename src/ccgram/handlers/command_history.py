"""Per-user per-topic command history for recall buttons.

Stores recently sent commands in memory (no disk persistence).
History resets on bot restart.

Functions:
  - record_command: Store a command in history
  - get_history: Retrieve recent commands (newest-first)
  - clear_history: Remove history for a topic (cleanup)
  - truncate_for_display: Truncate text for button labels
"""

from collections import deque

HISTORY_MAX = 20

# Telegram switch_inline_query_current_chat limit (256 UTF-8 chars)
INLINE_QUERY_MAX = 256

# (user_id, thread_id) -> deque of commands (oldest-first in deque)
_history: dict[tuple[int, int], deque[str]] = {}


def record_command(user_id: int, thread_id: int, text: str) -> None:
    """Append a command to the user's topic history.

    Deduplicates consecutive identical commands. Caps at HISTORY_MAX.
    """
    key = (user_id, thread_id)
    dq = _history.get(key)
    if dq is None:
        dq = deque(maxlen=HISTORY_MAX)
        _history[key] = dq

    # Deduplicate consecutive identical
    if dq and dq[-1] == text:
        return

    dq.append(text)


def get_history(user_id: int, thread_id: int, *, limit: int = 20) -> list[str]:
    """Return recent commands, newest-first."""
    key = (user_id, thread_id)
    dq = _history.get(key)
    if not dq:
        return []
    # Reverse to get newest-first, then slice
    return list(reversed(dq))[:limit]


def clear_history(user_id: int, thread_id: int) -> None:
    """Remove history entry for a topic."""
    _history.pop((user_id, thread_id), None)


def truncate_for_display(text: str, max_len: int) -> str:
    """Truncate text for button labels, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"
