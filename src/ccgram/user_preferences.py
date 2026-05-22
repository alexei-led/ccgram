"""User preferences — starred directories, MRU, and read offsets.

Extracted from SessionManager to reduce its surface area. The
``schedule_save`` callback is injected via the constructor — the store
cannot be built without an explicit callback.

Module-level access: ``get_user_preferences()`` returns the
SessionManager-owned instance (raises RuntimeError until SessionManager
has constructed the store). The legacy module attribute
``user_preferences`` is a thin proxy that delegates to the same instance
for backward compat.

Key class: UserPreferences.
Key data:
  - user_dir_favorites (user_id -> {"starred": [...], "mru": [...]})
  - user_window_offsets (user_id -> {window_id -> byte_offset})
"""

from __future__ import annotations

import structlog
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

logger = structlog.get_logger()


class UserPreferences:
    """Per-user directory favorites and transcript read offsets.

    Persistence is delegated: the ``schedule_save`` callback (provided
    by SessionManager) triggers a debounced save after mutations.
    """

    def __init__(self, *, schedule_save: Callable[[], None]) -> None:
        self.user_dir_favorites: dict[int, dict[str, list[str]]] = {}
        self.user_window_offsets: dict[int, dict[str, int]] = {}
        self._schedule_save: Callable[[], None] = schedule_save

    def reset(self) -> None:
        """Clear all state. Used for test isolation."""
        self.user_dir_favorites.clear()
        self.user_window_offsets.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize preferences for state.json persistence."""
        return {
            "user_window_offsets": {
                str(uid): offsets for uid, offsets in self.user_window_offsets.items()
            },
            "user_dir_favorites": {
                str(uid): favs for uid, favs in self.user_dir_favorites.items()
            },
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore preferences from persisted data.

        Does NOT call ``_schedule_save`` — loading from disk must not
        trigger a write.
        """
        self.user_window_offsets = {
            int(uid): offsets
            for uid, offsets in data.get("user_window_offsets", {}).items()
        }
        self.user_dir_favorites = {
            int(uid): favs for uid, favs in data.get("user_dir_favorites", {}).items()
        }

    # ------------------------------------------------------------------
    # Directory favorites
    # ------------------------------------------------------------------

    def get_user_starred(self, user_id: int) -> list[str]:
        """Get starred directories for a user."""
        return list(self.user_dir_favorites.get(user_id, {}).get("starred", []))

    def get_user_mru(self, user_id: int) -> list[str]:
        """Get MRU directories for a user."""
        return list(self.user_dir_favorites.get(user_id, {}).get("mru", []))

    def update_user_mru(self, user_id: int, path: str) -> None:
        """Insert path at front of MRU list, dedupe, cap at 5."""
        resolved = str(Path(path).resolve())
        favs = self.user_dir_favorites.setdefault(user_id, {})
        mru: list[str] = favs.get("mru", [])
        mru = [resolved] + [p for p in mru if p != resolved]
        favs["mru"] = mru[:5]
        self._schedule_save()

    def toggle_user_star(self, user_id: int, path: str) -> bool:
        """Toggle a directory in/out of starred list. Returns True if now starred."""
        resolved = str(Path(path).resolve())
        favs = self.user_dir_favorites.setdefault(user_id, {})
        starred: list[str] = favs.get("starred", [])
        if resolved in starred:
            starred.remove(resolved)
            now_starred = False
        else:
            starred.append(resolved)
            now_starred = True
        favs["starred"] = starred
        self._schedule_save()
        return now_starred

    # ------------------------------------------------------------------
    # Read offsets
    # ------------------------------------------------------------------

    def get_user_window_offset(self, user_id: int, window_id: str) -> int | None:
        """Get the user's last read offset for a window.

        Returns None if no offset has been recorded (first time).
        """
        user_offsets = self.user_window_offsets.get(user_id)
        if user_offsets is None:
            return None
        return user_offsets.get(window_id)

    def update_user_window_offset(
        self, user_id: int, window_id: str, offset: int
    ) -> None:
        """Update the user's last read offset for a window."""
        if user_id not in self.user_window_offsets:
            self.user_window_offsets[user_id] = {}
        self.user_window_offsets[user_id][window_id] = offset
        self._schedule_save()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_stale_offsets(self, known_window_ids: set[str]) -> bool:
        """Remove user_window_offsets entries for unknown windows.

        Returns True if any changes were made.
        """
        changed = False
        empty_users: list[int] = []
        pruned = 0
        for uid, offsets in self.user_window_offsets.items():
            stale = [wid for wid in offsets if wid not in known_window_ids]
            for wid in stale:
                logger.debug("Pruning stale offset: user %d, window %s", uid, wid)
                del offsets[wid]
                changed = True
                pruned += 1
            if not offsets:
                empty_users.append(uid)
        for uid in empty_users:
            del self.user_window_offsets[uid]
            changed = True
        if pruned:
            logger.info("Pruned %d stale window offset(s)", pruned)
        if changed:
            self._schedule_save()
        return changed


_active_prefs: UserPreferences | None = None


def get_user_preferences() -> UserPreferences:
    """Return the SessionManager-owned UserPreferences.

    Raises:
        RuntimeError: when called before SessionManager has constructed
        and installed the preferences store.
    """
    if _active_prefs is None:
        raise RuntimeError(
            "UserPreferences not yet wired. "
            "Instantiate SessionManager() before accessing user_preferences."
        )
    return _active_prefs


def install_user_preferences(prefs: UserPreferences) -> None:
    """Install the SessionManager-owned preferences as the module-level singleton.

    Called once by ``SessionManager.__post_init__``. Replaces any
    previously installed instance (used by tests that build a fresh
    SessionManager).
    """
    global _active_prefs
    _active_prefs = prefs


class _UserPreferencesProxy:
    """Backward-compat module-level facade that resolves to the wired prefs.

    All attribute access delegates to the SessionManager-owned
    ``UserPreferences``. Raises ``RuntimeError`` if accessed before
    SessionManager has installed an instance.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(get_user_preferences(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_user_preferences(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(get_user_preferences(), name)

    def __repr__(self) -> str:
        if _active_prefs is None:
            return "<UserPreferencesProxy unwired>"
        return f"<UserPreferencesProxy → {_active_prefs!r}>"


user_preferences: UserPreferences = cast("UserPreferences", _UserPreferencesProxy())
