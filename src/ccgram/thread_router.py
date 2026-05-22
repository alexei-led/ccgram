"""Thread routing — Telegram topic to tmux window binding.

Maps Telegram topics (user_id + thread_id) to tmux windows (window_id)
bidirectionally.  Manages group chat IDs for multi-group forum topic
routing and display names for windows.

Key class: ThreadRouter. Persistence and window-state queries are
injected via the constructor — the router cannot be built without
explicit callbacks.

Module-level access: ``get_thread_router()`` returns the
SessionManager-owned instance (raises RuntimeError until SessionManager
has constructed the router). The legacy module attribute
``thread_router`` is a thin proxy that delegates to the same instance
for backward compat.

Key data:
  - thread_bindings  (user_id -> {thread_id -> window_id})
  - _window_to_thread (reverse index for O(1) inbound lookups)
  - group_chat_ids   (composite key -> chat_id)
  - window_display_names (window_id -> display name)
"""

from __future__ import annotations

import structlog
from collections.abc import Callable, Iterator
from typing import Any, cast

logger = structlog.get_logger()


class ThreadRouter:
    """Bidirectional mapping between Telegram topics and tmux windows.

    Owns thread_bindings, group_chat_ids, window_display_names, and
    the reverse index _window_to_thread.

    Persistence and window-state queries are injected via the
    constructor:

    * ``schedule_save``: triggers a debounced save after mutations.
    * ``has_window_state``: returns True when a window has tracked
      WindowState — used to decide whether a display name is still
      load-bearing during ``unbind_thread``.
    """

    def __init__(
        self,
        *,
        schedule_save: Callable[[], None],
        has_window_state: Callable[[str], bool],
    ) -> None:
        self.thread_bindings: dict[int, dict[int, str]] = {}
        # "user_id:thread_id" -> chat_id (supports multiple groups per user)
        self.group_chat_ids: dict[str, int] = {}
        # window_id -> display name (window_name)
        self.window_display_names: dict[str, str] = {}
        # Reverse index: (user_id, window_id) -> thread_id for O(1) lookups
        self._window_to_thread: dict[tuple[int, str], int] = {}
        self._schedule_save: Callable[[], None] = schedule_save
        self._has_window_state: Callable[[str], bool] = has_window_state

    def reset(self) -> None:
        """Clear all state.  Used for test isolation."""
        self.thread_bindings.clear()
        self.group_chat_ids.clear()
        self.window_display_names.clear()
        self._window_to_thread.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_reverse_index(self) -> None:
        """Rebuild _window_to_thread from thread_bindings."""
        self._window_to_thread = {}
        for uid, bindings in self.thread_bindings.items():
            for tid, wid in bindings.items():
                self._window_to_thread[(uid, wid)] = tid

    def _dedup_thread_bindings(self) -> None:
        """Enforce 1 window = 1 thread.  Keep highest thread_id per window."""
        for _uid, bindings in self.thread_bindings.items():
            window_threads: dict[str, list[int]] = {}
            for tid, wid in bindings.items():
                window_threads.setdefault(wid, []).append(tid)
            for wid, tids in window_threads.items():
                if len(tids) > 1:
                    keep = max(tids)
                    for tid in tids:
                        if tid != keep:
                            del bindings[tid]
                            logger.warning(
                                "Startup: removed duplicate binding "
                                "thread %d -> window %s (keeping %d)",
                                tid,
                                wid,
                                keep,
                            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing state for state.json persistence."""
        return {
            "thread_bindings": {
                str(uid): {str(tid): wid for tid, wid in bindings.items()}
                for uid, bindings in self.thread_bindings.items()
            },
            "group_chat_ids": self.group_chat_ids,
            "window_display_names": self.window_display_names,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore routing state from persisted data.

        Does NOT call ``_schedule_save`` — loading from disk must not
        trigger a write.
        """
        self.thread_bindings = {
            int(uid): {int(tid): wid for tid, wid in bindings.items()}
            for uid, bindings in data.get("thread_bindings", {}).items()
        }
        self.group_chat_ids = data.get("group_chat_ids", {})
        self.window_display_names = data.get("window_display_names", {})
        self._dedup_thread_bindings()
        self._rebuild_reverse_index()

    # ------------------------------------------------------------------
    # Thread binding operations
    # ------------------------------------------------------------------

    def bind_thread(
        self, user_id: int, thread_id: int, window_id: str, window_name: str = ""
    ) -> None:
        """Bind a Telegram topic thread to a tmux window.

        Enforces 1 topic = 1 window: if another thread is already bound to
        the same window_id, that stale binding is removed first.
        """
        if user_id not in self.thread_bindings:
            self.thread_bindings[user_id] = {}

        # Enforce 1:1 — unbind any OTHER thread pointing to this window
        stale = [
            tid
            for tid, wid in self.thread_bindings[user_id].items()
            if wid == window_id and tid != thread_id
        ]
        for tid in stale:
            del self.thread_bindings[user_id][tid]
            logger.info(
                "Evicted stale binding: thread %d -> window_id %s "
                "(replaced by thread %d)",
                tid,
                window_id,
                thread_id,
            )

        # Clean up stale reverse index if this thread was previously bound elsewhere
        old_window = self.thread_bindings[user_id].get(thread_id)
        if old_window is not None and old_window != window_id:
            self._window_to_thread.pop((user_id, old_window), None)

        self.thread_bindings[user_id][thread_id] = window_id
        self._window_to_thread[(user_id, window_id)] = thread_id
        if window_name:
            self.window_display_names[window_id] = window_name
        self._schedule_save()
        display = window_name or self.get_display_name(window_id)
        logger.info(
            "Bound thread %d -> window_id %s (%s) for user %d",
            thread_id,
            window_id,
            display,
            user_id,
        )

    def unbind_thread(self, user_id: int, thread_id: int) -> str | None:
        """Remove a thread binding.  Returns the previously bound window_id.

        Cleans up the reverse index and group_chat_id.  Does NOT touch
        display names — the caller (SessionManager) handles display-name
        lifecycle because it requires window_states knowledge.
        """
        bindings = self.thread_bindings.get(user_id)
        if not bindings or thread_id not in bindings:
            return None
        window_id = bindings.pop(thread_id)
        self._window_to_thread.pop((user_id, window_id), None)
        if not bindings:
            del self.thread_bindings[user_id]
        logger.info(
            "Unbound thread %d (was %s) for user %d",
            thread_id,
            window_id,
            user_id,
        )

        # Clean up group_chat_id for the unbound thread
        chat_key = f"{user_id}:{thread_id}"
        self.group_chat_ids.pop(chat_key, None)

        # Clean up orphaned display name if nothing references this window
        still_bound = any(
            wid == window_id
            for ub in self.thread_bindings.values()
            for wid in ub.values()
        )
        if not still_bound and not self._has_window_state(window_id):
            self.window_display_names.pop(window_id, None)

        self._schedule_save()
        return window_id

    def get_window_for_thread(self, user_id: int, thread_id: int) -> str | None:
        """Look up the window_id bound to a thread."""
        bindings = self.thread_bindings.get(user_id)
        if not bindings:
            return None
        return bindings.get(thread_id)

    def get_thread_for_window(self, user_id: int, window_id: str) -> int | None:
        """Reverse lookup: get thread_id for a window (O(1) via reverse index)."""
        return self._window_to_thread.get((user_id, window_id))

    def get_all_thread_windows(self, user_id: int) -> dict[int, str]:
        """Get all thread bindings for a user."""
        return dict(self.thread_bindings.get(user_id, {}))

    def resolve_window_for_thread(
        self,
        user_id: int,
        thread_id: int | None,
    ) -> str | None:
        """Resolve the tmux window_id for a user's thread.

        Returns None if thread_id is None or the thread is not bound.
        """
        if thread_id is None:
            return None
        return self.get_window_for_thread(user_id, thread_id)

    def has_window(self, window_id: str) -> bool:
        """Check if any user has a binding to this window_id."""
        return any(wid == window_id for (_, wid) in self._window_to_thread)

    def iter_thread_bindings(self) -> Iterator[tuple[int, int, str]]:
        """Iterate all thread bindings as (user_id, thread_id, window_id)."""
        for user_id, bindings in self.thread_bindings.items():
            for thread_id, window_id in bindings.items():
                yield user_id, thread_id, window_id

    # ------------------------------------------------------------------
    # Group chat ID management
    # ------------------------------------------------------------------

    def set_group_chat_id(self, user_id: int, thread_id: int, chat_id: int) -> None:
        """Store the group chat ID for a user's thread.

        Uses composite key ``user_id:thread_id`` to support multiple
        groups per user.
        """
        key = f"{user_id}:{thread_id}"
        if self.group_chat_ids.get(key) != chat_id:
            self.group_chat_ids[key] = chat_id
            self._schedule_save()
            logger.debug(
                "Stored group chat_id %d for user %d, thread %d",
                chat_id,
                user_id,
                thread_id,
            )

    def resolve_chat_id(self, user_id: int, thread_id: int | None = None) -> int:
        """Resolve the chat_id for sending messages.

        In forum topics (thread_id is set), returns the stored group chat_id
        for that specific thread (user_id:thread_id).
        Falls back to user_id for direct messages or if no group_id stored.
        """
        if thread_id is not None:
            key = f"{user_id}:{thread_id}"
            group_id = self.group_chat_ids.get(key)
            if group_id is not None:
                return group_id
        return user_id

    def get_window_for_chat_thread(self, chat_id: int, thread_id: int) -> str | None:
        """Resolve window_id for a specific Telegram chat/thread pair."""
        for user_id, bindings in self.thread_bindings.items():
            window_id = bindings.get(thread_id)
            if not window_id:
                continue
            key = f"{user_id}:{thread_id}"
            resolved_chat = self.group_chat_ids.get(key, user_id)
            if resolved_chat == chat_id:
                return window_id
        return None

    # ------------------------------------------------------------------
    # Display name management
    # ------------------------------------------------------------------

    def get_display_name(self, window_id: str) -> str:
        """Get display name for a window_id, fallback to window_id itself."""
        return self.window_display_names.get(window_id, window_id)

    def pop_display_name(self, window_id: str) -> str:
        """Remove and return display name for window_id. Falls back to window_id."""
        if window_id not in self.window_display_names:
            return window_id
        name = self.window_display_names.pop(window_id)
        self._schedule_save()
        return name

    def set_display_name(self, window_id: str, window_name: str) -> None:
        """Update display name for a window_id."""
        if self.window_display_names.get(window_id) != window_name:
            self.window_display_names[window_id] = window_name
            self._schedule_save()

    def sync_display_names(self, live_windows: list[tuple[str, str]]) -> bool:
        """Sync display names from live tmux windows.  Returns True if changed.

        Saves state internally when changes are detected.
        """
        changed = False
        for window_id, window_name in live_windows:
            old = self.window_display_names.get(window_id)
            if old and old != window_name:
                self.window_display_names[window_id] = window_name
                changed = True
                logger.debug(
                    "Synced display name: %s %s → %s", window_id, old, window_name
                )
        if changed:
            self._schedule_save()
        return changed


_active_router: ThreadRouter | None = None


def get_thread_router() -> ThreadRouter:
    """Return the SessionManager-owned ThreadRouter.

    Raises:
        RuntimeError: when called before SessionManager has constructed
        and installed the router.
    """
    if _active_router is None:
        raise RuntimeError(
            "ThreadRouter not yet wired. "
            "Instantiate SessionManager() before accessing thread_router."
        )
    return _active_router


def install_thread_router(router: ThreadRouter) -> None:
    """Install the SessionManager-owned router as the module-level singleton.

    Called once by ``SessionManager.__post_init__``. Replaces any
    previously installed router (used by tests that build a fresh
    SessionManager).
    """
    global _active_router
    _active_router = router


class _ThreadRouterProxy:
    """Backward-compat module-level facade that resolves to the wired router.

    All attribute access delegates to the SessionManager-owned
    ``ThreadRouter``. Raises ``RuntimeError`` if accessed before
    SessionManager has installed an instance.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(get_thread_router(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_thread_router(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(get_thread_router(), name)

    def __repr__(self) -> str:
        if _active_router is None:
            return "<ThreadRouterProxy unwired>"
        return f"<ThreadRouterProxy → {_active_router!r}>"


thread_router: ThreadRouter = cast("ThreadRouter", _ThreadRouterProxy())
