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

import contextlib
import contextvars
from dataclasses import dataclass
import structlog
from collections.abc import Callable, Iterator
from typing import Any, cast

logger = structlog.get_logger()

_RETIRED_TOPIC_LIMIT = 100


@dataclass(frozen=True)
class RetiredTopic:
    """A known former forum-topic binding retained for Sync cleanup.

    This is deliberately local evidence only: a record means ccgram previously
    owned this exact chat/thread binding. It is not evidence that arbitrary
    Telegram topics can be enumerated or safely removed.
    """

    user_id: int
    chat_id: int
    thread_id: int
    reason: str
    cleanup_eligible: bool
    sequence: int


_active_chat_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "active_thread_chat_id", default=None
)


@contextlib.contextmanager
def chat_scope(chat_id: int | None):
    token = _active_chat_id.set(chat_id)
    try:
        yield
    finally:
        _active_chat_id.reset(token)


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
        default_group_id: int | None = None,
    ) -> None:
        self.thread_bindings: dict[int, dict[int, str]] = {}
        # Chat-scoped bindings preserve Telegram's chat-local thread identity.
        self.chat_thread_bindings: dict[tuple[int, int, int], str] = {}
        # "user_id:thread_id" -> chat_id for legacy/direct-message bindings.
        self.group_chat_ids: dict[str, int] = {}
        self.default_group_id = default_group_id
        # window_id -> display name (window_name)
        self.window_display_names: dict[str, str] = {}
        # Reverse index: (user_id, window_id) -> thread_id for O(1) lookups
        self._window_to_thread: dict[tuple[int, str], int] = {}
        self._chat_window_to_thread: dict[tuple[int, int, str], int] = {}
        self._retired_topics: list[RetiredTopic] = []
        self._next_retired_sequence = 1
        self._schedule_save: Callable[[], None] = schedule_save
        self._has_window_state: Callable[[str], bool] = has_window_state

    def reset(self) -> None:
        """Clear all state.  Used for test isolation."""
        self.thread_bindings.clear()
        self.group_chat_ids.clear()
        self.chat_thread_bindings.clear()
        self.window_display_names.clear()
        self._window_to_thread.clear()
        self._chat_window_to_thread.clear()
        self._retired_topics.clear()
        self._next_retired_sequence = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_reverse_index(self) -> None:
        """Rebuild _window_to_thread from thread_bindings."""
        self._window_to_thread = {}
        self._chat_window_to_thread = {}
        for uid, bindings in self.thread_bindings.items():
            for tid, wid in bindings.items():
                self._window_to_thread[(uid, wid)] = tid
        for (uid, chat_id, _tid), wid in self.chat_thread_bindings.items():
            self._chat_window_to_thread[(uid, chat_id, wid)] = _tid

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
            "chat_thread_bindings": {
                f"{uid}:{chat_id}:{tid}": wid
                for (uid, chat_id, tid), wid in self.chat_thread_bindings.items()
            },
            "window_display_names": self.window_display_names,
            "retired_topics": [
                {
                    "user_id": topic.user_id,
                    "chat_id": topic.chat_id,
                    "thread_id": topic.thread_id,
                    "reason": topic.reason,
                    "cleanup_eligible": topic.cleanup_eligible,
                    "sequence": topic.sequence,
                }
                for topic in self._retired_topics
            ],
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
        self.chat_thread_bindings = {}
        for key, wid in data.get("chat_thread_bindings", {}).items():
            uid, chat_id, tid = (int(part) for part in key.split(":", 2))
            self.chat_thread_bindings[(uid, chat_id, tid)] = wid
        self.window_display_names = data.get("window_display_names", {})
        self._retired_topics = self._load_retired_topics(data.get("retired_topics", []))
        self._next_retired_sequence = (
            max((topic.sequence for topic in self._retired_topics), default=0) + 1
        )
        self._dedup_thread_bindings()
        self._rebuild_reverse_index()
        for _user_id, chat_id, thread_id, _window_id in self.iter_thread_bindings_with_chat():
            self._restore_active_topic(chat_id, thread_id)

    @staticmethod
    def _load_retired_topics(raw_topics: Any) -> list[RetiredTopic]:
        """Load a bounded, validated retired-topic registry from persisted state."""
        if not isinstance(raw_topics, list):
            return []
        loaded: list[RetiredTopic] = []
        for raw in raw_topics[-_RETIRED_TOPIC_LIMIT:]:
            if not isinstance(raw, dict):
                continue
            try:
                reason = raw["reason"]
                cleanup_eligible = raw["cleanup_eligible"]
                topic = RetiredTopic(
                    user_id=int(raw["user_id"]),
                    chat_id=int(raw["chat_id"]),
                    thread_id=int(raw["thread_id"]),
                    reason=reason,
                    cleanup_eligible=cleanup_eligible,
                    sequence=int(raw["sequence"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not isinstance(reason, str)
                or not reason
                or not isinstance(cleanup_eligible, bool)
                or topic.chat_id == topic.user_id
                or topic.thread_id <= 0
                or topic.sequence <= 0
            ):
                continue
            loaded = [
                existing
                for existing in loaded
                if (existing.chat_id, existing.thread_id)
                != (topic.chat_id, topic.thread_id)
            ]
            loaded.append(topic)
        return loaded

    # ------------------------------------------------------------------
    # Retired topic registry
    # ------------------------------------------------------------------

    def iter_retired_topics(self) -> Iterator[RetiredTopic]:
        """Yield locally known retired topics in oldest-first retention order."""
        return iter(tuple(self._retired_topics))

    def discard_retired_topic(self, topic: RetiredTopic) -> bool:
        """Remove exactly *topic* after a terminal Telegram API outcome."""
        try:
            self._retired_topics.remove(topic)
        except ValueError:
            return False
        self._schedule_save()
        return True

    def _retire_topic(
        self,
        user_id: int,
        chat_id: int | None,
        thread_id: int,
        *,
        reason: str,
        cleanup_eligible: bool,
    ) -> None:
        """Remember a known local forum topic, never an inferred Telegram topic."""
        if chat_id is None or chat_id == user_id:
            return
        self._retired_topics = [
            topic
            for topic in self._retired_topics
            if (topic.chat_id, topic.thread_id) != (chat_id, thread_id)
        ]
        self._retired_topics.append(
            RetiredTopic(
                user_id=user_id,
                chat_id=chat_id,
                thread_id=thread_id,
                reason=reason,
                cleanup_eligible=cleanup_eligible,
                sequence=self._next_retired_sequence,
            )
        )
        self._next_retired_sequence += 1
        self._retired_topics = self._retired_topics[-_RETIRED_TOPIC_LIMIT:]

    def _restore_active_topic(self, chat_id: int | None, thread_id: int) -> None:
        """Forget a retired record when the same chat/topic is bound again."""
        if chat_id is None:
            return
        self._retired_topics = [
            topic
            for topic in self._retired_topics
            if (topic.chat_id, topic.thread_id) != (chat_id, thread_id)
        ]

    # ------------------------------------------------------------------
    # Thread binding operations
    # ------------------------------------------------------------------

    def bind_thread(
        self,
        user_id: int,
        thread_id: int,
        window_id: str,
        window_name: str = "",
        chat_id: int | None = None,
    ) -> None:
        """Bind a topic, using chat-scoped identity when ``chat_id`` is known."""
        if chat_id is not None:
            key = (user_id, chat_id, thread_id)
            old_thread = self._chat_window_to_thread.pop(
                (user_id, chat_id, window_id), None
            )
            if old_thread is not None and old_thread != thread_id:
                self.chat_thread_bindings.pop((user_id, chat_id, old_thread), None)
            stale = [
                candidate
                for candidate, wid in self.chat_thread_bindings.items()
                if candidate[0] == user_id
                and candidate[1] == chat_id
                and wid == window_id
                and candidate != key
            ]
            for candidate in stale:
                self.chat_thread_bindings.pop(candidate, None)
                self._chat_window_to_thread.pop(
                    (user_id, candidate[1], window_id), None
                )
            self.chat_thread_bindings[key] = window_id
            self._chat_window_to_thread[(user_id, chat_id, window_id)] = thread_id
        else:
            if user_id not in self.thread_bindings:
                self.thread_bindings[user_id] = {}
            stale = [
                tid
                for tid, wid in self.thread_bindings[user_id].items()
                if wid == window_id and tid != thread_id
            ]
            for tid in stale:
                del self.thread_bindings[user_id][tid]
            old_window = self.thread_bindings[user_id].get(thread_id)
            if old_window is not None and old_window != window_id:
                self._window_to_thread.pop((user_id, old_window), None)
            self.thread_bindings[user_id][thread_id] = window_id
            self._window_to_thread[(user_id, window_id)] = thread_id
        if window_name:
            self.window_display_names[window_id] = window_name
        self._restore_active_topic(chat_id, thread_id)
        self._schedule_save()

    def unbind_thread(
        self,
        user_id: int,
        thread_id: int,
        chat_id: int | None = None,
        *,
        retirement_reason: str = "keep_remote",
        cleanup_eligible: bool = False,
    ) -> str | None:
        """Remove a thread binding.  Returns the previously bound window_id.

        Cleans up the reverse index and group_chat_id.  Does NOT touch
        display names — the caller (SessionManager) handles display-name
        lifecycle because it requires window_states knowledge.
        """
        known_chat_id: int | None = chat_id
        if chat_id is not None:
            key = (user_id, chat_id, thread_id)
            window_id = self.chat_thread_bindings.pop(key, None)
            if window_id is None:
                return None
            self._chat_window_to_thread.pop((user_id, chat_id, window_id), None)
            self.group_chat_ids.pop(f"{user_id}:{thread_id}:{chat_id}", None)
        else:
            bindings = self.thread_bindings.get(user_id)
            if not bindings or thread_id not in bindings:
                candidates = [
                    (key, wid)
                    for key, wid in self.chat_thread_bindings.items()
                    if key[0] == user_id and key[2] == thread_id
                ]
                if len(candidates) != 1:
                    return None
                key, window_id = candidates[0]
                known_chat_id = key[1]
                self.chat_thread_bindings.pop(key, None)
                self._chat_window_to_thread.pop((user_id, key[1], window_id), None)
                self.group_chat_ids.pop(f"{user_id}:{thread_id}:{key[1]}", None)
            else:
                window_id = bindings.pop(thread_id)
                known_chat_id = self.group_chat_ids.get(f"{user_id}:{thread_id}")
                self._window_to_thread.pop((user_id, window_id), None)
                if not bindings:
                    del self.thread_bindings[user_id]
        if chat_id is None and user_id in self.thread_bindings:
            bindings = self.thread_bindings[user_id]
            if thread_id not in bindings:
                # Chat-scoped binding was removed above.
                bindings = None
            if bindings is not None and not bindings:
                del self.thread_bindings[user_id]
        logger.info(
            "Unbound thread %d (was %s) for user %d",
            thread_id,
            window_id,
            user_id,
        )

        self._retire_topic(
            user_id,
            known_chat_id,
            thread_id,
            reason=retirement_reason,
            cleanup_eligible=cleanup_eligible,
        )

        # Clean up group_chat_id for the unbound thread
        self.group_chat_ids.pop(f"{user_id}:{thread_id}", None)

        # Clean up orphaned display name if nothing references this window
        still_bound = (
            any(
                wid == window_id
                for ub in self.thread_bindings.values()
                for wid in ub.values()
            )
            or window_id in self.chat_thread_bindings.values()
        )
        if not still_bound and not self._has_window_state(window_id):
            self.window_display_names.pop(window_id, None)

        self._schedule_save()
        return window_id

    def get_window_for_thread(
        self, user_id: int, thread_id: int, chat_id: int | None = None
    ) -> str | None:
        """Look up a window, disambiguating chat-scoped Telegram threads."""
        if chat_id is not None:
            return self.chat_thread_bindings.get((user_id, chat_id, thread_id))
        bindings = self.thread_bindings.get(user_id, {})
        matches = {
            wid
            for (uid, _chat, tid), wid in self.chat_thread_bindings.items()
            if uid == user_id and tid == thread_id
        }
        legacy = bindings.get(thread_id)
        if legacy is not None:
            matches.add(legacy)
        return next(iter(matches)) if len(matches) == 1 else None

    def get_thread_for_window(
        self, user_id: int, window_id: str, chat_id: int | None = None
    ) -> int | None:
        """Reverse lookup for legacy or chat-scoped bindings."""
        if chat_id is not None:
            return self._chat_window_to_thread.get((user_id, chat_id, window_id))
        legacy = self._window_to_thread.get((user_id, window_id))
        if legacy is not None:
            return legacy
        matches = [
            tid
            for (uid, _chat, wid), tid in self._chat_window_to_thread.items()
            if uid == user_id and wid == window_id
        ]
        return matches[0] if len(matches) == 1 else None

    def get_all_thread_windows(self, user_id: int) -> dict[int, str]:
        """Get all thread bindings for a user, including chat-scoped ones."""
        result = dict(self.thread_bindings.get(user_id, {}))
        for (uid, _chat_id, thread_id), window_id in self.chat_thread_bindings.items():
            if uid == user_id:
                result[thread_id] = window_id
        return result

    def resolve_window_for_thread(
        self,
        user_id: int,
        thread_id: int | None,
        chat_id: int | None = None,
    ) -> str | None:
        """Resolve the tmux window_id for a user's thread.

        Returns None if thread_id is None or the thread is not bound.
        """
        if thread_id is None:
            return None
        return self.get_window_for_thread(user_id, thread_id, chat_id)

    def has_window(self, window_id: str) -> bool:
        """Check if any user has a binding to this window_id."""
        return (
            any(wid == window_id for (_, wid) in self._window_to_thread)
            or window_id in self.chat_thread_bindings.values()
        )

    def has_window_for_user(self, user_id: int, window_id: str) -> bool:
        return (
            any(
                uid == user_id and wid == window_id
                for (uid, _chat, _thread), wid in self.chat_thread_bindings.items()
            )
            or self._window_to_thread.get((user_id, window_id)) is not None
        )

    def iter_thread_bindings_with_chat(
        self,
    ) -> Iterator[tuple[int, int | None, int, str]]:
        """Iterate bindings with chat identity when available."""
        for user_id, bindings in self.thread_bindings.items():
            for thread_id, window_id in bindings.items():
                yield (
                    user_id,
                    self.group_chat_ids.get(f"{user_id}:{thread_id}"),
                    thread_id,
                    window_id,
                )
        for (
            user_id,
            chat_id,
            thread_id,
        ), window_id in self.chat_thread_bindings.items():
            yield user_id, chat_id, thread_id, window_id

    def iter_thread_bindings(self) -> Iterator[tuple[int, int, str]]:
        """Iterate all thread bindings as (user_id, thread_id, window_id)."""
        for user_id, bindings in self.thread_bindings.items():
            for thread_id, window_id in bindings.items():
                yield user_id, thread_id, window_id
        for (
            user_id,
            _chat_id,
            thread_id,
        ), window_id in self.chat_thread_bindings.items():
            yield user_id, thread_id, window_id

    def all_bound_window_ids(self) -> set[str]:
        return {window_id for _, _, window_id in self.iter_thread_bindings()}

    # ------------------------------------------------------------------
    # Group chat ID management
    # ------------------------------------------------------------------

    def set_group_chat_id(self, user_id: int, thread_id: int, chat_id: int) -> None:
        """Store the group chat ID and promote the binding to chat scope."""
        bindings = self.thread_bindings.get(user_id)
        if bindings and thread_id in bindings:
            window_id = bindings.pop(thread_id)
            if not bindings:
                self.thread_bindings.pop(user_id, None)
            self._window_to_thread.pop((user_id, window_id), None)
            self.chat_thread_bindings[(user_id, chat_id, thread_id)] = window_id
            self._chat_window_to_thread[(user_id, chat_id, window_id)] = thread_id
            self._restore_active_topic(chat_id, thread_id)
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
        Falls back to the configured group for an unbound topic.
        Falls back to user_id for direct messages.
        """
        active_chat_id = _active_chat_id.get()
        if (
            active_chat_id is not None
            and thread_id is not None
            and (user_id, active_chat_id, thread_id) in self.chat_thread_bindings
        ):
            return active_chat_id
        if thread_id is not None:
            key = f"{user_id}:{thread_id}"
            group_id = self.group_chat_ids.get(key)
            if group_id is not None:
                return group_id
            chats = {
                chat_id
                for (uid, chat_id, tid) in self.chat_thread_bindings
                if uid == user_id and tid == thread_id
            }
            if len(chats) == 1:
                return next(iter(chats))
            if self.default_group_id is not None:
                return self.default_group_id
        return user_id

    def get_window_for_chat_thread(self, chat_id: int, thread_id: int) -> str | None:
        """Resolve window_id for a specific Telegram chat/thread pair."""
        scoped = [
            wid
            for (
                user_id,
                bound_chat,
                bound_thread,
            ), wid in self.chat_thread_bindings.items()
            if bound_chat == chat_id and bound_thread == thread_id
        ]
        if len(scoped) == 1:
            return scoped[0]
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
