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
from dataclasses import dataclass, field, replace
import math
import structlog
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal, cast
import uuid

from .multiplexer.base import canonical_window_id

logger = structlog.get_logger()

_RETIRED_TOPIC_LIMIT = 100
_TOPIC_PROVISIONING_KINDS = {
    "topic_for_target",
    "target_for_topic",
    "replacement",
}

TopicProvisioningKind = Literal[
    "topic_for_target",
    "target_for_topic",
    "replacement",
]


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
    retry_at: float = 0.0
    closed: bool = False
    target_id: str | None = None


@dataclass(frozen=True)
class TopicProvisioning:
    """Durable evidence for one topic/target creation transaction.

    A record remains present until the caller can prove that creation either
    committed or failed safely.  ``created_at`` is diagnostic metadata only;
    it must never be used as an expiry or deletion decision.
    """

    claim_id: str
    user_id: int
    chat_id: int
    thread_id: int | None
    target_id: str | None
    previous_target_id: str | None
    kind: TopicProvisioningKind
    uncertain: bool = False
    created_at: float = field(default_factory=time.time)


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
        # Private chats observed to carry Telegram's regular topic-message
        # metadata. A positive chat ID alone is not enough to infer this.
        self.private_topic_chats: set[int] = set()
        # "user_id:thread_id" -> chat_id for legacy and chat-scoped bindings.
        self.group_chat_ids: dict[str, int] = {}
        self.default_group_id = default_group_id
        # window_id -> display name (window_name)
        self.window_display_names: dict[str, str] = {}
        # Reverse index: (user_id, window_id) -> thread_id for O(1) lookups
        self._window_to_thread: dict[tuple[int, str], int] = {}
        self._chat_window_to_thread: dict[tuple[int, int, str], int] = {}
        self._retired_topics: list[RetiredTopic] = []
        self._topic_deletion_claims: set[tuple[int, int]] = set()
        self._topic_deletion_targets: dict[tuple[int, int], str | None] = {}
        self._topic_provisionings: dict[str, TopicProvisioning] = {}
        # Ownership is intentionally process-local.  The durable records above
        # survive restart so recovery can resolve an interrupted flow, while
        # this set identifies claims still owned by the current flow.
        self._owned_provisioning_claims: set[str] = set()
        self._next_retired_sequence = 1
        self._schedule_save: Callable[[], None] = schedule_save
        self._has_window_state: Callable[[str], bool] = has_window_state

    def reset(self) -> None:
        """Clear all state.  Used for test isolation."""
        self.thread_bindings.clear()
        self.group_chat_ids.clear()
        self.chat_thread_bindings.clear()
        self.private_topic_chats.clear()
        self.window_display_names.clear()
        self._window_to_thread.clear()
        self._chat_window_to_thread.clear()
        self._retired_topics.clear()
        self._topic_deletion_claims.clear()
        self._topic_deletion_targets.clear()
        self._topic_provisionings.clear()
        self._owned_provisioning_claims.clear()
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

    def _remove_group_routing_metadata(self, user_id: int, thread_id: int) -> bool:
        """Remove routing metadata belonging to one evicted topic claim."""
        prefix = f"{user_id}:{thread_id}"
        stale_keys = [
            key
            for key in self.group_chat_ids
            if key == prefix or key.startswith(f"{prefix}:")
        ]
        for key in stale_keys:
            del self.group_chat_ids[key]
        return bool(stale_keys)

    def _normalize_group_backed_bindings(self) -> bool:
        """Promote legacy bindings with a persisted chat ID to chat scope.

        ``thread_bindings`` predates chat-local topic identity.  A matching
        ``group_chat_ids`` entry is sufficient evidence to promote one of
        those rows; an already chat-scoped row for the same topic wins over
        the older representation.  The routing metadata becomes redundant
        after promotion and must not survive as a stale fallback route.
        """
        changed = False
        for user_id, bindings in list(self.thread_bindings.items()):
            for thread_id, window_id in list(bindings.items()):
                metadata_key = f"{user_id}:{thread_id}"
                chat_id = self.group_chat_ids.get(metadata_key)
                if not isinstance(chat_id, int) or isinstance(chat_id, bool):
                    continue
                del bindings[thread_id]
                scoped_key = (user_id, chat_id, thread_id)
                # A natively scoped row is the newer, unambiguous record for
                # this exact topic, so never overwrite it based on JSON order.
                self.chat_thread_bindings.setdefault(scoped_key, window_id)
                self._remove_group_routing_metadata(user_id, thread_id)
                changed = True
            if not bindings:
                del self.thread_bindings[user_id]
        return changed

    def _dedup_thread_bindings(self) -> bool:
        """Enforce 1 window = 1 thread.  Keep highest thread_id per window."""
        changed = False
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
                            changed = True
                            logger.warning(
                                "Startup: removed duplicate binding "
                                "thread %d -> window %s (keeping %d)",
                                tid,
                                wid,
                                keep,
                            )

        return self._dedup_chat_thread_bindings() or changed

    def _dedup_chat_thread_bindings(self) -> bool:
        """Keep one deterministic binding per chat and window."""
        changed = False
        window_bindings: dict[tuple[int, str], list[tuple[int, int, int]]] = {}
        for key, wid in self.chat_thread_bindings.items():
            window_bindings.setdefault((key[1], wid), []).append(key)
        for (chat_id, wid), keys in window_bindings.items():
            if len(keys) <= 1:
                continue
            keep = max(keys, key=lambda key: (key[2], key[0]))
            for key in keys:
                if key == keep:
                    continue
                del self.chat_thread_bindings[key]
                self._remove_group_routing_metadata(key[0], key[2])
                changed = True
                logger.warning(
                    "Startup: removed duplicate binding chat %d thread %d -> "
                    "window %s (keeping thread %d)",
                    chat_id,
                    key[2],
                    wid,
                    keep[2],
                )
        return changed

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
            "private_topic_chats": sorted(self.private_topic_chats),
            "window_display_names": self.window_display_names,
            "retired_topics": [
                {
                    "user_id": topic.user_id,
                    "chat_id": topic.chat_id,
                    "thread_id": topic.thread_id,
                    "reason": topic.reason,
                    "cleanup_eligible": topic.cleanup_eligible,
                    "sequence": topic.sequence,
                    "retry_at": topic.retry_at,
                    "closed": topic.closed,
                    "target_id": topic.target_id,
                }
                for topic in self._retired_topics
            ],
            "topic_provisioning": [
                {
                    "claim_id": claim.claim_id,
                    "user_id": claim.user_id,
                    "chat_id": claim.chat_id,
                    "thread_id": claim.thread_id,
                    "target_id": claim.target_id,
                    "previous_target_id": claim.previous_target_id,
                    "kind": claim.kind,
                    "uncertain": claim.uncertain,
                    "created_at": claim.created_at,
                }
                for claim in self._topic_provisionings.values()
            ],
        }

    def from_dict(self, data: dict[str, Any]) -> bool:
        """Restore routing state and return whether persisted routing was repaired.

        Loading itself does not schedule persistence.  The owning
        ``SessionManager`` saves once when this method reports a normalization
        or de-duplication repair.
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
        raw_private_chats = data.get("private_topic_chats", [])
        self.private_topic_chats = {
            chat_id for chat_id in raw_private_chats if isinstance(chat_id, int)
        }
        # Migrate short-lived state written by the previous direct-message
        # topic implementation without retaining its incorrect API shape.
        for key in data.get("direct_message_topics", []):
            try:
                chat_id, _thread_id = (int(part) for part in key.split(":", 1))
            except AttributeError, TypeError, ValueError:
                continue
            self.private_topic_chats.add(chat_id)
        self.window_display_names = data.get("window_display_names", {})
        self._retired_topics = self._load_retired_topics(data.get("retired_topics", []))
        raw_provisioning = data.get("topic_provisioning", [])
        loaded_provisioning = self._load_topic_provisionings(raw_provisioning)
        self._topic_provisionings = {
            claim.claim_id: claim for claim in loaded_provisioning
        }
        # A restart cannot safely claim that an old flow is still running.
        # Keep its durable record for reconciliation, but clear process-local
        # ownership so recovery can handle it explicitly.
        self._owned_provisioning_claims.clear()
        self._next_retired_sequence = (
            max((topic.sequence for topic in self._retired_topics), default=0) + 1
        )
        repaired = self._normalize_group_backed_bindings()
        repaired = self._dedup_thread_bindings() or repaired
        if isinstance(raw_provisioning, list):
            repaired = len(loaded_provisioning) != len(raw_provisioning) or repaired
        elif raw_provisioning:
            repaired = True
        self._rebuild_reverse_index()
        for (
            _user_id,
            chat_id,
            thread_id,
            _window_id,
        ) in self.iter_thread_bindings_with_chat():
            self._restore_active_topic(chat_id, thread_id)
        return repaired

    @staticmethod
    def _load_topic_provisionings(raw_claims: Any) -> list[TopicProvisioning]:
        """Load validated provisioning records without age-based truncation."""
        if not isinstance(raw_claims, list):
            return []

        loaded: dict[str, TopicProvisioning] = {}
        for raw in raw_claims:
            if not isinstance(raw, dict):
                continue
            try:
                raw_claim_id = raw["claim_id"]
                claim_id = str(uuid.UUID(raw_claim_id))
                user_id = raw["user_id"]
                chat_id = raw["chat_id"]
                thread_id = raw.get("thread_id")
                target_id = raw.get("target_id")
                previous_target_id = raw.get("previous_target_id")
                kind = raw["kind"]
                uncertain = raw.get("uncertain", False)
                created_at = raw.get("created_at", 0.0)
            except KeyError, AttributeError, TypeError, ValueError:
                continue

            if (
                not isinstance(user_id, int)
                or isinstance(user_id, bool)
                or not isinstance(chat_id, int)
                or isinstance(chat_id, bool)
                or (thread_id is not None and not isinstance(thread_id, int))
                or isinstance(thread_id, bool)
                or (isinstance(thread_id, int) and thread_id <= 0)
                or (target_id is not None and not isinstance(target_id, str))
                or (isinstance(target_id, str) and not target_id)
                or (
                    previous_target_id is not None
                    and not isinstance(previous_target_id, str)
                )
                or (isinstance(previous_target_id, str) and not previous_target_id)
                or not isinstance(kind, str)
                or kind not in _TOPIC_PROVISIONING_KINDS
                or not isinstance(uncertain, bool)
                or isinstance(created_at, bool)
                or not isinstance(created_at, (int, float))
            ):
                continue

            created_at_value = float(created_at)
            if not math.isfinite(created_at_value):
                continue
            loaded[claim_id] = TopicProvisioning(
                claim_id=claim_id,
                user_id=user_id,
                chat_id=chat_id,
                thread_id=thread_id,
                target_id=target_id,
                previous_target_id=previous_target_id,
                kind=cast(TopicProvisioningKind, kind),
                uncertain=uncertain,
                created_at=created_at_value,
            )
        return list(loaded.values())

    @staticmethod
    def _load_retired_topics(raw_topics: Any) -> list[RetiredTopic]:
        """Load a bounded, validated retired-topic registry from persisted state."""
        if not isinstance(raw_topics, list):
            return []
        loaded: list[RetiredTopic] = []
        for raw in raw_topics:
            if not isinstance(raw, dict):
                continue
            try:
                reason = raw["reason"]
                cleanup_eligible = raw["cleanup_eligible"]
                raw_retry_at = raw.get("retry_at", 0.0)
                closed = raw.get("closed", False)
                raw_target_id = raw.get("target_id")
                if isinstance(raw_retry_at, bool) or not isinstance(
                    raw_retry_at, (int, float)
                ):
                    continue
                if raw_target_id is not None and (
                    not isinstance(raw_target_id, str) or not raw_target_id
                ):
                    continue
                retry_at = float(raw_retry_at)
                topic = RetiredTopic(
                    user_id=int(raw["user_id"]),
                    chat_id=int(raw["chat_id"]),
                    thread_id=int(raw["thread_id"]),
                    reason=reason,
                    cleanup_eligible=cleanup_eligible,
                    sequence=int(raw["sequence"]),
                    retry_at=retry_at,
                    closed=closed,
                    target_id=(
                        canonical_window_id(raw_target_id)
                        if raw_target_id is not None
                        else None
                    ),
                )
            except KeyError, OverflowError, TypeError, ValueError:
                continue
            if (
                not isinstance(reason, str)
                or not reason
                or not isinstance(cleanup_eligible, bool)
                or not math.isfinite(topic.retry_at)
                or topic.retry_at < 0
                or not isinstance(topic.closed, bool)
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
        return ThreadRouter._retain_retired_topics(loaded)

    @staticmethod
    def _retain_retired_topics(topics: list[RetiredTopic]) -> list[RetiredTopic]:
        """Keep all pending cleanup records and the newest history records."""
        noneligible_to_drop = max(
            0,
            sum(not topic.cleanup_eligible for topic in topics) - _RETIRED_TOPIC_LIMIT,
        )
        if noneligible_to_drop == 0:
            return topics

        retained: list[RetiredTopic] = []
        for topic in topics:
            if not topic.cleanup_eligible and noneligible_to_drop:
                noneligible_to_drop -= 1
                continue
            retained.append(topic)
        return retained

    # ------------------------------------------------------------------
    # Retired topic registry
    # ------------------------------------------------------------------

    def iter_retired_topics(self) -> Iterator[RetiredTopic]:
        """Yield locally known retired topics in oldest-first retention order."""
        return iter(tuple(self._retired_topics))

    def _retired_topic_index(self, topic: RetiredTopic) -> int | None:
        """Find a retired record by its stable chat/thread/sequence identity."""
        for index, current in enumerate(self._retired_topics):
            if (
                current.user_id,
                current.chat_id,
                current.thread_id,
                current.sequence,
            ) == (
                topic.user_id,
                topic.chat_id,
                topic.thread_id,
                topic.sequence,
            ):
                return index
        return None

    def discard_retired_topic(self, topic: RetiredTopic) -> bool:
        """Remove exactly *topic* after a terminal Telegram API outcome."""
        index = self._retired_topic_index(topic)
        if index is None:
            return False
        self._retired_topics.pop(index)
        self._schedule_save()
        return True

    def update_retired_topic(
        self,
        topic: RetiredTopic,
        *,
        retry_at: float,
        closed: bool,
        cleanup_eligible: bool | None = None,
    ) -> RetiredTopic | None:
        """Update a retired record if its exact previous value is still present."""
        index = self._retired_topic_index(topic)
        if index is None:
            return None

        if isinstance(retry_at, bool) or not isinstance(retry_at, (int, float)):
            raise TypeError("retry_at must be a number")
        retry_at_value = float(retry_at)
        if not math.isfinite(retry_at_value) or retry_at_value < 0:
            raise ValueError("retry_at must be finite and nonnegative")
        if not isinstance(closed, bool):
            raise TypeError("closed must be a bool")
        if cleanup_eligible is not None and not isinstance(cleanup_eligible, bool):
            raise TypeError("cleanup_eligible must be a bool or None")

        current = self._retired_topics[index]
        updated = replace(
            current,
            retry_at=retry_at_value,
            closed=closed,
            cleanup_eligible=(
                current.cleanup_eligible
                if cleanup_eligible is None
                else cleanup_eligible
            ),
        )
        self._retired_topics[index] = updated
        self._schedule_save()
        return updated

    @staticmethod
    def _validate_provisioning_thread_id(thread_id: int | None) -> int | None:
        if thread_id is not None and (
            not isinstance(thread_id, int)
            or isinstance(thread_id, bool)
            or thread_id <= 0
        ):
            raise ValueError("thread_id must be a positive integer or None")
        return thread_id

    @staticmethod
    def _validate_provisioning_target_id(
        target_id: str | None, *, field_name: str = "target_id"
    ) -> str | None:
        if target_id is not None and (not isinstance(target_id, str) or not target_id):
            raise ValueError(f"{field_name} must be a non-empty string or None")
        return target_id

    @staticmethod
    def _validate_provisioning_claim_id(claim_id: str) -> str:
        if not isinstance(claim_id, str):
            raise TypeError("claim_id must be a UUID string")
        try:
            return str(uuid.UUID(claim_id))
        except AttributeError, ValueError:
            raise ValueError("claim_id must be a UUID string") from None

    def _require_topic_provisioning(self, claim_id: str) -> TopicProvisioning:
        canonical_claim_id = self._validate_provisioning_claim_id(claim_id)
        claim = self._topic_provisionings.get(canonical_claim_id)
        if claim is None:
            raise KeyError(f"Unknown topic provisioning claim: {claim_id}")
        return claim

    @staticmethod
    def _provisioning_targets(claim: TopicProvisioning) -> tuple[str, ...]:
        return tuple(
            target
            for target in (claim.target_id, claim.previous_target_id)
            if target is not None
        )

    def _has_provisioning_target(
        self, target_id: str, *, except_claim_id: str | None = None
    ) -> bool:
        wanted = canonical_window_id(target_id)
        return any(
            claim.claim_id != except_claim_id
            and any(
                canonical_window_id(candidate) == wanted
                for candidate in self._provisioning_targets(claim)
            )
            for claim in self._topic_provisionings.values()
        )

    def _has_provisioning_topic(
        self,
        _user_id: int,
        chat_id: int,
        thread_id: int,
        *,
        except_claim_id: str | None = None,
    ) -> bool:
        return any(
            claim.claim_id != except_claim_id
            and claim.chat_id == chat_id
            and claim.thread_id == thread_id
            for claim in self._topic_provisionings.values()
        )

    def _target_deletion_claimed(self, _chat_id: int, target_id: str) -> bool:
        wanted = canonical_window_id(target_id)
        for key in self._topic_deletion_claims:
            claimed_target = self._topic_deletion_targets.get(key)
            if (
                claimed_target is not None
                and canonical_window_id(claimed_target) == wanted
            ):
                return True
            for topic in self._retired_topics:
                if (topic.chat_id, topic.thread_id) != key:
                    continue
                if (
                    topic.target_id is not None
                    and canonical_window_id(topic.target_id) == wanted
                ):
                    return True
        return False

    def begin_topic_provisioning(  # noqa: C901
        self,
        user_id: int,
        chat_id: int,
        *,
        thread_id: int | None = None,
        target_id: str | None = None,
        previous_target_id: str | None = None,
        kind: TopicProvisioningKind,
    ) -> TopicProvisioning:
        """Durably claim a topic/target pair before starting creation."""
        if (
            not isinstance(user_id, int)
            or isinstance(user_id, bool)
            or not isinstance(chat_id, int)
            or isinstance(chat_id, bool)
        ):
            raise TypeError("user_id and chat_id must be integers")
        thread_id = self._validate_provisioning_thread_id(thread_id)
        target_id = self._validate_provisioning_target_id(target_id)
        previous_target_id = self._validate_provisioning_target_id(
            previous_target_id, field_name="previous_target_id"
        )
        if not isinstance(kind, str) or kind not in _TOPIC_PROVISIONING_KINDS:
            raise ValueError(f"Unknown topic provisioning kind: {kind!r}")
        if thread_id is None and target_id is None:
            raise ValueError("topic provisioning needs a thread_id or target_id")
        if thread_id is not None:
            if self._is_topic_deletion_claimed(user_id, thread_id, chat_id):
                raise ValueError(
                    "Topic deletion is in progress; retry with a new topic"
                )
            if self._has_provisioning_topic(user_id, chat_id, thread_id):
                raise ValueError(
                    "Topic provisioning is in progress; retry with a new topic"
                )
        for candidate in (target_id, previous_target_id):
            if candidate is None:
                continue
            if self._has_provisioning_target(candidate):
                raise ValueError(
                    "Target provisioning is in progress; retry with a new target"
                )
            if self._target_deletion_claimed(chat_id, candidate):
                raise ValueError(
                    "Topic deletion is in progress; retry with a new target"
                )

        claim_id = str(uuid.uuid4())
        while claim_id in self._topic_provisionings:
            claim_id = str(uuid.uuid4())
        claim = TopicProvisioning(
            claim_id=claim_id,
            user_id=user_id,
            chat_id=chat_id,
            thread_id=thread_id,
            target_id=target_id,
            previous_target_id=previous_target_id,
            kind=cast(TopicProvisioningKind, kind),
        )
        self._topic_provisionings[claim_id] = claim
        self._owned_provisioning_claims.add(claim_id)
        self._schedule_save()
        return claim

    def attach_provisioning_topic(
        self, claim_id: str, thread_id: int
    ) -> TopicProvisioning:
        """Attach the Telegram topic ID returned by a creation request."""
        claim = self._require_topic_provisioning(claim_id)
        validated_thread_id = self._validate_provisioning_thread_id(thread_id)
        if validated_thread_id is None:
            raise ValueError("thread_id must be a positive integer")
        if (claim.chat_id, validated_thread_id) in self._topic_deletion_claims:
            raise ValueError("Topic deletion is in progress; retry with a new topic")
        if claim.thread_id == validated_thread_id:
            return claim
        if self._has_provisioning_topic(
            claim.user_id,
            claim.chat_id,
            validated_thread_id,
            except_claim_id=claim.claim_id,
        ):
            raise ValueError(
                "Topic provisioning is in progress; retry with a new topic"
            )
        updated = replace(claim, thread_id=validated_thread_id)
        self._topic_provisionings[claim.claim_id] = updated
        self._schedule_save()
        return updated

    def attach_provisioning_target(
        self, claim_id: str, target_id: str
    ) -> TopicProvisioning:
        """Attach or supersede the durable target returned by a backend."""
        claim = self._require_topic_provisioning(claim_id)
        validated_target_id = self._validate_provisioning_target_id(target_id)
        if validated_target_id is None:
            raise ValueError("target_id must be a non-empty string")
        if self._target_deletion_claimed(claim.chat_id, validated_target_id):
            raise ValueError("Topic deletion is in progress; retry with a new target")
        if claim.target_id == validated_target_id:
            return claim
        if self._has_provisioning_target(
            validated_target_id, except_claim_id=claim.claim_id
        ):
            raise ValueError(
                "Target provisioning is in progress; retry with a new target"
            )
        previous_target_id = claim.previous_target_id
        if claim.target_id is not None:
            previous_target_id = claim.target_id
        updated = replace(
            claim,
            target_id=validated_target_id,
            previous_target_id=previous_target_id,
        )
        self._topic_provisionings[claim.claim_id] = updated
        self._schedule_save()
        return updated

    def commit_topic_provisioning(
        self, claim_id: str, *, window_name: str = ""
    ) -> bool:
        """Bind both sides of a complete claim and release it atomically."""
        claim = self._require_topic_provisioning(claim_id)
        thread_id = claim.thread_id
        target_id = claim.target_id
        if thread_id is None or target_id is None:
            return False
        if self._target_deletion_claimed(claim.chat_id, target_id):
            raise ValueError("Topic deletion is in progress; retry with a new target")
        self._bind_thread(
            claim.user_id,
            thread_id,
            target_id,
            window_name=window_name,
            chat_id=claim.chat_id,
            provisioning_claim_id=claim.claim_id,
            schedule_save=False,
        )
        self._topic_provisionings.pop(claim.claim_id, None)
        self._owned_provisioning_claims.discard(claim.claim_id)
        self._schedule_save()
        return True

    def abort_topic_provisioning(
        self,
        claim_id: str,
        *,
        target_confirmed_absent: bool,
        topic_confirmed_absent: bool = False,
    ) -> TopicProvisioning | None:
        """Release a failed claim only after its remote outcome is known.

        A no-topic ``topic_for_target`` failure can release with explicit proof
        that no forum topic was created, even when its terminal target remains
        alive.  Any ambiguous outcome remains durable and is marked uncertain;
        the current owner stays attached until it explicitly calls
        ``mark_provisioning_uncertain`` after stopping the flow.
        """
        try:
            canonical_claim_id = self._validate_provisioning_claim_id(claim_id)
        except TypeError, ValueError:
            return None
        claim = self._topic_provisionings.get(canonical_claim_id)
        if claim is None:
            return None
        if not isinstance(target_confirmed_absent, bool):
            raise TypeError("target_confirmed_absent must be a bool")
        if not isinstance(topic_confirmed_absent, bool):
            raise TypeError("topic_confirmed_absent must be a bool")

        can_release = target_confirmed_absent or (
            topic_confirmed_absent
            and claim.thread_id is None
            and claim.kind == "topic_for_target"
        )
        if not can_release:
            if claim.uncertain:
                return claim
            updated = replace(claim, uncertain=True)
            self._topic_provisionings[claim.claim_id] = updated
            self._schedule_save()
            return updated

        active_window = (
            self.get_window_for_chat_thread(claim.chat_id, claim.thread_id)
            if claim.thread_id is not None
            else None
        )
        self._topic_provisionings.pop(claim.claim_id, None)
        self._owned_provisioning_claims.discard(claim.claim_id)
        if claim.thread_id is not None and active_window is None:
            self._retire_topic(
                claim.user_id,
                claim.chat_id,
                claim.thread_id,
                reason="creation_failed",
                cleanup_eligible=True,
                target_id=claim.target_id,
            )
        self._schedule_save()
        return claim

    def mark_provisioning_uncertain(self, claim_id: str) -> TopicProvisioning:
        """Retain a claim for recovery after its owner has stopped."""
        claim = self._require_topic_provisioning(claim_id)
        self._owned_provisioning_claims.discard(claim.claim_id)
        if claim.uncertain:
            return claim
        updated = replace(claim, uncertain=True)
        self._topic_provisionings[claim.claim_id] = updated
        self._schedule_save()
        return updated

    def has_topic_provisioning(self, chat_id: int, thread_id: int) -> bool:
        """Return whether any durable claim protects this exact topic."""
        return any(
            claim.chat_id == chat_id and claim.thread_id == thread_id
            for claim in self._topic_provisionings.values()
        )

    def has_target_provisioning(self, target_id: str) -> bool:
        """Return whether a durable claim protects this target or its alias."""
        if not isinstance(target_id, str):
            return False
        return self._has_provisioning_target(target_id)

    def iter_topic_provisionings(self) -> list[TopicProvisioning]:
        """Return a snapshot of all durable provisioning claims."""
        return list(self._topic_provisionings.values())

    def owns_topic_provisioning(self, claim_id: str) -> bool:
        """Return whether this process still owns the claim."""
        return isinstance(claim_id, str) and claim_id in self._owned_provisioning_claims

    def has_active_topic(self, chat_id: int, thread_id: int) -> bool:
        """Return whether any user currently owns this chat/thread pair."""
        if any(
            bound_chat == chat_id and bound_thread == thread_id
            for (_user_id, bound_chat, bound_thread) in self.chat_thread_bindings
        ):
            return True
        return any(
            thread_id in bindings
            and self.resolve_chat_id(user_id, thread_id) == chat_id
            for user_id, bindings in self.thread_bindings.items()
        )

    def begin_topic_deletion(self, topic: RetiredTopic) -> bool:
        """Claim an exact retired topic for one in-flight deletion attempt."""
        key = (topic.chat_id, topic.thread_id)
        if key in self._topic_deletion_claims:
            return False
        current_index = self._retired_topic_index(topic)
        if (
            current_index is None
            or self.has_active_topic(*key)
            or self.has_topic_provisioning(*key)
        ):
            return False
        current_topic = self._retired_topics[current_index]
        if current_topic.target_id is not None and self.has_target_provisioning(
            current_topic.target_id
        ):
            return False
        self._topic_deletion_claims.add(key)
        self._topic_deletion_targets[key] = (
            canonical_window_id(current_topic.target_id)
            if current_topic.target_id is not None
            else None
        )
        return True

    def end_topic_deletion(self, topic: RetiredTopic) -> None:
        """Release an in-flight deletion claim for a retired topic."""
        key = (topic.chat_id, topic.thread_id)
        self._topic_deletion_claims.discard(key)
        self._topic_deletion_targets.pop(key, None)

    def _retire_topic(
        self,
        user_id: int,
        chat_id: int | None,
        thread_id: int,
        *,
        reason: str,
        cleanup_eligible: bool,
        target_id: str | None = None,
    ) -> None:
        """Remember a known local topic, never an inferred Telegram topic."""
        if chat_id is None:
            return
        if target_id is not None:
            target_id = canonical_window_id(target_id)
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
                target_id=target_id,
            )
        )
        self._next_retired_sequence += 1
        self._retired_topics = self._retain_retired_topics(self._retired_topics)

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

    def _is_topic_deletion_claimed(
        self, user_id: int, thread_id: int, chat_id: int | None
    ) -> bool:
        """Check whether a bind would race with an in-flight deletion."""
        if chat_id is not None:
            return (chat_id, thread_id) in self._topic_deletion_claims

        matching_claims = {
            claimed_chat
            for claimed_chat, claimed_thread in self._topic_deletion_claims
            if claimed_thread == thread_id
        }
        if not matching_claims:
            return False

        metadata_key = f"{user_id}:{thread_id}"
        scoped_chats = {
            bound_chat
            for (bound_user, bound_chat, bound_thread) in self.chat_thread_bindings
            if bound_user == user_id and bound_thread == thread_id
        }
        has_resolved_chat = (
            metadata_key in self.group_chat_ids
            or len(scoped_chats) == 1
            or self.default_group_id is not None
        )
        if has_resolved_chat:
            resolved_chat_id = self.resolve_chat_id(user_id, thread_id)
            return (resolved_chat_id, thread_id) in self._topic_deletion_claims

        return True

    def _is_topic_provisioning_claimed(
        self, _user_id: int, thread_id: int, chat_id: int | None
    ) -> bool:
        """Check whether a bind would race with topic/target provisioning."""
        return any(
            claim.thread_id == thread_id
            and (chat_id is None or claim.chat_id == chat_id)
            for claim in self._topic_provisionings.values()
        )

    def _can_bind_for_provisioning(
        self,
        claim_id: str | None,
        user_id: int,
        chat_id: int | None,
        thread_id: int,
    ) -> bool:
        if claim_id is None or chat_id is None:
            return False
        claim = self._topic_provisionings.get(claim_id)
        return (
            claim is not None
            and claim.user_id == user_id
            and claim.chat_id == chat_id
            and claim.thread_id == thread_id
        )

    def _bind_chat_scoped(
        self, user_id: int, chat_id: int, thread_id: int, window_id: str
    ) -> None:
        """Claim a window once within a chat, evicting stale topic owners."""
        key = (user_id, chat_id, thread_id)
        old_window = self.chat_thread_bindings.get(key)
        if old_window is not None and old_window != window_id:
            self._chat_window_to_thread.pop((user_id, chat_id, old_window), None)
        old_thread = self._chat_window_to_thread.pop(
            (user_id, chat_id, window_id), None
        )
        if old_thread is not None and old_thread != thread_id:
            self.chat_thread_bindings.pop((user_id, chat_id, old_thread), None)
            self._remove_group_routing_metadata(user_id, old_thread)
        stale = [
            candidate
            for candidate, wid in self.chat_thread_bindings.items()
            if candidate[1] == chat_id and wid == window_id and candidate != key
        ]
        for candidate in stale:
            self.chat_thread_bindings.pop(candidate, None)
            self._chat_window_to_thread.pop(
                (candidate[0], candidate[1], window_id), None
            )
            self._remove_group_routing_metadata(candidate[0], candidate[2])
        self.chat_thread_bindings[key] = window_id
        self._chat_window_to_thread[(user_id, chat_id, window_id)] = thread_id
        self._remove_group_routing_metadata(user_id, thread_id)

    def _bind_thread(
        self,
        user_id: int,
        thread_id: int,
        window_id: str,
        *,
        window_name: str = "",
        chat_id: int | None = None,
        provisioning_claim_id: str | None = None,
        schedule_save: bool = True,
    ) -> None:
        """Bind a topic, optionally from its owning atomic provisioning claim."""
        if self._is_topic_deletion_claimed(user_id, thread_id, chat_id):
            raise ValueError("Topic deletion is in progress; retry with a new topic")
        if self._is_topic_provisioning_claimed(user_id, thread_id, chat_id) and not (
            self._can_bind_for_provisioning(
                provisioning_claim_id, user_id, chat_id, thread_id
            )
        ):
            raise ValueError(
                "Topic provisioning is in progress; retry with a new topic"
            )
        if chat_id is not None:
            self._bind_chat_scoped(user_id, chat_id, thread_id, window_id)
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
        if schedule_save:
            self._schedule_save()

    def bind_thread(
        self,
        user_id: int,
        thread_id: int,
        window_id: str,
        window_name: str = "",
        chat_id: int | None = None,
    ) -> None:
        """Bind a topic, using chat-scoped identity when ``chat_id`` is known."""
        self._bind_thread(
            user_id,
            thread_id,
            window_id,
            window_name=window_name,
            chat_id=chat_id,
        )

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
            target_id=window_id,
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
        wanted = canonical_window_id(window_id)
        return any(
            canonical_window_id(wid) == wanted for (_, wid) in self._window_to_thread
        ) or any(
            canonical_window_id(wid) == wanted
            for wid in self.chat_thread_bindings.values()
        )

    def has_window_for_user(self, user_id: int, window_id: str) -> bool:
        wanted = canonical_window_id(window_id)
        return any(
            uid == user_id and canonical_window_id(wid) == wanted
            for (uid, _chat, _thread), wid in self.chat_thread_bindings.items()
        ) or any(
            uid == user_id and canonical_window_id(wid) == wanted
            for (uid, wid) in self._window_to_thread
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
    # Chat capability and ID management
    # ------------------------------------------------------------------

    def mark_private_topic_chat(self, chat_id: int) -> None:
        """Record a private chat observed with valid topic-message metadata."""
        if not isinstance(chat_id, int) or chat_id in self.private_topic_chats:
            return
        self.private_topic_chats.add(chat_id)
        self._schedule_save()

    def is_private_topic_chat(self, chat_id: int) -> bool:
        """Return whether *chat_id* was observed as a private topic chat."""
        return chat_id in self.private_topic_chats

    def iter_private_topic_chat_ids(self) -> Iterator[int]:
        """Iterate private chats that have supplied topic-message metadata."""
        yield from sorted(self.private_topic_chats)

    def set_group_chat_id(self, user_id: int, thread_id: int, chat_id: int) -> None:
        """Record chat metadata without guessing a legacy topic's identity."""
        if self._is_topic_provisioning_claimed(user_id, thread_id, chat_id):
            raise ValueError(
                "Topic provisioning is in progress; retry with a new topic"
            )
        key = f"{user_id}:{thread_id}"
        bindings = self.thread_bindings.get(user_id)
        if bindings and thread_id in bindings:
            if self.group_chat_ids.get(key) != chat_id:
                return
            window_id = bindings.pop(thread_id)
            self._window_to_thread.pop((user_id, window_id), None)
            if not bindings:
                self.thread_bindings.pop(user_id, None)
            self.bind_thread(user_id, thread_id, window_id, chat_id=chat_id)
            return
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
