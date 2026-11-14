"""Credential-free codec for unfinished native marker side effects.

Native metadata is the only selected-state truth. These immutable records
authorize retrying one desired native write, not routing or terminal input.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

TOPIC_MARK_OPERATIONS_KEY = "topic_mark_operations"


@dataclass(frozen=True, slots=True)
class TopicMarkOperation:
    window_id: str
    owner: str | None
    desired: bool

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id:
            raise ValueError("Marker operation needs an original routing reference")
        if self.owner is not None and (
            not isinstance(self.owner, str) or not self.owner
        ):
            raise ValueError("Marker operation owner must be nonempty or null")
        if not isinstance(self.desired, bool):
            raise ValueError("Marker operation desired value must be boolean")

    @property
    def key(self) -> tuple[str, str]:
        return (
            ("owner", self.owner)
            if self.owner is not None
            else ("route", self.window_id.casefold())
        )

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "window_id": self.window_id,
            "owner": self.owner,
            "desired": self.desired,
        }


def operation_matches(
    operation: TopicMarkOperation, window_id: str, owner: str | None
) -> bool:
    if operation.owner is not None and owner is not None:
        return operation.owner == owner
    return operation.window_id.casefold() == window_id.casefold()


def parse_topic_mark_operations(
    state: Mapping[str, object],
) -> tuple[TopicMarkOperation, ...]:
    """Reject malformed or ambiguous intent rather than hiding it from callers."""
    raw = state.get(TOPIC_MARK_OPERATIONS_KEY, [])
    if not isinstance(raw, list):
        raise ValueError("Marker operation journal must be a list")
    operations: dict[tuple[str, str], TopicMarkOperation] = {}
    for row in raw:
        if not isinstance(row, dict) or set(row) != {"window_id", "owner", "desired"}:
            raise ValueError("Malformed marker operation record")
        operation = TopicMarkOperation(row["window_id"], row["owner"], row["desired"])
        if operation.key in operations:
            raise ValueError("Duplicate marker operation owner or quarantined route")
        operations[operation.key] = operation
    return tuple(operations.values())


def read_topic_mark_operations(path: str | Path) -> tuple[TopicMarkOperation, ...]:
    """Read state.json without bot initialization, native mutation, or writes."""
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    if not isinstance(state, dict):
        raise ValueError("State must be an object")
    return parse_topic_mark_operations(state)


def has_topic_mark_conflict(
    operations: Iterable[TopicMarkOperation],
    window_id: str,
    owner: str | None,
    desired: bool,
) -> bool:
    """Whether a manual write would contradict a matching unfinished operation."""
    return any(
        operation_matches(operation, window_id, owner) and operation.desired != desired
        for operation in operations
    )
