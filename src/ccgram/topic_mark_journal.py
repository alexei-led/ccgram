"""Credential-free codec for unfinished native marker side effects."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .topic_mark_types import TopicMarkWriteWitness

TOPIC_MARK_OPERATIONS_KEY = "topic_mark_operations"


@dataclass(frozen=True, slots=True)
class TopicMarkAttempt:
    desired: bool
    witness: TopicMarkWriteWitness

    def __post_init__(self) -> None:
        if not isinstance(self.desired, bool) or not isinstance(
            self.witness, TopicMarkWriteWitness
        ):
            raise ValueError("Malformed marker attempt")

    def to_dict(self) -> dict[str, object]:
        return {"desired": self.desired, "witness": self.witness.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> TopicMarkAttempt:
        if not isinstance(value, dict) or set(value) != {"desired", "witness"}:
            raise ValueError("Malformed marker attempt")
        return cls(value["desired"], TopicMarkWriteWitness.from_dict(value["witness"]))


@dataclass(frozen=True, slots=True)
class TopicMarkOperation:
    window_id: str
    owner: str | None
    desired: bool
    attempt: TopicMarkAttempt | None = None
    detach_clear: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id:
            raise ValueError("Marker operation needs an original routing reference")
        if self.owner is not None and (
            not isinstance(self.owner, str) or not self.owner
        ):
            raise ValueError("Marker operation owner must be nonempty or null")
        if not isinstance(self.desired, bool):
            raise ValueError("Marker operation desired value must be boolean")
        if not isinstance(self.detach_clear, bool) or (
            self.desired and self.detach_clear
        ):
            raise ValueError("Only a clear may carry boolean detach protection")
        if self.attempt is not None and not isinstance(self.attempt, TopicMarkAttempt):
            raise ValueError("Malformed marker attempt")

    @property
    def key(self) -> tuple[str, str]:
        return (
            ("owner", self.owner)
            if self.owner is not None
            else ("route", self.window_id.casefold())
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "owner": self.owner,
            "desired": self.desired,
            "attempt": self.attempt.to_dict() if self.attempt is not None else None,
            "detach_clear": self.detach_clear,
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
    """Missing v4.22 journal is valid; unpublished legacy rows are not."""
    raw = state.get(TOPIC_MARK_OPERATIONS_KEY, [])
    if not isinstance(raw, list):
        raise ValueError("Marker operation journal must be a list")
    operations: dict[tuple[str, str], TopicMarkOperation] = {}
    dispatches: set[str] = set()
    for row in raw:
        if not isinstance(row, dict) or set(row) != {
            "window_id",
            "owner",
            "desired",
            "attempt",
            "detach_clear",
        }:
            raise ValueError("Malformed or incompatible marker operation record")
        attempt = (
            None
            if row["attempt"] is None
            else TopicMarkAttempt.from_dict(row["attempt"])
        )
        operation = TopicMarkOperation(
            row["window_id"],
            row["owner"],
            row["desired"],
            attempt,
            detach_clear=row["detach_clear"],
        )
        if operation.key in operations:
            raise ValueError("Duplicate marker operation owner or quarantined route")
        if attempt is not None:
            dispatch_id = attempt.witness.dispatch_id
            if dispatch_id in dispatches:
                raise ValueError("Duplicate marker dispatch identity")
            dispatches.add(dispatch_id)
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
    """A possibly-live attempt conflicts even with the same desired value."""
    return any(
        operation_matches(operation, window_id, owner)
        and (operation.attempt is not None or operation.desired != desired)
        for operation in operations
    )
