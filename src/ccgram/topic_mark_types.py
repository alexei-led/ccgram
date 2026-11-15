"""Credential-free witnesses for possibly dispatched native marker writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import PurePosixPath
from typing import Protocol, Self

_NATIVE_BACKENDS = frozenset({"agterm", "herdr", "tmux"})
_IDENTITY_FIELDS = (
    "dispatch_id",
    "backend",
    "host_id",
    "boot_id",
    "namespace_id",
    "native_start",
    "code_id",
)


@dataclass(frozen=True, slots=True)
class TopicMarkWriteWitness:
    dispatch_id: str
    backend: str
    host_id: str
    boot_id: str
    namespace_id: str
    native_pid: int
    native_start: str
    code_id: str
    socket_dev: int
    socket_ino: int
    pin_path: str | None

    def __post_init__(self) -> None:
        for name in _IDENTITY_FIELDS:
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value
                or not value.isprintable()
                or any(character.isspace() for character in value)
            ):
                raise ValueError(f"Marker write {name} must be a canonical identity")
        if self.backend not in _NATIVE_BACKENDS:
            raise ValueError("Marker write backend is unsupported")
        for name, minimum in (("native_pid", 1), ("socket_dev", 0), ("socket_ino", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"Marker write {name} must be a valid integer identity"
                )
        if self.pin_path is not None:
            if type(self.pin_path) is not str or not self.pin_path.isprintable():
                raise ValueError("Marker write pin must be an absolute canonical path")
            path = PurePosixPath(self.pin_path)
            if (
                not path.is_absolute()
                or not path.name
                or path.root != "/"
                or ".." in path.parts
                or str(path) != self.pin_path
            ):
                raise ValueError("Marker write pin must be an absolute canonical path")

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        if type(raw) is not dict or raw.keys() != {field.name for field in fields(cls)}:
            raise ValueError("Malformed marker write witness")
        return cls(**raw)


class TopicMarkWriteGuard(Protocol):
    """The state owner durably fences dispatch separately from verified selection."""

    def arm(self, witness: TopicMarkWriteWitness) -> None: ...

    def terminal(self, dispatch_id: str) -> None: ...
