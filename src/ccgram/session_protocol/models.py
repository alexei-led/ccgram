"""Wire values for ccgram/1; serialization is shared by HTTP and events."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from typing import Any

PROTOCOL = "ccgram/1"
ACTION_KINDS = frozenset(
    {
        "message.send",
        "run.interrupt",
        "request.respond",
        "terminal.key",
        "terminal.paste",
        "session.close",
        "session.resume",
        "session.rename",
    }
)
CLOSE_REASONS = frozenset(
    {"answered", "dismissed", "interrupted", "turn_ended", "vanished", "expired"}
)
MAX_TEXT_BYTES = 65536
# Refusal when the live prompt no longer matches the one the input was aimed at
# (someone answered or moved it at the terminal first).
PROMPT_CHANGED_MESSAGE = "Interactive prompt changed before input"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def wire(value: Any) -> Any:
    """Convert domain values to JSON without exposing internal target locators."""
    if is_dataclass(value) and not isinstance(value, type):
        return {_camel(f.name): wire(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    return value


def _camel(name: str) -> str:
    first, *rest = name.split("_")
    return first + "".join(word.title() for word in rest)


class ProtocolError(Exception):
    def __init__(
        self, status: int, code: str, message: str, details: Any = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details if details is not None else {}

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": wire(self.details),
            }
        }


@dataclass(frozen=True, slots=True)
class Part:
    id: str
    type: str
    text: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    kind: str | None = None
    status: str | None = None
    title: str | None = None
    input: Any = None
    output: Any = None
    diff: Any = None
    url: str | None = None
    width: int | None = None
    height: int | None = None
    mime: str | None = None
    size: int | None = None
    level: str | None = None


@dataclass(frozen=True, slots=True)
class Message:
    id: str
    revision: int
    conversation_id: str
    role: str
    time: str | None
    complete: bool
    parts: tuple[Part, ...]
    action_id: str | None = None
    phase: str | None = None


@dataclass(frozen=True, slots=True)
class RequestOption:
    id: str
    label: str
    kind: str


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    session_id: str
    kind: str
    source: str
    title: str
    tool: dict[str, Any] | None
    options: tuple[RequestOption, ...]
    response_schema: dict[str, Any] | None
    opened_at: str
    expires_at: str | None = None


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    history_limit: str
    history_generation: int = 0


@dataclass(frozen=True, slots=True)
class Status:
    state: str
    detail: str
    source: str
    since: str


@dataclass(frozen=True, slots=True)
class Pane:
    id: str
    title: str
    active: bool


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    id: str
    binding_version: int
    title: str
    host: str
    cwd: str
    git_branch: str | None
    provider: str
    conversation: Conversation | None
    status: Status
    approval_mode: str
    panes: tuple[Pane, ...]
    open_requests: tuple[Request, ...]
    capabilities: tuple[str, ...]
    last_activity: str


@dataclass(frozen=True, slots=True)
class HistoryPage:
    conversation: str | None
    messages: tuple[Message, ...]
    prev_cursor: str | None
    history_limit: str
    history_generation: int = 0


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    binding_version: int
    kind: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ActionResult:
    action_id: str
    outcome: str
    error: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
