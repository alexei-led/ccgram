"""cmux sidecar JSON-RPC protocol — request/response DTOs and error codes.

The sidecar uses newline-delimited JSON over a local Unix socket. Each
request carries a stable ``method`` name and an ``id``; responses echo
the ``id`` and either include a ``result`` payload or an ``error``
object with one of :data:`SIDECAR_ERROR_CODES`. The codes map cleanly
onto :data:`ccgram.terminal_backends.base.TerminalBackendErrorCode` —
the mapping in :func:`map_sidecar_error_to_terminal` is the only
boundary that widens that taxonomy.

Event frames (``method`` set, ``id`` absent) describe asynchronous
state changes. They are parsed for future streaming work; the MVP
client only logs unexpected frames.

DTOs are frozen dataclasses with explicit ``from_wire``/``to_wire``
adapters so the wire schema remains the source of truth and tests can
assert it without depending on dataclass internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

from .base import TerminalBackendErrorCode

PROTOCOL_VERSION: Final[str] = "1"

SidecarErrorCode = Literal[
    "unavailable",
    "not_found",
    "unsupported",
    "no_terminal_surface",
    "timeout",
    "rejected",
    "internal_error",
    "incompatible_version",
    "malformed_request",
]

SIDECAR_ERROR_CODES: Final[frozenset[str]] = frozenset(get_args(SidecarErrorCode))

METHOD_HELLO: Final[str] = "hello"
METHOD_LIST_WORKSPACES: Final[str] = "list_workspaces"
METHOD_CAPTURE_SCREEN: Final[str] = "capture_screen"
METHOD_SEND_TEXT: Final[str] = "send_text"
METHOD_SEND_KEY: Final[str] = "send_key"
METHOD_CLOSE_WORKSPACE: Final[str] = "close_workspace"

KNOWN_METHODS: Final[frozenset[str]] = frozenset(
    {
        METHOD_HELLO,
        METHOD_LIST_WORKSPACES,
        METHOD_CAPTURE_SCREEN,
        METHOD_SEND_TEXT,
        METHOD_SEND_KEY,
        METHOD_CLOSE_WORKSPACE,
    }
)

EVENT_WORKSPACE_STATE: Final[str] = "workspace_state"
EVENT_WORKSPACE_CLOSED: Final[str] = "workspace_closed"

KNOWN_EVENTS: Final[frozenset[str]] = frozenset(
    {EVENT_WORKSPACE_STATE, EVENT_WORKSPACE_CLOSED}
)

# Map sidecar codes to the existing TerminalBackendErrorCode taxonomy.
# ``incompatible_version`` and ``malformed_request`` collapse to
# ``unsupported`` and ``rejected`` respectively — both signal a
# permanent mismatch the caller should not retry blindly.
_ERROR_MAP: Final[dict[str, TerminalBackendErrorCode]] = {
    "unavailable": "unavailable",
    "not_found": "not_found",
    "unsupported": "unsupported",
    "no_terminal_surface": "no_terminal_surface",
    "timeout": "timeout",
    "rejected": "rejected",
    "internal_error": "internal_error",
    "incompatible_version": "unsupported",
    "malformed_request": "rejected",
}


def map_sidecar_error_to_terminal(code: str) -> TerminalBackendErrorCode:
    """Translate a sidecar error code into the normalised backend code."""
    mapped = _ERROR_MAP.get(code)
    if mapped is None:
        return "internal_error"
    return mapped


class CmuxProtocolError(Exception):
    """Wire-level violation (malformed JSON, missing fields, bad id)."""


@dataclass(frozen=True, slots=True)
class CmuxRequest:
    """JSON-RPC-shaped request frame sent to the sidecar."""

    id: int
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("CmuxRequest.id must be non-negative")
        if not self.method or not self.method.strip():
            raise ValueError("CmuxRequest.method must be a non-empty string")

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "method": self.method,
            "params": dict(self.params),
        }


@dataclass(frozen=True, slots=True)
class CmuxError:
    """Structured error payload returned in a response frame."""

    code: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("CmuxError.code must be a non-empty string")

    def to_terminal_code(self) -> TerminalBackendErrorCode:
        return map_sidecar_error_to_terminal(self.code)

    def to_wire(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "data": dict(self.data),
        }


@dataclass(frozen=True, slots=True)
class CmuxResponse:
    """Response frame keyed by the originating request ``id``."""

    id: int
    result: dict[str, Any] | None = None
    error: CmuxError | None = None

    def __post_init__(self) -> None:
        if self.result is None and self.error is None:
            raise CmuxProtocolError(
                f"CmuxResponse(id={self.id}) must carry either result or error"
            )
        if self.result is not None and self.error is not None:
            raise CmuxProtocolError(
                f"CmuxResponse(id={self.id}) must not carry both result and error"
            )

    @classmethod
    def from_wire(cls, payload: Any) -> CmuxResponse:
        if not isinstance(payload, dict):
            raise CmuxProtocolError("response payload must be an object")
        raw_id = _require_int_id(payload, key="id", context="response")
        error_payload = payload.get("error")
        if error_payload is not None:
            return cls(id=raw_id, error=_parse_error(error_payload))
        return cls(id=raw_id, result=_parse_result(payload.get("result"), raw_id))


def _require_int_id(payload: dict[str, Any], *, key: str, context: str) -> int:
    if key not in payload:
        raise CmuxProtocolError(f"{context} missing {key!r} field")
    raw = payload[key]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise CmuxProtocolError(f"{context} {key!r} must be int, got {raw!r}")
    return raw


def _parse_error(error_payload: Any) -> CmuxError:
    if not isinstance(error_payload, dict):
        raise CmuxProtocolError("response 'error' must be an object")
    code = error_payload.get("code")
    message = error_payload.get("message", "")
    data = error_payload.get("data") or {}
    if not isinstance(code, str) or not code:
        raise CmuxProtocolError("response error.code must be a non-empty string")
    if not isinstance(message, str):
        raise CmuxProtocolError("response error.message must be a string")
    if not isinstance(data, dict):
        raise CmuxProtocolError("response error.data must be an object")
    return CmuxError(code=code, message=message, data=data)


def _parse_result(result_payload: Any, request_id: int) -> dict[str, Any]:
    if result_payload is None:
        raise CmuxProtocolError(
            f"response id={request_id} carries neither result nor error"
        )
    if not isinstance(result_payload, dict):
        raise CmuxProtocolError("response 'result' must be an object")
    return dict(result_payload)


@dataclass(frozen=True, slots=True)
class CmuxEvent:
    """Asynchronous event frame (no ``id``)."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: Any) -> CmuxEvent:
        if not isinstance(payload, dict):
            raise CmuxProtocolError("event payload must be an object")
        if "method" not in payload:
            raise CmuxProtocolError("event payload missing 'method'")
        method = payload["method"]
        if not isinstance(method, str) or not method:
            raise CmuxProtocolError("event 'method' must be a non-empty string")
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise CmuxProtocolError("event 'params' must be an object")
        return cls(method=method, params=dict(params))


@dataclass(frozen=True, slots=True)
class CmuxHelloResult:
    """Capability handshake returned by ``hello``."""

    protocol_version: str
    sidecar_version: str
    supports_create: bool = False
    supports_capture: bool = True
    supports_send_text: bool = True
    supports_send_key: bool = True
    supports_close: bool = False
    supports_event_stream: bool = False

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> CmuxHelloResult:
        protocol = result.get("protocol_version")
        sidecar = result.get("sidecar_version", "")
        if not isinstance(protocol, str) or not protocol:
            raise CmuxProtocolError("hello result missing protocol_version")
        if not isinstance(sidecar, str):
            raise CmuxProtocolError("hello result sidecar_version must be a string")
        return cls(
            protocol_version=protocol,
            sidecar_version=sidecar,
            supports_create=bool(result.get("supports_create", False)),
            supports_capture=bool(result.get("supports_capture", True)),
            supports_send_text=bool(result.get("supports_send_text", True)),
            supports_send_key=bool(result.get("supports_send_key", True)),
            supports_close=bool(result.get("supports_close", False)),
            supports_event_stream=bool(result.get("supports_event_stream", False)),
        )


@dataclass(frozen=True, slots=True)
class CmuxWorkspace:
    """Workspace description returned by ``list_workspaces``."""

    workspace_id: str
    title: str = ""
    cwd: str | None = None
    provider_name: str | None = None
    state: str = "unknown"
    has_terminal_surface: bool = True

    @classmethod
    def from_wire(cls, payload: Any) -> CmuxWorkspace:
        if not isinstance(payload, dict):
            raise CmuxProtocolError("workspace entry must be an object")
        workspace_id = payload.get("workspace_id") or payload.get("id")
        if not isinstance(workspace_id, str) or not workspace_id:
            raise CmuxProtocolError("workspace entry missing workspace_id")
        cwd = payload.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise CmuxProtocolError("workspace cwd must be a string when set")
        provider_name = payload.get("provider_name")
        if provider_name is not None and not isinstance(provider_name, str):
            raise CmuxProtocolError("workspace provider_name must be a string when set")
        return cls(
            workspace_id=workspace_id,
            title=str(payload.get("title", "")),
            cwd=cwd,
            provider_name=provider_name,
            state=str(payload.get("state", "unknown")),
            has_terminal_surface=bool(payload.get("has_terminal_surface", True)),
        )


__all__ = [
    "EVENT_WORKSPACE_CLOSED",
    "EVENT_WORKSPACE_STATE",
    "KNOWN_EVENTS",
    "KNOWN_METHODS",
    "METHOD_CAPTURE_SCREEN",
    "METHOD_CLOSE_WORKSPACE",
    "METHOD_HELLO",
    "METHOD_LIST_WORKSPACES",
    "METHOD_SEND_KEY",
    "METHOD_SEND_TEXT",
    "PROTOCOL_VERSION",
    "SIDECAR_ERROR_CODES",
    "CmuxError",
    "CmuxEvent",
    "CmuxHelloResult",
    "CmuxProtocolError",
    "CmuxRequest",
    "CmuxResponse",
    "CmuxWorkspace",
    "SidecarErrorCode",
    "map_sidecar_error_to_terminal",
]
