"""cmux sidecar JSON-RPC protocol — request/response DTOs and errors.

The sidecar speaks newline-delimited JSON over a local Unix socket.
Protocol v2 treats a cmux terminal tab/panel as the bindable unit;
workspaces are metadata only, because cmux tabs can move between them.
DTOs use explicit ``from_wire``/``to_wire`` adapters so tests assert the
wire schema instead of dataclass internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

from .base import TerminalBackendErrorCode

PROTOCOL_VERSION: Final[str] = "2"

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
METHOD_LIST_TERMINAL_SESSIONS: Final[str] = "list_terminal_sessions"
METHOD_CAPTURE_SCREEN: Final[str] = "capture_screen"
METHOD_SEND_TEXT: Final[str] = "send_text"
METHOD_SEND_KEY: Final[str] = "send_key"
METHOD_CLOSE_TERMINAL_SESSION: Final[str] = "close_terminal_session"

KNOWN_METHODS: Final[frozenset[str]] = frozenset(
    {
        METHOD_HELLO,
        METHOD_LIST_TERMINAL_SESSIONS,
        METHOD_CAPTURE_SCREEN,
        METHOD_SEND_TEXT,
        METHOD_SEND_KEY,
        METHOD_CLOSE_TERMINAL_SESSION,
    }
)

EVENT_TERMINAL_SESSION_STATE: Final[str] = "terminal_session_state"
EVENT_TERMINAL_SESSION_CLOSED: Final[str] = "terminal_session_closed"

KNOWN_EVENTS: Final[frozenset[str]] = frozenset(
    {EVENT_TERMINAL_SESSION_STATE, EVENT_TERMINAL_SESSION_CLOSED}
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


def _optional_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise CmuxProtocolError(f"{key} must be a boolean")
    return value


def _optional_str(payload: dict[str, Any], key: str, default: str) -> str:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, str):
        raise CmuxProtocolError(f"{key} must be a string")
    return value


def _optional_nonempty_str(payload: dict[str, Any], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise CmuxProtocolError(f"{key} must be a non-empty string when set")
    return value


def _first_nonempty_str(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, str) or not value:
            raise CmuxProtocolError(f"{key} must be a non-empty string")
        return value
    raise CmuxProtocolError(f"terminal entry missing {keys[0]}")


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
            supports_create=_optional_bool(result, "supports_create", False),
            supports_capture=_optional_bool(result, "supports_capture", True),
            supports_send_text=_optional_bool(result, "supports_send_text", True),
            supports_send_key=_optional_bool(result, "supports_send_key", True),
            supports_close=_optional_bool(result, "supports_close", False),
            supports_event_stream=_optional_bool(
                result, "supports_event_stream", False
            ),
        )


@dataclass(frozen=True, slots=True)
class CmuxTerminalSession:
    """Terminal tab/panel returned by ``list_terminal_sessions``."""

    terminal_id: str
    title: str = ""
    cwd: str | None = None
    provider_name: str | None = None
    state: str = "unknown"
    workspace_id: str | None = None
    workspace_title: str = ""
    pane_id: str | None = None
    surface_id: str | None = None
    panel_id: str | None = None

    @classmethod
    def from_wire(cls, payload: Any) -> CmuxTerminalSession:
        if not isinstance(payload, dict):
            raise CmuxProtocolError("terminal entry must be an object")
        terminal_id = _first_nonempty_str(
            payload, ("terminal_id", "surface_id", "panel_id", "id")
        )
        cwd = payload.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise CmuxProtocolError("terminal cwd must be a string when set")
        provider_name = payload.get("provider_name")
        if provider_name is not None and not isinstance(provider_name, str):
            raise CmuxProtocolError("terminal provider_name must be a string when set")
        return cls(
            terminal_id=terminal_id,
            title=_optional_str(payload, "title", ""),
            cwd=cwd,
            provider_name=provider_name,
            state=_optional_str(payload, "state", "unknown"),
            workspace_id=_optional_nonempty_str(payload, "workspace_id"),
            workspace_title=_optional_str(payload, "workspace_title", ""),
            pane_id=_optional_nonempty_str(payload, "pane_id"),
            surface_id=_optional_nonempty_str(payload, "surface_id"),
            panel_id=_optional_nonempty_str(payload, "panel_id"),
        )


__all__ = [
    "EVENT_TERMINAL_SESSION_CLOSED",
    "EVENT_TERMINAL_SESSION_STATE",
    "KNOWN_EVENTS",
    "KNOWN_METHODS",
    "METHOD_CAPTURE_SCREEN",
    "METHOD_CLOSE_TERMINAL_SESSION",
    "METHOD_HELLO",
    "METHOD_LIST_TERMINAL_SESSIONS",
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
    "CmuxTerminalSession",
    "SidecarErrorCode",
    "map_sidecar_error_to_terminal",
]
