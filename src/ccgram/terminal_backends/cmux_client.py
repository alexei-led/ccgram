"""cmux sidecar client — JSON-RPC over a local Unix socket (NDJSON framing).

The client is transport-agnostic: tests inject a fake
:class:`CmuxTransport` that records requests and yields canned
responses, while production wires a real ``asyncio.open_unix_connection``
through :class:`UnixSocketTransport`. Both shapes serialise requests as
one JSON object per line and parse responses the same way.

Responsibilities are kept narrow:

* Assign request ids and round-trip them through the transport.
* Translate sidecar errors into :class:`TerminalBackendError` subclasses
  using the stable code mapping in :mod:`.cmux_protocol`.
* Enforce per-request timeouts.
* Run the ``hello`` handshake on first use and reject incompatible
  protocol versions.

The client knows nothing about workspace bind UX or capabilities gating
— that lives in :class:`CmuxBackend`. It also never logs raw chat
payloads; only method names and ids are emitted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any, Protocol

import structlog

from .base import (
    TerminalBackendError,
    TerminalBackendUnavailableError,
    TerminalNotFoundError,
    TerminalOperationRejectedError,
    TerminalOperationTimeoutError,
    TerminalUnsupportedOperationError,
)
from .cmux_protocol import (
    METHOD_CAPTURE_SCREEN,
    METHOD_CLOSE_WORKSPACE,
    METHOD_HELLO,
    METHOD_LIST_WORKSPACES,
    METHOD_SEND_KEY,
    METHOD_SEND_TEXT,
    PROTOCOL_VERSION,
    CmuxError,
    CmuxHelloResult,
    CmuxProtocolError,
    CmuxRequest,
    CmuxResponse,
    CmuxWorkspace,
)

logger = structlog.get_logger()


class CmuxTransport(Protocol):
    """Send a single request frame and return the response frame.

    Implementations must enforce framing (NDJSON) and may multiplex
    multiple in-flight requests internally. The supplied ``timeout``
    is per-call and applies to the full request/response round-trip.
    """

    async def request(
        self, request: CmuxRequest, *, timeout: float
    ) -> CmuxResponse: ...

    async def close(self) -> None: ...


class UnixSocketTransport:
    """NDJSON-framed JSON-RPC transport over an ``asyncio`` Unix socket.

    Each :meth:`request` opens a fresh connection so concurrent calls
    don't have to share a reader/writer pair. That keeps the MVP simple
    and avoids correlating responses by id — a follow-up plan introduces
    a multiplexed long-lived connection when the event stream lands.
    """

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path

    async def request(self, request: CmuxRequest, *, timeout: float) -> CmuxResponse:
        try:
            return await asyncio.wait_for(self._round_trip(request), timeout=timeout)
        except TimeoutError as exc:
            raise TerminalOperationTimeoutError(
                f"cmux sidecar timed out after {timeout:.2f}s for {request.method}"
            ) from exc
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            raise TerminalBackendUnavailableError(
                f"cmux sidecar unreachable at {self._socket_path}: {exc}"
            ) from exc

    async def _round_trip(self, request: CmuxRequest) -> CmuxResponse:
        reader, writer = await asyncio.open_unix_connection(self._socket_path)
        try:
            payload = (json.dumps(request.to_wire()) + "\n").encode("utf-8")
            writer.write(payload)
            await writer.drain()
            line = await reader.readline()
            if not line:
                raise TerminalBackendUnavailableError(
                    "cmux sidecar closed the connection before responding"
                )
            return _decode_response_line(line)
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    async def close(self) -> None:  # pragma: no cover - nothing to close
        return None


def _decode_response_line(line: bytes) -> CmuxResponse:
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CmuxProtocolError("response was not valid UTF-8") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CmuxProtocolError(f"response was not valid JSON: {exc.msg}") from exc
    return CmuxResponse.from_wire(payload)


_ErrorMapper = Callable[[CmuxError], TerminalBackendError]


def _default_error_mapper(error: CmuxError) -> TerminalBackendError:
    code = error.to_terminal_code()
    message = error.message or error.code
    if code == "unavailable":
        return TerminalBackendUnavailableError(message)
    if code == "not_found":
        return TerminalNotFoundError(message)
    if code == "unsupported":
        return TerminalUnsupportedOperationError(message)
    if code == "timeout":
        return TerminalOperationTimeoutError(message)
    if code == "rejected":
        return TerminalOperationRejectedError(message)
    return TerminalBackendError(message, code=code)


class CmuxSidecarClient:
    """JSON-RPC client wrapping a :class:`CmuxTransport`.

    Construct directly with an injected transport for tests; the
    convenience constructor :meth:`with_unix_socket` builds the real
    socket transport. ``hello`` is invoked lazily on the first
    non-handshake request and the result cached.
    """

    def __init__(
        self,
        transport: CmuxTransport,
        *,
        timeout: float,
        required_protocol_version: str | None = None,
        clock: Callable[[], float] | None = None,
        error_mapper: _ErrorMapper | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        self._transport = transport
        self._timeout = timeout
        self._required_protocol_version = required_protocol_version
        self._error_mapper = error_mapper or _default_error_mapper
        self._next_id = 0
        self._hello: CmuxHelloResult | None = None
        self._hello_lock = asyncio.Lock()
        self._clock = clock  # reserved for future deadline tracking

    @classmethod
    def with_unix_socket(
        cls,
        socket_path: str,
        *,
        timeout_ms: int,
        required_protocol_version: str | None = None,
    ) -> CmuxSidecarClient:
        return cls(
            transport=UnixSocketTransport(socket_path),
            timeout=max(timeout_ms, 1) / 1000.0,
            required_protocol_version=required_protocol_version,
        )

    def _next_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    async def hello(self) -> CmuxHelloResult:
        """Negotiate protocol version and capabilities (cached)."""
        if self._hello is not None:
            return self._hello
        async with self._hello_lock:
            if self._hello is not None:
                return self._hello
            response = await self._send(METHOD_HELLO, {})
            result = CmuxHelloResult.from_result(self._require_result(response))
            self._validate_protocol(result)
            self._hello = result
            return result

    def cached_hello(self) -> CmuxHelloResult | None:
        return self._hello

    def _validate_protocol(self, result: CmuxHelloResult) -> None:
        required = self._required_protocol_version or PROTOCOL_VERSION
        if result.protocol_version != required:
            raise TerminalUnsupportedOperationError(
                f"cmux sidecar protocol version {result.protocol_version!r} "
                f"does not match required {required!r}"
            )

    async def _ensure_hello(self) -> CmuxHelloResult:
        return await self.hello()

    async def _send(self, method: str, params: dict[str, Any]) -> CmuxResponse:
        request = CmuxRequest(id=self._next_request_id(), method=method, params=params)
        logger.debug("cmux.request", method=method, request_id=request.id)
        response = await self._transport.request(request, timeout=self._timeout)
        if response.id != request.id:
            raise CmuxProtocolError(
                f"response id mismatch: expected {request.id}, got {response.id}"
            )
        return response

    def _require_result(self, response: CmuxResponse) -> dict[str, Any]:
        if response.error is not None:
            raise self._error_mapper(response.error)
        assert response.result is not None
        return response.result

    async def list_workspaces(self) -> list[CmuxWorkspace]:
        await self._ensure_hello()
        response = await self._send(METHOD_LIST_WORKSPACES, {})
        result = self._require_result(response)
        entries = result.get("workspaces", [])
        if not isinstance(entries, list):
            raise CmuxProtocolError("list_workspaces: result.workspaces must be a list")
        return [CmuxWorkspace.from_wire(entry) for entry in entries]

    async def capture_screen(
        self, workspace_id: str, *, with_ansi: bool = False
    ) -> str:
        await self._ensure_hello()
        response = await self._send(
            METHOD_CAPTURE_SCREEN,
            {"workspace_id": workspace_id, "with_ansi": with_ansi},
        )
        result = self._require_result(response)
        screen = result.get("screen", "")
        if not isinstance(screen, str):
            raise CmuxProtocolError("capture_screen: result.screen must be a string")
        return screen

    async def send_text(
        self, workspace_id: str, text: str, *, raw: bool = False
    ) -> bool:
        await self._ensure_hello()
        response = await self._send(
            METHOD_SEND_TEXT,
            {"workspace_id": workspace_id, "text": text, "raw": raw},
        )
        result = self._require_result(response)
        return bool(result.get("ok", True))

    async def send_key(self, workspace_id: str, key: str) -> bool:
        await self._ensure_hello()
        response = await self._send(
            METHOD_SEND_KEY,
            {"workspace_id": workspace_id, "key": key},
        )
        result = self._require_result(response)
        return bool(result.get("ok", True))

    async def close_workspace(self, workspace_id: str) -> bool:
        await self._ensure_hello()
        response = await self._send(
            METHOD_CLOSE_WORKSPACE, {"workspace_id": workspace_id}
        )
        result = self._require_result(response)
        return bool(result.get("ok", True))

    async def close(self) -> None:
        await self._transport.close()


__all__ = [
    "CmuxSidecarClient",
    "CmuxTransport",
    "UnixSocketTransport",
]
