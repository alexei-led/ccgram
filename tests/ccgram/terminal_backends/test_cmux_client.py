from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ccgram.terminal_backends.base import (
    TerminalBackendUnavailableError,
    TerminalNotFoundError,
    TerminalOperationRejectedError,
    TerminalOperationTimeoutError,
    TerminalUnsupportedOperationError,
)
from ccgram.terminal_backends.cmux_client import (
    CmuxSidecarClient,
    CmuxTransport,
    UnixSocketTransport,
)
from ccgram.terminal_backends.cmux_protocol import (
    METHOD_CAPTURE_SCREEN,
    METHOD_CLOSE_WORKSPACE,
    METHOD_HELLO,
    METHOD_LIST_WORKSPACES,
    METHOD_SEND_KEY,
    METHOD_SEND_TEXT,
    PROTOCOL_VERSION,
    CmuxError,
    CmuxProtocolError,
    CmuxRequest,
    CmuxResponse,
)


@pytest.fixture
def short_socket() -> Iterator[Path]:
    # AF_UNIX paths are capped near 100 bytes on macOS; pytest's tmp_path
    # often exceeds that. Use /tmp with a short random name instead.
    sock = Path("/tmp") / f"ccgram-cmux-{uuid.uuid4().hex[:8]}.sock"
    try:
        yield sock
    finally:
        if sock.exists():
            sock.unlink(missing_ok=True)


FakeResponder = Callable[[CmuxRequest], Awaitable[CmuxResponse] | CmuxResponse]


class FakeTransport(CmuxTransport):
    """Test transport — records every request and yields canned responses."""

    def __init__(self, responder: FakeResponder) -> None:
        self._responder = responder
        self.requests: list[CmuxRequest] = []
        self.closed = False

    async def request(self, request: CmuxRequest, *, timeout: float) -> CmuxResponse:
        del timeout  # FakeTransport bypasses real timeouts
        self.requests.append(request)
        result = self._responder(request)
        if asyncio.iscoroutine(result):
            return await result
        assert isinstance(result, CmuxResponse)
        return result

    async def close(self) -> None:
        self.closed = True


def _hello_result(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "sidecar_version": "0.1.0",
    }
    base.update(overrides)
    return base


def _scripted_responder(
    handlers: dict[str, Callable[[CmuxRequest], Any]],
) -> FakeResponder:
    def respond(request: CmuxRequest) -> Any:
        handler = handlers.get(request.method)
        if handler is None:
            raise AssertionError(f"unexpected method {request.method!r}")
        return handler(request)

    return respond


def _ok(request: CmuxRequest, result: dict[str, Any]) -> CmuxResponse:
    return CmuxResponse(id=request.id, result=result)


def _err(request: CmuxRequest, code: str, message: str = "") -> CmuxResponse:
    return CmuxResponse(id=request.id, error=CmuxError(code=code, message=message))


def _make_client(
    responder: FakeResponder,
    *,
    timeout: float = 0.5,
    required_version: str | None = None,
) -> tuple[CmuxSidecarClient, FakeTransport]:
    transport = FakeTransport(responder)
    client = CmuxSidecarClient(
        transport,
        timeout=timeout,
        required_protocol_version=required_version,
    )
    return client, transport


class TestHandshake:
    async def test_hello_caches_result(self) -> None:
        calls = {"hello": 0}

        def respond(request: CmuxRequest) -> CmuxResponse:
            calls["hello"] += 1
            return _ok(request, _hello_result())

        client, transport = _make_client(_scripted_responder({METHOD_HELLO: respond}))
        a = await client.hello()
        b = await client.hello()
        assert a is b
        assert calls["hello"] == 1
        assert transport.requests[0].method == METHOD_HELLO

    async def test_hello_is_run_lazily_before_first_op(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello_result())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, {"workspaces": []})

        client, transport = _make_client(
            _scripted_responder(
                {METHOD_HELLO: respond_hello, METHOD_LIST_WORKSPACES: respond_list}
            )
        )
        await client.list_workspaces()
        methods = [r.method for r in transport.requests]
        assert methods == [METHOD_HELLO, METHOD_LIST_WORKSPACES]

    async def test_incompatible_version_raises_unsupported(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello_result(protocol_version="999"))

        client, _ = _make_client(_scripted_responder({METHOD_HELLO: respond}))
        with pytest.raises(TerminalUnsupportedOperationError):
            await client.hello()

    async def test_required_version_override(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello_result(protocol_version="2"))

        client, _ = _make_client(
            _scripted_responder({METHOD_HELLO: respond}), required_version="2"
        )
        hello = await client.hello()
        assert hello.protocol_version == "2"

    async def test_concurrent_hello_calls_share_one_handshake(self) -> None:
        calls = {"count": 0}

        async def respond(request: CmuxRequest) -> CmuxResponse:
            calls["count"] += 1
            await asyncio.sleep(0)
            return _ok(request, _hello_result())

        transport = FakeTransport(_scripted_responder({METHOD_HELLO: respond}))
        client = CmuxSidecarClient(transport, timeout=0.5)
        results = await asyncio.gather(client.hello(), client.hello(), client.hello())
        assert all(r is results[0] for r in results)
        assert calls["count"] == 1


class TestRequestIdMismatch:
    async def test_response_id_mismatch_raises_protocol_error(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            if request.method == METHOD_HELLO:
                return _ok(request, _hello_result())
            return CmuxResponse(id=request.id + 9, result={"workspaces": []})

        client, _ = _make_client(
            _scripted_responder(
                {METHOD_HELLO: respond, METHOD_LIST_WORKSPACES: respond}
            )
        )
        with pytest.raises(CmuxProtocolError, match="id mismatch"):
            await client.list_workspaces()


class TestErrorMapping:
    @pytest.mark.parametrize(
        "code, exc",
        [
            ("unavailable", TerminalBackendUnavailableError),
            ("not_found", TerminalNotFoundError),
            ("unsupported", TerminalUnsupportedOperationError),
            ("timeout", TerminalOperationTimeoutError),
            ("rejected", TerminalOperationRejectedError),
            ("malformed_request", TerminalOperationRejectedError),
            ("incompatible_version", TerminalUnsupportedOperationError),
        ],
    )
    async def test_sidecar_error_codes_translate(self, code: str, exc) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            if request.method == METHOD_HELLO:
                return _ok(request, _hello_result())
            return _err(request, code)

        client, _ = _make_client(
            _scripted_responder(
                {
                    METHOD_HELLO: respond,
                    METHOD_LIST_WORKSPACES: respond,
                }
            )
        )
        with pytest.raises(exc):
            await client.list_workspaces()


class TestOperations:
    async def test_list_workspaces_returns_parsed_entries(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            if request.method == METHOD_HELLO:
                return _ok(request, _hello_result())
            return _ok(
                request,
                {
                    "workspaces": [
                        {"workspace_id": "a", "title": "Alpha"},
                        {"workspace_id": "b"},
                    ]
                },
            )

        client, _ = _make_client(
            _scripted_responder(
                {METHOD_HELLO: respond, METHOD_LIST_WORKSPACES: respond}
            )
        )
        workspaces = await client.list_workspaces()
        ids = [w.workspace_id for w in workspaces]
        assert ids == ["a", "b"]

    async def test_capture_screen_returns_string(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            if request.method == METHOD_HELLO:
                return _ok(request, _hello_result())
            return _ok(request, {"screen": "hello\n$ "})

        client, transport = _make_client(
            _scripted_responder({METHOD_HELLO: respond, METHOD_CAPTURE_SCREEN: respond})
        )
        screen = await client.capture_screen("ws-1", with_ansi=True)
        assert screen == "hello\n$ "
        capture_call = transport.requests[1]
        assert capture_call.params == {"workspace_id": "ws-1", "with_ansi": True}

    async def test_send_text_passes_params(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            if request.method == METHOD_HELLO:
                return _ok(request, _hello_result())
            return _ok(request, {"ok": True})

        client, transport = _make_client(
            _scripted_responder({METHOD_HELLO: respond, METHOD_SEND_TEXT: respond})
        )
        ok = await client.send_text("ws-1", "hi", raw=True)
        assert ok is True
        params = transport.requests[1].params
        assert params == {"workspace_id": "ws-1", "text": "hi", "raw": True}

    async def test_send_key_passes_key(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            if request.method == METHOD_HELLO:
                return _ok(request, _hello_result())
            return _ok(request, {"ok": True})

        client, transport = _make_client(
            _scripted_responder({METHOD_HELLO: respond, METHOD_SEND_KEY: respond})
        )
        await client.send_key("ws-1", "Escape")
        assert transport.requests[1].params == {"workspace_id": "ws-1", "key": "Escape"}

    async def test_close_workspace(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            if request.method == METHOD_HELLO:
                return _ok(request, _hello_result())
            return _ok(request, {"ok": True})

        client, _ = _make_client(
            _scripted_responder(
                {METHOD_HELLO: respond, METHOD_CLOSE_WORKSPACE: respond}
            )
        )
        assert await client.close_workspace("ws-1") is True

    async def test_list_workspaces_rejects_non_list(self) -> None:
        def respond(request: CmuxRequest) -> CmuxResponse:
            if request.method == METHOD_HELLO:
                return _ok(request, _hello_result())
            return _ok(request, {"workspaces": "nope"})

        client, _ = _make_client(
            _scripted_responder(
                {METHOD_HELLO: respond, METHOD_LIST_WORKSPACES: respond}
            )
        )
        with pytest.raises(CmuxProtocolError):
            await client.list_workspaces()


class TestUnixSocketTransport:
    async def test_round_trip_with_local_server(self, short_socket: Path) -> None:
        socket_path = short_socket

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            line = await reader.readline()
            request = json.loads(line.decode("utf-8"))
            response = {
                "id": request["id"],
                "result": {"echo": request["params"]},
            }
            writer.write((json.dumps(response) + "\n").encode("utf-8"))
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=str(socket_path))
        try:
            transport = UnixSocketTransport(str(socket_path))
            request = CmuxRequest(id=11, method=METHOD_HELLO, params={"x": 1})
            response = await transport.request(request, timeout=1.0)
            assert response.id == 11
            assert response.result == {"echo": {"x": 1}}
        finally:
            server.close()
            await server.wait_closed()

    async def test_unreachable_socket_raises_unavailable(
        self, short_socket: Path
    ) -> None:
        transport = UnixSocketTransport(str(short_socket))
        with pytest.raises(TerminalBackendUnavailableError):
            await transport.request(CmuxRequest(id=0, method=METHOD_HELLO), timeout=0.2)

    async def test_truncated_response_raises_unavailable(
        self, short_socket: Path
    ) -> None:
        socket_path = short_socket

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            await reader.readline()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=str(socket_path))
        try:
            transport = UnixSocketTransport(str(socket_path))
            with pytest.raises(TerminalBackendUnavailableError):
                await transport.request(
                    CmuxRequest(id=0, method=METHOD_HELLO), timeout=1.0
                )
        finally:
            server.close()
            await server.wait_closed()

    async def test_malformed_json_raises_protocol_error(
        self, short_socket: Path
    ) -> None:
        socket_path = short_socket

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            await reader.readline()
            writer.write(b"not json at all\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=str(socket_path))
        try:
            transport = UnixSocketTransport(str(socket_path))
            with pytest.raises(CmuxProtocolError):
                await transport.request(
                    CmuxRequest(id=0, method=METHOD_HELLO), timeout=1.0
                )
        finally:
            server.close()
            await server.wait_closed()

    async def test_event_before_response_is_ignored(self, short_socket: Path) -> None:
        socket_path = short_socket

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            line = await reader.readline()
            request = json.loads(line.decode("utf-8"))
            writer.write(
                b'{"method":"workspace_state","params":{"workspace_id":"ws-1"}}\n'
            )
            writer.write(
                (
                    json.dumps({"id": request["id"], "result": {"ok": True}}) + "\n"
                ).encode("utf-8")
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=str(socket_path))
        try:
            transport = UnixSocketTransport(str(socket_path))
            response = await transport.request(
                CmuxRequest(id=7, method=METHOD_HELLO), timeout=1.0
            )
            assert response.id == 7
            assert response.result == {"ok": True}
        finally:
            server.close()
            await server.wait_closed()

    async def test_timeout_when_server_never_responds(self, short_socket: Path) -> None:
        socket_path = short_socket
        stop = asyncio.Event()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            await reader.readline()
            try:
                await stop.wait()
            finally:
                writer.close()

        server = await asyncio.start_unix_server(handle, path=str(socket_path))
        try:
            transport = UnixSocketTransport(str(socket_path))
            with pytest.raises(TerminalOperationTimeoutError):
                await transport.request(
                    CmuxRequest(id=0, method=METHOD_HELLO), timeout=0.05
                )
        finally:
            stop.set()
            server.close()
            await server.wait_closed()


class TestConstruction:
    def test_rejects_non_positive_timeout(self) -> None:
        transport = FakeTransport(lambda r: _ok(r, {}))
        with pytest.raises(ValueError):
            CmuxSidecarClient(transport, timeout=0)

    def test_with_unix_socket_factory(self, tmp_path) -> None:
        client = CmuxSidecarClient.with_unix_socket(
            str(tmp_path / "x.sock"), timeout_ms=1000
        )
        assert client._timeout == 1.0  # type: ignore[attr-defined]
