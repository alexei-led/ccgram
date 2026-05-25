from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from ccgram.terminal_backends.base import (
    BACKEND_CMUX,
    BACKEND_TMUX,
    TerminalBackendUnavailableError,
    TerminalNotFoundError,
    TerminalUnitRef,
    TerminalUnsupportedOperationError,
)
from ccgram.terminal_backends.cmux import CmuxBackend
from ccgram.terminal_backends.cmux_client import CmuxSidecarClient, CmuxTransport
from ccgram.terminal_backends.cmux_protocol import (
    METHOD_CAPTURE_SCREEN,
    METHOD_CLOSE_WORKSPACE,
    METHOD_HELLO,
    METHOD_LIST_WORKSPACES,
    METHOD_SEND_KEY,
    METHOD_SEND_TEXT,
    PROTOCOL_VERSION,
    CmuxError,
    CmuxRequest,
    CmuxResponse,
)


FakeResponder = Callable[[CmuxRequest], Awaitable[CmuxResponse] | CmuxResponse]


class FakeTransport(CmuxTransport):
    def __init__(self, responder: FakeResponder) -> None:
        self._responder = responder
        self.requests: list[CmuxRequest] = []

    async def request(self, request: CmuxRequest, *, timeout: float) -> CmuxResponse:
        del timeout
        self.requests.append(request)
        result = self._responder(request)
        if asyncio.iscoroutine(result):
            return await result
        assert isinstance(result, CmuxResponse)
        return result

    async def close(self) -> None:
        return None


def _hello(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "sidecar_version": "0.1.0",
    }
    base.update(overrides)
    return base


def _make_backend(
    responses: dict[str, Callable[[CmuxRequest], CmuxResponse]],
) -> tuple[CmuxBackend, FakeTransport]:
    def respond(request: CmuxRequest) -> CmuxResponse:
        handler = responses.get(request.method)
        if handler is None:
            raise AssertionError(f"unexpected method {request.method!r}")
        return handler(request)

    transport = FakeTransport(respond)
    client = CmuxSidecarClient(transport, timeout=0.5)
    return CmuxBackend(client), transport


def _ok(request: CmuxRequest, result: dict[str, Any]) -> CmuxResponse:
    return CmuxResponse(id=request.id, result=result)


def _err(request: CmuxRequest, code: str, message: str = "") -> CmuxResponse:
    return CmuxResponse(id=request.id, error=CmuxError(code=code, message=message))


class TestBackendName:
    def test_name(self) -> None:
        backend, _ = _make_backend({})
        assert backend.name == BACKEND_CMUX


class TestCapabilitiesBeforeHandshake:
    def test_defaults_are_conservative(self) -> None:
        backend, _ = _make_backend({})
        caps = backend.capabilities()
        assert caps.backend == BACKEND_CMUX
        assert caps.supports_create is False
        assert caps.supports_close is False
        assert caps.supports_resume is False
        assert caps.supports_capture is True
        assert caps.supports_send_text is True

    async def test_capabilities_after_negotiate_reflect_hello(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                _hello(
                    supports_create=True,
                    supports_close=True,
                    supports_event_stream=True,
                ),
            )

        backend, _ = _make_backend({METHOD_HELLO: respond_hello})
        caps = await backend.negotiate()
        assert caps.supports_create is True
        assert caps.supports_close is True
        assert caps.supports_event_stream is True
        # Subsequent sync read uses cached hello.
        synced = backend.capabilities()
        assert synced == caps


class TestBackendChecks:
    async def test_capture_rejects_tmux_ref(self) -> None:
        backend, _ = _make_backend({METHOD_HELLO: lambda r: _ok(r, _hello())})
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@1")
        with pytest.raises(TerminalUnsupportedOperationError):
            await backend.capture(ref)


class TestListUnits:
    async def test_list_units_translates_workspaces(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                {
                    "workspaces": [
                        {
                            "workspace_id": "ws-a",
                            "title": "alpha",
                            "cwd": "/tmp/a",
                            "provider_name": "claude",
                            "state": "ready",
                        },
                        {
                            "workspace_id": "ws-b",
                            "has_terminal_surface": False,
                            "state": "unknown",
                        },
                    ]
                },
            )

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_LIST_WORKSPACES: respond_list}
        )
        units = await backend.list_units()
        assert len(units) == 2
        first, second = units
        assert first.ref == TerminalUnitRef(backend=BACKEND_CMUX, unit_id="ws-a")
        assert first.cwd == "/tmp/a"
        assert first.provider_name == "claude"
        assert first.state == "ready"
        assert first.supports_capture is True

        assert second.ref.unit_id == "ws-b"
        assert second.supports_capture is False
        assert second.state == "unknown"

    async def test_unknown_state_normalised_to_unknown(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                {"workspaces": [{"workspace_id": "ws-x", "state": "exploded"}]},
            )

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_LIST_WORKSPACES: respond_list}
        )
        units = await backend.list_units()
        assert units[0].state == "unknown"


class TestSendAndCapture:
    async def test_capture_forwards_with_ansi(self) -> None:
        captured: dict[str, Any] = {}

        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_capture(request: CmuxRequest) -> CmuxResponse:
            captured.update(request.params)
            return _ok(request, {"screen": "hi"})

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_CAPTURE_SCREEN: respond_capture}
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="ws-1")
        screen = await backend.capture(ref, with_ansi=True)
        assert screen == "hi"
        assert captured == {"workspace_id": "ws-1", "with_ansi": True}

    async def test_send_text_forwards_raw_flag(self) -> None:
        captured: dict[str, Any] = {}

        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_send(request: CmuxRequest) -> CmuxResponse:
            captured.update(request.params)
            return _ok(request, {"ok": True})

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_SEND_TEXT: respond_send}
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="ws-1")
        assert await backend.send_text(ref, "hello", raw=True) is True
        assert captured == {"workspace_id": "ws-1", "text": "hello", "raw": True}

    async def test_send_key_forwards_key(self) -> None:
        captured: dict[str, Any] = {}

        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_send_key(request: CmuxRequest) -> CmuxResponse:
            captured.update(request.params)
            return _ok(request, {"ok": True})

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_SEND_KEY: respond_send_key}
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="ws-1")
        await backend.send_key(ref, "Up")
        assert captured == {"workspace_id": "ws-1", "key": "Up"}

    async def test_close_forwards(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_close(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, {"ok": True})

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_CLOSE_WORKSPACE: respond_close}
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="ws-1")
        assert await backend.close(ref) is True


class TestErrorMapping:
    async def test_unavailable_during_list(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _err(request, "unavailable", "sidecar down")

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_LIST_WORKSPACES: respond_list}
        )
        with pytest.raises(TerminalBackendUnavailableError):
            await backend.list_units()

    async def test_not_found_during_capture(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_capture(request: CmuxRequest) -> CmuxResponse:
            return _err(request, "not_found", "missing")

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_CAPTURE_SCREEN: respond_capture}
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="ws-missing")
        with pytest.raises(TerminalNotFoundError):
            await backend.capture(ref)
