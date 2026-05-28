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
    METHOD_CLOSE_TERMINAL_SESSION,
    METHOD_HELLO,
    METHOD_LIST_TERMINAL_SESSIONS,
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
    *,
    workspace_id: str | None = None,
) -> tuple[CmuxBackend, FakeTransport]:
    def respond(request: CmuxRequest) -> CmuxResponse:
        handler = responses.get(request.method)
        if handler is None:
            raise AssertionError(f"unexpected method {request.method!r}")
        return handler(request)

    transport = FakeTransport(respond)
    client = CmuxSidecarClient(transport, timeout=0.5)
    return CmuxBackend(client, workspace_id=workspace_id), transport


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
        assert caps.supports_capture is False
        assert caps.supports_send_text is False
        assert caps.supports_send_key is False

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
        synced = backend.capabilities()
        assert synced == caps


class TestBackendChecks:
    async def test_capture_rejects_tmux_ref(self) -> None:
        backend, _ = _make_backend({METHOD_HELLO: lambda r: _ok(r, _hello())})
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@1")
        with pytest.raises(TerminalUnsupportedOperationError):
            await backend.capture(ref)


class TestListUnits:
    async def test_list_units_translates_terminal_sessions(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                {
                    "terminal_sessions": [
                        {
                            "terminal_id": "term-a",
                            "title": "alpha",
                            "cwd": "/tmp/a",
                            "provider_name": "claude",
                            "state": "ready",
                            "workspace_id": "ws-a",
                            "workspace_title": "Feature A",
                        },
                        {
                            "terminal_id": "term-b",
                            "state": "unknown",
                        },
                    ]
                },
            )

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_LIST_TERMINAL_SESSIONS: respond_list,
            }
        )
        units = await backend.list_units()
        assert len(units) == 2
        first, second = units
        assert first.ref == TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-a")
        assert first.cwd == "/tmp/a"
        assert first.provider_name == "claude"
        assert first.state == "ready"
        assert first.supports_capture is True
        assert first.backend_metadata["workspace_id"] == "ws-a"
        assert first.backend_metadata["workspace_title"] == "Feature A"

        assert second.ref.unit_id == "term-b"
        assert second.supports_capture is True
        assert second.state == "unknown"

    async def test_list_units_reflects_negotiated_operation_capabilities(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                _hello(
                    supports_capture=False,
                    supports_send_text=False,
                    supports_send_key=False,
                    supports_close=True,
                ),
            )

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                {"terminal_sessions": [{"terminal_id": "term-a", "state": "ready"}]},
            )

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_LIST_TERMINAL_SESSIONS: respond_list,
            }
        )
        units = await backend.list_units()
        unit = units[0]
        assert unit.supports_capture is False
        assert unit.supports_send_text is False
        assert unit.supports_send_key is False
        assert unit.supports_close is True

    async def test_unknown_state_normalised_to_unknown(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                {"terminal_sessions": [{"terminal_id": "term-x", "state": "exploded"}]},
            )

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_LIST_TERMINAL_SESSIONS: respond_list,
            }
        )
        units = await backend.list_units()
        assert units[0].state == "unknown"

    async def test_workspace_scoped_backend_lists_only_matching_workspace(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                {
                    "terminal_sessions": [
                        {"terminal_id": "term-a", "workspace_id": "ws-a"},
                        {"terminal_id": "term-b", "workspace_id": "ws-b"},
                        {"terminal_id": "term-c"},
                    ]
                },
            )

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_LIST_TERMINAL_SESSIONS: respond_list,
            },
            workspace_id="ws-a",
        )
        units = await backend.list_units()
        assert [unit.ref.unit_id for unit in units] == ["term-a"]


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
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-1")
        screen = await backend.capture(ref, with_ansi=True)
        assert screen == "hi"
        assert captured == {"terminal_id": "term-1", "with_ansi": True}

    async def test_workspace_scope_checks_membership_before_capture(self) -> None:
        captured: dict[str, Any] = {}

        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                {
                    "terminal_sessions": [
                        {"terminal_id": "term-1", "workspace_id": "ws-a"}
                    ]
                },
            )

        def respond_capture(request: CmuxRequest) -> CmuxResponse:
            captured.update(request.params)
            return _ok(request, {"screen": "hi"})

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_LIST_TERMINAL_SESSIONS: respond_list,
                METHOD_CAPTURE_SCREEN: respond_capture,
            },
            workspace_id="ws-a",
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-1")
        assert await backend.capture(ref) == "hi"
        assert captured == {"terminal_id": "term-1", "with_ansi": False}

    async def test_workspace_scope_rejects_moved_out_terminal(self) -> None:
        capture_called = False

        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(
                request,
                {
                    "terminal_sessions": [
                        {"terminal_id": "term-1", "workspace_id": "ws-b"}
                    ]
                },
            )

        def respond_capture(request: CmuxRequest) -> CmuxResponse:
            nonlocal capture_called
            capture_called = True
            return _ok(request, {"screen": "hi"})

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_LIST_TERMINAL_SESSIONS: respond_list,
                METHOD_CAPTURE_SCREEN: respond_capture,
            },
            workspace_id="ws-a",
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-1")
        with pytest.raises(TerminalNotFoundError):
            await backend.capture(ref)
        assert capture_called is False

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
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-1")
        assert await backend.send_text(ref, "hello", raw=True) is True
        assert captured == {"terminal_id": "term-1", "text": "hello", "raw": True}

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
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-1")
        await backend.send_key(ref, "Up")
        assert captured == {"terminal_id": "term-1", "key": "Up"}

    async def test_close_forwards(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_close(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, {"ok": True})

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_CLOSE_TERMINAL_SESSION: respond_close,
            }
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-1")
        assert await backend.close(ref) is True


class TestErrorMapping:
    async def test_unavailable_during_list(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _err(request, "unavailable", "sidecar down")

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_LIST_TERMINAL_SESSIONS: respond_list,
            }
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
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-missing")
        with pytest.raises(TerminalNotFoundError) as excinfo:
            await backend.capture(ref)
        assert excinfo.value.ref == ref


class TestProtocolErrorTranslation:
    async def test_list_units_wraps_malformed_response(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_list(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, {"terminal_sessions": "not-a-list"})

        from ccgram.terminal_backends.base import TerminalBackendError

        backend, _ = _make_backend(
            {
                METHOD_HELLO: respond_hello,
                METHOD_LIST_TERMINAL_SESSIONS: respond_list,
            }
        )
        with pytest.raises(TerminalBackendError) as excinfo:
            await backend.list_units()
        assert excinfo.value.code == "internal_error"

    async def test_capture_wraps_malformed_screen(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, _hello())

        def respond_capture(request: CmuxRequest) -> CmuxResponse:
            return _ok(request, {"screen": 123})

        from ccgram.terminal_backends.base import TerminalBackendError

        backend, _ = _make_backend(
            {METHOD_HELLO: respond_hello, METHOD_CAPTURE_SCREEN: respond_capture}
        )
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-a")
        with pytest.raises(TerminalBackendError) as excinfo:
            await backend.capture(ref)
        assert excinfo.value.code == "internal_error"
        assert excinfo.value.ref == ref

    async def test_negotiate_wraps_id_mismatch(self) -> None:
        def respond_hello(request: CmuxRequest) -> CmuxResponse:
            return CmuxResponse(id=request.id + 999, result=_hello())

        from ccgram.terminal_backends.base import TerminalBackendError

        backend, _ = _make_backend({METHOD_HELLO: respond_hello})
        with pytest.raises(TerminalBackendError) as excinfo:
            await backend.negotiate()
        assert excinfo.value.code == "internal_error"
