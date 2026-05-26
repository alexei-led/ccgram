from __future__ import annotations

import pytest

from ccgram.terminal_backends.base import TerminalBackendErrorCode
from ccgram.terminal_backends.cmux_protocol import (
    EVENT_WORKSPACE_CLOSED,
    EVENT_WORKSPACE_STATE,
    KNOWN_EVENTS,
    KNOWN_METHODS,
    METHOD_HELLO,
    PROTOCOL_VERSION,
    SIDECAR_ERROR_CODES,
    CmuxError,
    CmuxEvent,
    CmuxHelloResult,
    CmuxProtocolError,
    CmuxRequest,
    CmuxResponse,
    CmuxWorkspace,
    map_sidecar_error_to_terminal,
)


class TestCmuxRequest:
    def test_to_wire_round_trip(self) -> None:
        req = CmuxRequest(id=7, method="hello", params={"client": "ccgram"})
        wire = req.to_wire()
        assert wire == {"id": 7, "method": "hello", "params": {"client": "ccgram"}}

    def test_default_params_independent_per_instance(self) -> None:
        a = CmuxRequest(id=1, method=METHOD_HELLO)
        b = CmuxRequest(id=2, method=METHOD_HELLO)
        a.params["x"] = 1  # type: ignore[index]
        assert "x" not in b.params

    def test_to_wire_copies_params(self) -> None:
        params = {"workspace_id": "abc"}
        req = CmuxRequest(id=1, method="x", params=params)
        wire = req.to_wire()
        wire["params"]["mutated"] = True
        assert "mutated" not in req.params

    def test_rejects_negative_id(self) -> None:
        with pytest.raises(ValueError):
            CmuxRequest(id=-1, method="hello")

    @pytest.mark.parametrize("method", ["", "   "])
    def test_rejects_blank_method(self, method: str) -> None:
        with pytest.raises(ValueError):
            CmuxRequest(id=0, method=method)


class TestCmuxError:
    def test_to_terminal_code_standard(self) -> None:
        assert (
            CmuxError(code="not_found", message="gone").to_terminal_code()
            == "not_found"
        )

    def test_to_terminal_code_incompatible_collapses_to_unsupported(self) -> None:
        err = CmuxError(code="incompatible_version", message="old")
        assert err.to_terminal_code() == "unsupported"

    def test_to_terminal_code_malformed_collapses_to_rejected(self) -> None:
        err = CmuxError(code="malformed_request", message="bad params")
        assert err.to_terminal_code() == "rejected"

    def test_unknown_code_falls_back_to_internal_error(self) -> None:
        err = CmuxError(code="exotic_failure", message="???")
        assert err.to_terminal_code() == "internal_error"

    def test_rejects_blank_code(self) -> None:
        with pytest.raises(ValueError):
            CmuxError(code="", message="x")


class TestCmuxResponse:
    def test_from_wire_result(self) -> None:
        response = CmuxResponse.from_wire({"id": 3, "result": {"ok": True}})
        assert response.id == 3
        assert response.result == {"ok": True}
        assert response.error is None

    def test_from_wire_error(self) -> None:
        response = CmuxResponse.from_wire(
            {
                "id": 5,
                "error": {"code": "timeout", "message": "too slow"},
            }
        )
        assert response.id == 5
        assert response.result is None
        assert response.error is not None
        assert response.error.code == "timeout"

    @pytest.mark.parametrize(
        "payload, fragment",
        [
            ({}, "id"),
            ({"id": "1", "result": {}}, "id"),
            ({"id": 1}, "neither result nor error"),
            ({"id": 1, "result": "stringy"}, "result"),
            ({"id": 1, "error": "string"}, "error"),
            ({"id": 1, "error": {"code": "", "message": ""}}, "code"),
            ({"id": 1, "error": {"code": "x", "message": 5}}, "message"),
            ({"id": 1, "error": {"code": "x", "message": "", "data": 1}}, "data"),
        ],
    )
    def test_from_wire_malformed(self, payload, fragment: str) -> None:
        with pytest.raises(CmuxProtocolError, match=fragment):
            CmuxResponse.from_wire(payload)

    def test_from_wire_rejects_bool_id(self) -> None:
        with pytest.raises(CmuxProtocolError, match="id"):
            CmuxResponse.from_wire({"id": True, "result": {}})

    def test_post_init_rejects_both_result_and_error(self) -> None:
        with pytest.raises(CmuxProtocolError):
            CmuxResponse(id=1, result={}, error=CmuxError(code="x", message="y"))

    def test_post_init_rejects_neither(self) -> None:
        with pytest.raises(CmuxProtocolError):
            CmuxResponse(id=1)


class TestCmuxEvent:
    def test_from_wire(self) -> None:
        event = CmuxEvent.from_wire(
            {"method": EVENT_WORKSPACE_STATE, "params": {"workspace_id": "abc"}}
        )
        assert event.method == EVENT_WORKSPACE_STATE
        assert event.params == {"workspace_id": "abc"}

    def test_from_wire_defaults_params_empty(self) -> None:
        event = CmuxEvent.from_wire({"method": EVENT_WORKSPACE_CLOSED})
        assert event.params == {}

    def test_from_wire_malformed(self) -> None:
        with pytest.raises(CmuxProtocolError):
            CmuxEvent.from_wire({"params": {}})

    def test_known_events_listed(self) -> None:
        assert EVENT_WORKSPACE_STATE in KNOWN_EVENTS
        assert EVENT_WORKSPACE_CLOSED in KNOWN_EVENTS


class TestCmuxHelloResult:
    def test_from_result_minimal(self) -> None:
        hello = CmuxHelloResult.from_result(
            {"protocol_version": "1", "sidecar_version": "0.1.0"}
        )
        assert hello.protocol_version == "1"
        assert hello.sidecar_version == "0.1.0"
        assert hello.supports_capture is True

    def test_from_result_full(self) -> None:
        hello = CmuxHelloResult.from_result(
            {
                "protocol_version": "1",
                "sidecar_version": "0.2.0",
                "supports_create": True,
                "supports_capture": False,
                "supports_send_text": False,
                "supports_send_key": False,
                "supports_close": True,
                "supports_event_stream": True,
            }
        )
        assert hello.supports_create is True
        assert hello.supports_event_stream is True
        assert hello.supports_send_text is False

    def test_from_result_missing_protocol_version(self) -> None:
        with pytest.raises(CmuxProtocolError):
            CmuxHelloResult.from_result({"sidecar_version": "x"})

    @pytest.mark.parametrize(
        "field",
        [
            "supports_create",
            "supports_capture",
            "supports_send_text",
            "supports_send_key",
            "supports_close",
            "supports_event_stream",
        ],
    )
    def test_from_result_rejects_non_bool_capability(self, field: str) -> None:
        with pytest.raises(CmuxProtocolError, match=field):
            CmuxHelloResult.from_result(
                {
                    "protocol_version": "1",
                    "sidecar_version": "0.2.0",
                    field: "false",
                }
            )


class TestCmuxWorkspace:
    def test_from_wire_minimal(self) -> None:
        ws = CmuxWorkspace.from_wire({"workspace_id": "abc"})
        assert ws.workspace_id == "abc"
        assert ws.has_terminal_surface is True
        assert ws.state == "unknown"

    def test_from_wire_full(self) -> None:
        ws = CmuxWorkspace.from_wire(
            {
                "workspace_id": "ws-1",
                "title": "demo",
                "cwd": "/tmp",
                "provider_name": "claude",
                "state": "ready",
                "has_terminal_surface": False,
            }
        )
        assert ws.title == "demo"
        assert ws.cwd == "/tmp"
        assert ws.state == "ready"
        assert ws.has_terminal_surface is False

    def test_from_wire_accepts_legacy_id_alias(self) -> None:
        ws = CmuxWorkspace.from_wire({"id": "legacy-1"})
        assert ws.workspace_id == "legacy-1"

    @pytest.mark.parametrize(
        "payload, fragment",
        [
            ({}, "workspace_id"),
            ({"workspace_id": ""}, "workspace_id"),
            ({"workspace_id": "ok", "cwd": 5}, "cwd"),
            ({"workspace_id": "ok", "provider_name": 5}, "provider_name"),
            ({"workspace_id": "ok", "title": 5}, "title"),
            ({"workspace_id": "ok", "state": 5}, "state"),
            (
                {"workspace_id": "ok", "has_terminal_surface": "0"},
                "has_terminal_surface",
            ),
        ],
    )
    def test_from_wire_malformed(self, payload, fragment: str) -> None:
        with pytest.raises(CmuxProtocolError, match=fragment):
            CmuxWorkspace.from_wire(payload)


class TestErrorMapping:
    def test_every_sidecar_code_maps_to_terminal_code(self) -> None:
        for code in SIDECAR_ERROR_CODES:
            mapped = map_sidecar_error_to_terminal(code)
            assert isinstance(mapped, str)
            assert mapped in {
                "unavailable",
                "not_found",
                "unsupported",
                "no_terminal_surface",
                "timeout",
                "rejected",
                "internal_error",
            }

    def test_unknown_code_maps_to_internal_error(self) -> None:
        code: TerminalBackendErrorCode = map_sidecar_error_to_terminal("totally_new")
        assert code == "internal_error"


class TestConstants:
    def test_protocol_version_is_one(self) -> None:
        assert PROTOCOL_VERSION == "1"

    def test_method_constants_known(self) -> None:
        assert METHOD_HELLO in KNOWN_METHODS
        assert len(KNOWN_METHODS) >= 6
