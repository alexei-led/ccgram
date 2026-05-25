from __future__ import annotations

import pytest

from ccgram.terminal_backends import (
    BACKEND_CMUX,
    BACKEND_TMUX,
    KNOWN_BACKENDS,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalBackendError,
    TerminalBackendUnavailableError,
    TerminalNotFoundError,
    TerminalOperationRejectedError,
    TerminalOperationTimeoutError,
    TerminalUnit,
    TerminalUnitRef,
    TerminalUnsupportedOperationError,
)


class TestTerminalUnitRef:
    def test_construct_tmux(self) -> None:
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@3")
        assert ref.backend == "tmux"
        assert ref.unit_id == "@3"
        assert ref.display_id == "tmux:@3"

    def test_construct_cmux(self) -> None:
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="workspace-uuid")
        assert ref.display_id == "cmux:workspace-uuid"

    def test_frozen(self) -> None:
        ref = TerminalUnitRef(backend="tmux", unit_id="@0")
        with pytest.raises(AttributeError):
            ref.unit_id = "@1"  # type: ignore[misc]

    @pytest.mark.parametrize("backend", ["", "   ", "tmuxx", "foo"])
    def test_rejects_invalid_backend(self, backend: str) -> None:
        with pytest.raises(ValueError):
            TerminalUnitRef(backend=backend, unit_id="@0")

    @pytest.mark.parametrize("unit_id", ["", "   "])
    def test_rejects_blank_unit_id(self, unit_id: str) -> None:
        with pytest.raises(ValueError):
            TerminalUnitRef(backend="tmux", unit_id=unit_id)

    def test_known_backends(self) -> None:
        assert {"tmux", "cmux"} == KNOWN_BACKENDS


class TestTerminalUnit:
    def test_defaults(self) -> None:
        ref = TerminalUnitRef(backend="tmux", unit_id="@0")
        unit = TerminalUnit(ref=ref)
        assert unit.title == ""
        assert unit.cwd is None
        assert unit.state == "unknown"
        assert unit.supports_capture is False
        assert unit.backend_metadata == {}

    def test_full(self) -> None:
        ref = TerminalUnitRef(backend="cmux", unit_id="ws-1")
        unit = TerminalUnit(
            ref=ref,
            title="proj",
            cwd="/tmp",
            provider_name="claude",
            state="ready",
            supports_capture=True,
            supports_send_text=True,
            supports_send_key=True,
            supports_close=True,
            supports_resume=False,
            backend_metadata={"event_seq": 42},
        )
        assert unit.state == "ready"
        assert unit.backend_metadata == {"event_seq": 42}


class TestTerminalBackendCapabilities:
    def test_defaults(self) -> None:
        caps = TerminalBackendCapabilities(backend="tmux")
        assert caps.supports_create is False
        assert caps.protocol_version == ""

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError):
            TerminalBackendCapabilities(backend="not-a-backend")


class TestErrorTaxonomy:
    @pytest.mark.parametrize(
        "exc_cls, code",
        [
            (TerminalBackendUnavailableError, "unavailable"),
            (TerminalNotFoundError, "not_found"),
            (TerminalUnsupportedOperationError, "unsupported"),
            (TerminalOperationTimeoutError, "timeout"),
            (TerminalOperationRejectedError, "rejected"),
        ],
    )
    def test_stable_codes(
        self,
        exc_cls: type[TerminalBackendError],
        code: str,
    ) -> None:
        err = exc_cls("boom")
        assert err.code == code
        assert isinstance(err, TerminalBackendError)

    def test_base_default_code(self) -> None:
        err = TerminalBackendError("boom")
        assert err.code == "internal_error"

    def test_str_includes_code_and_ref(self) -> None:
        ref = TerminalUnitRef(backend="cmux", unit_id="ws-1")
        err = TerminalNotFoundError("workspace gone", ref=ref)
        msg = str(err)
        assert "not_found" in msg
        assert "cmux:ws-1" in msg

    def test_explicit_code_override(self) -> None:
        err = TerminalBackendError("noop", code="rejected")
        assert err.code == "rejected"


class _FakeBackend:
    """Minimal protocol-conforming class for runtime_checkable test."""

    @property
    def name(self) -> str:
        return "tmux"

    def capabilities(self) -> TerminalBackendCapabilities:
        return TerminalBackendCapabilities(backend="tmux")

    async def list_units(self) -> list[TerminalUnit]:
        return []

    async def capture(
        self, ref: TerminalUnitRef, *, with_ansi: bool = False
    ) -> str | None:
        return None

    async def send_text(
        self, ref: TerminalUnitRef, text: str, *, raw: bool = False
    ) -> bool:
        return True

    async def send_key(self, ref: TerminalUnitRef, key: str) -> bool:
        return True

    async def close(self, ref: TerminalUnitRef) -> bool:
        return True


def test_terminal_backend_protocol_runtime_check() -> None:
    backend = _FakeBackend()
    assert isinstance(backend, TerminalBackend)


def test_terminal_backend_protocol_rejects_incomplete() -> None:
    class _Bare:
        pass

    assert not isinstance(_Bare(), TerminalBackend)
