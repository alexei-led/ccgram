from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

import ccgram.terminal_operations as terminal_operations
from ccgram.terminal_backends import (
    BACKEND_CMUX,
    BACKEND_TMUX,
    TerminalBackendCapabilities,
    TerminalBackendError,
    TerminalBackendUnavailableError,
    TerminalNotFoundError,
    TerminalUnit,
    TerminalUnitRef,
)
from ccgram.terminal_backends.router import (
    TerminalBackendRouter,
    get_router,
    reset_router_for_testing,
)


class _SpyBackend:
    """Duck-typed ``TerminalBackend`` whose ops are AsyncMocks for spying."""

    send_text: AsyncMock
    capture: AsyncMock
    send_key: AsyncMock
    close: AsyncMock
    list_units: AsyncMock

    def __init__(self, name: str = BACKEND_TMUX) -> None:
        self._name = name
        self.send_text = AsyncMock(return_value=True)
        self.capture = AsyncMock(return_value="snap")
        self.send_key = AsyncMock(return_value=True)
        self.close = AsyncMock(return_value=True)
        self.list_units = AsyncMock(return_value=[])

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> TerminalBackendCapabilities:
        return TerminalBackendCapabilities(backend=self._name, supports_send_text=True)


@pytest.fixture
def fake_backend() -> Iterator[_SpyBackend]:
    reset_router_for_testing()
    router: TerminalBackendRouter = get_router()
    # Replace tmux with the spy so we never hit libtmux.
    backend = _SpyBackend(BACKEND_TMUX)
    router.register(backend)
    yield backend
    reset_router_for_testing()


@pytest.fixture
def patch_identity(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Allow tests to choose what get_terminal_identity returns."""

    def _set(result: Any) -> None:
        monkeypatch.setattr(
            terminal_operations, "get_terminal_identity", lambda window_id: result
        )

    return _set


@pytest.fixture(autouse=True)
def patch_display_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        @staticmethod
        def get_display_name(window_id: str) -> str:
            return f"name({window_id})"

    monkeypatch.setattr(terminal_operations, "thread_router", _Stub())


class TestResolveRef:
    def test_legacy_window_resolves_to_tmux(self, patch_identity: Any) -> None:
        patch_identity(None)
        ref = terminal_operations._resolve_ref("@3")
        assert ref.backend == BACKEND_TMUX
        assert ref.unit_id == "@3"

    def test_identity_overrides_to_cmux(self, patch_identity: Any) -> None:
        from ccgram.window_state_ports.terminal_identity import (
            TerminalIdentityProjection,
        )

        patch_identity(
            TerminalIdentityProjection(
                window_id="@7",
                backend=BACKEND_CMUX,
                unit_id="ws-1",
                legacy_window_id="@7",
            )
        )
        ref = terminal_operations._resolve_ref("@7")
        assert ref.backend == BACKEND_CMUX
        assert ref.unit_id == "ws-1"

    def test_blank_unit_id_falls_back_to_window_id(self, patch_identity: Any) -> None:
        from ccgram.window_state_ports.terminal_identity import (
            TerminalIdentityProjection,
        )

        patch_identity(
            TerminalIdentityProjection(
                window_id="@7",
                backend=BACKEND_TMUX,
                unit_id="",
                legacy_window_id="@7",
            )
        )
        ref = terminal_operations._resolve_ref("@7")
        assert ref.unit_id == "@7"


class TestSendText:
    async def test_success_returns_sent_to_display(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        ok, msg = await terminal_operations.send_text("@1", "hi")
        assert ok is True
        assert msg == "Sent to name(@1)"
        fake_backend.send_text.assert_awaited_once()
        call_ref: TerminalUnitRef = fake_backend.send_text.call_args.args[0]
        assert call_ref.unit_id == "@1"
        assert call_ref.backend == BACKEND_TMUX
        assert fake_backend.send_text.call_args.kwargs == {"raw": False}

    async def test_raw_flag_passed_through(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        await terminal_operations.send_text("@1", "ls", raw=True)
        assert fake_backend.send_text.call_args.kwargs == {"raw": True}

    async def test_failure_returns_failed(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        fake_backend.send_text.return_value = False
        ok, msg = await terminal_operations.send_text("@1", "hi")
        assert ok is False
        assert msg == "Failed to send keys"

    async def test_not_found_maps_to_legacy_message(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        fake_backend.send_text.side_effect = TerminalNotFoundError(
            "gone", ref=TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@1")
        )
        ok, msg = await terminal_operations.send_text("@1", "hi")
        assert ok is False
        assert msg == "Window not found (may have been closed)"

    async def test_backend_unavailable_returns_scoped_error(
        self, patch_identity: Any
    ) -> None:
        reset_router_for_testing()
        # Empty router — tmux not registered.
        router = TerminalBackendRouter()

        def _empty_router() -> TerminalBackendRouter:
            return router

        import ccgram.terminal_backends.router as router_mod

        original = router_mod.get_router
        router_mod.get_router = _empty_router  # type: ignore[assignment]
        try:
            patch_identity(None)
            ok, msg = await terminal_operations.send_text("@1", "hi")
            assert ok is False
            assert "unavailable" in msg
        finally:
            router_mod.get_router = original  # type: ignore[assignment]
            reset_router_for_testing()

    async def test_backend_error_returns_scoped_code(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        fake_backend.send_text.side_effect = TerminalBackendError(
            "kaboom", code="rejected"
        )
        ok, msg = await terminal_operations.send_text("@1", "hi")
        assert ok is False
        assert "rejected" in msg


class TestCapture:
    async def test_capture_returns_text(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        text = await terminal_operations.capture("@1", with_ansi=True)
        assert text == "snap"
        fake_backend.capture.assert_awaited_once()
        assert fake_backend.capture.call_args.kwargs == {"with_ansi": True}

    async def test_capture_propagates_not_found(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        fake_backend.capture.side_effect = TerminalNotFoundError("gone")
        with pytest.raises(TerminalNotFoundError):
            await terminal_operations.capture("@1")

    async def test_capture_returns_none_on_failure(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        fake_backend.capture.return_value = None
        assert await terminal_operations.capture("@1") is None


class TestSendKey:
    async def test_send_key_delegates(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        assert await terminal_operations.send_key("@1", "Escape") is True
        fake_backend.send_key.assert_awaited_once()
        assert fake_backend.send_key.call_args.args[1] == "Escape"

    async def test_send_key_not_found_returns_false(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        fake_backend.send_key.side_effect = TerminalNotFoundError("gone")
        assert await terminal_operations.send_key("@1", "Escape") is False


class TestClose:
    async def test_close_delegates(
        self, fake_backend: _SpyBackend, patch_identity: Any
    ) -> None:
        patch_identity(None)
        assert await terminal_operations.close("@1") is True
        fake_backend.close.assert_awaited_once()


class TestListUnits:
    async def test_list_units_returns_backend_projection(
        self, fake_backend: _SpyBackend
    ) -> None:
        fake_backend.list_units.return_value = [
            TerminalUnit(ref=TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@0"))
        ]
        units = await terminal_operations.list_units(BACKEND_TMUX)
        assert [u.ref.unit_id for u in units] == ["@0"]

    async def test_list_units_unknown_backend_raises(
        self, fake_backend: _SpyBackend
    ) -> None:
        with pytest.raises(TerminalBackendUnavailableError):
            await terminal_operations.list_units(BACKEND_CMUX)
