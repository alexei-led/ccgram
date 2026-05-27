from __future__ import annotations

import pytest

from ccgram.terminal_backends import (
    BACKEND_CMUX,
    BACKEND_TMUX,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalBackendUnavailableError,
    TerminalUnit,
    TerminalUnitRef,
)
from ccgram.terminal_backends.router import (
    TerminalBackendRouter,
    get_router,
    reset_router_for_testing,
)


class _FakeBackend(TerminalBackend):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> TerminalBackendCapabilities:
        return TerminalBackendCapabilities(backend=self._name)

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


class TestTerminalBackendRouter:
    def test_register_and_lookup_by_name(self) -> None:
        router = TerminalBackendRouter()
        tmux = _FakeBackend(BACKEND_TMUX)
        router.register(tmux)
        assert router.get(BACKEND_TMUX) is tmux
        assert router.known() == frozenset({BACKEND_TMUX})

    def test_for_ref_dispatches_by_backend(self) -> None:
        router = TerminalBackendRouter()
        tmux = _FakeBackend(BACKEND_TMUX)
        cmux = _FakeBackend(BACKEND_CMUX)
        router.register(tmux)
        router.register(cmux)

        tmux_ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@3")
        cmux_ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="term-1")

        assert router.for_ref(tmux_ref) is tmux
        assert router.for_ref(cmux_ref) is cmux

    def test_register_replaces_existing(self) -> None:
        router = TerminalBackendRouter()
        a = _FakeBackend(BACKEND_TMUX)
        b = _FakeBackend(BACKEND_TMUX)
        router.register(a)
        router.register(b)
        assert router.get(BACKEND_TMUX) is b

    def test_unregister(self) -> None:
        router = TerminalBackendRouter()
        router.register(_FakeBackend(BACKEND_TMUX))
        router.unregister(BACKEND_TMUX)
        assert BACKEND_TMUX not in router.known()
        with pytest.raises(TerminalBackendUnavailableError):
            router.get(BACKEND_TMUX)

    def test_register_rejects_unknown_backend_name(self) -> None:
        router = TerminalBackendRouter()
        with pytest.raises(ValueError):
            router.register(_FakeBackend("not-a-backend"))

    def test_get_missing_raises_unavailable(self) -> None:
        router = TerminalBackendRouter()
        with pytest.raises(TerminalBackendUnavailableError) as excinfo:
            router.get(BACKEND_CMUX)
        assert excinfo.value.code == "unavailable"


class TestRouterSingleton:
    def setup_method(self) -> None:
        reset_router_for_testing()

    def teardown_method(self) -> None:
        reset_router_for_testing()

    def test_default_registers_tmux_backend(self) -> None:
        router = get_router()
        assert BACKEND_TMUX in router.known()
        backend = router.get(BACKEND_TMUX)
        assert backend.name == BACKEND_TMUX
        assert backend.capabilities().supports_send_text is True

    def test_reset_drops_registrations(self) -> None:
        first = get_router()
        first.register(_FakeBackend(BACKEND_CMUX))
        assert BACKEND_CMUX in first.known()
        reset_router_for_testing()
        second = get_router()
        assert second is not first
        assert BACKEND_CMUX not in second.known()

    def test_singleton_returns_same_instance(self) -> None:
        a = get_router()
        b = get_router()
        assert a is b
