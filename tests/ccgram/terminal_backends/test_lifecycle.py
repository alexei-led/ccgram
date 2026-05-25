"""Tests for terminal backend lifecycle wiring (config → router)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ccgram.terminal_backends.base import BACKEND_CMUX
from ccgram.terminal_backends.config import TerminalBackendConfig
from ccgram.terminal_backends.lifecycle import (
    register_cmux_backend_if_enabled,
    unregister_cmux_backend,
)
from ccgram.terminal_backends.router import get_router, reset_router_for_testing


@pytest.fixture(autouse=True)
def _isolated_router():
    reset_router_for_testing()
    yield
    reset_router_for_testing()


def _enabled() -> TerminalBackendConfig:
    return TerminalBackendConfig(
        cmux_enabled=True,
        cmux_sidecar_socket=Path("/tmp/cmux-sidecar.sock"),
    )


def _disabled() -> TerminalBackendConfig:
    return TerminalBackendConfig()


class TestRegisterCmuxBackendIfEnabled:
    def test_disabled_config_is_a_no_op(self) -> None:
        result = register_cmux_backend_if_enabled(_disabled())
        assert result is None
        assert BACKEND_CMUX not in get_router().known()

    def test_enabled_config_registers_injected_backend(self) -> None:
        fake = MagicMock()
        fake.name = BACKEND_CMUX
        result = register_cmux_backend_if_enabled(_enabled(), backend=fake)
        assert result is fake
        assert BACKEND_CMUX in get_router().known()

    def test_injected_non_cmux_backend_rejected(self) -> None:
        fake = MagicMock()
        fake.name = "tmux"
        with pytest.raises(ValueError, match="cmux"):
            register_cmux_backend_if_enabled(_enabled(), backend=fake)


class TestUnregisterCmuxBackend:
    def test_unregister_removes_backend(self) -> None:
        fake = MagicMock()
        fake.name = BACKEND_CMUX
        register_cmux_backend_if_enabled(_enabled(), backend=fake)
        assert BACKEND_CMUX in get_router().known()
        unregister_cmux_backend()
        assert BACKEND_CMUX not in get_router().known()

    def test_unregister_when_absent_is_noop(self) -> None:
        unregister_cmux_backend()  # must not raise
        assert BACKEND_CMUX not in get_router().known()
