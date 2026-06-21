"""Task 3 tests: registry resolution, the module-level proxy, and the config
``CCGRAM_MULTIPLEXER`` switch (tmux-only)."""

from __future__ import annotations

from typing import cast

import pytest

from ccgram import multiplexer as mux_pkg
from ccgram.config import Config
from ccgram.multiplexer import (
    Multiplexer,
    get_active_multiplexer,
    get_multiplexer,
    install_multiplexer,
    multiplexer,
)
from ccgram.multiplexer.registry import (
    UnknownMultiplexerError,
    multiplexer_names,
)


@pytest.fixture(autouse=True)
def _unwire_multiplexer():
    """Each test starts with the proxy unwired and leaves it unwired."""
    mux_pkg._reset_multiplexer_for_testing()
    yield
    mux_pkg._reset_multiplexer_for_testing()


class _FakeBackend:
    """Minimal stand-in to prove the proxy forwards attribute access."""

    @property
    def capabilities(self) -> str:
        return "fake-caps"

    def ping(self) -> str:
        return "pong"


class TestRegistryResolution:
    def test_tmux_registered(self) -> None:
        assert "tmux" in multiplexer_names()

    def test_herdr_registered(self) -> None:
        assert "herdr" in multiplexer_names()

    def test_get_tmux_returns_tmux_backend(self) -> None:
        backend = get_multiplexer("tmux")
        assert backend.capabilities.name == "tmux"

    def test_get_herdr_returns_herdr_backend(self) -> None:
        # Construction is I/O-free, so this resolves without a running herdr.
        backend = get_multiplexer("herdr")
        assert backend.capabilities.name == "herdr"

    def test_get_caches_one_instance_per_name(self) -> None:
        assert get_multiplexer("tmux") is get_multiplexer("tmux")

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(UnknownMultiplexerError, match="screen"):
            get_multiplexer("screen")

    def test_unknown_name_lists_available(self) -> None:
        with pytest.raises(UnknownMultiplexerError, match="tmux"):
            get_multiplexer("nope")


class TestProxy:
    def test_proxy_raises_before_wiring(self) -> None:
        with pytest.raises(RuntimeError, match="not yet wired"):
            _ = multiplexer.capabilities

    def test_get_active_raises_before_wiring(self) -> None:
        with pytest.raises(RuntimeError, match="not yet wired"):
            get_active_multiplexer()

    def test_proxy_forwards_after_wiring(self) -> None:
        fake = _FakeBackend()
        install_multiplexer(cast("Multiplexer", fake))
        assert multiplexer.capabilities == "fake-caps"
        assert getattr(multiplexer, "ping")() == "pong"
        assert get_active_multiplexer() is fake

    def test_repr_reflects_wiring_state(self) -> None:
        assert "unwired" in repr(multiplexer)
        install_multiplexer(cast("Multiplexer", _FakeBackend()))
        assert "unwired" not in repr(multiplexer)

    def test_wire_tmux_via_registry(self) -> None:
        install_multiplexer(get_multiplexer("tmux"))
        assert multiplexer.capabilities.name == "tmux"


@pytest.fixture
def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test:token")
    monkeypatch.setenv("ALLOWED_USERS", "12345")
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))


@pytest.mark.usefixtures("_base_env")
class TestConfigSwitch:
    def test_default_is_tmux(self, monkeypatch) -> None:
        monkeypatch.delenv("CCGRAM_MULTIPLEXER", raising=False)
        assert Config().multiplexer_name == "tmux"

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", "herdr")
        assert Config().multiplexer_name == "herdr"
