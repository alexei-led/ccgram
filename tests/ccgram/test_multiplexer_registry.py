"""Task 3 tests: registry resolution, the module-level proxy, and the config
``CCGRAM_MULTIPLEXER`` switch (tmux-only)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
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
    @pytest.mark.parametrize("name", ["tmux", "herdr", "agterm"])
    def test_registered_backend_resolves_to_itself(self, name: str) -> None:
        # Both constructors are I/O-free, so this resolves without a running
        # multiplexer.
        assert name in multiplexer_names()
        assert get_multiplexer(name).capabilities.name == name

    def test_get_caches_one_instance_per_name(self) -> None:
        assert get_multiplexer("tmux") is get_multiplexer("tmux")

    def test_agterm_registry_loads_without_bot_credentials(self, tmp_path) -> None:
        source_root = Path(__file__).parents[2] / "src"
        env = {
            "HOME": str(tmp_path),
            "PYTHONPATH": str(source_root),
            "PATH": os.environ.get("PATH", ""),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from ccgram.multiplexer import get_multiplexer; "
                "assert get_multiplexer('agterm').capabilities.name == 'agterm'",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_unknown_name_raises_and_lists_the_registered_ones(self) -> None:
        with pytest.raises(UnknownMultiplexerError) as excinfo:
            get_multiplexer("screen")
        message = str(excinfo.value)
        assert "screen" in message
        assert all(name in message for name in multiplexer_names())


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
    def test_default_falls_back_to_tmux_when_no_mux_vars(self, monkeypatch) -> None:
        """With no CCGRAM_MULTIPLEXER and no backend-specific vars, auto → tmux."""
        monkeypatch.delenv("CCGRAM_MULTIPLEXER", raising=False)
        monkeypatch.delenv("HERDR_PANE_ID", raising=False)
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.delenv("AGTERM_SESSION_ID", raising=False)
        assert Config().multiplexer_name == "tmux"

    def test_auto_detects_herdr(self, monkeypatch) -> None:
        monkeypatch.delenv("CCGRAM_MULTIPLEXER", raising=False)
        monkeypatch.setenv("HERDR_PANE_ID", "w1:p1")
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.delenv("AGTERM_SESSION_ID", raising=False)
        assert Config().multiplexer_name == "herdr"

    def test_auto_detects_tmux(self, monkeypatch) -> None:
        monkeypatch.delenv("CCGRAM_MULTIPLEXER", raising=False)
        monkeypatch.delenv("HERDR_PANE_ID", raising=False)
        monkeypatch.setenv("TMUX_PANE", "%0")
        monkeypatch.delenv("AGTERM_SESSION_ID", raising=False)
        assert Config().multiplexer_name == "tmux"

    def test_auto_detects_agterm(self, monkeypatch) -> None:
        monkeypatch.delenv("CCGRAM_MULTIPLEXER", raising=False)
        monkeypatch.delenv("HERDR_PANE_ID", raising=False)
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.setenv("AGTERM_SESSION_ID", "abc-uuid")
        assert Config().multiplexer_name == "agterm"

    def test_herdr_wins_over_tmux(self, monkeypatch) -> None:
        """herdr has higher precedence than tmux when both vars are present."""
        monkeypatch.delenv("CCGRAM_MULTIPLEXER", raising=False)
        monkeypatch.setenv("HERDR_PANE_ID", "w1:p1")
        monkeypatch.setenv("TMUX_PANE", "%0")
        monkeypatch.delenv("AGTERM_SESSION_ID", raising=False)
        assert Config().multiplexer_name == "herdr"

    def test_explicit_auto_value(self, monkeypatch) -> None:
        """CCGRAM_MULTIPLEXER=auto triggers detection (same as unset)."""
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", "auto")
        monkeypatch.delenv("HERDR_PANE_ID", raising=False)
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.delenv("AGTERM_SESSION_ID", raising=False)
        assert Config().multiplexer_name == "tmux"

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", "herdr")
        assert Config().multiplexer_name == "herdr"
