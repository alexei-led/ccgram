from __future__ import annotations

from pathlib import Path

import pytest

from ccgram.terminal_backends.config import (
    DEFAULT_BACKEND,
    DEFAULT_CMUX_TIMEOUT_MS,
    ENV_CMUX_ENABLED,
    ENV_CMUX_PROTOCOL_VERSION,
    ENV_CMUX_SIDECAR_SOCKET,
    ENV_CMUX_SIDECAR_TIMEOUT_MS,
    ENV_TERMINAL_BACKEND_DEFAULT,
    TerminalBackendConfig,
    load_terminal_backend_config,
)


class TestLoadDefaults:
    def test_empty_env_yields_tmux_only_defaults(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config({}, config_dir=tmp_path)
        assert config.default_backend == DEFAULT_BACKEND == "tmux"
        assert config.cmux_enabled is False
        assert config.cmux_sidecar_timeout_ms == DEFAULT_CMUX_TIMEOUT_MS
        assert config.cmux_protocol_version is None
        assert config.cmux_active is False
        # Socket path is derived from config_dir even when cmux is disabled
        # so the bootstrap reporter can show "would be at ..." messages.
        assert config.cmux_sidecar_socket == tmp_path / "cmux-sidecar.sock"

    def test_no_config_dir_means_no_default_socket(self) -> None:
        config = load_terminal_backend_config({}, config_dir=None)
        assert config.cmux_sidecar_socket is None
        assert config.cmux_active is False


class TestDefaultBackendOverride:
    def test_explicit_tmux(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_TERMINAL_BACKEND_DEFAULT: "tmux"}, config_dir=tmp_path
        )
        assert config.default_backend == "tmux"

    def test_explicit_cmux(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_TERMINAL_BACKEND_DEFAULT: "cmux"}, config_dir=tmp_path
        )
        assert config.default_backend == "cmux"

    @pytest.mark.parametrize("bad", ["zellij", "TMUX2", "  "])
    def test_unknown_backend_rejected(self, bad: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            load_terminal_backend_config(
                {ENV_TERMINAL_BACKEND_DEFAULT: bad}, config_dir=tmp_path
            )


class TestCmuxEnabled:
    def test_truthy_values_enable(self, tmp_path: Path) -> None:
        for raw in ("1", "true", "TRUE", "yes", "on"):
            config = load_terminal_backend_config(
                {ENV_CMUX_ENABLED: raw}, config_dir=tmp_path
            )
            assert config.cmux_enabled is True, raw
            assert config.cmux_active is True
            assert config.cmux_sidecar_socket == tmp_path / "cmux-sidecar.sock"

    def test_falsy_values_disable(self, tmp_path: Path) -> None:
        for raw in ("", "0", "false", "FALSE", "no", "off"):
            config = load_terminal_backend_config(
                {ENV_CMUX_ENABLED: raw}, config_dir=tmp_path
            )
            assert config.cmux_enabled is False, raw
            assert config.cmux_active is False

    def test_invalid_value_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_terminal_backend_config(
                {ENV_CMUX_ENABLED: "maybe"}, config_dir=tmp_path
            )

    def test_enabled_without_socket_or_config_dir_rejected(self) -> None:
        with pytest.raises(ValueError, match="no sidecar socket"):
            load_terminal_backend_config({ENV_CMUX_ENABLED: "true"}, config_dir=None)


class TestSocketPath:
    def test_explicit_socket_path(self, tmp_path: Path) -> None:
        socket = tmp_path / "custom.sock"
        config = load_terminal_backend_config(
            {
                ENV_CMUX_ENABLED: "true",
                ENV_CMUX_SIDECAR_SOCKET: str(socket),
            },
            config_dir=None,
        )
        assert config.cmux_sidecar_socket == socket
        assert config.cmux_active is True

    def test_explicit_socket_path_expands_user(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_CMUX_SIDECAR_SOCKET: "~/cmux.sock"}, config_dir=tmp_path
        )
        assert config.cmux_sidecar_socket is not None
        assert "~" not in str(config.cmux_sidecar_socket)


class TestTimeout:
    def test_default_when_unset(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config({}, config_dir=tmp_path)
        assert config.cmux_sidecar_timeout_ms == DEFAULT_CMUX_TIMEOUT_MS

    def test_explicit_value(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_CMUX_SIDECAR_TIMEOUT_MS: "2500"}, config_dir=tmp_path
        )
        assert config.cmux_sidecar_timeout_ms == 2500

    def test_zero_allowed(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_CMUX_SIDECAR_TIMEOUT_MS: "0"}, config_dir=tmp_path
        )
        assert config.cmux_sidecar_timeout_ms == 0

    def test_negative_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            load_terminal_backend_config(
                {ENV_CMUX_SIDECAR_TIMEOUT_MS: "-5"}, config_dir=tmp_path
            )

    def test_non_integer_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_terminal_backend_config(
                {ENV_CMUX_SIDECAR_TIMEOUT_MS: "fast"}, config_dir=tmp_path
            )


class TestProtocolVersion:
    def test_none_when_unset(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config({}, config_dir=tmp_path)
        assert config.cmux_protocol_version is None

    def test_explicit_value(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_CMUX_PROTOCOL_VERSION: "2"}, config_dir=tmp_path
        )
        assert config.cmux_protocol_version == "2"

    def test_whitespace_stripped(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_CMUX_PROTOCOL_VERSION: "  1  "}, config_dir=tmp_path
        )
        assert config.cmux_protocol_version == "1"

    def test_blank_treated_as_unset(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_CMUX_PROTOCOL_VERSION: "   "}, config_dir=tmp_path
        )
        assert config.cmux_protocol_version is None


class TestRawEnvCapture:
    def test_only_terminal_backend_vars_captured(self, tmp_path: Path) -> None:
        env = {
            ENV_CMUX_ENABLED: "true",
            ENV_CMUX_SIDECAR_TIMEOUT_MS: "1500",
            "TELEGRAM_BOT_TOKEN": "secret",
            "PATH": "/usr/bin",
        }
        config = load_terminal_backend_config(env, config_dir=tmp_path)
        assert config.raw_env == {
            ENV_CMUX_ENABLED: "true",
            ENV_CMUX_SIDECAR_TIMEOUT_MS: "1500",
        }


class TestDataclass:
    def test_frozen(self) -> None:
        config = TerminalBackendConfig()
        with pytest.raises(AttributeError):
            config.cmux_enabled = True  # type: ignore[misc]

    def test_post_init_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError):
            TerminalBackendConfig(default_backend="unknown")

    def test_post_init_rejects_negative_timeout(self) -> None:
        with pytest.raises(ValueError):
            TerminalBackendConfig(cmux_sidecar_timeout_ms=-1)

    def test_post_init_rejects_enabled_without_socket(self) -> None:
        with pytest.raises(ValueError, match="cmux_sidecar_socket"):
            TerminalBackendConfig(cmux_enabled=True)

    def test_with_default_returns_copy(self) -> None:
        config = TerminalBackendConfig()
        switched = config.with_default("cmux")
        assert config.default_backend == "tmux"
        assert switched.default_backend == "cmux"
