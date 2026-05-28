from __future__ import annotations

from pathlib import Path

import pytest

from ccgram.terminal_backends.config import (
    DEFAULT_BACKEND,
    ENV_CMUX_ENABLED,
    ENV_CMUX_WORKSPACE_ID,
    ENV_TERMINAL_BACKEND,
    ENV_TERMINAL_BACKEND_DEFAULT,
    TerminalBackendConfig,
    load_terminal_backend_config,
)


class TestLoadDefaults:
    def test_empty_env_yields_tmux_defaults(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config({}, config_dir=tmp_path)
        assert config.default_backend == DEFAULT_BACKEND == "tmux"
        assert config.cmux_enabled is False
        assert config.cmux_active is False
        assert config.cmux_workspace_id is None


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

    def test_short_alias_wins(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {
                ENV_TERMINAL_BACKEND: "cmux",
                ENV_TERMINAL_BACKEND_DEFAULT: "tmux",
            },
            config_dir=tmp_path,
        )
        assert config.default_backend == "cmux"

    @pytest.mark.parametrize("bad", ["zellij", "TMUX2", "  "])
    def test_unknown_backend_rejected(self, bad: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            load_terminal_backend_config(
                {ENV_TERMINAL_BACKEND: bad}, config_dir=tmp_path
            )


class TestCmuxEnabled:
    def test_truthy_values_enable(self, tmp_path: Path) -> None:
        for raw in ("1", "true", "TRUE", "yes", "on"):
            config = load_terminal_backend_config(
                {ENV_CMUX_ENABLED: raw}, config_dir=tmp_path
            )
            assert config.cmux_enabled is True, raw
            assert config.cmux_active is True

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


class TestCmuxWorkspace:
    def test_workspace_id_is_optional(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config({}, config_dir=tmp_path)
        assert config.cmux_workspace_id is None

    def test_workspace_id_is_trimmed(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_CMUX_WORKSPACE_ID: " ws-uuid "}, config_dir=tmp_path
        )
        assert config.cmux_workspace_id == "ws-uuid"

    def test_blank_workspace_id_is_none(self, tmp_path: Path) -> None:
        config = load_terminal_backend_config(
            {ENV_CMUX_WORKSPACE_ID: "  "}, config_dir=tmp_path
        )
        assert config.cmux_workspace_id is None


class TestRawEnvCapture:
    def test_only_terminal_backend_vars_captured(self, tmp_path: Path) -> None:
        env = {
            ENV_TERMINAL_BACKEND: "cmux",
            ENV_CMUX_ENABLED: "true",
            ENV_CMUX_WORKSPACE_ID: "ws-uuid",
            "TELEGRAM_BOT_TOKEN": "secret",
            "PATH": "/usr/bin",
        }
        config = load_terminal_backend_config(env, config_dir=tmp_path)
        assert config.raw_env == {
            ENV_TERMINAL_BACKEND: "cmux",
            ENV_CMUX_ENABLED: "true",
            ENV_CMUX_WORKSPACE_ID: "ws-uuid",
        }


class TestDataclass:
    def test_frozen(self) -> None:
        config = TerminalBackendConfig()
        with pytest.raises(AttributeError):
            config.cmux_enabled = True  # type: ignore[misc]

    def test_post_init_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError):
            TerminalBackendConfig(default_backend="unknown")

    def test_with_default_returns_copy(self) -> None:
        config = TerminalBackendConfig()
        switched = config.with_default("cmux")
        assert config.default_backend == "tmux"
        assert switched.default_backend == "cmux"

    def test_with_default_keeps_workspace_id(self) -> None:
        config = TerminalBackendConfig(cmux_workspace_id="ws-uuid")
        switched = config.with_default("cmux")
        assert switched.cmux_workspace_id == "ws-uuid"
