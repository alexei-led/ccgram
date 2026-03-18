"""Tests for provider registry, config integration, and per-window resolution."""

import json
import shlex
from unittest.mock import patch

import pytest

from ccgram.providers.base import ProviderCapabilities
from ccgram.providers.registry import ProviderRegistry, UnknownProviderError
from test_provider_contracts import StubProvider as _StubProvider

# ── Registry tests ──────────────────────────────────────────────────────


class TestProviderRegistry:
    def test_register_and_get(self) -> None:
        reg = ProviderRegistry()
        reg.register("stub", _StubProvider)
        provider = reg.get("stub")
        assert provider.capabilities.name == "stub"

    def test_get_unknown_raises(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(UnknownProviderError, match="nope"):
            reg.get("nope")

    def test_register_overwrites(self) -> None:
        class _OtherProvider(_StubProvider):
            _CAPS = ProviderCapabilities(name="other", launch_command="other-cli")

        reg = ProviderRegistry()
        reg.register("stub", _StubProvider)
        reg.register("stub", _OtherProvider)
        assert reg.get("stub").capabilities.name == "other"

    def test_get_caches_instance_per_name(self) -> None:
        reg = ProviderRegistry()
        reg.register("stub", _StubProvider)
        a = reg.get("stub")
        b = reg.get("stub")
        assert a is b

    def test_re_register_invalidates_cache(self) -> None:
        reg = ProviderRegistry()
        reg.register("stub", _StubProvider)
        a = reg.get("stub")
        reg.register("stub", _StubProvider)
        b = reg.get("stub")
        assert a is not b

    def test_error_message_lists_available(self) -> None:
        reg = ProviderRegistry()
        reg.register("alpha", _StubProvider)
        reg.register("bravo", _StubProvider)
        with pytest.raises(UnknownProviderError, match="alpha, bravo"):
            reg.get("missing")


# ── Config integration tests ────────────────────────────────────────────


class TestConfigProviderSettings:
    def test_default_provider_name(self) -> None:
        env = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "ALLOWED_USERS": "123",
            "HOME": "/tmp",
        }
        with patch.dict("os.environ", env, clear=True):
            from ccgram.config import Config

            cfg = Config()
            assert cfg.provider_name == "claude"

    def test_override_provider_via_env(self) -> None:
        env = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "ALLOWED_USERS": "123",
            "HOME": "/tmp",
            "CCGRAM_PROVIDER": "codex",
        }
        with patch.dict("os.environ", env, clear=True):
            from ccgram.config import Config

            cfg = Config()
            assert cfg.provider_name == "codex"


# ── resolve_launch_command tests ──────────────────────────────────────────


class TestResolveLaunchCommand:
    @pytest.fixture(autouse=True)
    def _reset(self):
        from ccgram.providers import _reset_provider

        _reset_provider()
        yield
        _reset_provider()

    def test_default_returns_provider_command(self) -> None:
        from ccgram.providers import resolve_launch_command

        assert resolve_launch_command("claude") == "claude"
        assert resolve_launch_command("codex") == "codex"
        gemini_cmd = resolve_launch_command("gemini")
        assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH=" in gemini_cmd
        assert gemini_cmd.endswith(" gemini")

    def test_per_provider_env_override(self, monkeypatch) -> None:
        from ccgram.providers import resolve_launch_command

        monkeypatch.setenv("CCGRAM_CLAUDE_COMMAND", "ce --current")
        assert resolve_launch_command("claude") == "ce --current"
        assert resolve_launch_command("codex") == "codex"

    def test_override_does_not_affect_other_providers(self, monkeypatch) -> None:
        from ccgram.providers import resolve_launch_command

        monkeypatch.setenv("CCGRAM_CODEX_COMMAND", "my-codex")
        assert resolve_launch_command("codex") == "my-codex"
        assert resolve_launch_command("claude") == "claude"
        gemini_cmd = resolve_launch_command("gemini")
        assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH=" in gemini_cmd
        assert gemini_cmd.endswith(" gemini")

    def test_unknown_provider_falls_back_to_claude_default(self) -> None:
        from ccgram.providers import resolve_launch_command

        assert resolve_launch_command("nonexistent") == "claude"

    def test_all_three_providers_independently(self, monkeypatch) -> None:
        from ccgram.providers import resolve_launch_command

        monkeypatch.setenv("CCGRAM_CLAUDE_COMMAND", "ce --current")
        monkeypatch.setenv("CCGRAM_CODEX_COMMAND", "my-codex --flag")
        monkeypatch.setenv("CCGRAM_GEMINI_COMMAND", "/opt/gemini/run")
        assert resolve_launch_command("claude") == "ce --current"
        assert resolve_launch_command("codex") == "my-codex --flag"
        assert resolve_launch_command("gemini") == "/opt/gemini/run"

    def test_yolo_mode_appends_provider_specific_flags(self) -> None:
        from ccgram.providers import resolve_launch_command

        assert (
            resolve_launch_command("claude", approval_mode="yolo")
            == "claude --dangerously-skip-permissions"
        )
        assert (
            resolve_launch_command("codex", approval_mode="yolo")
            == "codex --dangerously-bypass-approvals-and-sandbox"
        )
        gemini_cmd = resolve_launch_command("gemini", approval_mode="yolo")
        assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH=" in gemini_cmd
        assert gemini_cmd.endswith(" gemini --yolo")

    def test_gemini_hardening_writes_system_settings_file(
        self, tmp_path, monkeypatch
    ) -> None:
        from ccgram.providers import resolve_launch_command

        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        cmd = resolve_launch_command("gemini")

        settings_path = tmp_path / "gemini-system-settings.json"
        assert settings_path.exists()
        assert json.loads(settings_path.read_text()) == {
            "tools": {"shell": {"enableInteractiveShell": False}}
        }
        assert (
            f"GEMINI_CLI_SYSTEM_SETTINGS_PATH={shlex.quote(str(settings_path))}" in cmd
        )
        assert cmd.endswith(" gemini")

    def test_yolo_mode_does_not_duplicate_flag(self, monkeypatch) -> None:
        from ccgram.providers import resolve_launch_command

        monkeypatch.setenv(
            "CCGRAM_CLAUDE_COMMAND", "claude --dangerously-skip-permissions"
        )
        assert (
            resolve_launch_command("claude", approval_mode="yolo")
            == "claude --dangerously-skip-permissions"
        )


# ── Integration: registry wired together ─────────────────────────────────


class TestModuleLevelRegistry:
    def test_singleton_exists_with_claude(self, monkeypatch) -> None:
        from ccgram.providers import _reset_provider, get_provider, registry

        _reset_provider()
        try:
            get_provider()
            assert isinstance(registry, ProviderRegistry)
            assert "claude" in sorted(registry._providers)
        finally:
            _reset_provider()

    def test_unknown_provider_falls_back_to_claude(self, monkeypatch) -> None:
        from ccgram.providers import _reset_provider, get_provider

        _reset_provider()
        monkeypatch.setenv("CCGRAM_PROVIDER", "doesnotexist")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USERS", "123")
        try:
            provider = get_provider()
            assert provider.capabilities.name == "claude"
        finally:
            _reset_provider()

    def test_resolve_capabilities_unknown_falls_back(self) -> None:
        from ccgram.providers import _reset_provider, resolve_capabilities

        _reset_provider()
        try:
            caps = resolve_capabilities("nonexistent")
            assert caps.name == "claude"
        finally:
            _reset_provider()


class TestRegistryIsValid:
    def test_valid_name(self) -> None:
        reg = ProviderRegistry()
        reg.register("stub", _StubProvider)
        assert reg.is_valid("stub") is True

    def test_invalid_name(self) -> None:
        reg = ProviderRegistry()
        assert reg.is_valid("nonexistent") is False


class TestGetProviderForWindow:
    @pytest.fixture(autouse=True)
    def _reset(self):
        from ccgram.providers import _reset_provider

        _reset_provider()
        yield
        _reset_provider()

    def test_returns_window_specific_provider(self, monkeypatch) -> None:
        from ccgram.providers import get_provider_for_window
        from ccgram.session import SessionManager, WindowState, session_manager

        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)

        session_manager.window_states["@1"] = WindowState(
            session_id="s1", cwd="/tmp", provider_name="codex"
        )
        provider = get_provider_for_window("@1")
        assert provider.capabilities.name == "codex"

        session_manager.window_states.pop("@1", None)

    def test_falls_back_to_global_when_empty(self, monkeypatch) -> None:
        from ccgram.providers import get_provider_for_window
        from ccgram.session import SessionManager, WindowState, session_manager

        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)

        session_manager.window_states["@2"] = WindowState(
            session_id="s2", cwd="/tmp", provider_name=""
        )
        provider = get_provider_for_window("@2")
        assert provider.capabilities.name == "claude"

        session_manager.window_states.pop("@2", None)

    def test_falls_back_when_window_not_in_state(self, monkeypatch) -> None:
        from ccgram.providers import get_provider_for_window
        from ccgram.session import SessionManager

        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)

        provider = get_provider_for_window("@999")
        assert provider.capabilities.name == "claude"

    def test_falls_back_on_invalid_provider_name(self, monkeypatch) -> None:
        from ccgram.providers import get_provider_for_window
        from ccgram.session import SessionManager, WindowState, session_manager

        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)

        session_manager.window_states["@3"] = WindowState(
            session_id="s3", cwd="/tmp", provider_name="nonexistent"
        )
        provider = get_provider_for_window("@3")
        assert provider.capabilities.name == "claude"

        session_manager.window_states.pop("@3", None)

    def test_different_windows_resolve_different_providers(self, monkeypatch) -> None:
        from ccgram.providers import get_provider_for_window
        from ccgram.session import SessionManager, WindowState, session_manager

        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)

        session_manager.window_states["@10"] = WindowState(
            session_id="s10", cwd="/tmp", provider_name="claude"
        )
        session_manager.window_states["@11"] = WindowState(
            session_id="s11", cwd="/tmp", provider_name="codex"
        )
        session_manager.window_states["@12"] = WindowState(
            session_id="s12", cwd="/tmp", provider_name="gemini"
        )

        assert get_provider_for_window("@10").capabilities.name == "claude"
        assert get_provider_for_window("@11").capabilities.name == "codex"
        assert get_provider_for_window("@12").capabilities.name == "gemini"

        session_manager.window_states.pop("@10", None)
        session_manager.window_states.pop("@11", None)
        session_manager.window_states.pop("@12", None)
