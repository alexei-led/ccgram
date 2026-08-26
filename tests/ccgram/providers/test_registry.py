import json
import shlex
from unittest.mock import patch

import pytest

from ccgram.providers.base import ProviderCapabilities
from ccgram.providers.registry import ProviderRegistry, UnknownProviderError
from test_contracts import StubProvider as _StubProvider


@pytest.fixture
def _fresh_registry():
    from ccgram.providers import _reset_provider

    _reset_provider()
    yield
    _reset_provider()


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


@pytest.mark.usefixtures("_fresh_registry")
class TestResolveLaunchCommand:
    def test_default_returns_provider_command(self) -> None:
        from ccgram.providers import resolve_launch_command

        assert resolve_launch_command("claude") == "claude"
        assert resolve_launch_command("codex") == "codex"
        gemini_cmd = resolve_launch_command("gemini")
        assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH=" in gemini_cmd
        assert gemini_cmd.endswith(" gemini")

    def test_all_providers_can_be_overridden_independently(self, monkeypatch) -> None:
        from ccgram.providers import resolve_launch_command

        monkeypatch.setenv("CCGRAM_CLAUDE_COMMAND", "ce --current")
        monkeypatch.setenv("CCGRAM_CODEX_COMMAND", "my-codex --flag")
        monkeypatch.setenv("CCGRAM_GEMINI_COMMAND", "/opt/gemini/run")
        assert resolve_launch_command("claude") == "ce --current"
        assert resolve_launch_command("codex") == "my-codex --flag"
        assert resolve_launch_command("gemini") == "/opt/gemini/run"

    @pytest.mark.parametrize(
        ("overridden", "untouched"),
        [
            pytest.param("claude", "codex", id="claude_override"),
            pytest.param("codex", "claude", id="codex_override"),
        ],
    )
    def test_overriding_one_provider_leaves_the_others_alone(
        self, monkeypatch, overridden: str, untouched: str
    ) -> None:
        from ccgram.providers import resolve_launch_command

        monkeypatch.setenv(f"CCGRAM_{overridden.upper()}_COMMAND", "custom-cli")
        assert resolve_launch_command(overridden) == "custom-cli"
        assert resolve_launch_command(untouched) == untouched
        assert resolve_launch_command("gemini").endswith(" gemini")

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

    @pytest.mark.parametrize(
        ("provider", "expected"),
        [
            pytest.param(
                "claude", "claude --dangerously-skip-permissions", id="claude"
            ),
            pytest.param(
                "codex",
                "codex --dangerously-bypass-approvals-and-sandbox",
                id="codex",
            ),
            pytest.param(
                "antigravity", "agy --dangerously-skip-permissions", id="antigravity"
            ),
        ],
    )
    def test_yolo_mode_appends_provider_specific_flags(
        self, provider: str, expected: str
    ) -> None:
        from ccgram.providers import resolve_launch_command

        assert resolve_launch_command(provider, approval_mode="yolo") == expected

    def test_gemini_yolo_keeps_the_hardened_settings_env(self) -> None:
        from ccgram.providers import resolve_launch_command

        cmd = resolve_launch_command("gemini", approval_mode="yolo")
        assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH=" in cmd
        assert cmd.endswith(" gemini --yolo")

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
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param("stub", True, id="registered"),
            pytest.param("nope", False, id="unknown"),
        ],
    )
    def test_is_valid(self, name: str, expected: bool) -> None:
        reg = ProviderRegistry()
        reg.register("stub", _StubProvider)
        assert reg.is_valid(name) is expected


@pytest.mark.usefixtures("_fresh_registry")
class TestEnsureRegistered:
    @pytest.mark.parametrize(
        "name",
        ["antigravity", "claude", "codex", "gemini", "pi", "shell"],
    )
    def test_all_providers_registered(self, name: str) -> None:
        from ccgram.providers import _ensure_registered, registry

        _ensure_registered()
        assert registry.is_valid(name), f"Provider {name!r} not registered"


@pytest.mark.usefixtures("_fresh_registry")
class TestGetProviderForWindow:
    @pytest.mark.parametrize(
        ("provider_name", "expected"),
        [
            pytest.param("codex", "codex", id="explicit_codex"),
            pytest.param("gemini", "gemini", id="explicit_gemini"),
            pytest.param("claude", "claude", id="explicit_claude"),
            pytest.param("", "claude", id="empty_falls_back_to_global"),
            pytest.param(None, "claude", id="unset_falls_back_to_global"),
            pytest.param("nonexistent", "claude", id="invalid_falls_back_to_global"),
        ],
    )
    def test_resolves_provider_name_with_global_fallback(
        self, provider_name: str | None, expected: str
    ) -> None:
        from ccgram.providers import get_provider_for_window

        provider = get_provider_for_window("@1", provider_name=provider_name)
        assert provider.capabilities.name == expected
