"""Tests for provider-agnostic command catalog and discovery sources."""

from pathlib import Path

from ccgram.command_catalog import CommandCatalog, discover_user_defined_commands
from ccgram.providers.base import DiscoveredCommand, ProviderCapabilities


class _DummyProvider:
    def __init__(
        self,
        *,
        name: str,
        commands: list[DiscoveredCommand],
        supports_user_command_discovery: bool,
    ) -> None:
        self._commands = commands
        self.calls = 0
        self._caps = ProviderCapabilities(
            name=name,
            launch_command=name,
            builtin_commands=tuple(cmd.name for cmd in commands),
            supports_user_command_discovery=supports_user_command_discovery,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._caps

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        self.calls += 1
        return list(self._commands)


class TestDiscoverUserDefinedCommands:
    def test_discovers_skills_and_commands(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "commit"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: committing-code\ndescription: Commit changes\nuser-invocable: true\n---\n"
        )

        cmd_dir = tmp_path / "commands" / "spec"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "work.md").write_text("---\ndescription: plan work\n---\n")

        discovered = discover_user_defined_commands(tmp_path)
        names = {cmd.name for cmd in discovered}
        assert "committing-code" in names
        assert "spec:work" in names


class TestCommandCatalog:
    def test_merges_provider_builtins_with_user_commands(self, tmp_path: Path) -> None:
        cmd_dir = tmp_path / "commands" / "spec"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "work.md").write_text("---\ndescription: plan work\n---\n")

        provider = _DummyProvider(
            name="codex",
            commands=[
                DiscoveredCommand(
                    name="/status", description="status", source="builtin"
                )
            ],
            supports_user_command_discovery=True,
        )
        catalog = CommandCatalog(ttl_seconds=60.0)

        discovered = catalog.get_provider_commands(provider, str(tmp_path))
        names = {cmd.name for cmd in discovered}
        assert "/status" in names
        assert "spec:work" in names

    def test_does_not_merge_user_commands_when_disabled(self, tmp_path: Path) -> None:
        cmd_dir = tmp_path / "commands" / "spec"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "work.md").write_text("---\ndescription: plan work\n---\n")

        provider = _DummyProvider(
            name="gemini",
            commands=[
                DiscoveredCommand(name="/help", description="help", source="builtin")
            ],
            supports_user_command_discovery=False,
        )
        catalog = CommandCatalog(ttl_seconds=60.0)

        discovered = catalog.get_provider_commands(provider, str(tmp_path))
        names = {cmd.name for cmd in discovered}
        assert "/help" in names
        assert "spec:work" not in names

    def test_uses_ttl_cache_for_same_provider_and_dir(self, tmp_path: Path) -> None:
        provider = _DummyProvider(
            name="codex",
            commands=[
                DiscoveredCommand(
                    name="/status", description="status", source="builtin"
                )
            ],
            supports_user_command_discovery=False,
        )
        catalog = CommandCatalog(ttl_seconds=60.0)

        catalog.get_provider_commands(provider, str(tmp_path))
        catalog.get_provider_commands(provider, str(tmp_path))
        assert provider.calls == 1

    def test_invalidate_provider_clears_cached_entries(self, tmp_path: Path) -> None:
        provider = _DummyProvider(
            name="codex",
            commands=[
                DiscoveredCommand(
                    name="/status", description="status", source="builtin"
                )
            ],
            supports_user_command_discovery=False,
        )
        catalog = CommandCatalog(ttl_seconds=60.0)

        catalog.get_provider_commands(provider, str(tmp_path))
        catalog.invalidate("codex")
        catalog.get_provider_commands(provider, str(tmp_path))
        assert provider.calls == 2
