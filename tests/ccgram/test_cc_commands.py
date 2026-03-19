"""Tests for CC command discovery and menu registration."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import ccgram.cc_commands as cc_mod
from ccgram.cc_commands import (
    CC_BUILTINS,
    _sanitize_telegram_name,
    discover_provider_commands,
    discover_cc_commands,
    get_cc_name,
    get_provider_command_map,
    get_provider_supported_commands,
    parse_frontmatter,
    register_commands,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset module-level cache before each test."""
    cc_mod._name_map = {}
    yield
    cc_mod._name_map = {}


class TestSanitizeTelegramName:
    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("clear", "clear"),
            ("committing-code", "committing_code"),
            ("spec:work", "spec_work"),
            ("UPPER-Case", "upper_case"),
            ("a" * 50, "a" * 32),
            ("hello--world", "hello__world"),
            ("name with spaces", "namewithspaces"),
            ("...", ""),
            ("", ""),
        ],
        ids=[
            "simple",
            "hyphens",
            "colons",
            "uppercase",
            "truncate-32",
            "double-hyphen",
            "spaces-stripped",
            "all-special-chars",
            "empty",
        ],
    )
    def test_sanitize(self, input_name: str, expected: str) -> None:
        assert _sanitize_telegram_name(input_name) == expected


class TestParseFrontmatter:
    def test_valid_frontmatter(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_text(
            "---\nname: my-skill\ndescription: Does things\nuser-invocable: true\n---\n# Content\n"
        )
        result = parse_frontmatter(p)
        assert result == {
            "name": "my-skill",
            "description": "Does things",
            "user-invocable": "true",
        }

    def test_no_frontmatter(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_text("# Just a markdown file\nNo frontmatter here.\n")
        assert parse_frontmatter(p) == {}

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_text("")
        assert parse_frontmatter(p) == {}

    def test_missing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.md"
        assert parse_frontmatter(p) == {}

    def test_unclosed_frontmatter(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_text("---\nname: broken\nno closing delimiter")
        assert parse_frontmatter(p) == {}

    def test_quoted_values(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_text("---\nname: \"quoted-name\"\ndescription: 'single'\n---\n")
        result = parse_frontmatter(p)
        assert result["name"] == "quoted-name"
        assert result["description"] == "single"

    def test_blank_lines_in_frontmatter(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_text("---\nname: test\n\ndescription: hello\n---\n")
        result = parse_frontmatter(p)
        assert result == {"name": "test", "description": "hello"}

    def test_non_utf8_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.md"
        p.write_bytes(b"---\nname: \xff\xfe\n---\n")
        assert parse_frontmatter(p) == {}


class TestDiscoverBuiltins:
    def test_builtins_always_present(self, tmp_path: Path) -> None:
        commands = discover_cc_commands(claude_dir=tmp_path)
        builtin_names = {c.name for c in commands if c.source == "builtin"}
        assert builtin_names == set(CC_BUILTINS.keys())

    def test_builtin_telegram_names(self, tmp_path: Path) -> None:
        commands = discover_cc_commands(claude_dir=tmp_path)
        for cmd in commands:
            if cmd.source == "builtin":
                assert cmd.telegram_name == cmd.name


class TestDiscoverSkills:
    def _make_skill(
        self,
        claude_dir: Path,
        name: str,
        *,
        user_invocable: bool = True,
        desc: str = "A skill",
    ) -> None:
        skill_dir = claude_dir / "skills" / name
        skill_dir.mkdir(parents=True)
        fm = f"---\nname: {name}\ndescription: {desc}\nuser-invocable: {'true' if user_invocable else 'false'}\n---\n# Skill\n"
        (skill_dir / "SKILL.md").write_text(fm)

    def test_user_invocable_skill_discovered(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "my-skill")
        commands = discover_cc_commands(claude_dir=tmp_path)
        skill_cmds = [c for c in commands if c.source == "skill"]
        assert len(skill_cmds) == 1
        assert skill_cmds[0].name == "my-skill"
        assert skill_cmds[0].telegram_name == "my_skill"
        assert "A skill" in skill_cmds[0].description

    def test_non_invocable_skill_skipped(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "internal-skill", user_invocable=False)
        commands = discover_cc_commands(claude_dir=tmp_path)
        skill_cmds = [c for c in commands if c.source == "skill"]
        assert skill_cmds == []

    def test_system_dir_skipped(self, tmp_path: Path) -> None:
        system_dir = tmp_path / "skills" / ".system"
        system_dir.mkdir(parents=True)
        (system_dir / "SKILL.md").write_text(
            "---\nname: system\nuser-invocable: true\n---\n"
        )
        commands = discover_cc_commands(claude_dir=tmp_path)
        assert not any(c.name == "system" for c in commands)

    def test_missing_skill_md_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "skills" / "no-skill-file").mkdir(parents=True)
        commands = discover_cc_commands(claude_dir=tmp_path)
        skill_cmds = [c for c in commands if c.source == "skill"]
        assert skill_cmds == []

    def test_arrow_prefix_not_duplicated(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "prefixed", desc="↗ Already prefixed")
        commands = discover_cc_commands(claude_dir=tmp_path)
        skill = next(c for c in commands if c.name == "prefixed")
        assert skill.description == "↗ Already prefixed"
        assert skill.description.count("↗") == 1


class TestDiscoverCustomCommands:
    def _make_command(
        self, claude_dir: Path, group: str, name: str, *, desc: str = "A command"
    ) -> None:
        group_dir = claude_dir / "commands" / group
        group_dir.mkdir(parents=True, exist_ok=True)
        fm = f"---\ndescription: {desc}\n---\n# Command\n"
        (group_dir / f"{name}.md").write_text(fm)

    def test_custom_command_discovered(self, tmp_path: Path) -> None:
        self._make_command(tmp_path, "spec", "work")
        commands = discover_cc_commands(claude_dir=tmp_path)
        cmd_cmds = [c for c in commands if c.source == "command"]
        assert len(cmd_cmds) == 1
        assert cmd_cmds[0].name == "spec:work"
        assert cmd_cmds[0].telegram_name == "spec_work"

    def test_dotfile_skipped(self, tmp_path: Path) -> None:
        group_dir = tmp_path / "commands" / "group"
        group_dir.mkdir(parents=True)
        (group_dir / ".hidden.md").write_text("---\ndescription: hidden\n---\n")
        commands = discover_cc_commands(claude_dir=tmp_path)
        cmd_cmds = [c for c in commands if c.source == "command"]
        assert cmd_cmds == []

    def test_dot_group_skipped(self, tmp_path: Path) -> None:
        dot_group = tmp_path / "commands" / ".hidden"
        dot_group.mkdir(parents=True)
        (dot_group / "cmd.md").write_text("---\ndescription: hidden\n---\n")
        commands = discover_cc_commands(claude_dir=tmp_path)
        cmd_cmds = [c for c in commands if c.source == "command"]
        assert cmd_cmds == []

    def test_multiple_groups(self, tmp_path: Path) -> None:
        self._make_command(tmp_path, "alpha", "one")
        self._make_command(tmp_path, "beta", "two")
        commands = discover_cc_commands(claude_dir=tmp_path)
        cmd_cmds = [c for c in commands if c.source == "command"]
        names = {c.name for c in cmd_cmds}
        assert names == {"alpha:one", "beta:two"}


class TestGetCCName:
    def _make_skill(self, claude_dir: Path, name: str) -> None:
        skill_dir = claude_dir / "skills" / name
        skill_dir.mkdir(parents=True)
        fm = f"---\nname: {name}\nuser-invocable: true\n---\n"
        (skill_dir / "SKILL.md").write_text(fm)

    def _make_command(self, claude_dir: Path, group: str, name: str) -> None:
        group_dir = claude_dir / "commands" / group
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / f"{name}.md").write_text("---\ndescription: Test\n---\n")

    async def test_builtin_lookup(self, tmp_path: Path) -> None:
        await register_commands(AsyncMock(), claude_dir=tmp_path)
        assert get_cc_name("clear") == "clear"

    async def test_skill_lookup(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path, "committing-code")
        await register_commands(AsyncMock(), claude_dir=tmp_path)
        assert get_cc_name("committing_code") == "committing-code"

    async def test_command_lookup(self, tmp_path: Path) -> None:
        self._make_command(tmp_path, "spec", "work")
        await register_commands(AsyncMock(), claude_dir=tmp_path)
        assert get_cc_name("spec_work") == "spec:work"

    async def test_not_found(self, tmp_path: Path) -> None:
        await register_commands(AsyncMock(), claude_dir=tmp_path)
        assert get_cc_name("nonexistent") is None


class TestRegisterCommands:
    async def test_registers_bot_and_cc_commands(self, tmp_path: Path) -> None:
        bot = AsyncMock()
        await register_commands(bot, claude_dir=tmp_path)

        bot.delete_my_commands.assert_called_once()
        bot.set_my_commands.assert_called_once()

        registered = bot.set_my_commands.call_args[0][0]
        names = [c.command for c in registered]
        assert names[0] == "new"
        assert "clear" in names
        assert "compact" in names

    async def test_registers_commands_from_multiple_providers(
        self, tmp_path: Path
    ) -> None:
        from ccgram.providers.claude import ClaudeProvider
        from ccgram.providers.codex import CodexProvider

        bot = AsyncMock()
        await register_commands(
            bot,
            claude_dir=tmp_path,
            providers=[ClaudeProvider(), CodexProvider()],
        )

        registered = bot.set_my_commands.call_args[0][0]
        names = [c.command for c in registered]
        assert "status" in names
        assert get_cc_name("status") == "/status"

    async def test_description_truncation(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "verbose"
        skill_dir.mkdir(parents=True)
        long_desc = "x" * 300
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: verbose\ndescription: {long_desc}\nuser-invocable: true\n---\n"
        )

        bot = AsyncMock()
        await register_commands(bot, claude_dir=tmp_path)

        registered = bot.set_my_commands.call_args[0][0]
        for cmd in registered:
            assert len(cmd.description) <= 256

    async def test_telegram_command_limit(self, tmp_path: Path) -> None:
        commands_dir = tmp_path / "commands" / "bulk"
        commands_dir.mkdir(parents=True)
        for i in range(120):
            (commands_dir / f"cmd{i}.md").write_text(
                f"---\ndescription: Command {i}\n---\n"
            )

        bot = AsyncMock()
        await register_commands(bot, claude_dir=tmp_path)

        registered = bot.set_my_commands.call_args[0][0]
        assert len(registered) <= 100

    async def test_duplicate_telegram_names_deduplicated(self, tmp_path: Path) -> None:
        # skill "foo-bar" and command "foo:bar" both sanitize to "foo_bar"
        skill_dir = tmp_path / "skills" / "foo-bar"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: foo-bar\nuser-invocable: true\n---\n"
        )
        cmd_dir = tmp_path / "commands" / "foo"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "bar.md").write_text("---\ndescription: Dup\n---\n")

        bot = AsyncMock()
        await register_commands(bot, claude_dir=tmp_path)

        registered = bot.set_my_commands.call_args[0][0]
        tg_names = [c.command for c in registered]
        assert tg_names.count("foo_bar") == 1

    async def test_can_register_bot_commands_only(self, tmp_path: Path) -> None:
        bot = AsyncMock()
        await register_commands(bot, claude_dir=tmp_path, include_cc_commands=False)

        registered = bot.set_my_commands.call_args[0][0]
        names = [c.command for c in registered]
        assert "new" in names
        assert "commands" in names
        assert "clear" not in names

    async def test_register_commands_supports_scope(self, tmp_path: Path) -> None:
        bot = AsyncMock()
        scope = object()
        await register_commands(
            bot,
            claude_dir=tmp_path,
            include_cc_commands=False,
            scope=scope,
        )

        bot.delete_my_commands.assert_called_once_with(scope=scope)
        bot.set_my_commands.assert_called_once()
        assert bot.set_my_commands.call_args.kwargs.get("scope") is scope

    async def test_bot_native_name_collision_skipped(self, tmp_path: Path) -> None:
        # A skill that sanitizes to "new" should not create a duplicate
        skill_dir = tmp_path / "skills" / "new"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: new\nuser-invocable: true\n---\n"
        )

        bot = AsyncMock()
        await register_commands(bot, claude_dir=tmp_path)

        registered = bot.set_my_commands.call_args[0][0]
        tg_names = [c.command for c in registered]
        assert tg_names.count("new") == 1


class TestProviderCommandHelpers:
    def test_discovers_codex_builtin_commands(self, tmp_path: Path) -> None:
        from ccgram.providers.codex import CodexProvider

        commands = discover_provider_commands(CodexProvider(), claude_dir=tmp_path)
        names = {c.name for c in commands}
        assert "/status" in names
        assert "/mcp" in names
        assert "/help" not in names

    def test_builds_provider_command_map(self, tmp_path: Path) -> None:
        from ccgram.providers.codex import CodexProvider

        mapping = get_provider_command_map(CodexProvider(), claude_dir=tmp_path)
        assert mapping["status"] == "/status"
        assert mapping["permissions"] == "/permissions"

    def test_provider_supported_commands_include_slash_form(
        self, tmp_path: Path
    ) -> None:
        from ccgram.providers.claude import ClaudeProvider

        supported = get_provider_supported_commands(
            ClaudeProvider(), claude_dir=tmp_path
        )
        assert "/clear" in supported
        assert "/compact" in supported

    def test_codex_provider_discovery_includes_user_commands(
        self, tmp_path: Path
    ) -> None:
        from ccgram.providers.codex import CodexProvider

        cmd_dir = tmp_path / "commands" / "spec"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "work.md").write_text("---\ndescription: work\n---\n")

        discovered = discover_provider_commands(CodexProvider(), claude_dir=tmp_path)
        names = {cmd.name for cmd in discovered}
        assert "/status" in names
        assert "spec:work" in names
