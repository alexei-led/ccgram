"""Tests for agent first-run readiness: trust seeding and prompt answering.

Nothing here may touch the operator's real agent config or the real pane:
every test points HOME/CODEX_HOME at a temp directory and mocks the
multiplexer.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram import agent_startup


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Claude's and Codex's config paths into a temp home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    claude_json = home / ".claude.json"
    claude_json.write_text(json.dumps({"projects": {}, "numStartups": 3}))
    from ccgram.config import config

    monkeypatch.setattr(config, "claude_config_dir", home / ".claude", raising=False)
    monkeypatch.setattr(config, "autostart_trust", True, raising=False)
    return home


class TestPreseedClaudeTrust:
    def test_marks_the_project_trusted(self, fake_home: Path, tmp_path: Path) -> None:
        project = tmp_path / "ccgram"
        project.mkdir()

        assert agent_startup.preseed_trust("claude", project) is True

        data = json.loads((fake_home / ".claude.json").read_text())
        assert data["projects"][str(project)]["hasTrustDialogAccepted"] is True
        assert data["numStartups"] == 3  # unrelated keys survive

    def test_second_call_is_a_noop(self, fake_home: Path, tmp_path: Path) -> None:
        project = tmp_path / "ccgram"
        project.mkdir()
        agent_startup.preseed_trust("claude", project)
        assert agent_startup.preseed_trust("claude", project) is False

    def test_missing_config_is_not_created(
        self, fake_home: Path, tmp_path: Path
    ) -> None:
        (fake_home / ".claude.json").unlink()
        project = tmp_path / "ccgram"
        project.mkdir()

        assert agent_startup.preseed_trust("claude", project) is False
        assert not (fake_home / ".claude.json").exists()


class TestPreseedCodexTrust:
    def test_appends_a_project_section(self, fake_home: Path, tmp_path: Path) -> None:
        config_file = fake_home / ".codex" / "config.toml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text('model = "gpt-5-codex"\n')
        project = tmp_path / "ccgram"
        project.mkdir()

        assert agent_startup.preseed_trust("codex", project) is True

        body = config_file.read_text()
        assert 'model = "gpt-5-codex"' in body
        assert f'[projects."{project}"]' in body
        assert 'trust_level = "trusted"' in body

    def test_existing_section_is_left_alone(
        self, fake_home: Path, tmp_path: Path
    ) -> None:
        project = tmp_path / "ccgram"
        project.mkdir()
        config_file = fake_home / ".codex" / "config.toml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(f'[projects."{project}"]\ntrust_level = "trusted"\n')

        assert agent_startup.preseed_trust("codex", project) is False

    def test_single_quoted_existing_entry_is_not_duplicated(
        self, fake_home: Path, tmp_path: Path
    ) -> None:
        project = tmp_path / "ccgram"
        project.mkdir()
        config_file = fake_home / ".codex" / "config.toml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(f"[projects.'{project}']\ntrust_level = \"trusted\"\n")

        assert agent_startup.preseed_trust("codex", project) is False
        assert config_file.read_text().count("[projects.") == 1

    def test_a_path_with_a_quote_produces_valid_toml(
        self, fake_home: Path, tmp_path: Path
    ) -> None:
        project = tmp_path / 'we"ird'
        project.mkdir()
        config_file = fake_home / ".codex" / "config.toml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text('model = "gpt-5-codex"\n')

        assert agent_startup.preseed_trust("codex", project) is True

        import tomllib

        parsed = tomllib.loads(config_file.read_text())
        assert parsed["projects"][str(project)]["trust_level"] == "trusted"

    def test_an_unparseable_config_is_left_alone(
        self, fake_home: Path, tmp_path: Path
    ) -> None:
        project = tmp_path / "ccgram"
        project.mkdir()
        config_file = fake_home / ".codex" / "config.toml"
        config_file.parent.mkdir(parents=True)
        original = "this is = not = toml [[[\n"
        config_file.write_text(original)

        assert agent_startup.preseed_trust("codex", project) is False
        assert config_file.read_text() == original

    def test_an_untrusted_entry_is_not_rewritten(
        self, fake_home: Path, tmp_path: Path
    ) -> None:
        project = tmp_path / "ccgram"
        project.mkdir()
        config_file = fake_home / ".codex" / "config.toml"
        config_file.parent.mkdir(parents=True)
        original = f'[projects."{project}"]\ntrust_level = "untrusted"\n'
        config_file.write_text(original)

        assert agent_startup.preseed_trust("codex", project) is False
        assert config_file.read_text() == original

    def test_global_off_beats_a_per_agent_on(
        self, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented global switch is final, not a default."""
        from ccgram.config import config

        monkeypatch.setattr(config, "autostart_trust", False, raising=False)
        monkeypatch.setenv("CCGRAM_CLAUDE_AUTOSTART", "on")
        project = tmp_path / "ccgram"
        project.mkdir()

        assert agent_startup.preseed_trust("claude", project) is False

    def test_a_launch_command_override_skips_seeding(
        self, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An override can point the agent at a profile this process cannot see."""
        monkeypatch.setenv("CCGRAM_CODEX_COMMAND", "CODEX_HOME=/profiles/work codex")
        project = tmp_path / "ccgram"
        project.mkdir()

        assert agent_startup.preseed_trust("codex", project) is False
        assert not (fake_home / ".codex" / "config.toml").exists()

    def test_a_configured_profile_without_a_file_is_not_seeded_elsewhere(
        self, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLAUDE_CONFIG_DIR names the profile the launch reads, and only that."""
        work = fake_home / ".claude-work"
        work.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(work))
        from ccgram.config import config

        monkeypatch.setattr(config, "claude_config_dir", work, raising=False)
        project = tmp_path / "ccgram"
        project.mkdir()

        assert agent_startup.preseed_trust("claude", project) is False
        home_config = json.loads((fake_home / ".claude.json").read_text())
        assert str(project) not in home_config.get("projects", {})


class TestAutostartGating:
    def test_global_switch_off_disables_seeding(
        self, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "autostart_trust", False, raising=False)
        project = tmp_path / "ccgram"
        project.mkdir()

        assert agent_startup.preseed_trust("claude", project) is False

    def test_per_agent_switch_beats_the_global_one(
        self, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CCGRAM_CLAUDE_AUTOSTART", "off")
        project = tmp_path / "ccgram"
        project.mkdir()

        assert agent_startup.preseed_trust("claude", project) is False

    def test_provider_without_a_recipe_is_untouched(
        self, fake_home: Path, tmp_path: Path
    ) -> None:
        project = tmp_path / "ccgram"
        project.mkdir()
        assert agent_startup.preseed_trust("pi", project) is False


class TestAnswerFirstRunPrompt:
    @pytest.fixture
    def adapter(self) -> MagicMock:
        adapter = MagicMock()
        adapter.send_keys = AsyncMock(return_value=(True, "ok"))
        return adapter

    async def test_sends_declared_keys_on_the_exact_fingerprint(
        self, fake_home: Path, adapter: MagicMock
    ) -> None:
        pane = "Do you trust the files in this folder?\n 1. Yes, proceed"
        with patch.object(agent_startup, "_capture_pane", AsyncMock(return_value=pane)):
            answered = await agent_startup.answer_first_run_prompt(
                "@1", "claude", adapter=adapter, poll=False
            )

        assert answered is True
        adapter.send_keys.assert_awaited_once_with("@1", ["Enter"])

    async def test_unknown_screen_sends_nothing(
        self, fake_home: Path, adapter: MagicMock
    ) -> None:
        with patch.object(
            agent_startup, "_capture_pane", AsyncMock(return_value="ready\n> ")
        ):
            answered = await agent_startup.answer_first_run_prompt(
                "@1", "codex", adapter=adapter, poll=False
            )

        assert answered is False
        adapter.send_keys.assert_not_awaited()

    async def test_settled_pane_short_circuits(
        self, fake_home: Path, adapter: MagicMock
    ) -> None:
        pane = "codex\n  for shortcuts, type /help"
        with patch.object(agent_startup, "_capture_pane", AsyncMock(return_value=pane)):
            assert (
                await agent_startup.answer_first_run_prompt(
                    "@1", "codex", adapter=adapter, poll=False
                )
                is False
            )

    async def test_disabled_autostart_skips_capture(
        self, fake_home: Path, adapter: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CCGRAM_CLAUDE_AUTOSTART", "off")
        capture = AsyncMock(return_value="Do you trust the files in this folder?")
        with patch.object(agent_startup, "_capture_pane", capture):
            answered = await agent_startup.answer_first_run_prompt(
                "@1", "claude", adapter=adapter, poll=False
            )

        assert answered is False
        capture.assert_not_awaited()
