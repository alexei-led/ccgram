"""Integration tests for Config — real .env files and filesystem.

The value-parsing rules are unit-tested in ``tests/ccgram/test_config.py``;
what needs a real filesystem is where a value is allowed to come from, and
which source wins.
"""

import pytest

from ccgram.config import Config

pytestmark = pytest.mark.integration


class TestConfigIntegration:
    def test_reads_env_file_from_config_dir(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=from-dotenv-token\nALLOWED_USERS=99999\n"
        )
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("ALLOWED_USERS", raising=False)
        cfg = Config()
        assert cfg.telegram_bot_token == "from-dotenv-token"
        assert cfg.is_user_allowed(99999)

    def test_process_env_wins_over_env_file(self, tmp_path, monkeypatch):
        """load_dotenv runs with override=False, so an exported value stands."""
        (tmp_path / ".env").write_text(
            "TELEGRAM_BOT_TOKEN=from-dotenv-token\nALLOWED_USERS=1\n"
        )
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "from-process-env")
        monkeypatch.setenv("ALLOWED_USERS", "1")

        assert Config().telegram_bot_token == "from-process-env"

    def test_cwd_env_file_wins_over_config_dir_env_file(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / ".env").write_text(
            "TELEGRAM_BOT_TOKEN=from-config-dir\nALLOWED_USERS=1\n"
        )
        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text("TELEGRAM_BOT_TOKEN=from-cwd\nALLOWED_USERS=2\n")
        monkeypatch.chdir(project)
        monkeypatch.setenv("CCGRAM_DIR", str(config_dir))
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("ALLOWED_USERS", raising=False)

        cfg = Config()

        assert cfg.telegram_bot_token == "from-cwd"
        assert cfg.is_user_allowed(2)
        assert not cfg.is_user_allowed(1)

    def test_creates_config_dir_if_missing(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "nonexistent"
        monkeypatch.setenv("CCGRAM_DIR", str(new_dir))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-create-dir")
        monkeypatch.setenv("ALLOWED_USERS", "1")
        Config()
        assert new_dir.is_dir()

    def test_multiple_comma_separated_allowed_users(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-multi")
        monkeypatch.setenv("ALLOWED_USERS", "123,456,789")
        cfg = Config()
        assert cfg.is_user_allowed(123)
        assert cfg.is_user_allowed(456)
        assert cfg.is_user_allowed(789)
        assert not cfg.is_user_allowed(999)
