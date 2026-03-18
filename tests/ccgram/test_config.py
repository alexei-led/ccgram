"""Unit tests for Config — env var loading, validation, and user access."""

import socket
from pathlib import Path

import pytest

from ccgram.config import Config


@pytest.fixture
def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test:token")
    monkeypatch.setenv("ALLOWED_USERS", "12345")
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))


@pytest.mark.usefixtures("_base_env")
class TestConfigValid:
    def test_valid_config(self):
        cfg = Config()
        assert cfg.telegram_bot_token == "test:token"
        assert cfg.allowed_users == {12345}

    def test_custom_tmux_session_name(self, monkeypatch):
        monkeypatch.setenv("TMUX_SESSION_NAME", "mysession")
        cfg = Config()
        assert cfg.tmux_session_name == "mysession"

    def test_custom_monitor_poll_interval(self, monkeypatch):
        monkeypatch.setenv("MONITOR_POLL_INTERVAL", "5.0")
        cfg = Config()
        assert cfg.monitor_poll_interval == 5.0

    def test_is_user_allowed_true(self):
        cfg = Config()
        assert cfg.is_user_allowed(12345) is True

    def test_is_user_allowed_false(self):
        cfg = Config()
        assert cfg.is_user_allowed(99999) is False

    def test_group_id_default_none(self):
        cfg = Config()
        assert cfg.group_id is None

    def test_group_id_parsed_as_int(self, monkeypatch):
        monkeypatch.setenv("CCGRAM_GROUP_ID", "-1001234567890")
        cfg = Config()
        assert cfg.group_id == -1001234567890

    def test_instance_name_defaults_to_hostname(self):
        cfg = Config()
        assert cfg.instance_name == socket.gethostname()

    def test_instance_name_from_env(self, monkeypatch):
        monkeypatch.setenv("CCGRAM_INSTANCE_NAME", "bot-1")
        cfg = Config()
        assert cfg.instance_name == "bot-1"


@pytest.mark.usefixtures("_base_env")
class TestOwnWindowId:
    def test_own_window_id_default_none(self):
        cfg = Config()
        assert cfg.own_window_id is None

    def test_own_window_id_set_directly(self):
        cfg = Config()
        cfg.own_window_id = "@3"
        assert cfg.own_window_id == "@3"


@pytest.mark.usefixtures("_base_env")
class TestConfigMissingEnv:
    def test_missing_telegram_bot_token(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
            Config()

    def test_missing_allowed_users(self, monkeypatch):
        monkeypatch.delenv("ALLOWED_USERS", raising=False)
        with pytest.raises(ValueError, match="ALLOWED_USERS"):
            Config()

    def test_non_numeric_allowed_users(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_USERS", "abc")
        with pytest.raises(ValueError, match="non-numeric"):
            Config()

    def test_non_numeric_group_id(self, monkeypatch):
        monkeypatch.setenv("CCGRAM_GROUP_ID", "not-a-number")
        with pytest.raises(ValueError, match="CCGRAM_GROUP_ID must be a valid integer"):
            Config()


@pytest.mark.usefixtures("_base_env")
class TestClaudeConfigDir:
    def test_claude_config_dir_default(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        cfg = Config()
        assert cfg.claude_config_dir == Path.home() / ".claude"
        assert cfg.claude_projects_path == Path.home() / ".claude" / "projects"

    def test_claude_config_dir_override(self, monkeypatch, tmp_path):
        custom_dir = tmp_path / "custom-claude"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_dir))
        cfg = Config()
        assert cfg.claude_config_dir == custom_dir
        assert cfg.claude_projects_path == custom_dir / "projects"


@pytest.mark.usefixtures("_base_env")
class TestShowHiddenDirs:
    def test_show_hidden_dirs_default_false(self):
        cfg = Config()
        assert cfg.show_hidden_dirs is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "True", "YES"])
    def test_show_hidden_dirs_enabled(self, monkeypatch, value):
        monkeypatch.setenv("CCGRAM_SHOW_HIDDEN_DIRS", value)
        cfg = Config()
        assert cfg.show_hidden_dirs is True
