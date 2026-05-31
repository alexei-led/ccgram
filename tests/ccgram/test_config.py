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


@pytest.mark.usefixtures("_base_env")
class TestHideToolCalls:
    def test_hide_tool_calls_default_false(self, monkeypatch):
        monkeypatch.delenv("CCGRAM_HIDE_TOOL_CALLS", raising=False)
        cfg = Config()
        assert cfg.hide_tool_calls is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "True", "YES", "Yes"])
    def test_hide_tool_calls_enabled(self, monkeypatch, value):
        monkeypatch.setenv("CCGRAM_HIDE_TOOL_CALLS", value)
        cfg = Config()
        assert cfg.hide_tool_calls is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_hide_tool_calls_disabled(self, monkeypatch, value):
        monkeypatch.setenv("CCGRAM_HIDE_TOOL_CALLS", value)
        cfg = Config()
        assert cfg.hide_tool_calls is False


@pytest.mark.usefixtures("_base_env")
class TestStatusMode:
    def test_default_is_system(self, monkeypatch):
        monkeypatch.delenv("CCGRAM_STATUS_MODE", raising=False)
        assert Config().status_mode == "system"

    @pytest.mark.parametrize("value", ["user", "USER", "User", " user "])
    def test_user_mode_normalized(self, monkeypatch, value):
        monkeypatch.setenv("CCGRAM_STATUS_MODE", value)
        assert Config().status_mode == "user"

    @pytest.mark.parametrize("value", ["system", "SYSTEM", " System "])
    def test_system_mode_normalized(self, monkeypatch, value):
        monkeypatch.setenv("CCGRAM_STATUS_MODE", value)
        assert Config().status_mode == "system"

    @pytest.mark.parametrize("value", ["", "garbage", "1", "off", "default"])
    def test_invalid_falls_back_to_system(self, monkeypatch, value):
        monkeypatch.setenv("CCGRAM_STATUS_MODE", value)
        assert Config().status_mode == "system"


@pytest.mark.usefixtures("_base_env")
class TestLiveViewConfig:
    @pytest.mark.parametrize(
        ("attr", "env_var", "default", "env_str", "expected"),
        [
            ("live_view_interval", "CCGRAM_LIVE_VIEW_INTERVAL", 5, "10", 10),
            ("live_view_timeout", "CCGRAM_LIVE_VIEW_TIMEOUT", 300, "600", 600),
        ],
    )
    def test_default_and_override(
        self, monkeypatch, attr, env_var, default, env_str, expected
    ):
        assert getattr(Config(), attr) == default
        monkeypatch.setenv(env_var, env_str)
        assert getattr(Config(), attr) == expected

    @pytest.mark.parametrize(
        ("attr", "env_var"),
        [
            ("live_view_interval", "CCGRAM_LIVE_VIEW_INTERVAL"),
            ("live_view_timeout", "CCGRAM_LIVE_VIEW_TIMEOUT"),
        ],
    )
    def test_zero_clamped_to_one(self, monkeypatch, attr, env_var):
        monkeypatch.setenv(env_var, "0")
        assert getattr(Config(), attr) == 1

    @pytest.mark.parametrize(
        "env_var",
        ["CCGRAM_LIVE_VIEW_INTERVAL", "CCGRAM_LIVE_VIEW_TIMEOUT"],
    )
    def test_invalid_raises(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "not-a-number")
        with pytest.raises(ValueError, match=env_var):
            Config()


@pytest.mark.usefixtures("_base_env")
class TestPollingConfig:
    @pytest.mark.parametrize(
        ("attr", "env_var", "default", "env_str", "expected", "clamp_str", "clamped"),
        [
            (
                "monitor_poll_interval",
                "MONITOR_POLL_INTERVAL",
                1.0,
                "0.8",
                0.8,
                "0.1",
                0.5,
            ),
            (
                "status_poll_interval",
                "CCGRAM_STATUS_POLL_INTERVAL",
                1.0,
                "2.0",
                2.0,
                "0.2",
                0.5,
            ),
        ],
    )
    def test_default_override_and_clamp(
        self, monkeypatch, attr, env_var, default, env_str, expected, clamp_str, clamped
    ):
        assert getattr(Config(), attr) == default
        monkeypatch.setenv(env_var, env_str)
        assert getattr(Config(), attr) == expected
        monkeypatch.setenv(env_var, clamp_str)
        assert getattr(Config(), attr) == clamped
