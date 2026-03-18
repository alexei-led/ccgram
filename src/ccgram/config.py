"""Application configuration — reads env vars and exposes a singleton.

Loads TELEGRAM_BOT_TOKEN, ALLOWED_USERS, tmux/Claude paths, and
monitoring intervals from environment variables (with .env support).
.env loading priority: local .env (cwd) > $CCGRAM_DIR/.env (default ~/.ccgram).
The module-level `config` instance is imported by nearly every other module.

Key class: Config (singleton instantiated as `config`).
"""

import structlog
import os
import socket
from pathlib import Path

from dotenv import load_dotenv

from .utils import ccgram_dir

logger = structlog.get_logger()


def _env_with_fallback(new_name: str, old_name: str, default: str = "") -> str:
    """Read env var with fallback to legacy CCBOT_* name."""
    val = os.getenv(new_name)
    if val is not None:
        return val
    val = os.getenv(old_name)
    if val is not None:
        logger.warning("%s is deprecated, use %s instead", old_name, new_name)
        return val
    return default


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.config_dir = ccgram_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load .env: local (cwd) takes priority over config_dir
        # load_dotenv default override=False means first-loaded wins
        local_env = Path(".env")
        global_env = self.config_dir / ".env"
        if local_env.is_file():
            load_dotenv(local_env)
            logger.debug("Loaded env from %s", local_env.resolve())
        if global_env.is_file():
            load_dotenv(global_env)
            logger.debug("Loaded env from %s", global_env)

        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN") or ""
        if not self.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

        allowed_users_str = os.getenv("ALLOWED_USERS", "")
        if not allowed_users_str:
            raise ValueError("ALLOWED_USERS environment variable is required")
        try:
            self.allowed_users: set[int] = {
                int(uid.strip()) for uid in allowed_users_str.split(",") if uid.strip()
            }
        except ValueError as e:
            raise ValueError(
                f"ALLOWED_USERS contains non-numeric value: {e}. "
                "Expected comma-separated Telegram user IDs."
            ) from e

        # Tmux session name and window naming
        self.tmux_session_name = os.getenv("TMUX_SESSION_NAME", "ccgram")
        self.tmux_main_window_name = "__main__"
        # Own tmux window ID (set by run_bot() after auto-detect, used to skip self in list_windows)
        self.own_window_id: str | None = None

        # All state files live under config_dir
        self.state_file = self.config_dir / "state.json"
        self.session_map_file = self.config_dir / "session_map.json"
        self.monitor_state_file = self.config_dir / "monitor_state.json"
        self.events_file = self.config_dir / "events.jsonl"

        # Claude Code session monitoring configuration
        _claude_config_dir = os.getenv("CLAUDE_CONFIG_DIR")
        self.claude_config_dir: Path = (
            Path(_claude_config_dir).expanduser()
            if _claude_config_dir
            else Path.home() / ".claude"
        )
        self.claude_projects_path = self.claude_config_dir / "projects"
        self.monitor_poll_interval = float(os.getenv("MONITOR_POLL_INTERVAL", "2.0"))

        # Multi-instance support
        group_id_str = _env_with_fallback("CCGRAM_GROUP_ID", "CCBOT_GROUP_ID")
        if group_id_str:
            try:
                self.group_id: int | None = int(group_id_str)
            except ValueError as e:
                raise ValueError(f"CCGRAM_GROUP_ID must be a valid integer: {e}") from e
        else:
            self.group_id = None

        self.instance_name: str = (
            _env_with_fallback("CCGRAM_INSTANCE_NAME", "CCBOT_INSTANCE_NAME")
            or socket.gethostname()
        )

        # Provider selection
        self.provider_name: str = _env_with_fallback(
            "CCGRAM_PROVIDER", "CCBOT_PROVIDER", "claude"
        )

        # Directory browser: show hidden (dot) directories
        self.show_hidden_dirs: bool = _env_with_fallback(
            "CCGRAM_SHOW_HIDDEN_DIRS", "CCBOT_SHOW_HIDDEN_DIRS"
        ).lower() in ("1", "true", "yes")

        # Whisper transcription
        self.whisper_provider: str = _env_with_fallback(
            "CCGRAM_WHISPER_PROVIDER", "CCBOT_WHISPER_PROVIDER"
        )
        self.whisper_api_key: str = _env_with_fallback(
            "CCGRAM_WHISPER_API_KEY", "CCBOT_WHISPER_API_KEY"
        )
        self.whisper_base_url: str = _env_with_fallback(
            "CCGRAM_WHISPER_BASE_URL", "CCBOT_WHISPER_BASE_URL"
        )
        self.whisper_model: str = _env_with_fallback(
            "CCGRAM_WHISPER_MODEL", "CCBOT_WHISPER_MODEL"
        )
        self.whisper_language: str = _env_with_fallback(
            "CCGRAM_WHISPER_LANGUAGE", "CCBOT_WHISPER_LANGUAGE"
        )

        # Auto-close stale topics (minutes; 0 = disabled)
        self.autoclose_done_minutes = int(os.getenv("AUTOCLOSE_DONE_MINUTES", "30"))
        self.autoclose_dead_minutes = int(os.getenv("AUTOCLOSE_DEAD_MINUTES", "10"))

        logger.debug(
            "Config initialized: dir=%s, token=%s..., allowed_users=%d, "
            "tmux_session=%s",
            self.config_dir,
            self.telegram_bot_token[:8],
            len(self.allowed_users),
            self.tmux_session_name,
        )

    def is_user_allowed(self, user_id: int) -> bool:
        """Check if a user is in the allowed list."""
        return user_id in self.allowed_users


config = Config()
