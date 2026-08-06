"""Tests for process exit behavior after bot polling stops."""

import os
from unittest.mock import MagicMock, patch

import pytest

from ccgram.main import run_bot


def test_run_bot_exits_nonzero_after_sustained_polling_conflict() -> None:
    config = MagicMock(
        allowed_users=set(),
        claude_projects_path="/tmp/claude",
        multiplexer_name="herdr",
    )
    application = MagicMock()

    with (
        patch.dict(os.environ, {"TMUX_SESSION_NAME": "test"}),
        patch("ccgram.main.setup_logging"),
        patch("ccgram.config.config", config),
        patch("ccgram.bot.create_bot", return_value=application),
        patch("ccgram.bot.polling_conflict_requires_restart", return_value=True),
        patch("ccgram.main._install_signal_handlers"),
        pytest.raises(SystemExit, match="1") as exc_info,
    ):
        run_bot()

    assert exc_info.value.code == 1
    application.run_polling.assert_called_once()
