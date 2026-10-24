"""Tests for process exit behavior after bot polling stops."""

import os
import sys
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


def test_run_bot_execs_again_after_restart_callback() -> None:
    config = MagicMock(
        allowed_users=set(),
        claude_projects_path="/tmp/claude",
        multiplexer_name="herdr",
    )
    application = MagicMock()
    application.bot_data = {}
    application.run_polling.side_effect = lambda **_kwargs: application.bot_data[
        "request_restart"
    ]()

    with (
        patch.dict(os.environ, {"TMUX_SESSION_NAME": "test"}),
        patch("ccgram.main.setup_logging"),
        patch("ccgram.config.config", config),
        patch("ccgram.bot.create_bot", return_value=application),
        patch("ccgram.bot.polling_conflict_requires_restart", return_value=False),
        patch("ccgram.main._install_signal_handlers"),
        patch("ccgram.main.os.execv", side_effect=SystemExit(0)) as execv,
        patch("ccgram.main._reraise_shutdown_signal") as reraise_signal,
        pytest.raises(SystemExit, match="0"),
    ):
        run_bot()

    execv.assert_called_once_with(sys.argv[0], sys.argv)
    reraise_signal.assert_not_called()
