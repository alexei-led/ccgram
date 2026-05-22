"""Handler test fixtures — short-circuit slow real waits."""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _instant_session_map_wait(monkeypatch):
    from ccgram.session_map import session_map_sync

    monkeypatch.setattr(
        session_map_sync,
        "wait_for_session_map_entry",
        AsyncMock(return_value=True),
    )


@pytest.fixture(autouse=True)
def _wire_runtime_callbacks_for_tests():
    """Wire safe no-op defaults for the three startup-registered callbacks.

    Production wires these in bot.post_init via ``bootstrap.wire_runtime_callbacks``;
    unit tests bypass that path entirely.  Without this fixture, every test
    exercising a Stop event, status bubble render, or shell approval would
    have to wire the callback itself.  Tests that exercise the unwired-state
    failure mode call ``bootstrap.reset_for_testing()`` themselves.
    """
    from ccgram import bootstrap
    from ccgram.handlers import hook_events
    from ccgram.handlers.shell import shell_capture

    bootstrap.reset_for_testing()

    hook_events.register_stop_callback(AsyncMock())
    shell_capture.register_approval_callback(AsyncMock(return_value=False))

    yield

    bootstrap.reset_for_testing()


@pytest.fixture(autouse=True)
def _disable_send_rate_limit(monkeypatch):
    """Zero out MESSAGE_SEND_INTERVAL so back-to-back sends don't sleep.

    Tests in TestRateLimitSend re-patch the interval inline when they
    need to assert on the wait calculation.
    """
    monkeypatch.setattr(
        "ccgram.handlers.messaging_pipeline.message_sender.MESSAGE_SEND_INTERVAL", 0
    )


@pytest.fixture(autouse=True)
def _zero_command_orchestration_delays(monkeypatch):
    """Collapse command-orchestration probe / fallback delays."""
    monkeypatch.setattr(
        "ccgram.handlers.commands.status_snapshot._CODEX_STATUS_FALLBACK_DELAY_SECONDS",
        0,
    )
    monkeypatch.setattr(
        "ccgram.handlers.commands.failure_probe._COMMAND_ERROR_PROBE_DELAY_SECONDS",
        0,
    )
