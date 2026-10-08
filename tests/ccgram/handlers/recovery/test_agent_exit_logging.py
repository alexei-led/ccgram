"""An exited agent remains recoverable without logging on every poll."""

import pytest
from structlog.testing import capture_logs

from ccgram.handlers.recovery.transcript_discovery import (
    discover_and_register_transcript,
)
from ccgram.multiplexer.base import WindowRef
from ccgram.window_state_store import window_store


@pytest.mark.parametrize("provider_name", ["pi", "codex", "claude"])
@pytest.mark.parametrize("shell", ["fish", "bash"])
async def test_repeated_shell_polls_preserve_recovery_without_log_spam(
    provider_name: str, shell: str
) -> None:
    window_id = "@7"
    state = window_store.get_window_state(window_id)
    state.provider_name = provider_name
    state.initial_provider_name = provider_name
    state.cwd = "/project"
    state.session_id = "recoverable-session"
    state.transcript_path = "/project/transcript.jsonl"
    window = WindowRef(window_id, "agent", state.cwd, shell)

    with capture_logs() as logs:
        results = [
            await discover_and_register_transcript(window_id, _window=window)
            for _ in range(3)
        ]

    assert results == [True, True, True]
    assert state.provider_name == provider_name
    assert state.initial_provider_name == provider_name
    assert state.session_id == "recoverable-session"
    assert state.transcript_path == "/project/transcript.jsonl"
    assert not any(
        record["event"] == "Agent exited to shell; keeping provider for recovery"
        for record in logs
    )
