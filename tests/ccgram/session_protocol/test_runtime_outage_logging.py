"""A backend outage must log its edges, not once per retry.

The poll loop retries every second. Logging a full traceback for every failed
attempt buried the journal under identical stacks during a one-minute
multiplexer hiccup and hid the errors around it.
"""

from __future__ import annotations

import asyncio
from tempfile import TemporaryFile
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.session_protocol.models import ProtocolError
from ccgram.session_protocol.runtime import ProtocolRuntime


def _sleeper(iterations: int) -> AsyncMock:
    """Sleep *iterations* times, then cancel the loop like a shutdown would."""
    return AsyncMock(side_effect=[None] * iterations + [asyncio.CancelledError()])


async def test_repeated_outage_logs_one_warning(service, backend, store, monkeypatch):
    backend.unavailable = True
    logger = MagicMock()
    with TemporaryFile() as lock:
        monkeypatch.setattr("ccgram.session_protocol.runtime.logger", logger)
        monkeypatch.setattr(
            "ccgram.session_protocol.runtime.asyncio.sleep", _sleeper(4)
        )
        runtime = ProtocolRuntime(store, service, lock)
        with pytest.raises(asyncio.CancelledError):
            await runtime.poll("")
    assert logger.warning.call_count == 1
    message, error = logger.warning.call_args.args
    assert "source unavailable" in message
    assert "offline" in str(error)


async def test_recovery_logs_once_after_an_outage(service, store, monkeypatch):
    attempts = {"count": 0}

    async def refresh() -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ProtocolError(503, "backend_unavailable", "offline")

    monkeypatch.setattr(service, "refresh", refresh)
    logger = MagicMock()
    with TemporaryFile() as lock:
        monkeypatch.setattr("ccgram.session_protocol.runtime.logger", logger)
        monkeypatch.setattr(
            "ccgram.session_protocol.runtime.asyncio.sleep", _sleeper(3)
        )
        runtime = ProtocolRuntime(store, service, lock)
        with pytest.raises(asyncio.CancelledError):
            await runtime.poll("")
    assert logger.warning.call_count == 1
    recovered = [
        call for call in logger.info.call_args_list if "recovered" in call.args[0]
    ]
    assert len(recovered) == 1
