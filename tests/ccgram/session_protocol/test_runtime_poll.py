"""The shared poll survives unexpected errors from one pass."""

import asyncio
from contextlib import suppress
from typing import Any

from ccgram.session_protocol import runtime as runtime_module
from ccgram.session_protocol.runtime import ProtocolRuntime


async def test_poll_keeps_refreshing_after_an_unexpected_error(monkeypatch):
    real_sleep = asyncio.sleep
    attempts: list[int] = []

    class FlakyService:
        async def refresh(self) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                raise ValueError("unexpected")

    async def no_wait(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(runtime_module.asyncio, "sleep", no_wait)
    service: Any = FlakyService()
    runtime = ProtocolRuntime(None, service, None, prefix="ccgram")  # type: ignore[arg-type]
    task = asyncio.create_task(runtime.poll("ccgram"))
    for _ in range(200):
        if len(attempts) >= 3:
            break
        await real_sleep(0)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert len(attempts) >= 3
