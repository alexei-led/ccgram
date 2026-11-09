"""Cancelling the shared poll stops it, even while a window scan runs in a thread."""

import asyncio
import contextlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from time import time

import pytest

from ccgram.session_protocol import runtime as runtime_module
from ccgram.session_protocol.runtime import ProtocolRuntime
from ccgram.session_protocol.service import LocalSessionService
from ccgram.session_protocol.transcript_source import TranscriptSource


def _write_user_rows(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for i in range(count):
            row = {
                "type": "user",
                "uuid": f"u{i}",
                "timestamp": "2026-11-01T10:00:00Z",
                "message": {"role": "user", "content": f"text {i}"},
            }
            stream.write(json.dumps(row) + "\n")


@pytest.fixture
def lean(backend, store, tmp_path):
    path = tmp_path / "transcript.jsonl"
    _write_user_rows(path, 50)
    backend.target = replace(backend.target, transcript=path)
    return LocalSessionService(backend, store)


async def test_cancel_during_lookback_scan_propagates(lean, monkeypatch):
    service = lean
    session = (await service.list_sessions())[0]
    await service.remember_delivery("act_1", session.id, "conv", "hello", time())
    entered, release = threading.Event(), threading.Event()
    real_scan = TranscriptSource.scan

    def blocking_scan(self, start, end, *, newest_first=False):
        for builder in real_scan(self, start, end, newest_first=newest_first):
            entered.set()
            release.wait(5)
            yield builder

    monkeypatch.setattr(TranscriptSource, "scan", blocking_scan)
    task = asyncio.create_task(service.refresh())
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancel_during_hook_scan_propagates(lean, monkeypatch):
    service = lean
    session = (await service.list_sessions())[0]
    await service.refresh()
    target = service._targets[session.id]
    entered, release = threading.Event(), threading.Event()
    real_scan = TranscriptSource.scan

    def blocking_scan(self, start, end, *, newest_first=False):
        for builder in real_scan(self, start, end, newest_first=newest_first):
            entered.set()
            release.wait(5)
            yield builder

    monkeypatch.setattr(TranscriptSource, "scan", blocking_scan)
    from ccgram.providers.base import HookEvent

    event = HookEvent(
        "PermissionRequest",
        "ccgram:@1",
        "conv",
        {"tool_name": "Bash", "tool_input_hash": "nope"},
        time(),
    )
    task = asyncio.create_task(service._match_hook(session.id, target, event))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_error_raised_while_cancelling_is_not_swallowed(lean, monkeypatch):
    service = lean
    task: asyncio.Task | None = None

    async def failing_update(target):
        assert task is not None
        task.cancel()
        raise ValueError("raised while cancelling")

    monkeypatch.setattr(service, "_update", failing_update)
    task = asyncio.create_task(service.refresh())
    with pytest.raises(ValueError, match="while cancelling"):
        await task


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


async def test_close_returns_when_the_poll_ignores_cancellation(monkeypatch):
    monkeypatch.setattr(runtime_module, "CLOSE_TIMEOUT_SECONDS", 0.05)
    released = asyncio.Event()

    async def stubborn() -> None:
        while not released.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0.01)

    store, lock = _Closable(), _Closable()
    runtime = ProtocolRuntime(store, None, lock, prefix="")  # type: ignore[arg-type]
    runtime.task = asyncio.create_task(stubborn())
    await asyncio.sleep(0)
    await asyncio.wait_for(runtime.close(), 2)
    assert store.closed and lock.closed
    released.set()
    await runtime.task
