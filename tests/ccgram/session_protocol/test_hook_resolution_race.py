import asyncio
import contextlib
import json
import threading
import time
from dataclasses import replace
from tempfile import TemporaryFile
from unittest.mock import AsyncMock

import pytest

from ccgram.providers.base import HookEvent
from ccgram.session_protocol import service as service_module
from ccgram.session_protocol.backend import Prompt
from ccgram.session_protocol.requests import hook_input_hash
from ccgram.session_protocol.runtime import ProtocolRuntime


async def test_obsolete_unmatched_resolution_preserves_newer_queued_evidence(
    service, backend, store, tmp_path, monkeypatch
):
    tool_input = {"command": "newer"}
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        json.dumps(
            {
                "uuid": "tool",
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call",
                            "name": "Bash",
                            "input": tool_input,
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    backend.target = replace(backend.target, transcript=path)
    session = (await service.list_sessions())[0]
    backend.current_prompt = Prompt("newer\n1. Yes\n2. No", "permission")
    await service.get_session(session.id)
    advance = service_module.MessageBuilder.advance
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def delayed_read(builder, *args):
        if not release.is_set():
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise TimeoutError("test did not release transcript read")
        return advance(builder, *args)

    monkeypatch.setattr(service_module.MessageBuilder, "advance", delayed_read)
    monkeypatch.setattr(
        "ccgram.session_protocol.runtime.asyncio.sleep",
        AsyncMock(side_effect=asyncio.CancelledError),
    )
    now = time.time()
    with TemporaryFile() as lock:
        runtime = ProtocolRuntime(store, service, lock)
        runtime.observe_hook(
            HookEvent(
                "PermissionRequest",
                "@1",
                "conv",
                {"tool_name": "Bash", "tool_input_hash": "missing"},
                now,
            )
        )
        pending = asyncio.create_task(runtime.poll(""))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            runtime.observe_hook(
                HookEvent(
                    "PermissionRequest",
                    "@1",
                    "conv",
                    {
                        "tool_name": "Bash",
                        "tool_input_hash": hook_input_hash(tool_input),
                    },
                    now + 0.001,
                )
            )
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            release.set()
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
    snapshot = await service.get_session(session.id)
    request = snapshot.open_requests[0]
    assert request.source == "hook"
    assert request.tool is not None and request.tool["input"] == tool_input
    assert request.options
