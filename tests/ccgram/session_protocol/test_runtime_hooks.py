import asyncio
import hashlib
import json
import time
from dataclasses import replace
from tempfile import TemporaryFile
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ccgram.providers.base import HookEvent
from ccgram.session_protocol.backend import Prompt
from ccgram.session_protocol.models import Action
from ccgram.session_protocol.requests import PendingHook
from ccgram.session_protocol.runtime import ProtocolRuntime


@pytest.mark.parametrize("kind", ["request.respond", "message.send"])
async def test_queued_hook_blocks_stale_actions_before_polling(
    service, backend, store, kind
):
    session = (await service.list_sessions())[0]
    backend.current_prompt = Prompt("/tmp/file\n1. Yes\n2. No", "permission")
    service.requests.observe_hook(
        session.id,
        PendingHook(
            "conv",
            "permission",
            "Edit",
            {"file_path": "/tmp/file", "content": "old"},
            time.time(),
        ),
    )
    snapshot = await service.get_session(session.id)
    data = (
        {"requestId": snapshot.open_requests[0].id, "optionId": "1"}
        if kind == "request.respond"
        else {"text": "continue", "dismissRequest": True}
    )
    with TemporaryFile() as lock:
        runtime = ProtocolRuntime(store, service, lock)
        runtime.observe_hook(
            HookEvent(
                "PermissionRequest",
                "@1",
                "conv",
                {"tool_name": "Edit", "tool_input_hash": "replacement"},
                time.time(),
            )
        )
        result, status = await service.execute(
            session.id, Action(str(uuid4()), snapshot.binding_version, kind, data)
        )
    assert status == 409 and result.outcome == "rejected"
    assert backend.sent == []


@pytest.mark.parametrize("matched", [False, True])
async def test_pending_hooks_release_only_after_transcript_processing(
    service, backend, store, tmp_path, monkeypatch, matched
):
    path = tmp_path / "transcript.jsonl"
    tool_input = {"command": "git status"}
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
    backend.current_prompt = Prompt("git status\n1. Yes\n2. No", "permission")
    stamp = time.time()
    service.requests.observe_hook(
        session.id, PendingHook("conv", "permission", "Bash", tool_input, stamp)
    )
    await service.get_session(session.id)
    original = service.requests.get(session.id)
    assert original is not None
    digest = hashlib.sha256(
        json.dumps(tool_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(
        "ccgram.session_protocol.runtime.asyncio.sleep",
        AsyncMock(side_effect=asyncio.CancelledError),
    )
    with TemporaryFile() as lock:
        runtime = ProtocolRuntime(store, service, lock)
        for _ in range(2):
            runtime.observe_hook(
                HookEvent(
                    "PermissionRequest",
                    "@1",
                    "conv",
                    {
                        "tool_name": "Bash",
                        "tool_input_hash": digest if matched else "missing",
                    },
                    stamp,
                )
            )
        assert not service.requests.is_current(session.id, original)
        with pytest.raises(asyncio.CancelledError):
            await runtime.poll("")
    assert service.requests.is_current(session.id, original) is matched
    if not matched:
        assert service.requests.get(session.id) is None


@pytest.mark.parametrize("failure", ["overflow", "processing"])
async def test_lost_hook_processing_keeps_session_blocked(
    service, backend, store, tmp_path, monkeypatch, failure
):
    path = tmp_path / "transcript.jsonl"
    path.touch()
    backend.target = replace(backend.target, transcript=path)
    session = (await service.list_sessions())[0]
    monkeypatch.setattr(
        "ccgram.session_protocol.runtime.asyncio.sleep",
        AsyncMock(side_effect=asyncio.CancelledError),
    )
    with TemporaryFile() as lock:
        runtime = ProtocolRuntime(store, service, lock)
        event = HookEvent(
            "PermissionRequest" if failure == "processing" else "Stop",
            "@1",
            "conv",
            {"tool_name": "Bash"},
            time.time(),
        )
        for _ in range(257 if failure == "overflow" else 1):
            runtime.observe_hook(event)
        if failure == "processing":

            def unavailable(self, *args):
                raise OSError("transcript unavailable")

            monkeypatch.setattr(
                "ccgram.session_protocol.service.MessageBuilder.advance", unavailable
            )
        with pytest.raises(asyncio.CancelledError):
            await runtime.poll("")
    assert service.requests.has_deferred(session.id)
