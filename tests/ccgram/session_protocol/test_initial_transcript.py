from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ccgram.session_protocol.service import LocalSessionService
from ccgram.session_protocol.store import ProtocolStore
from ccgram.session_protocol.telegram import TelegramSessionAdapter


async def test_first_claude_message_creates_transcript_without_readiness_deadlock(
    service, backend, tmp_path, monkeypatch
):
    path = tmp_path / "new-session.jsonl"
    backend.target = replace(backend.target, transcript=path)
    target, snapshot = await service.for_target(backend.target.id)

    assert target.foreground_verified
    assert "message.send" in snapshot.capabilities
    assert not path.exists()
    assert not (await service.history(snapshot.id, None, 10)).messages
    assert not path.exists()

    original_send = backend.send_text

    async def send(target, text, *, enter, raw=None):
        result = await original_send(target, text, enter=enter, raw=raw)
        now = datetime.now(UTC).isoformat()
        records = [
            {
                "type": "user",
                "uuid": "first-user",
                "timestamp": now,
                "message": {"content": text},
            },
            {
                "type": "assistant",
                "uuid": "first-reply",
                "timestamp": now,
                "message": {"content": [{"type": "text", "text": "Hello"}]},
            },
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return result

    monkeypatch.setattr(backend, "send_text", send)
    success, message = await TelegramSessionAdapter(service).send_text(
        backend.target.id, "Hi", action_id="telegram-message:first-claude"
    )

    assert (success, message) == (True, "Sent")
    assert backend.sent == [("text", "Hi", True)]
    assert path.exists()
    history = await service.history(snapshot.id, None, 10)
    assert [item.role for item in history.messages] == ["user", "agent"]
    assert [item.id for item in history.messages] == [
        "conv/first-user",
        "conv/first-reply",
    ]


async def test_observed_transcript_disappearance_stays_unavailable_after_restart(
    service, backend, store, tmp_path
):
    path = tmp_path / "seen.jsonl"
    path.touch()
    backend.target = replace(backend.target, transcript=path)
    assert (await service.for_target(backend.target.id))[0].foreground_verified
    path.unlink()

    for current in [service, LocalSessionService(backend, store)]:
        target, snapshot = await current.for_target(backend.target.id)
        assert not target.available
        assert not target.foreground_verified
        assert "message.send" not in snapshot.capabilities
    assert backend.sent == []
    store.close()
    reopened = ProtocolStore(store.path)
    try:
        target, snapshot = await LocalSessionService(backend, reopened).for_target(
            backend.target.id
        )
        assert not target.available
        assert "message.send" not in snapshot.capabilities
    finally:
        reopened.close()


@pytest.mark.parametrize("failure", [PermissionError, IsADirectoryError])
async def test_unreadable_initial_transcript_does_not_admit_input(
    service, backend, tmp_path, monkeypatch, failure
):
    path = tmp_path / "unreadable.jsonl"
    path.touch()
    backend.target = replace(backend.target, transcript=path)
    original_open = Path.open

    def open_path(self, *args, **kwargs):
        if self == path:
            raise failure("source cannot be read")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    target, snapshot = await service.for_target(backend.target.id)

    assert not target.available
    assert "message.send" not in snapshot.capabilities
    assert backend.sent == []


@pytest.mark.parametrize(
    ("provider", "argv"),
    [("claude", ()), ("unknown", ("wrapper",)), ("shell", ("bash",))],
)
async def test_pending_transcript_cannot_supply_foreground_evidence(
    service, backend, tmp_path, provider, argv
):
    backend.target = replace(
        backend.target,
        transcript=tmp_path / "missing.jsonl",
        provider=provider,
        foreground_argv=argv,
    )

    target, snapshot = await service.for_target(backend.target.id)

    assert not target.foreground_verified
    assert "message.send" not in snapshot.capabilities
    assert backend.sent == []


async def test_pending_history_does_not_hide_prompt_probe_failure(
    service, backend, tmp_path, monkeypatch
):
    backend.target = replace(backend.target, transcript=tmp_path / "missing.jsonl")
    from ccgram.session_protocol.models import ProtocolError

    monkeypatch.setattr(
        backend,
        "prompt",
        AsyncMock(
            side_effect=ProtocolError(503, "backend_unavailable", "screen unavailable")
        ),
    )

    target, snapshot = await service.for_target(backend.target.id)

    assert not target.available
    assert "message.send" not in snapshot.capabilities
    assert backend.sent == []
