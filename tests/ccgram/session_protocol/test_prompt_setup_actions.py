import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ccgram.session_protocol.models import Action, ProtocolError
from ccgram.session_protocol.telegram import TelegramSessionAdapter


@pytest.mark.parametrize("source", ["missing", "unreadable"])
async def test_close_ignores_unavailable_transcript(
    service, backend, tmp_path, monkeypatch, source
):
    path = tmp_path / "transcript.jsonl"
    if source == "unreadable":
        path.mkdir()
    else:
        path.touch()
    backend.target = replace(backend.target, transcript=path)
    if source == "missing":
        await service.refresh()
        path.unlink()
    snapshot = (await service.list_sessions())[0]
    assert "terminal.paste" not in snapshot.capabilities
    assert "session.close" in snapshot.capabilities
    denied, _ = await service.execute(
        snapshot.id,
        Action(
            str(uuid4()), snapshot.binding_version, "terminal.paste", {"text": "unsafe"}
        ),
    )
    assert denied.outcome == "rejected"
    assert backend.sent == []
    close = AsyncMock(wraps=backend.close)
    monkeypatch.setattr(backend, "close", close)
    adapter = TelegramSessionAdapter(service)
    assert (await adapter.close("@1"))[0]
    close.assert_awaited_once()


@pytest.mark.parametrize("clear", [True, False])
async def test_setup_holds_shared_lock_for_entire_sequence(
    service, backend, monkeypatch, clear
):
    backend.target = replace(
        backend.target, provider="shell", foreground_argv=("bash",)
    )
    snapshot = (await service.list_sessions())[0]
    reached = asyncio.Event()
    release = asyncio.Event()
    original_key = backend.send_key

    async def block_first_key(target, key):
        result = await original_key(target, key)
        reached.set()
        await release.wait()
        return result

    monkeypatch.setattr(backend, "send_key", block_first_key)
    setup = Action(
        str(uuid4()),
        snapshot.binding_version,
        "terminal.setup_prompt",
        {"text": "set prompt", "clear": clear},
    )
    pending = asyncio.create_task(service.execute_internal(snapshot.id, setup))
    await asyncio.wait_for(reached.wait(), 1)
    paste = asyncio.create_task(
        service.execute(
            snapshot.id,
            Action(
                str(uuid4()),
                snapshot.binding_version,
                "terminal.paste",
                {"text": "mobile"},
            ),
        )
    )
    await asyncio.sleep(0)
    assert backend.sent == [("key", "C-c", False)]
    release.set()
    assert (await pending)[0].outcome == "sent"
    assert (await paste)[0].outcome == "sent"
    assert backend.sent == [
        ("key", "C-c", False),
        ("text", "set prompt", True),
        *([("text", "clear", True)] if clear else []),
        ("text", "mobile", False),
    ]


@pytest.mark.parametrize("state", ["stale", "gone", "unavailable"])
async def test_close_keeps_liveness_and_binding_checks(
    service, backend, monkeypatch, state
):
    snapshot = (await service.list_sessions())[0]
    close = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "close", close)
    if state == "stale":
        backend.target = replace(backend.target, process_id=999)
    elif state == "gone":
        backend.disappear = True
    else:
        monkeypatch.setattr(
            backend,
            "inspect",
            AsyncMock(side_effect=ProtocolError(503, "backend_unavailable", "offline")),
        )
    result, _ = await service.execute(
        snapshot.id, Action(str(uuid4()), snapshot.binding_version, "session.close", {})
    )
    assert result.outcome == "rejected"
    close.assert_not_awaited()


async def test_cancelled_setup_releases_lock_without_replaying_partial_input(
    service, backend, monkeypatch
):
    backend.target = replace(
        backend.target, provider="shell", foreground_argv=("bash",)
    )
    snapshot = (await service.list_sessions())[0]
    reached = asyncio.Event()
    original_key = backend.send_key

    async def block_first_key(target, key):
        await original_key(target, key)
        reached.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(backend, "send_key", block_first_key)
    setup = Action(
        str(uuid4()),
        snapshot.binding_version,
        "terminal.setup_prompt",
        {"text": "setup", "clear": True},
    )
    pending = asyncio.create_task(service.execute_internal(snapshot.id, setup))
    await reached.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    result, _ = await service.execute_internal(snapshot.id, setup)
    assert result.outcome == "unknown"
    paste, _ = await asyncio.wait_for(
        service.execute(
            snapshot.id,
            Action(
                str(uuid4()),
                snapshot.binding_version,
                "terminal.paste",
                {"text": "mobile"},
            ),
        ),
        1,
    )
    assert paste.outcome == "sent"
    assert backend.sent == [("key", "C-c", False), ("text", "mobile", False)]


async def test_prompt_setup_is_not_a_public_wire_action(service, backend):
    snapshot = (await service.list_sessions())[0]
    with pytest.raises(ProtocolError) as error:
        await service.execute(
            snapshot.id,
            Action(
                str(uuid4()),
                snapshot.binding_version,
                "terminal.setup_prompt",
                {"text": "setup", "clear": True},
            ),
        )
    assert error.value.code == "invalid_request"
    assert backend.sent == []


@pytest.mark.parametrize("failure", ["interrupt", "command", "clear"])
async def test_prompt_setup_stops_at_first_backend_failure(
    service, backend, monkeypatch, failure
):
    backend.target = replace(
        backend.target, provider="shell", foreground_argv=("bash",)
    )
    snapshot = (await service.list_sessions())[0]
    original_text = backend.send_text
    if failure == "interrupt":
        backend.fail = True
    else:

        async def fail_text(target, text, **kwargs):
            await original_text(target, text, **kwargs)
            return text != ("setup" if failure == "command" else "clear")

        monkeypatch.setattr(backend, "send_text", fail_text)
    result, _ = await service.execute_internal(
        snapshot.id,
        Action(
            str(uuid4()),
            snapshot.binding_version,
            "terminal.setup_prompt",
            {"text": "setup", "clear": True},
        ),
    )
    assert result.outcome == "unknown"
    assert backend.sent == [
        ("key", "C-c", False),
        *([("text", "setup", True)] if failure != "interrupt" else []),
        *([("text", "clear", True)] if failure == "clear" else []),
    ]


async def test_close_works_after_transcript_generation_changes(
    service, backend, tmp_path, monkeypatch
):
    path = tmp_path / "transcript.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    backend.target = replace(backend.target, transcript=path)
    before = (await service.list_sessions())[0]
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text("{}\n", encoding="utf-8")
    replacement.replace(path)
    _, current = await service.for_target("@1")
    assert current.binding_version > before.binding_version
    close = AsyncMock(wraps=backend.close)
    monkeypatch.setattr(backend, "close", close)
    result, _ = await service.execute(
        current.id, Action(str(uuid4()), current.binding_version, "session.close", {})
    )
    assert result.outcome == "sent"
    close.assert_awaited_once()
