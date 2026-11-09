"""Without a subscriber or a recent read, the protocol keeps no message index."""

import json
import tracemalloc
from typing import Any
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import time

import pytest

from ccgram.providers.base import HookEvent
from ccgram.session_protocol import service as service_module
from ccgram.session_protocol.messages import MessageBuilder
from ccgram.session_protocol.models import wire
from ccgram.session_protocol.requests import hook_input_hash
from ccgram.session_protocol.service import LocalSessionService
from ccgram.session_protocol.transcript_source import TranscriptSource

from .conftest import FakeBackend


class ManyTargets(FakeBackend):
    def __init__(self, targets) -> None:
        super().__init__()
        self._targets = targets

    async def targets(self):
        return self._targets

    async def inspect(self, target_id):
        return next(t for t in self._targets if t.id == target_id)


def _user(uuid: str, text: str, timestamp: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": {"role": "user", "content": text},
    }


def _write(path: Path, rows: list[dict], mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)


@pytest.fixture
def clock():
    now = [1000.0]
    return now


@pytest.fixture
def lean(backend, store, clock, tmp_path):
    path = tmp_path / "transcript.jsonl"
    old = datetime(2026, 11, 1, 10, tzinfo=UTC).isoformat()
    _write(path, [_user(f"u{i}", f"text {i}", old) for i in range(300)])
    backend.target = replace(backend.target, transcript=path)
    service = LocalSessionService(backend, store, clock=lambda: clock[0])
    return service, path


def _source(service: LocalSessionService, session_id: str) -> TranscriptSource:
    target = service._targets[session_id]
    return service._sources[target.id.casefold()]


async def test_telegram_only_refresh_keeps_no_index_and_publishes_nothing(lean):
    service, _ = lean
    session = (await service.list_sessions())[0]
    await service.refresh()
    source = _source(service, session.id)
    assert source.indexed is False
    assert source.builder is None
    _, events = service.hub.replay(service.hub.epoch, 0)
    assert not any(event.type == "message.upsert" for event in events)


async def test_delivery_is_correlated_without_an_index_and_materializes_one_message(
    lean, monkeypatch
):
    service, path = lean
    session = (await service.list_sessions())[0]
    await service.refresh()
    sent_at = time()
    await service.remember_delivery("act_1", session.id, "conv", "hello there", sent_at)
    sent = datetime.fromtimestamp(sent_at + 1, UTC).isoformat()
    _write(path, [_user("hello", "hello   there", sent)], mode="a")
    materialized: list[str] = []
    real_message = MessageBuilder.message

    def counting(self, message_id):
        materialized.append(message_id)
        return real_message(self, message_id)

    monkeypatch.setattr(MessageBuilder, "message", counting)
    await service.refresh()
    _, events = service.hub.replay(service.hub.epoch, 0)
    assert any(
        event.type == "action.updated"
        and event.data.get("actionId") == "act_1"
        and event.data.get("messageId") == "conv/hello"
        for event in events
    )
    assert set(materialized) == {"conv/hello"}
    assert _source(service, session.id).builder is None


async def test_history_read_builds_an_index_that_expires_when_idle(lean, clock):
    service, _ = lean
    session = (await service.list_sessions())[0]
    page = await service.history(session.id, None, 50)
    assert len(page.messages) == 50
    source = _source(service, session.id)
    assert source.indexed is True and source.builder is not None
    _, events = service.hub.replay(service.hub.epoch, 0)
    assert not any(event.type == "message.upsert" for event in events)
    clock[0] += service_module.INDEX_IDLE_SECONDS + 1
    await service.refresh()
    assert source.indexed is False
    assert source.builder is None


async def test_on_demand_history_matches_a_resident_index(lean):
    service, _ = lean
    session = (await service.list_sessions())[0]

    async def walk() -> list:
        pages, cursor = [], None
        while True:
            page = await service.history(session.id, cursor, 70)
            pages.append(wire(page))
            if not page.prev_cursor:
                return pages
            cursor = page.prev_cursor

    on_demand = await walk()
    service.hub.subscribe(None)
    await service.refresh()
    assert _source(service, session.id).builder is not None
    assert await walk() == on_demand


async def test_hook_matches_a_tool_call_without_an_index(lean):
    service, path = lean
    _write(
        path,
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "timestamp": "2026-11-01T10:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        }
                    ],
                },
            }
        ],
        mode="a",
    )
    session = (await service.list_sessions())[0]
    await service.refresh()
    target = service._targets[session.id]
    digest = hook_input_hash({"command": "ls"})

    def hook(input_hash: str) -> HookEvent:
        return HookEvent(
            "PermissionRequest",
            "ccgram:@1",
            "conv",
            {"tool_name": "Bash", "tool_input_hash": input_hash},
            time(),
        )

    assert await service._match_hook(session.id, target, hook(digest)) is True
    assert await service._match_hook(session.id, target, hook("nope")) is False
    assert _source(service, session.id).builder is None


async def test_replacement_without_an_index_bumps_the_generation(lean):
    service, path = lean
    session = (await service.list_sessions())[0]
    await service.refresh()
    replacement = path.with_name("replacement.jsonl")
    _write(replacement, [_user("new", "fresh", "2026-11-02T10:00:00Z")])
    replacement.replace(path)
    page = await service.history(session.id, None, 50)
    assert page.history_generation == 1
    assert [m.id for m in page.messages] == ["conv/g1/new"]


def test_telegram_only_retained_memory_is_constant_in_transcript_size(tmp_path):
    import asyncio

    async def retained(size_rows: int) -> int:
        paths = []
        for n in range(4):
            path = tmp_path / f"t{size_rows}-{n}.jsonl"
            _write(
                path,
                [
                    _user(f"u{i}", "x" * 200, "2026-11-01T10:00:00Z")
                    for i in range(size_rows)
                ],
            )
            paths.append(path)
        from ccgram.session_protocol.store import ProtocolStore

        store = ProtocolStore(tmp_path / f"protocol-{size_rows}.sqlite3")
        targets = [
            replace(
                FakeBackend().target,
                id=f"@{n + 1}",
                transcript=p,
                conversation_id=f"c{n}",
            )
            for n, p in enumerate(paths)
        ]
        backend: Any = ManyTargets(targets)
        service = LocalSessionService(backend, store)
        tracemalloc.start()
        try:
            await service.refresh()
            await service.refresh()
            current, _ = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            store.close()
        return current

    small = asyncio.run(retained(1_000))
    large = asyncio.run(retained(8_000))
    assert large < 512 * 1024, f"retained {large} bytes"
    assert large < small + 128 * 1024, f"retained grew {small} -> {large}"
