"""A subscriber's disconnect keeps publishing for a grace period; after it, resumes
never bridge a gap silently."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ccgram.session_protocol import service as service_module
from ccgram.session_protocol.service import LocalSessionService
from ccgram.session_protocol.transcript_source import TranscriptSource

GRACE = service_module.SUBSCRIBER_GRACE_SECONDS


def _user(uuid: str, text: str, timestamp: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": {"role": "user", "content": text},
    }


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


@pytest.fixture
def connected(backend, store, tmp_path):
    path = tmp_path / "transcript.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for i in range(200):
            stream.write(
                json.dumps(_user(f"u{i}", f"text {i}", "2026-11-01T10:00:00Z")) + "\n"
            )
    backend.target = replace(backend.target, transcript=path)
    now = [1000.0]
    service = LocalSessionService(backend, store, clock=lambda: now[0])
    return service, path, now


def _source(service: LocalSessionService) -> TranscriptSource:
    return next(iter(service._sources.values()))


def _upserts(events) -> list[str]:
    return [e.data["id"] for e in events if e.type == "message.upsert"]


async def test_reconnect_within_grace_replays_exactly_without_rebuild(connected):
    service, path, now = connected
    await service.list_sessions()
    subscription = service.hub.subscribe(None)
    await service.refresh()
    await service.refresh()
    builder = _source(service).builder
    assert builder is not None
    epoch, cursor = service.hub.epoch, service.hub.seq

    service.hub.unsubscribe(subscription)
    now[0] += GRACE / 10
    _append(path, _user("new", "fresh", "2026-11-02T10:00:00Z"))
    await service.refresh()

    assert _source(service).builder is builder
    resumed, replay = service.hub.replay(epoch, cursor)
    assert resumed is True
    assert _upserts(replay) == ["conv/new"]


async def test_reconnect_after_grace_is_a_resync_and_no_message_is_lost(connected):
    service, path, now = connected
    await service.list_sessions()
    subscription = service.hub.subscribe(None)
    await service.refresh()
    await service.refresh()
    epoch, cursor = service.hub.epoch, service.hub.seq

    service.hub.unsubscribe(subscription)
    now[0] += GRACE + 1
    await service.refresh()
    expired_at = service.hub.seq
    _append(path, _user("gap", "during the gap", "2026-11-02T10:00:00Z"))
    await service.refresh()

    assert service.hub.replay(epoch, cursor)[0] is False
    assert service.hub.replay(epoch, expired_at)[0] is False

    published: list[str] = []
    real_publish = service.hub.publish

    def record(kind, session_id, data):
        if kind == "message.upsert":
            published.append(data.id)
        return real_publish(kind, session_id, data)

    service.hub.publish = record
    service.hub.subscribe(None)
    await service.refresh()
    assert "conv/gap" in published


async def test_memory_returns_to_lean_after_grace(connected):
    service, _, now = connected
    await service.list_sessions()
    subscription = service.hub.subscribe(None)
    await service.refresh()
    assert _source(service).builder is not None

    service.hub.unsubscribe(subscription)
    now[0] += GRACE + 1
    await service.refresh()
    source = _source(service)
    assert source.publishing is False
    assert source.indexed is False
    assert source.builder is None
    assert not service._revisions.get(next(iter(service._targets)))
