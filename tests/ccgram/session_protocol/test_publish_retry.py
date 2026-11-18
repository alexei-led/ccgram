"""Publish and correlation state is cleared only once the work succeeded."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import time

import pytest

from ccgram.session_protocol import service as service_module
from ccgram.session_protocol.messages import SourceChangedError
from ccgram.session_protocol.service import LocalSessionService
from ccgram.session_protocol.transcript_source import TranscriptSource

GRACE = service_module.SUBSCRIBER_GRACE_SECONDS
OLD = datetime(2026, 11, 1, 10, tzinfo=UTC).isoformat()


def _user(uuid: str, text: str, timestamp: str = OLD) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": {"role": "user", "content": text},
    }


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _upserts(events) -> set[str]:
    return {e.data["id"] for e in events if e.type == "message.upsert"}


@pytest.fixture
def connected(backend, store, tmp_path):
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        "".join(json.dumps(_user(f"u{i}", f"text {i}")) + "\n" for i in range(300)),
        encoding="utf-8",
    )
    backend.target = replace(backend.target, transcript=path)
    now = [1000.0]
    service = LocalSessionService(backend, store, clock=lambda: now[0])
    return service, path, now


async def test_failed_first_publish_is_retried_in_full(connected, monkeypatch):
    service, path, now = connected
    session = (await service.list_sessions())[0]
    first = service.hub.subscribe(None)
    await service.refresh()
    service.hub.unsubscribe(first)
    # Grace ends while a recent read keeps the index resident, so the builder
    # survives with an empty dirty set and the next publish starts over.
    now[0] += GRACE + 1
    service.request_index(session.id)
    await service.refresh()
    _append(path, _user("new", "fresh"))
    await service.refresh()

    published: list[str] = []
    real_publish = service.hub.publish

    def record(kind, session_id, data, **kwargs):
        if kind == "message.upsert":
            published.append(data.id)
        return real_publish(kind, session_id, data, **kwargs)

    service.hub.publish = record
    service.hub.subscribe(None)
    calls = 0
    real = service_module._materialize_messages

    def flaky(builder, ids):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SourceChangedError("Indexed record moved")
        return real(builder, ids)

    monkeypatch.setattr(service_module, "_materialize_messages", flaky)
    await service.refresh()
    monkeypatch.setattr(service_module, "_materialize_messages", real)
    await service.refresh()

    expected = {f"conv/u{i}" for i in range(300)} | {"conv/new"}
    assert expected <= set(published)


async def test_failed_correlation_scan_is_retried(
    backend, store, tmp_path, monkeypatch
):
    path = tmp_path / "transcript.jsonl"
    path.write_text(json.dumps(_user("u0", "first")) + "\n", encoding="utf-8")
    backend.target = replace(backend.target, transcript=path)
    service = LocalSessionService(backend, store)
    session = (await service.list_sessions())[0]
    await service.refresh()

    sent_at = time()
    await service.remember_delivery("act_1", session.id, "conv", "hello there", sent_at)
    # The send's lookback runs before the echo is in the transcript.
    await service.refresh()
    echo = datetime.fromtimestamp(sent_at + 1, UTC).isoformat()
    _append(path, _user("echo", "hello there", echo))

    real_scan = TranscriptSource.scan
    failing = True

    def flaky_scan(self, start, end, *, newest_first=False):
        if failing:
            raise OSError("read failed")
        return real_scan(self, start, end, newest_first=newest_first)

    monkeypatch.setattr(TranscriptSource, "scan", flaky_scan)
    await service.refresh()
    failing = False
    await service.refresh()

    _, events = service.hub.replay(service.hub.epoch, 0)
    assert any(
        e.type == "action.updated"
        and e.data.get("actionId") == "act_1"
        and e.data.get("messageId") == "conv/echo"
        for e in events
    )


async def test_full_republish_does_not_lag_a_draining_subscriber(connected):
    service, path, _ = connected
    # 256 upserts plus the trailing session.updated is one event over a queue.
    path.write_text(
        "".join(json.dumps(_user(f"u{i}", f"text {i}")) + "\n" for i in range(256)),
        encoding="utf-8",
    )
    await service.list_sessions()
    subscription = service.hub.subscribe(None)
    received: list = []

    async def drain() -> None:
        while True:
            received.append(await subscription.get())

    task = asyncio.create_task(drain())
    await service.refresh()
    await asyncio.sleep(0)
    task.cancel()
    assert subscription.lagged is False
    assert len(_upserts(received)) == 256


def _lean(backend, store, tmp_path):
    path = tmp_path / "transcript.jsonl"
    path.write_text(json.dumps(_user("u0", "first")) + "\n", encoding="utf-8")
    backend.target = replace(backend.target, transcript=path)
    return LocalSessionService(backend, store), path


@pytest.mark.parametrize(
    ("error", "kept"), [(OSError("io"), True), (ValueError("bad record"), False)]
)
async def test_failed_appended_scan_is_retried_only_after_io_errors(
    backend, store, tmp_path, monkeypatch, error, kept
):
    service, path = _lean(backend, store, tmp_path)
    await service.list_sessions()
    await service.refresh()
    _append(path, _user("u1", "second"))

    async def failing(*args, **kwargs):
        raise error

    monkeypatch.setattr(service, "_correlate_windows", failing)
    await service.refresh()
    source = next(iter(service._sources.values()))
    assert (source.appended is not None) is kept


async def test_publishing_leaves_no_appended_range_pending(connected):
    service, path, _ = connected
    await service.list_sessions()
    service.hub.subscribe(None)
    await service.refresh()
    _append(path, _user("later", "more"))
    await service.refresh()
    assert next(iter(service._sources.values())).appended is None
