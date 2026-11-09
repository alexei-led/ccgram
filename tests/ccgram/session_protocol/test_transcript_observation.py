"""Observation cost: unchanged sources are one stat, appends read only new bytes."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import time

import pytest

from ccgram.session_protocol.messages import MessageBuilder
from ccgram.session_protocol.transcript_source import (
    MARKER_BYTES,
    TranscriptSource,
)


def _record(uuid, text, timestamp=None):
    row = {"type": "user", "uuid": uuid, "message": {"role": "user", "content": text}}
    if timestamp is not None:
        row["timestamp"] = timestamp
    return json.dumps(row) + "\n"


def _write(path: Path, lines: list[str], mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8") as stream:
        stream.writelines(lines)


@pytest.fixture
def read_meter(monkeypatch):
    """Count opens and bytes read from one transcript through Path.open."""
    meter = {"path": None, "opens": 0, "bytes": 0}
    real_open = Path.open

    class Metered:
        def __init__(self, stream):
            self._stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._stream.__exit__(*exc)

        def read(self, size=-1):
            data = self._stream.read(size)
            meter["bytes"] += len(data)
            return data

        def __getattr__(self, name):
            return getattr(self._stream, name)

    def metered_open(self, mode="r", *args, **kwargs):
        stream = real_open(self, mode, *args, **kwargs)
        if self == meter["path"] and "r" in mode and "+" not in mode:
            meter["opens"] += 1
            return Metered(stream)
        return stream

    monkeypatch.setattr(Path, "open", metered_open)
    return meter


@pytest.fixture
def materialized(monkeypatch):
    """Message IDs materialized by the builder, in call order."""
    calls: list[str] = []
    real_message = MessageBuilder.message

    def counting(self, message_id):
        calls.append(message_id)
        return real_message(self, message_id)

    monkeypatch.setattr(MessageBuilder, "message", counting)
    return calls


def _builder(source: TranscriptSource) -> MessageBuilder:
    assert source.builder is not None
    return source.builder


def _source(store, path, key="conv"):
    return TranscriptSource(key, path, "conv", "claude", store.transcript_cursor(key))


async def test_unchanged_transcript_costs_one_stat_and_no_reads(
    store, tmp_path, read_meter
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record(f"u{i}", "text") for i in range(200)])
    source = _source(store, path)
    await source.sync(store)
    read_meter["path"] = path
    read_meter["opens"] = read_meter["bytes"] = 0
    for _ in range(10):
        await source.sync(store)
    assert read_meter["opens"] == 0
    assert read_meter["bytes"] == 0


async def test_append_reads_only_the_new_bytes_and_marker(
    store, tmp_path, read_meter, materialized
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record(f"u{i}", "x" * 200) for i in range(200)])
    source = _source(store, path)
    await source.sync(store)
    appended = _record("new", "y" * 1000)
    _write(path, [appended], mode="a")
    read_meter["path"] = path
    read_meter["opens"] = read_meter["bytes"] = 0
    materialized.clear()
    await source.sync(store)
    assert read_meter["bytes"] <= len(appended) + 2 * MARKER_BYTES
    assert read_meter["bytes"] >= len(appended)
    assert _builder(source).position("conv/new") == 200
    assert len(_builder(source).ids) == 201


async def test_partial_trailing_record_waits_for_its_newline(store, tmp_path):
    path = tmp_path / "transcript.jsonl"
    complete = _record("first", "one")
    partial = _record("second", "two")
    _write(path, [complete, partial[:20]])
    source = _source(store, path)
    await source.sync(store)
    assert _builder(source).ids == ["conv/first"]
    _write(path, [partial[20:]], mode="a")
    await source.sync(store)
    assert _builder(source).ids == ["conv/first", "conv/second"]
    assert source.generation == 0


async def test_replaced_transcript_rebuilds_under_a_new_generation(store, tmp_path):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record("old", "before")])
    source = _source(store, path)
    await source.sync(store)
    replacement = tmp_path / "replacement.jsonl"
    _write(replacement, [_record("new", "after")])
    replacement.replace(path)
    await source.sync(store)
    assert source.generation == 1
    assert _builder(source).ids == ["conv/g1/new"]


async def test_truncated_transcript_rebuilds_under_a_new_generation(store, tmp_path):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record("a", "one"), _record("b", "two")])
    source = _source(store, path)
    await source.sync(store)
    _write(path, [_record("c", "three")])
    await source.sync(store)
    assert source.generation == 1
    assert _builder(source).ids == ["conv/g1/c"]


async def test_unreadable_source_is_not_retried_until_it_changes(
    store, tmp_path, monkeypatch
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record("a", "one")])
    source = _source(store, path)
    attempts = 0
    real_open = Path.open

    def failing_open(self, mode="r", *args, **kwargs):
        nonlocal attempts
        if self == path and "r" in mode:
            attempts += 1
            raise PermissionError("denied")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError):
        await source.sync(store)
    with pytest.raises(OSError):
        await source.sync(store)
    assert attempts == 1
    _write(path, [_record("b", "two")], mode="a")
    with pytest.raises(OSError):
        await source.sync(store)
    assert attempts == 2


async def test_idle_refresh_publishes_nothing_and_reads_nothing(
    service, backend, tmp_path, read_meter
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record(f"u{i}", "text") for i in range(100)])
    backend.target = replace(backend.target, transcript=path)
    await service.list_sessions()
    published = []
    real_publish = service.hub.publish
    service.hub.publish = lambda *args, **kwargs: (
        published.append(args) or real_publish(*args, **kwargs)
    )
    read_meter["path"] = path
    read_meter["opens"] = read_meter["bytes"] = 0
    await service.refresh()
    assert published == []
    assert read_meter["opens"] == 0


async def test_history_page_materializes_only_its_messages(
    service, backend, tmp_path, materialized
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record(f"u{i}", f"text {i}") for i in range(500)])
    backend.target = replace(backend.target, transcript=path)
    session = (await service.list_sessions())[0]
    materialized.clear()
    page = await service.history(session.id, None, 5)
    assert len(page.messages) == 5
    assert len(materialized) == 5


async def test_history_cursor_walks_every_message_once(service, backend, tmp_path):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record(f"u{i}", f"text {i}") for i in range(12)])
    backend.target = replace(backend.target, transcript=path)
    session = (await service.list_sessions())[0]
    seen: list[str] = []
    cursor = None
    while True:
        page = await service.history(session.id, cursor, 5)
        seen = [m.id for m in page.messages] + seen
        cursor = page.prev_cursor
        if cursor is None:
            break
    assert seen == [f"conv/u{i}" for i in range(12)]


async def test_delivery_publishes_and_materializes_only_the_matching_message(
    service, backend, tmp_path, materialized
):
    path = tmp_path / "transcript.jsonl"
    old = datetime(2026, 11, 1, 10, tzinfo=UTC).isoformat()
    _write(path, [_record(f"u{i}", f"text {i}", old) for i in range(300)])
    backend.target = replace(backend.target, transcript=path)
    session = (await service.list_sessions())[0]
    sent_at = time()
    await service.remember_delivery("act_1", session.id, "conv", "hello there", sent_at)
    sent = datetime.fromtimestamp(sent_at + 1, UTC).isoformat()
    _write(path, [_record("hello", "hello   there", sent)], mode="a")
    materialized.clear()
    upserts = []
    real_publish = service.hub.publish

    def record(kind, session_id, data):
        if kind == "message.upsert":
            upserts.append(data)
        return real_publish(kind, session_id, data)

    service.hub.publish = record
    await service.refresh()
    assert materialized == ["conv/hello"]
    assert [u.id for u in upserts] == ["conv/hello"]
    assert upserts[0].action_id == "act_1"
