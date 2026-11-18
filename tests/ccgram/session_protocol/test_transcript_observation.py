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
    MAX_RETRY_SECONDS,
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


class _Flaky:
    """Make reads of one path fail while `failing` is set; count attempts."""

    def __init__(self, monkeypatch, path):
        self.path = path
        self.failing = True
        self.attempts = 0
        real_open = Path.open

        def flaky_open(source, mode="r", *args, **kwargs):
            if source == path and "r" in mode:
                self.attempts += 1
                if self.failing:
                    raise OSError(24, "Too many open files")
            return real_open(source, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", flaky_open)


async def test_failed_read_backs_off_then_recovers_on_an_idle_transcript(
    store, tmp_path, monkeypatch
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record("a", "one")])
    now = [100.0]
    source = TranscriptSource(
        "conv", path, "conv", "claude", None, clock=lambda: now[0]
    )
    flaky = _Flaky(monkeypatch, path)
    with pytest.raises(OSError):
        await source.sync(store)
    assert source.failing
    with pytest.raises(OSError):
        await source.sync(store)
    assert flaky.attempts == 1
    flaky.failing = False
    now[0] += 1.0
    await source.sync(store)
    assert flaky.attempts == 2
    assert not source.failing
    assert _builder(source).ids == ["conv/a"]


async def test_backoff_doubles_and_caps(store, tmp_path, monkeypatch):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record("a", "one")])
    now = [0.0]
    source = TranscriptSource(
        "conv", path, "conv", "claude", None, clock=lambda: now[0]
    )
    flaky = _Flaky(monkeypatch, path)
    waits = []
    for _ in range(8):
        before = flaky.attempts
        waited = 0.0
        while flaky.attempts == before:
            with pytest.raises(OSError):
                await source.sync(store)
            now[0] += 0.5
            waited += 0.5
        waits.append(waited)
    assert waits[1:] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


async def test_changed_stamp_retries_a_failed_read_at_once(
    store, tmp_path, monkeypatch
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record("a", "one")])
    source = _source(store, path)
    flaky = _Flaky(monkeypatch, path)
    with pytest.raises(OSError):
        await source.sync(store)
    _write(path, [_record("b", "two")], mode="a")
    flaky.failing = False
    await source.sync(store)
    assert flaky.attempts == 2
    assert _builder(source).ids == ["conv/a", "conv/b"]


async def test_failure_mid_append_rebuilds_under_the_same_generation(
    store, tmp_path, monkeypatch
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record("a", "one")])
    source = _source(store, path)
    await source.sync(store)
    _write(path, [_record("b", "two")], mode="a")
    real_feed = MessageBuilder.feed
    calls = {"n": 0}

    def failing_feed(self, entries):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(5, "Input/output error")
        return real_feed(self, entries)

    monkeypatch.setattr(MessageBuilder, "feed", failing_feed)
    with pytest.raises(OSError):
        await source.sync(store)
    _write(path, [_record("c", "three")], mode="a")
    await source.sync(store)
    assert source.generation == 0
    assert _builder(source).ids == ["conv/a", "conv/b", "conv/c"]


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
    service.hub.subscribe()  # resident publish: subscribers receive message events
    await service.refresh()  # baseline publish, outside the case under test
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


async def test_new_conversation_replaces_the_previous_source(
    service, backend, tmp_path
):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(first, [_record("a", "one")])
    _write(second, [_record("b", "two")])
    backend.target = replace(backend.target, transcript=first)
    await service.list_sessions()
    backend.target = replace(
        backend.target, transcript=second, conversation_id="conv-2"
    )
    await service.refresh()
    assert [source.key for source in service._sources.values()] == ["@1:conv-2"]


async def test_gemini_chat_appends_read_only_new_lines(store, tmp_path, read_meter):
    path = tmp_path / "session-chat.jsonl"
    first = {"id": "m1", "type": "user", "content": "x" * 20_000}
    _write(path, [json.dumps(first) + "\n"])
    read_meter["path"] = path
    source = TranscriptSource("conv", path, "conv", "gemini", None)
    await source.sync(store)
    appended = {"id": "m2", "type": "gemini", "content": "more"}
    _write(path, [json.dumps(appended) + "\n"], mode="a")
    read_meter["opens"] = read_meter["bytes"] = 0
    await source.sync(store)
    assert _builder(source).ids == ["conv/m1", "conv/m2"]
    assert source.generation == 0
    assert read_meter["bytes"] <= len(json.dumps(appended)) + 1 + 2 * MARKER_BYTES


def test_backoff_stays_capped_through_a_long_outage(tmp_path, monkeypatch):
    path = tmp_path / "claude.jsonl"
    path.write_text("")
    now = [0.0]
    source = TranscriptSource(
        "conv", path, "conv", "claude", None, clock=lambda: now[0]
    )

    def unreadable(stat):
        raise OSError("unreadable")

    monkeypatch.setattr(source, "_observe", unreadable)
    for _ in range(1100):
        now[0] = source._retry_at
        with pytest.raises(OSError, match="unreadable"):
            source._sync()
    assert source._failures == 1100
    assert source._retry_at - now[0] == MAX_RETRY_SECONDS


async def test_refresh_keeps_sessions_of_a_target_that_failed_unexpectedly(
    service, backend, tmp_path, monkeypatch
):
    path = tmp_path / "transcript.jsonl"
    _write(path, [_record("u1", "hello")])
    backend.target = replace(backend.target, transcript=path)
    session = (await service.list_sessions())[0]

    async def broken(target):
        raise RuntimeError("bug in one target")

    monkeypatch.setattr(service, "_update", broken)
    await service.refresh()
    assert session.id in service._snapshots
