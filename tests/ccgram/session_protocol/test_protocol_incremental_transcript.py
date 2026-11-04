"""Per-poll transcript cost is O(appended bytes), never O(file size)."""

import hashlib
import json
import asyncio
import os
import threading
import tracemalloc
from dataclasses import replace
from pathlib import Path

import pytest

from ccgram.session_protocol.message_builder import MessageBuilder
from ccgram.session_protocol.messages import build_messages, read_entries
from ccgram.session_protocol.store import ProtocolStore

WINDOW = 4096
AUDIT = 1 << 20


@pytest.fixture
def reads(monkeypatch):
    """Counts file opens and bytes read through ``Path.open``."""
    counter = {"opens": 0, "bytes": 0}
    real_open = Path.open

    class Counting:
        def __init__(self, handle):
            self._handle = handle

        def read(self, size=-1):
            data = self._handle.read(size)
            counter["bytes"] += len(data)
            return data

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    def counting_open(self, mode="r", *args, **kwargs):
        if "b" in mode and "r" in mode and "+" not in mode:
            counter["opens"] += 1
            return Counting(real_open(self, mode, *args, **kwargs))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    return counter


def reset(counter):
    counter["opens"] = counter["bytes"] = 0


async def test_unchanged_stamp_reads_nothing(tmp_path, reads):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"x" * 100_000)
    store = ProtocolStore(tmp_path / "db")
    first = await store.transcript_snapshot("k", path)
    assert reads["bytes"] == 100_000
    reset(reads)
    second = await store.transcript_snapshot("k", path)
    assert reads == {"opens": 0, "bytes": 0}
    assert second.stamp == first.stamp and second.generation == 0
    store.close()


async def test_append_reads_only_appended_bytes(tmp_path, reads):
    path = tmp_path / "t.jsonl"
    path.write_bytes(os.urandom(2_000_000))
    store = ProtocolStore(tmp_path / "db")
    await store.transcript_snapshot("k", path)
    reset(reads)
    with path.open("ab") as stream:
        stream.write(b"y" * 1000)
    snapshot = await store.transcript_snapshot("k", path)
    assert snapshot.generation == 0
    assert snapshot.stamp[1] == 2_001_000
    assert reads["bytes"] <= 1000 + 3 * WINDOW + AUDIT
    store.close()


async def test_hash_chain_survives_many_appends_and_restart(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"a" * 10_000)
    db = tmp_path / "db"
    store = ProtocolStore(db)
    await store.transcript_snapshot("k", path)
    for _ in range(3):
        with path.open("ab") as stream:
            stream.write(b"b" * 777)
        assert (await store.transcript_snapshot("k", path)).generation == 0
    store.close()
    store = ProtocolStore(db)
    assert (await store.transcript_snapshot("k", path)).generation == 0
    path.write_bytes(b"z" * 20_000)
    assert (await store.transcript_snapshot("k", path)).generation == 1
    store.close()


@pytest.mark.parametrize("edge", ["tail", "head"])
async def test_in_place_edit_of_known_bytes_bumps_generation(tmp_path, edge):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"a" * 20_000)
    store = ProtocolStore(tmp_path / "db")
    await store.transcript_snapshot("k", path)
    data = bytearray(path.read_bytes())
    data[-10 if edge == "tail" else 10] = ord("Q")
    path.write_bytes(bytes(data) + b"appended")
    assert (await store.transcript_snapshot("k", path)).generation == 1
    store.close()


async def test_shrink_and_new_inode_bump_generation(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"a" * 20_000)
    store = ProtocolStore(tmp_path / "db")
    await store.transcript_snapshot("k", path)
    path.write_bytes(b"a" * 10_000)
    assert (await store.transcript_snapshot("k", path)).generation == 1
    replacement = tmp_path / "new"
    replacement.write_bytes(b"a" * 30_000)
    os.replace(replacement, path)
    assert (await store.transcript_snapshot("k", path)).generation == 2
    store.close()


def claude_line(kind, uuid, content, stamp="2026-10-20T10:00:00Z"):
    return {
        "type": kind,
        "uuid": uuid,
        "timestamp": stamp,
        "message": {"role": kind, "content": content},
    }


def claude_session():
    """Tool call, text, repeated uuid (replacement), and a late tool result."""
    tool = {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}
    result = {"type": "tool_result", "tool_use_id": "t1", "content": "out"}
    again = {"type": "tool_result", "tool_use_id": "t1", "content": "out2"}
    return [
        claude_line("user", "u1", "hello"),
        claude_line("assistant", "a1", [tool]),
        claude_line("assistant", "a2", [{"type": "text", "text": "waiting"}]),
        claude_line("user", "r1", [result]),
        claude_line("assistant", "a2", [{"type": "text", "text": "waiting more"}]),
        claude_line("user", "r2", [again]),
        claude_line("user", "u2", "next"),
        claude_line("assistant", "a3", [dict(tool, id="t2")]),
    ]


def codex_session():
    def item(payload):
        return {
            "type": "response_item",
            "timestamp": "2026-10-20T10:00:00Z",
            "payload": payload,
        }

    return [
        item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "go"}],
            }
        ),
        item(
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "shell",
                "arguments": '{"cmd": "ls"}',
            }
        ),
        item({"type": "function_call_output", "call_id": "c1", "output": "ok"}),
        item(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ),
    ]


@pytest.mark.parametrize(
    ("provider", "lines"), [("claude", claude_session()), ("codex", codex_session())]
)
@pytest.mark.parametrize("cut", range(0, 9))
def test_incremental_builder_equals_full_reparse(tmp_path, provider, lines, cut):
    raw = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    full = build_messages(read_entries(raw, provider), "conv", provider, 2)
    assert len(full) >= 3 and any(p.output for m in full for p in m.parts)
    path = tmp_path / "t.jsonl"
    builder = MessageBuilder("conv", provider, 2)
    split = [0, *sorted({cut * len(raw) // 8, (cut * 3) % (len(raw) + 1)}), len(raw)]
    for end in split[1:]:
        path.write_bytes(raw[:end])
        builder.advance(path, end, path.stat().st_ino)
    assert builder.messages() == full
    for window in [(0, None), (200, 600), (0, 1)]:
        expected = build_messages(
            read_entries(raw, provider), "conv", provider, 2, source_range=window
        )
        assert builder.messages(window) == expected


def test_incremental_builder_waits_for_unfinished_line(tmp_path):
    raw = b"".join(json.dumps(line).encode() + b"\n" for line in claude_session())
    path = tmp_path / "t.jsonl"
    builder = MessageBuilder("conv", "claude", 0)
    path.write_bytes(raw[:-10])
    builder.advance(path, len(raw) - 10, path.stat().st_ino)
    partial = builder.messages()
    path.write_bytes(raw)
    builder.advance(path, len(raw), path.stat().st_ino)
    assert len(partial) < len(builder.messages())
    assert builder.messages() == build_messages(
        read_entries(raw, "claude"), "conv", "claude"
    )


def test_incremental_builder_reports_only_touched_messages(tmp_path):
    lines = claude_session()
    path = tmp_path / "t.jsonl"
    builder = MessageBuilder("conv", "claude", 0)
    head = b"".join(json.dumps(line).encode() + b"\n" for line in lines[:3])
    path.write_bytes(head)
    builder.advance(path, len(head), path.stat().st_ino)
    builder.dirty.clear()
    late = json.dumps(lines[3]).encode() + b"\n"
    path.write_bytes(head + late)
    builder.advance(path, len(head + late), path.stat().st_ino)
    assert builder.dirty == {"conv/a1"}


def test_gemini_document_is_rebuilt_from_the_whole_file(tmp_path):
    path = tmp_path / "g.json"
    builder = MessageBuilder("conv", "gemini", 0)
    for count in (1, 2):
        doc = {
            "messages": [
                {"id": f"m{i}", "type": "gemini", "content": f"t{i}"}
                for i in range(count)
            ]
        }
        raw = json.dumps(doc).encode()
        path.write_bytes(raw)
        builder.advance(path, len(raw), path.stat().st_ino)
        assert builder.messages() == build_messages(
            read_entries(raw, "gemini"), "conv", "gemini"
        )


def upserts(service):
    _, events = service.hub.replay(service.hub.epoch, 0)
    return [e.data["id"] for e in events if e.type == "message.upsert"]


async def test_service_poll_cost_follows_appended_bytes(
    service, backend, tmp_path, reads, monkeypatch
):
    monkeypatch.setattr("ccgram.session_protocol.store._AUDIT", 2 * WINDOW)
    path = tmp_path / "t.jsonl"
    lines = [claude_line("assistant", f"a{i}", "x" * 1000) for i in range(300)]
    path.write_bytes(b"".join(json.dumps(line).encode() + b"\n" for line in lines))
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    assert len(upserts(service)) == 300
    reset(reads)
    await service.refresh()
    assert reads == {"opens": 0, "bytes": 0}
    extra = json.dumps(claude_line("assistant", "new", "tail")).encode() + b"\n"
    with path.open("ab") as stream:
        stream.write(extra)
    reset(reads)
    await service.refresh()
    assert reads["bytes"] <= 2 * len(extra) + 3 * WINDOW + 2 * WINDOW
    assert upserts(service)[300:] == ["conv/new"]
    assert not hasattr(backend.target, "transcript_content")


def test_builder_chunking_handles_lines_longer_than_a_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr("ccgram.session_protocol.message_builder._CHUNK", 64)
    lines = [claude_line("assistant", f"a{i}", "y" * (50 * i)) for i in range(6)]
    raw = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    path = tmp_path / "t.jsonl"
    path.write_bytes(raw)
    builder = MessageBuilder("conv", "claude", 0)
    builder.advance(path, len(raw), path.stat().st_ino)
    assert builder.offset == len(raw)
    assert builder.messages() == build_messages(
        read_entries(raw, "claude"), "conv", "claude"
    )


async def test_rotation_between_observation_and_read_never_mixes_generations(
    service, backend, tmp_path, monkeypatch
):
    path = tmp_path / "t.jsonl"
    rows = [claude_line("assistant", f"o{i}", f"old{i}") for i in range(3)]
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    new = [claude_line("assistant", f"n{i}", f"new{i}") for i in range(6)]
    with path.open("ab") as stream:
        stream.write(b'{"type": "user", "uuid": "pad"}\n')
    real = MessageBuilder.advance

    def rotate_then_advance(builder, target_path, size, ino, *extra):
        replacement = tmp_path / "rotated"
        replacement.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in new))
        replacement.replace(target_path)
        monkeypatch.setattr(MessageBuilder, "advance", real)
        return real(builder, target_path, size, ino, *extra)

    monkeypatch.setattr(MessageBuilder, "advance", rotate_then_advance)
    await service.refresh()
    await service.refresh()
    _, events = service.hub.replay(service.hub.epoch, 0)
    stale = [
        e.data
        for e in events
        if e.type == "message.upsert" and "/g1/" not in e.data["id"]
    ]
    assert {m["parts"][0]["text"] for m in stale} <= {"old0", "old1", "old2"}


async def test_rebinding_republishes_the_history(service, backend, tmp_path):
    path = tmp_path / "t.jsonl"
    rows = [claude_line("assistant", f"a{i}", f"t{i}") for i in range(3)]
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    assert len(upserts(service)) == 3
    backend.target = replace(backend.target, process_id=999)
    await service.refresh()
    assert len(upserts(service)) == 6


async def test_mid_file_edit_is_found_by_the_audit_and_survives_restart(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("ccgram.session_protocol.store._AUDIT", 2 * WINDOW)
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"a" * 40_000)
    db = tmp_path / "db"
    store = ProtocolStore(db)
    await store.transcript_snapshot("k", path)
    data = bytearray(path.read_bytes())
    data[30_000] = ord("Q")
    path.write_bytes(bytes(data))
    live = []
    for _ in range(6):
        with path.open("ab") as stream:
            stream.write(b"b")
        live.append((await store.transcript_snapshot("k", path)).generation)
    assert live[0] == 0 and live[-1] == 1
    store.close()
    store = ProtocolStore(db)
    assert (await store.transcript_snapshot("k", path)).generation == live[-1]
    store.close()


def test_reassigned_call_id_matches_a_full_reparse(tmp_path):
    tool = {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}
    result = {"type": "tool_result", "tool_use_id": "t1", "content": "out"}
    lines = [
        claude_line("assistant", "a1", [tool]),
        claude_line("user", "r1", [result]),
        claude_line("assistant", "a2", [tool]),
    ]
    raw = [json.dumps(line).encode() + b"\n" for line in lines]
    path = tmp_path / "t.jsonl"
    builder = MessageBuilder("conv", "claude", 0)
    for end in (2, 3):
        path.write_bytes(b"".join(raw[:end]))
        builder.advance(path, len(b"".join(raw[:end])), path.stat().st_ino)
    full = build_messages(read_entries(b"".join(raw), "claude"), "conv", "claude")
    assert builder.messages() == full


def test_builder_keeps_large_fields_out_of_memory(tmp_path):
    big = "z" * 6000
    tool = {"type": "tool_use", "id": "t0", "name": "Bash", "input": {"command": "ls"}}
    rows = []
    for i in range(1000):
        call = dict(tool, id=f"t{i}")
        result = {
            "type": "tool_result",
            "tool_use_id": f"t{i}",
            "content": big + str(i),
        }
        rows += [
            claude_line("assistant", f"a{i}", [call]),
            claude_line("user", f"r{i}", [result]),
        ]
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in rows)
    path = tmp_path / "t.jsonl"
    path.write_bytes(raw)
    tracemalloc.start()
    builder = MessageBuilder("conv", "claude", 0)
    builder.advance(path, len(raw), path.stat().st_ino)
    retained, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert retained < 0.15 * len(raw)
    assert builder.messages() == build_messages(
        read_entries(raw, "claude"), "conv", "claude"
    )


def latest_revisions(service):
    _, events = service.hub.replay(service.hub.epoch, 0)
    return {
        e.data["id"]: e.data["revision"] for e in events if e.type == "message.upsert"
    }


@pytest.mark.parametrize("failure", ["enospc", "short"])
async def test_spill_failure_mid_advance_never_corrupts_the_builder(
    service, backend, tmp_path, monkeypatch, failure
):
    lines = claude_session()
    head = b"".join(json.dumps(line).encode() + b"\n" for line in lines[:4])
    raw = head + b"".join(json.dumps(line).encode() + b"\n" for line in lines[4:])
    path = tmp_path / "t.jsonl"
    path.write_bytes(head)
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    with path.open("ab") as stream:
        stream.write(raw[len(head) :])
    real, calls = os.pwrite, {"n": 0}

    def flaky(fd, data, offset):
        calls["n"] += 1
        if calls["n"] != 3:
            return real(fd, data, offset)
        if failure == "enospc":
            raise OSError(28, "No space left on device")
        return real(fd, data[: len(data) // 2], offset)

    monkeypatch.setattr("ccgram.session_protocol.message_builder.os.pwrite", flaky)
    await service.refresh()
    await service.refresh()
    await service.refresh()
    full = build_messages(read_entries(raw, "claude"), "conv", "claude", 0)
    assert latest_revisions(service) == {m.id: m.revision for m in full}


async def test_spill_write_failure_is_a_503_not_an_escape(
    service, backend, tmp_path, monkeypatch
):
    lines = claude_session()
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"".join(json.dumps(line).encode() + b"\n" for line in lines))
    backend.target = replace(backend.target, transcript=path)
    real = os.pwrite
    monkeypatch.setattr(
        "ccgram.session_protocol.message_builder.os.pwrite",
        lambda fd, data, offset: real(fd, data[: len(data) // 2], offset),
    )
    await service.refresh()
    monkeypatch.undo()
    await service.refresh()
    assert latest_revisions(service)


async def test_edit_past_the_last_full_scan_is_found_after_restart(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"a" * 4096)
    db = tmp_path / "db"
    store = ProtocolStore(db)
    await store.transcript_snapshot("k", path)
    for _ in range(4):
        with path.open("ab") as stream:
            stream.write(b"b" * 20_000)
        await store.transcript_snapshot("k", path)
    store.close()
    data = bytearray(path.read_bytes())
    data[50_000] = ord("Q")
    path.write_bytes(bytes(data))
    store = ProtocolStore(db)
    assert (await store.transcript_snapshot("k", path)).generation == 1
    store.close()


async def test_vanishing_session_never_closes_a_builder_mid_update(
    service, backend, tmp_path, monkeypatch
):
    lines = claude_session()
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"".join(json.dumps(line).encode() + b"\n" for line in lines[:4]))
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    session_id = next(iter(service._snapshots))
    with path.open("ab") as stream:
        stream.write(b"".join(json.dumps(line).encode() + b"\n" for line in lines[4:]))
    entered, release, real = (
        threading.Event(),
        threading.Event(),
        MessageBuilder.advance,
    )

    def blocked(builder, target_path, size, ino, *extra):
        entered.set()
        release.wait(5)
        return real(builder, target_path, size, ino, *extra)

    monkeypatch.setattr(MessageBuilder, "advance", blocked)
    history = asyncio.create_task(service.history(session_id, None, 50))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    backend.disappear = True
    refresh = asyncio.create_task(service.refresh())
    await asyncio.sleep(0.1)
    release.set()
    outcome = await asyncio.gather(history, refresh, return_exceptions=True)
    assert not any(isinstance(item, ValueError) for item in outcome)


async def test_same_size_edit_outside_the_audit_window_is_not_cached_as_clean(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("ccgram.session_protocol.store._AUDIT", 2 * WINDOW)
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"a" * 40_000)
    store = ProtocolStore(tmp_path / "db")
    await store.transcript_snapshot("k", path)
    data = bytearray(path.read_bytes())
    data[30_000] = ord("Q")
    path.write_bytes(bytes(data))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
    generations = [
        (await store.transcript_snapshot("k", path)).generation for _ in range(3)
    ]
    assert generations == [1, 1, 1]
    store.close()


async def test_touch_without_content_change_keeps_the_generation(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"a" * 20_000)
    store = ProtocolStore(tmp_path / "db")
    await store.transcript_snapshot("k", path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
    assert (await store.transcript_snapshot("k", path)).generation == 0
    store.close()


async def test_same_size_rewrite_between_observation_and_read_is_not_published(
    service, backend, tmp_path, monkeypatch
):
    path = tmp_path / "t.jsonl"
    rows = [claude_line("assistant", f"o{i}", f"old{i}") for i in range(3)]
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    backend.target = replace(backend.target, transcript=path)
    real = MessageBuilder.advance

    def rewrite_then_advance(builder, target_path, *args, **kwargs):
        raw = target_path.read_bytes().replace(b"old", b"new")
        target_path.write_bytes(raw)
        stat = target_path.stat()
        os.utime(target_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**10))
        monkeypatch.setattr(MessageBuilder, "advance", real)
        return real(builder, target_path, *args, **kwargs)

    monkeypatch.setattr(MessageBuilder, "advance", rewrite_then_advance)
    await service.refresh()
    await service.refresh()
    _, events = service.hub.replay(service.hub.epoch, 0)
    stale = [
        e.data
        for e in events
        if e.type == "message.upsert" and "/g1/" not in e.data["id"]
    ]
    assert not {m["parts"][0]["text"] for m in stale} & {"new0", "new1", "new2"}


async def test_cancelled_caller_keeps_the_builder_lock_until_the_worker_ends(
    service, backend, tmp_path, monkeypatch
):
    lines = claude_session()
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"".join(json.dumps(line).encode() + b"\n" for line in lines[:4]))
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    session_id = next(iter(service._snapshots))
    with path.open("ab") as stream:
        stream.write(b"".join(json.dumps(line).encode() + b"\n" for line in lines[4:]))
    guard, active = threading.Lock(), {"now": 0, "max": 0}
    entered, release, real = (
        threading.Event(),
        threading.Event(),
        MessageBuilder.advance,
    )

    def tracked(builder, *args, **kwargs):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        entered.set()
        release.wait(5)
        try:
            return real(builder, *args, **kwargs)
        finally:
            with guard:
                active["now"] -= 1

    monkeypatch.setattr(MessageBuilder, "advance", tracked)
    first = asyncio.create_task(service.history(session_id, None, 50))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    first.cancel()
    await asyncio.sleep(0.05)
    second = asyncio.create_task(service.history(session_id, None, 50))
    await asyncio.sleep(0.1)
    release.set()
    await asyncio.gather(first, second, return_exceptions=True)
    assert active["max"] == 1


def stamp_of(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def jsonl(rows):
    return b"".join(json.dumps(r).encode() + b"\n" for r in rows)


def test_file_that_grows_after_observation_is_read_up_to_the_observed_size(tmp_path):
    path = tmp_path / "t.jsonl"
    rows = [claude_line("assistant", f"a{i}", f"t{i}") for i in range(3)]
    observed = jsonl(rows)
    path.write_bytes(observed)
    stamp, digest = stamp_of(path), hashlib.sha256(observed).hexdigest()
    path.write_bytes(observed + jsonl([claude_line("assistant", "late", "late")]))
    builder = MessageBuilder("conv", "claude", 0)
    messages = builder.update(path, stamp, digest=digest)
    assert [m.parts[0].text for m in messages] == ["t0", "t1", "t2"]
    assert builder.offset == len(observed)


def test_rewrite_plus_append_is_caught_by_the_observed_digest(tmp_path):
    path = tmp_path / "t.jsonl"
    observed = jsonl([claude_line("assistant", f"a{i}", f"old{i}") for i in range(3)])
    path.write_bytes(observed)
    stamp, digest = stamp_of(path), hashlib.sha256(observed).hexdigest()
    rewritten = observed.replace(b"old", b"new") + jsonl(
        [claude_line("assistant", "late", "late")]
    )
    path.write_bytes(rewritten)
    with pytest.raises(OSError, match="differ"):
        MessageBuilder("conv", "claude", 0).update(path, stamp, digest=digest)


async def test_cancelled_caller_keeps_a_builder_whose_worker_succeeded(
    service, backend, tmp_path, monkeypatch
):
    lines = claude_session()
    path = tmp_path / "t.jsonl"
    path.write_bytes(jsonl(lines[:4]))
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    session_id = next(iter(service._snapshots))
    (builder,) = service._builders.values()
    with path.open("ab") as stream:
        stream.write(jsonl(lines[4:]))
    entered, release, real = threading.Event(), threading.Event(), MessageBuilder.advance

    def blocked(self, *args):
        entered.set()
        release.wait(5)
        return real(self, *args)

    monkeypatch.setattr(MessageBuilder, "advance", blocked)
    first = asyncio.create_task(service.history(session_id, None, 50))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    first.cancel()
    release.set()
    await asyncio.gather(first, return_exceptions=True)
    assert list(service._builders.values()) == [builder]


async def test_cancelled_refresh_does_not_lose_the_messages_it_consumed(
    service, backend, tmp_path, monkeypatch
):
    lines = claude_session()
    path = tmp_path / "t.jsonl"
    path.write_bytes(jsonl(lines[:4]))
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    seen = set(latest_revisions(service))
    with path.open("ab") as stream:
        stream.write(jsonl(lines[4:]))
    entered, release, real = threading.Event(), threading.Event(), MessageBuilder.advance

    def blocked(self, *args):
        entered.set()
        release.wait(5)
        return real(self, *args)

    monkeypatch.setattr(MessageBuilder, "advance", blocked)
    refresh = asyncio.create_task(service.refresh())
    while not entered.is_set():
        await asyncio.sleep(0.01)
    refresh.cancel()
    release.set()
    await asyncio.gather(refresh, return_exceptions=True)
    monkeypatch.setattr(MessageBuilder, "advance", real)
    await service.refresh()
    full = build_messages(read_entries(jsonl(lines), "claude"), "conv", "claude", 0)
    assert latest_revisions(service) == {m.id: m.revision for m in full}
    assert set(latest_revisions(service)) - seen


async def test_history_verifies_its_own_observation_when_a_refresh_interleaves(
    service, backend, tmp_path, monkeypatch
):
    path = tmp_path / "t.jsonl"
    rows = [claude_line("assistant", f"o{i}", f"old{i}") for i in range(3)]
    old = jsonl(rows)
    path.write_bytes(old)
    backend.target = replace(backend.target, transcript=path)
    await service.refresh()
    session_id = next(iter(service._snapshots))
    # Drop the refresh builder so history starts a fresh one (digest check applies).
    service._drop_builder(next(iter(service._builders)))
    real = service._built

    async def refresh_first(*args, **kwargs):
        monkeypatch.setattr(service, "_built", real)
        path.write_bytes(
            old.replace(b"old", b"new") + jsonl([claude_line("user", "x", "late")])
        )
        await service.refresh()
        return await real(*args, **kwargs)

    monkeypatch.setattr(service, "_built", refresh_first)
    outcome = await asyncio.gather(
        service.history(session_id, None, 50), return_exceptions=True
    )
    texts = {
        part.text
        for page in outcome
        if not isinstance(page, BaseException)
        for message in page.messages
        for part in message.parts
    }
    assert not texts & {"new0", "new1", "new2"}
