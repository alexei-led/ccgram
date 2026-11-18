"""Windowed scans read only the windows they yield, and a failed scan is retried."""

import json
from pathlib import Path

import pytest

from ccgram.session_protocol import transcript_source
from ccgram.session_protocol.transcript_source import TranscriptSource

WINDOW = 4096


def _row(i: int) -> str:
    row = {
        "type": "user",
        "uuid": f"u{i}",
        "timestamp": "2026-11-01T10:00:00Z",
        "message": {"role": "user", "content": f"text {i} " + "x" * 150},
    }
    return json.dumps(row) + "\n"


@pytest.fixture
async def source(store, tmp_path, monkeypatch):
    monkeypatch.setattr(transcript_source, "WINDOW_BYTES", WINDOW)
    path = tmp_path / "transcript.jsonl"
    path.write_text("".join(_row(i) for i in range(400)), encoding="utf-8")
    instance = TranscriptSource("k", path, "conv", "claude", None)
    instance.set_indexed(False)
    await instance.sync(store)
    return instance


def _ids(windows) -> list[list[str]]:
    return [list(builder.ids) for builder in windows]


async def test_newest_first_parses_only_the_windows_taken(source, monkeypatch):
    assert source.consumed > 10 * WINDOW
    parsed: list[int] = []
    real_split = transcript_source.split_jsonl

    def counting(raw, offset):
        parsed.append(len(raw))
        return real_split(raw, offset)

    monkeypatch.setattr(transcript_source, "split_jsonl", counting)
    windows = source.scan(0, source.consumed, newest_first=True)
    newest = next(windows)
    windows.close()
    assert len(parsed) == 1
    assert newest.ids[-1] == "conv/u399"


async def test_newest_first_reads_the_file_from_the_end(source, monkeypatch):
    monkeypatch.setattr(transcript_source, "BACKSCAN_BYTES", 1024)
    reads: list[int] = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self != source.path:
            return handle
        real_seek = handle.seek

        def seek(offset, whence=0):
            reads.append(offset)
            return real_seek(offset, whence)

        handle.seek = seek
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)
    windows = source.scan(0, source.consumed, newest_first=True)
    next(windows)
    windows.close()
    assert min(reads) > source.consumed - 3 * WINDOW


async def test_both_directions_cover_every_record_once_in_order(source):
    forward = _ids(source.scan(0, source.consumed))
    newest = _ids(source.scan(0, source.consumed, newest_first=True))
    flat = [mid for window in reversed(newest) for mid in window]
    assert flat == [mid for window in forward for mid in window]
    assert len(flat) == 400
    assert len(newest) > 10


async def test_a_record_longer_than_a_window_is_not_split(store, tmp_path, monkeypatch):
    monkeypatch.setattr(transcript_source, "WINDOW_BYTES", 512)
    path = tmp_path / "long.jsonl"
    big = {
        "type": "user",
        "uuid": "big",
        "timestamp": "2026-11-01T10:00:00Z",
        "message": {"role": "user", "content": "y" * 5000},
    }
    path.write_text(_row(0) + json.dumps(big) + "\n" + _row(1), encoding="utf-8")
    instance = TranscriptSource("k", path, "conv", "claude", None)
    instance.set_indexed(False)
    await instance.sync(store)
    newest = _ids(instance.scan(0, instance.consumed, newest_first=True))
    assert [mid for window in reversed(newest) for mid in window] == [
        "conv/u0",
        "conv/big",
        "conv/u1",
    ]


async def test_appended_range_survives_until_cleared(source, store):
    assert source.appended == (0, source.consumed)
    before = source.consumed
    with source.path.open("a", encoding="utf-8") as stream:
        stream.write(_row(400))
    await source.sync(store)
    assert source.appended == (0, source.consumed)
    assert source.consumed > before
    source.clear_appended()
    assert source.appended is None


async def test_replacement_discards_a_pending_appended_range(source, store):
    replacement = source.path.with_name("replacement.jsonl")
    replacement.write_text(_row(1000), encoding="utf-8")
    replacement.replace(source.path)
    await source.sync(store)
    assert source.appended == (0, source.consumed)


async def test_pending_range_is_kept_when_the_index_is_resident(store, tmp_path):
    path = tmp_path / "transcript.jsonl"
    path.write_text(_row(0), encoding="utf-8")
    indexed = TranscriptSource("k", path, "conv", "claude", None)
    await indexed.sync(store)
    assert indexed.appended is None
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_row(1))
    await indexed.sync(store)
    first, _ = indexed.appended or (None, None)
    assert first is not None and first > 0
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_row(2))
    await indexed.sync(store)
    assert indexed.appended == (first, indexed.consumed)
