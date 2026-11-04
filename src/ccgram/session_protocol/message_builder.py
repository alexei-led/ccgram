"""Incremental, disk-spilled message builder over an append-only transcript."""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO

from .messages import (
    _PART_FIELDS,
    SourceEntry,
    TranscriptParserState,
    TranscriptRecord,
    _text,
    parse_records,
    read_entries,
)
from .models import Message, Part

_CHUNK = 1 << 20
_TAIL = 4096
_SPILL_MIN = 512


class TranscriptChangedError(OSError):
    """The source no longer matches what was observed; raised before any byte is consumed."""


class _Spilled:
    """Where a pickled value lives in the builder's scratch file."""

    __slots__ = ("length", "offset")

    def __init__(self, offset: int, length: int) -> None:
        self.offset = offset
        self.length = length


@dataclass(slots=True)
class _Stored:
    message: Message
    offsets: list[int]
    results: dict[str, list[tuple[int, Any, bool]]] = field(default_factory=dict)
    contributors: tuple[int, ...] = ()


class MessageBuilder:
    """Incremental ``build_messages``: feed appended lines, never re-parse the file.

    For every JSONL provider the result equals ``build_messages(read_entries(whole
    file))``; Gemini keeps one JSON document, so it is rebuilt on each advance.
    Memory holds only an index: each message, and each part field over
    ``_SPILL_MIN`` bytes, is pickled into an unlinked scratch file and loaded when
    touched or handed out. Not thread-safe: callers serialise ``update``.
    """

    def __init__(
        self,
        conversation_id: str,
        provider: str,
        generation: int,
        spill_dir: Path | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.provider = provider
        self.generation = generation
        self.dirty: set[str] = set()
        self._spill = tempfile.TemporaryFile(dir=spill_dir)  # noqa: SIM115 (scratch file outlives __init__, closed by close())
        self._ino: int | None = None
        self._reset()

    def close(self) -> None:
        """Release the scratch file; the builder is unusable afterwards."""
        self._spill.close()

    def _reset(self) -> None:
        self.offset = 0
        self._tail = b""
        self._state = TranscriptParserState()
        self._spill.truncate(0)
        self._spill_end = 0
        self._index: dict[str, _Spilled | None] = {}
        self._cache: dict[str, _Stored] = {}
        self._calls: dict[str, tuple[str, int]] = {}
        self._orphans: dict[str, list[tuple[int, Any, bool]]] = {}

    def update(
        self,
        path: Path,
        stamp: tuple[int, int, int, int],
        source_range: tuple[int, int | None] | None = None,
        *,
        dirty_only: bool = False,
        digest: str | None = None,
    ) -> tuple[Message, ...]:
        """Advance to the observed ``stamp`` of the file, then select messages."""
        self.advance(path, stamp[1], stamp[0], stamp, digest)
        selected = self.messages(
            source_range, only=set(self.dirty) if dirty_only else None
        )
        if dirty_only:
            self.dirty = set()
        return selected

    def advance(
        self,
        path: Path,
        size: int,
        ino: int,
        stamp: tuple[int, int, int, int] | None = None,
        digest: str | None = None,
    ) -> None:
        """Consume transcript bytes up to ``size``; a partial last line waits.

        Raises ``TranscriptChangedError`` (nothing consumed) unless the one open file is
        still the observed source: a same-size file must carry the observed ``stamp``,
        and a growing one is read only up to ``size``. A builder starting from zero
        also checks the sha256 of those ``size`` bytes against the observed ``digest``;
        a mismatch raises OSError after consuming, so the builder must be dropped.
        """
        with path.open("rb") as source:
            status = os.fstat(source.fileno())
            if (
                status.st_ino != ino
                or status.st_size < size
                or self._ino not in (None, ino)
            ):
                raise TranscriptChangedError("Transcript source was replaced")
            if stamp and status.st_size == size and _stat_stamp(status) != stamp:
                raise TranscriptChangedError("Transcript source was rewritten in place")
            fresh = self.offset == 0 or self.provider == "gemini"
            tail = len(self._tail)
            if (
                self.provider != "gemini"
                and tail
                and _read_at(source, self.offset - tail, tail) != self._tail
            ):
                raise TranscriptChangedError("Transcript source was rewritten")
            self._ino = ino
            if self.provider == "gemini":
                self._reset()
                self.feed(read_entries(_read_at(source, 0, size), self.provider))
                self.offset = size
            else:
                self._consume(source, size)
            if fresh and digest and _digest(source, size) != digest:
                raise OSError("Transcript bytes differ from the observed content")

    def _consume(self, source: BinaryIO, size: int) -> None:
        span = _CHUNK
        while self.offset < size:
            end = min(size, self.offset + span)
            raw = _read_at(source, self.offset, end - self.offset)
            consumed = raw.rfind(b"\n") + 1
            if not consumed:
                if end == size:
                    return
                span *= 2
                continue
            self.feed(read_entries(raw[:consumed], self.provider, base=self.offset))
            self.offset += consumed
            self._tail = (self._tail + raw[:consumed])[-_TAIL:]

    def feed(self, entries: list[SourceEntry]) -> None:
        records, state = parse_records(entries, self.provider, self._state)
        # Written by parse_records for the live reader; nothing here reads them.
        for scratch in (
            state.tool_names,
            state.tool_summaries,
            state.tool_diffs,
            state.pending_tools,
        ):
            scratch.clear()
        touched: set[str] = set()
        for entry, record in records:
            for call_id, output, failed in record.results:
                result = (
                    entry.offset,
                    self._offload({"text": _text(output), "truncated": False}),
                    failed,
                )
                owner = self._calls.get(call_id)
                if owner:
                    self._get(owner[0]).results.setdefault(call_id, []).append(result)
                    touched.add(owner[0])
                else:
                    self._orphans.setdefault(call_id, []).append(result)
            if record.parts:
                touched.add(self._record(entry, record, touched))
        for message_id in touched:
            stored = self._get(message_id)
            self._resolve(message_id, stored)
            self._index[message_id] = self._blob(stored)
        self._cache.clear()
        self.dirty |= touched

    def _get(self, message_id: str) -> _Stored:
        if message_id not in self._cache:
            ref = self._index[message_id]
            assert ref is not None
            self._cache[message_id] = self._load(ref)
        return self._cache[message_id]

    def _record(
        self, entry: SourceEntry, record: TranscriptRecord, touched: set[str]
    ) -> str:
        generation = f"g{self.generation}/" if self.generation else ""
        message_id = f"{self.conversation_id}/{generation}{record.key}"
        known = message_id in self._cache or bool(self._index.get(message_id))
        old = self._get(message_id) if known else None
        offsets = [*old.offsets, entry.offset] if old else [entry.offset]
        results = old.results if old else {}
        self._index.setdefault(message_id, None)
        parts = tuple(
            self._offload_part(
                Part(
                    id=f"{message_id}#{i}",
                    **{n: v for n, v in part.items() if n in _PART_FIELDS},
                )
            )
            for i, part in enumerate(record.parts)
        )
        message = Message(
            message_id,
            len(offsets),
            self.conversation_id,
            record.role,
            record.time,
            record.complete,
            parts,
            phase=record.phase,
        )
        self._cache[message_id] = _Stored(message, offsets, results)
        for i, part in enumerate(parts):
            call_id = part.tool_call_id
            if not call_id:
                continue
            previous = self._calls.get(call_id)
            moved = []
            if previous and previous[0] != message_id:
                moved = self._get(previous[0]).results.pop(call_id, [])
                touched.add(previous[0])
            merged = [
                *self._orphans.pop(call_id, []),
                *moved,
                *results.get(call_id, []),
            ]
            if merged:
                results[call_id] = merged
            self._calls[call_id] = message_id, i
        return message_id

    def _resolve(self, message_id: str, stored: _Stored) -> None:
        """Rebuild tool results from the calls this message currently owns."""
        parts = list(stored.message.parts)
        contributors = set(stored.offsets)
        applied = False
        for index, part in enumerate(parts):
            call_id = part.tool_call_id
            if not call_id:
                continue
            if self._calls.get(call_id) != (message_id, index):
                if part.output is not None:
                    parts[index] = replace(part, status="running", output=None)
                continue
            for offset, output, failed in stored.results.get(call_id, ()):
                parts[index] = replace(
                    parts[index],
                    status="failed" if failed else "completed",
                    output=output,
                )
                contributors.add(offset)
                applied = True
        stored.contributors = tuple(contributors)
        stored.message = replace(
            stored.message,
            parts=tuple(parts),
            revision=len(contributors) if applied else len(stored.offsets),
        )

    def _blob(self, value: Any) -> _Spilled:
        data = pickle.dumps(value, pickle.HIGHEST_PROTOCOL)
        if os.pwrite(self._spill.fileno(), data, self._spill_end) != len(data):
            raise OSError("Spill write was incomplete")
        self._spill_end += len(data)
        return _Spilled(self._spill_end - len(data), len(data))

    def _load(self, ref: _Spilled) -> Any:
        data = os.pread(self._spill.fileno(), ref.length, ref.offset)
        try:
            if len(data) != ref.length:
                raise EOFError("short spill read")
            return pickle.loads(data)
        except (pickle.PickleError, EOFError) as exc:
            raise OSError("Spill is unreadable") from exc

    def _offload(self, value: Any) -> Any:
        if value is None or isinstance(value, bool | int):
            return value
        if isinstance(value, str) and len(value) < _SPILL_MIN:
            return value
        if len(pickle.dumps(value, pickle.HIGHEST_PROTOCOL)) < _SPILL_MIN:
            return value
        return self._blob(value)

    def _offload_part(self, part: Part) -> Part:
        changes = {}
        for name in _PART_FIELDS:
            value = getattr(part, name)
            stored = self._offload(value)
            if stored is not value:
                changes[name] = stored
        return replace(part, **changes) if changes else part

    def _restore(self, message: Message) -> Message:
        parts = []
        for part in message.parts:
            changes = {
                name: self._load(value)
                for name in _PART_FIELDS
                if isinstance(value := getattr(part, name), _Spilled)
            }
            parts.append(replace(part, **changes) if changes else part)
        return replace(message, parts=tuple(parts))

    def messages(
        self,
        source_range: tuple[int, int | None] | None = None,
        *,
        only: set[str] | None = None,
    ) -> tuple[Message, ...]:
        """Messages in first-seen order, optionally the ``only`` ids or a byte range."""
        start, end = source_range or (0, None)
        selected = []
        for message_id, ref in self._index.items():
            if (only is not None and message_id not in only) or ref is None:
                continue
            stored: _Stored = self._load(ref)
            if source_range is None or any(
                offset >= start and (end is None or offset < end)
                for offset in stored.contributors
            ):
                selected.append(self._restore(stored.message))
        return tuple(selected)


def _read_at(source: BinaryIO, start: int, length: int) -> bytes:
    source.seek(start)
    return source.read(length)


def _stat_stamp(status: os.stat_result) -> tuple[int, int, int, int]:
    return status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _digest(source: BinaryIO, size: int) -> str:
    whole, done = hashlib.sha256(), 0
    source.seek(0)
    while done < size:
        chunk = source.read(min(_CHUNK, size - done))
        if not chunk:
            break
        whole.update(chunk)
        done += len(chunk)
    return whole.hexdigest()
