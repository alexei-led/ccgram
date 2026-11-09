"""Stat-first, append-only view of one transcript for the protocol core.

An unchanged transcript costs one stat. A change opens the file once: the same
handle checks the previous observation's identity, then parses only the complete
records appended since the last line boundary into the existing builder and
digests the new tail. A replacement (inode change, shrink, or a changed marker
before the last observed size) rebuilds from byte 0 under a new generation.
Every provider, Gemini included, writes JSONL; a Gemini chat is one session-*.jsonl
file, so it takes the same incremental path.

Detection is a tail marker, not a full-content hash: a rewrite that keeps the
inode, the size, and the last 4 KiB before the observed size goes unnoticed, as
does an in-place rewrite between the identity check and the parse on one handle.
Claude and Codex transcripts are append-only, so this is the accepted trade for
not rehashing the whole file on every append. Upgrade trigger: a field report of a
rewrite that keeps the tail. The fix is a rolling full-prefix digest, which costs one
read of the file per rebuild and O(appended bytes) per append.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from .messages import MessageBuilder, SourceChangedError, split_jsonl
from .store import ProtocolStore, TranscriptCursor

MARKER_BYTES = 4096
READ_CHUNK_BYTES = 4 * 1024 * 1024
MAX_RETRY_SECONDS = 30.0
_SYNC_ATTEMPTS = 3
_BACKOFF_DOUBLINGS = 5  # 2**5 = 32 s, past MAX_RETRY_SECONDS

type Stamp = tuple[int, int, int, int, int]


def source_stamp(stat: os.stat_result) -> Stamp:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _digest(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


class TranscriptSource:
    """Builder plus the identity it was built from. Not thread-safe.

    Callers serialize `sync` calls. A failed read is retried when the stamp
    changes or after an exponential backoff (1 s doubling to 30 s), so a
    persistent error costs one stat per poll instead of a full re-parse, and a
    transient error on an idle transcript still recovers. A replacement during
    the read is retried at once.
    """

    def __init__(
        self,
        key: str,
        path: Path,
        conversation_id: str,
        provider: str,
        cursor: TranscriptCursor | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.key = key
        self.path = path
        self.conversation_id = conversation_id
        self.provider = provider
        self.cursor = cursor
        self.stamp: Stamp | None = None
        self.builder: MessageBuilder | None = None
        self._clock = clock
        self._consumed = 0
        self._poisoned = False
        self._failures = 0
        self._failed_stamp: Stamp | None = None
        self._retry_at = 0.0
        self._saved = cursor
        self._handle: BinaryIO | None = None

    @property
    def generation(self) -> int:
        return self.cursor.generation if self.cursor else 0

    @property
    def failing(self) -> bool:
        """Whether the last read attempt failed."""
        return self._failures > 0

    async def sync(self, store: ProtocolStore) -> None:
        for _ in range(_SYNC_ATTEMPTS):
            try:
                await self._run_sync()
            except SourceChangedError:
                continue
            break
        else:
            raise OSError("Transcript changed repeatedly while reading")
        if self.cursor is not None and self.cursor != self._saved:
            store.save_transcript_cursor(self.key, self.cursor)
            self._saved = self.cursor

    async def _run_sync(self) -> None:
        work = asyncio.ensure_future(asyncio.to_thread(self._sync))
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            # The worker mutates the builder. Keep the caller's lock until it stops.
            await asyncio.wait([work])
            raise

    def _sync(self) -> None:
        stat = self.path.stat()
        stamp = source_stamp(stat)
        if self.builder is not None and not self._poisoned and stamp == self.stamp:
            return
        if (
            self._failures
            and stamp == self._failed_stamp
            and self._clock() < self._retry_at
        ):
            raise OSError("Transcript read failed; retry pending")
        try:
            self._observe(stat)
        except SourceChangedError:
            raise
        except OSError:
            self._failures += 1
            self._failed_stamp = stamp
            # Cap the exponent before exponentiating: a long outage must not overflow.
            delay = min(
                MAX_RETRY_SECONDS, 2.0 ** min(self._failures - 1, _BACKOFF_DOUBLINGS)
            )
            self._retry_at = self._clock() + delay
            raise
        self.stamp = stamp
        self._failures = 0
        self._failed_stamp = None

    def _observe(self, stat: os.stat_result) -> None:
        prior = self.cursor
        with self.path.open("rb") as handle:
            if os.fstat(handle.fileno()).st_ino != stat.st_ino:
                raise SourceChangedError("Transcript was replaced before reading")
            intact = _tail_intact(handle, stat, prior)
            if intact and self.builder is not None and not self._poisoned:
                builder, start = self.builder, self._consumed
                # A failure mid-feed leaves the builder partial; the next sync
                # rebuilds it under the same generation if the file is intact.
                self._poisoned = True
            else:
                builder, start = (
                    MessageBuilder(
                        self.conversation_id,
                        self.provider,
                        _generation(prior, intact),
                        reader=self._read_record,
                    ),
                    0,
                )
            consumed, marker = _consume(handle, builder, start, stat.st_size)
            if os.fstat(handle.fileno()).st_ino != stat.st_ino:
                raise SourceChangedError("Transcript was replaced while reading")
        self._commit(builder, consumed, stat, marker)

    @contextmanager
    def reading(self) -> Iterator[None]:
        """Pin one handle for a batch of materializations, such as one publish.

        The handle pins the inode it opened, so reads stay on the bytes the index
        describes even if the path is replaced during the batch. The identity check
        runs once here, before any message is read.
        """
        with self.path.open("rb") as handle:
            self._check_indexed(handle)
            self._handle = handle
            try:
                yield
            finally:
                self._handle = None

    def _read_record(self, offset: int) -> bytes:
        """The record line at `offset`. Only valid inside `reading`."""
        if self._handle is None:
            raise RuntimeError("Transcript records are read only inside reading()")
        return self._line(self._handle, offset)

    def _check_indexed(self, handle: BinaryIO) -> None:
        prior = self.cursor
        if prior is None or not _tail_intact(handle, os.fstat(handle.fileno()), prior):
            raise SourceChangedError("Transcript changed after it was indexed")

    def _line(self, handle: BinaryIO, offset: int) -> bytes:
        assert self.cursor is not None
        handle.seek(offset)
        line = handle.readline(self.cursor.observed - offset)
        if not line.endswith(b"\n"):
            raise SourceChangedError("Indexed record is no longer complete")
        return line

    def _commit(
        self,
        builder: MessageBuilder,
        consumed: int,
        stat: os.stat_result,
        marker: str,
    ) -> None:
        self.builder = builder
        self._consumed = consumed
        self._poisoned = False
        self.cursor = TranscriptCursor(
            stat.st_ino, stat.st_size, marker, builder.generation
        )


def _generation(prior: TranscriptCursor | None, intact: bool) -> int:
    if prior is None:
        return 0
    return prior.generation if intact else prior.generation + 1


def _tail_intact(
    handle: BinaryIO, stat: os.stat_result, prior: TranscriptCursor | None
) -> bool:
    """Whether the bytes before the previously observed size are unchanged."""
    if prior is None or prior.inode != stat.st_ino or stat.st_size < prior.observed:
        return False
    start = max(0, prior.observed - MARKER_BYTES)
    handle.seek(start)
    data = handle.read(prior.observed - start)
    return len(data) == prior.observed - start and _digest(data) == prior.marker


def _consume(
    handle: BinaryIO, builder: MessageBuilder, start: int, limit: int
) -> tuple[int, str]:
    """Feed complete records in [start, limit) and digest the tail before limit.

    Returns the new line boundary and the marker for `limit`. The read starts up
    to MARKER_BYTES before `start` when needed, so the marker comes from the same
    read as the parsed records.
    """
    tail_from = max(0, limit - MARKER_BYTES)
    position = min(start, tail_from)
    handle.seek(position)
    tail = bytearray()
    pending = b""
    consumed = start
    while position < limit:
        data = handle.read(min(READ_CHUNK_BYTES, limit - position))
        if not data:
            raise SourceChangedError("Transcript shrank while reading")
        chunk_start = position
        position += len(data)
        if position > tail_from:
            tail += data[max(0, tail_from - chunk_start) :]
        if position <= start:
            continue
        buffered = pending + data[max(0, start - chunk_start) :]
        cut = buffered.rfind(b"\n") + 1
        pending = buffered[cut:]
        if cut:
            entries, _ = split_jsonl(buffered[:cut], consumed)
            builder.feed(entries)
            consumed += cut
    return consumed, _digest(tail)
