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
import contextlib
import hashlib
import os
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from .messages import MessageBuilder, SourceChangedError, split_jsonl
from .store import ProtocolStore, TranscriptCursor

MARKER_BYTES = 4096
READ_CHUNK_BYTES = 4 * 1024 * 1024
WINDOW_BYTES = 1024 * 1024
# A record rarely spans more than this, so a backward search for a line boundary
# stays short.
BACKSCAN_BYTES = 64 * 1024
MAX_RETRY_SECONDS = 30.0
_SYNC_ATTEMPTS = 3
_BACKOFF_DOUBLINGS = 5  # 2**5 = 32 s, past MAX_RETRY_SECONDS

type Stamp = tuple[int, int, int, int, int]


def source_stamp(stat: os.stat_result) -> Stamp:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _digest(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


async def run_in_worker[**P, T](
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run `func` in a thread; a cancelled caller waits for the thread to stop.

    The thread may mutate the index or step a window generator, and the caller
    holds the refresh lock until it returns. Cancellation therefore waits for the
    thread, so the lock is never released under a running worker and a generator
    is never closed while it executes.
    """
    work = asyncio.ensure_future(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        # A repeated cancel must not cut the wait short either.
        while not work.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait([work])
        if not work.cancelled():
            # Retrieve the result so a failed worker does not log an unretrieved error.
            work.exception()
        raise


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
        # Resident index. The service turns it off for sessions nobody reads, so
        # only the cursor is kept and windows are parsed on demand by `scan`.
        self.indexed = True
        # Message events are published for this source. Set by the service.
        self.publishing = False
        self._appended: tuple[int, int] | None = None

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
        await run_in_worker(self._sync)

    def _sync(self) -> None:
        stat = self.path.stat()
        stamp = source_stamp(stat)
        if (
            not self._poisoned
            and stamp == self.stamp
            and (self.builder is not None or not self.indexed)
        ):
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

    def set_indexed(self, on: bool) -> None:
        """Keep the resident index, or drop it and keep only the cursor."""
        self.indexed = on
        if not on:
            self.builder = None
            self._poisoned = False

    @property
    def consumed(self) -> int:
        """Line boundary up to which the transcript has been observed."""
        return self._consumed

    def tail_boundary(self, nbytes: int) -> int:
        """Line boundary at or after `consumed - nbytes`, for a bounded tail scan."""
        begin = max(0, self._consumed - nbytes)
        if begin == 0 or self.cursor is None:
            return 0
        with self.path.open("rb") as handle:
            self._check_indexed(handle)
            return _after_newline(handle, begin, self._consumed)

    @property
    def appended(self) -> tuple[int, int] | None:
        """Line-aligned byte range appended since the last `clear_appended`.

        Only tracked while not indexed. It stays set until a caller has correlated
        the range, so a failed scan is retried on the next poll.
        """
        return self._appended

    def clear_appended(self) -> None:
        self._appended = None

    def _observe(self, stat: os.stat_result) -> None:
        prior = self.cursor
        with self.path.open("rb") as handle:
            if os.fstat(handle.fileno()).st_ino != stat.st_ino:
                raise SourceChangedError("Transcript was replaced before reading")
            intact = _tail_intact(handle, stat, prior)
            generation = _generation(prior, intact)
            if not self.indexed:
                start = self._consumed if intact else 0
                consumed, marker = _consume(handle, None, start, stat.st_size)
                if os.fstat(handle.fileno()).st_ino != stat.st_ino:
                    raise SourceChangedError("Transcript was replaced while reading")
                # A replacement starts over: a range pending from the old transcript
                # must not extend the new one.
                pending = self._appended if intact else None
                first = pending[0] if pending else start
                self._appended = (first, consumed)
                self._commit(None, consumed, stat, marker, generation)
                return
            if intact and self.builder is not None and not self._poisoned:
                builder, start = self.builder, self._consumed
                # A failure mid-feed leaves the builder partial; the next sync
                # rebuilds it under the same generation if the file is intact.
                self._poisoned = True
                incremental = True
            else:
                incremental = False
                builder, start = (
                    MessageBuilder(
                        self.conversation_id,
                        self.provider,
                        generation,
                        reader=self._read_record,
                    ),
                    0,
                )
            consumed, marker = _consume(handle, builder, start, stat.st_size)
            if os.fstat(handle.fileno()).st_ino != stat.st_ino:
                raise SourceChangedError("Transcript was replaced while reading")
        # Only an append that extends the resident index is "appended". A rebuild
        # covers the whole transcript and is not correlated as new messages. A
        # range still pending from a failed correlation stays part of the range.
        pending = self._appended if incremental else None
        self._appended = (
            (pending[0] if pending else start, consumed) if incremental else None
        )
        self._commit(builder, consumed, stat, marker, builder.generation)

    def scan(
        self, start: int, end: int, *, newest_first: bool = False
    ) -> Generator[MessageBuilder, None, None]:
        """Canonical builders over line-aligned windows of [start, end).

        Each window is a transient builder with inline values, for correlation and
        hook matching when no index is resident. Windows are at most WINDOW_BYTES
        of transcript, so memory does not grow with the transcript.
        """
        prior = self.cursor
        if prior is None or end <= start:
            return
        with self.path.open("rb") as handle:
            self._check_indexed(handle)
            for begin, stop in _line_windows(
                handle, start, end, newest_first=newest_first
            ):
                handle.seek(begin)
                raw = handle.read(stop - begin)
                builder = MessageBuilder(
                    self.conversation_id, self.provider, prior.generation
                )
                entries, _ = split_jsonl(raw, begin)
                builder.feed(entries)
                yield builder

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
        builder: MessageBuilder | None,
        consumed: int,
        stat: os.stat_result,
        marker: str,
        generation: int,
    ) -> None:
        self.builder = builder
        self._consumed = consumed
        self._poisoned = False
        self.cursor = TranscriptCursor(stat.st_ino, stat.st_size, marker, generation)


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


def _line_windows(
    handle: BinaryIO, start: int, end: int, *, newest_first: bool = False
) -> Iterator[tuple[int, int]]:
    """Split [start, end) lazily into line-aligned windows.

    Each window starts and ends on a line boundary. Newest first, the windows are
    found by reading backwards from `end`, so a caller that stops early reads only
    the windows it took.
    """
    if newest_first:
        stop = end
        while stop > start:
            begin = _before_newline(handle, start, max(start, stop - WINDOW_BYTES))
            yield begin, stop
            stop = begin
        return
    position = start
    while position < end:
        stop = min(position + WINDOW_BYTES, end)
        if stop < end:
            stop = _after_newline(handle, stop, end)
        yield position, stop
        position = stop


def _before_newline(handle: BinaryIO, start: int, offset: int) -> int:
    """Line boundary in [start, offset]: just after the last newline before it."""
    position = offset
    while position > start:
        chunk_start = max(start, position - BACKSCAN_BYTES)
        handle.seek(chunk_start)
        data = handle.read(position - chunk_start)
        found = data.rfind(b"\n")
        if found >= 0:
            return chunk_start + found + 1
        position = chunk_start
    return start


def _after_newline(handle: BinaryIO, offset: int, limit: int) -> int:
    handle.seek(offset)
    position = offset
    while position < limit:
        data = handle.read(min(READ_CHUNK_BYTES, limit - position))
        if not data:
            break
        found = data.find(b"\n")
        if found >= 0:
            return position + found + 1
        position += len(data)
    return limit


def _consume(
    handle: BinaryIO, builder: MessageBuilder | None, start: int, limit: int
) -> tuple[int, str]:
    """Feed complete records in [start, limit) and digest the tail before limit.

    Returns the new line boundary and the marker for `limit`. The read starts up
    to MARKER_BYTES before `start` when needed, so the marker comes from the same
    read as the parsed records. With no builder, only the boundary advances and
    nothing is parsed.
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
            if builder is not None:
                entries, _ = split_jsonl(buffered[:cut], consumed)
                builder.feed(entries)
            consumed += cut
    return consumed, _digest(tail)
