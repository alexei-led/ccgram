"""Stat-first, append-only view of one transcript for the protocol core.

An unchanged transcript costs one stat. An append reads only the new complete
records into the existing builder. A replacement (inode change, shrink, or a
changed marker before the last observed size) rebuilds from byte 0 and bumps the
history generation. Gemini stores one JSON document, so any change rebuilds it.

Detection is a tail marker, not a full-content hash: a rewrite that keeps the
inode, the size, and the last 4 KiB before the observed size goes unnoticed.
Claude and Codex transcripts are append-only, so this is the accepted trade for
not rehashing the whole file on every append. Upgrade trigger: a field report of a
rewrite that keeps the tail. The fix is a rolling full-prefix digest, which costs one
read of the file per rebuild and O(appended bytes) per append.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from .messages import MessageBuilder, read_entries, split_jsonl
from .store import ProtocolStore, TranscriptCursor

MARKER_BYTES = 4096
READ_CHUNK_BYTES = 4 * 1024 * 1024
_SYNC_ATTEMPTS = 3

type Stamp = tuple[int, int, int, int, int]


def source_stamp(stat: os.stat_result) -> Stamp:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _marker(path: Path, provider: str, observed: int) -> str:
    start = 0 if provider == "gemini" else max(0, observed - MARKER_BYTES)
    digest = hashlib.sha256()
    remaining = observed - start
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(remaining, READ_CHUNK_BYTES))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


class _SourceChangedError(OSError):
    """The file was replaced or truncated while it was being read. Retryable."""


class TranscriptSource:
    """Builder plus the identity it was built from. Not thread-safe.

    Callers serialize `sync` calls. A read error is not retried until the
    transcript's stamp changes, so a persistent error costs one stat per poll
    instead of a full re-parse. A replacement during the read is retried.
    """

    def __init__(
        self,
        key: str,
        path: Path,
        conversation_id: str,
        provider: str,
        cursor: TranscriptCursor | None,
    ) -> None:
        self.key = key
        self.path = path
        self.conversation_id = conversation_id
        self.provider = provider
        self.cursor = cursor
        self.stamp: Stamp | None = None
        self.builder: MessageBuilder | None = None
        self._consumed = 0
        self._poisoned = False
        self._failed_stamp: Stamp | None = None
        self._saved = cursor

    @property
    def generation(self) -> int:
        return self.cursor.generation if self.cursor else 0

    async def sync(self, store: ProtocolStore) -> None:
        for _ in range(_SYNC_ATTEMPTS):
            try:
                await self._run_sync()
            except _SourceChangedError:
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
        if stamp == self._failed_stamp:
            raise OSError("Transcript is unreadable at its current stamp")
        try:
            if self._extends(stat):
                self._append(stat)
            else:
                self._rebuild(stat)
        except _SourceChangedError:
            raise
        except OSError:
            self._failed_stamp = stamp
            raise
        self.stamp = stamp
        self._failed_stamp = None

    def _extends(self, stat: os.stat_result) -> bool:
        return (
            self.provider != "gemini"
            and self.builder is not None
            and not self._poisoned
            and self._identity_holds(self.cursor, stat)
        )

    def _identity_holds(
        self, prior: TranscriptCursor | None, stat: os.stat_result
    ) -> bool:
        return (
            prior is not None
            and prior.inode == stat.st_ino
            and stat.st_size >= prior.observed
            and _marker(self.path, self.provider, prior.observed) == prior.marker
        )

    def _append(self, stat: os.stat_result) -> None:
        assert self.builder is not None
        # A failure mid-feed leaves the builder partial; the next change rebuilds.
        self._poisoned = True
        self._consumed = self._consume(
            self.builder, self._consumed, stat.st_size, stat.st_ino
        )
        self._poisoned = False
        self.cursor = self._commit_cursor(stat.st_ino, stat.st_size)

    def _rebuild(self, stat: os.stat_result) -> None:
        prior = self.cursor
        if prior is None:
            generation = 0
        elif self._poisoned or not self._identity_holds(prior, stat):
            generation = prior.generation + 1
        else:
            generation = prior.generation
        builder = MessageBuilder(self.conversation_id, self.provider, generation)
        if self.provider == "gemini":
            raw = self.path.read_bytes()
            builder.feed(read_entries(raw, "gemini"))
            consumed, observed = len(raw), len(raw)
        else:
            consumed = self._consume(builder, 0, stat.st_size, stat.st_ino)
            observed = stat.st_size
        self.builder = builder
        self._consumed = consumed
        self._poisoned = False
        self.cursor = TranscriptCursor(
            stat.st_ino,
            observed,
            _marker(self.path, self.provider, observed),
            generation,
        )

    def _commit_cursor(self, inode: int, observed: int) -> TranscriptCursor:
        return TranscriptCursor(
            inode,
            observed,
            _marker(self.path, self.provider, observed),
            self.generation,
        )

    def _consume(
        self, builder: MessageBuilder, start: int, limit: int, inode: int
    ) -> int:
        """Feed complete records in [start, limit); return the new line boundary."""
        consumed = start
        position = start
        pending = b""
        with self.path.open("rb") as handle:
            handle.seek(start)
            while position < limit:
                data = handle.read(min(READ_CHUNK_BYTES, limit - position))
                if not data:
                    raise _SourceChangedError("Transcript shrank while reading")
                position += len(data)
                buffered = pending + data
                cut = buffered.rfind(b"\n") + 1
                pending = buffered[cut:]
                if cut:
                    entries, _ = split_jsonl(buffered[:cut], consumed)
                    builder.feed(entries)
                    consumed += cut
            if os.fstat(handle.fileno()).st_ino != inode:
                raise _SourceChangedError("Transcript was replaced while reading")
        return consumed
