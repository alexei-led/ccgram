"""Shared lifecycle for the in-process session protocol core."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from pathlib import Path
from typing import IO

from ..event_reader import read_new_events
from ..multiplexer.base import Multiplexer
from ..providers.base import HookEvent
from .backend import MultiplexerSessionBackend
from .models import ProtocolError
from .service import LocalSessionService
from .store import ProtocolStore

logger = logging.getLogger(__name__)


class ProtocolRuntime:
    def __init__(
        self,
        store: ProtocolStore,
        service: LocalSessionService,
        lock: IO[bytes],
        *,
        prefix: str = "",
    ) -> None:
        self.store = store
        self.service = service
        self.lock = lock
        self.prefix = prefix
        self.task: asyncio.Task[None] | None = None
        self._listener_cleanup: Callable[[], Awaitable[None]] | None = None
        self.hooks: asyncio.Queue[tuple[HookEvent, tuple[str, ...]]] = asyncio.Queue(
            256
        )

    def set_listener_cleanup(
        self, cleanup: Callable[[], Awaitable[None]] | None
    ) -> None:
        self._listener_cleanup = cleanup

    def observe_hook(self, event: HookEvent) -> None:
        # A remote observer never blocks the Telegram monitor or its receipts.
        sessions = self.service.defer_hook(event, self.prefix)
        try:
            self.hooks.put_nowait((event, sessions))
        except asyncio.QueueFull:
            # Lost evidence must keep affected approvals blocked until restart.
            logger.error("Protocol hook queue full; affected approvals require restart")

    async def poll(self, prefix: str) -> None:
        unavailable = False
        while True:
            try:
                await self.service.refresh()
                if unavailable:
                    unavailable = False
                    logger.info("Session protocol source recovered")
                while not self.hooks.empty():
                    event, sessions = self.hooks.get_nowait()
                    await self.service.observe_hook(event, prefix)
                    self.service.resolve_hook(sessions)
            except (ProtocolError, OSError) as exc:
                # A multiplexer that is briefly unreachable or restarting is
                # expected, and the next pass retries. Log the incident's start
                # and end instead of one traceback per second: a sustained
                # outage buried real errors under thousands of identical stacks.
                if not unavailable:
                    unavailable = True
                    logger.warning(
                        "Session protocol source unavailable: %s", exc, exc_info=True
                    )
            await asyncio.sleep(1)

    async def close(self) -> None:
        try:
            if self._listener_cleanup is not None:
                await self._listener_cleanup()
        finally:
            try:
                if self.task is not None:
                    self.task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.task
            finally:
                self.store.close()
                self.lock.close()


async def start_runtime(
    multiplexer: Multiplexer,
    directory: Path,
    command_dir: str,
    prefix: str,
    *,
    recovered_targets: list[tuple[str, str]] | None = None,
) -> ProtocolRuntime:
    """Create the shared core, recover fresh hooks, and start its poller."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_dir = Path.home() / ".cache" / "ccgram" / "locks"
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_dir / f"protocol-{multiplexer.capabilities.name}.lock"
    resources = ExitStack()
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        lock = os.fdopen(fd, "a+b")
        resources.callback(lock.close)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "A session protocol core is already using this multiplexer backend"
            ) from exc
        store = ProtocolStore(directory / "protocol.sqlite3")
        resources.callback(store.close)
        for previous, current in recovered_targets or []:
            store.retarget(previous, current)
        service = LocalSessionService(
            MultiplexerSessionBackend(multiplexer, command_dir), store
        )
        runtime = ProtocolRuntime(store, service, lock, prefix=prefix)
    except BaseException:
        resources.close()
        raise
    resources.pop_all()
    try:
        await service.refresh()
        # Recover only fresh observations. Old hooks never recreate approvals.
        events_file = directory / "events.jsonl"
        offset = (
            max(0, events_file.stat().st_size - 65536) if events_file.exists() else 0
        )
        events, _ = await read_new_events(events_file, offset)
        for event in events:
            await service.observe_hook(event, prefix)
        await service.refresh()
        runtime.task = asyncio.create_task(
            runtime.poll(prefix), name="ccgram-session-protocol"
        )
    except BaseException:
        await runtime.close()
        raise
    return runtime
