from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from _topic_mark_helpers import make_witness
from ccgram import topic_selection
from ccgram.multiplexer.base import WindowRef


class AdmissionRace:
    def __init__(self, store, router, pause_read):
        self.store = store
        self.router = router
        self.window = WindowRef(
            "@5",
            "selected",
            "/tmp",
            pane_current_command="claude",
            topic_markable=True,
            topic_marked=True,
            topic_mark_owner="owner:@5",
        )
        self.pause_read = pause_read
        self.reads = 0
        self.captured = asyncio.Event()
        self.release = asyncio.Event()
        self.attempted = asyncio.Event()
        self.bound_when_unmark_completed = None
        self.native = MagicMock(
            capabilities=SimpleNamespace(name="tmux"),
            find_window_by_id=AsyncMock(side_effect=self.read),
            list_windows_for_reconciliation=AsyncMock(side_effect=self.list_windows),
            set_topic_mark=AsyncMock(side_effect=self.write),
            topic_mark_endpoint_scope=AsyncMock(
                return_value=make_witness().endpoint_scope
            ),
            recover_topic_mark_write=AsyncMock(return_value=False),
        )

    async def read(self, window_id):
        assert window_id == self.window.window_id
        self.reads += 1
        snapshot = self.window
        if self.reads == self.pause_read:
            self.captured.set()
            await self.release.wait()
        return snapshot

    async def list_windows(self):
        return [await self.read(self.window.window_id)]

    async def write(self, window_id, marked, *, owner, write_guard):
        assert (window_id, owner) == (
            self.window.window_id,
            self.window.topic_mark_owner,
        )
        witness = make_witness()
        write_guard.arm(witness)
        self.window = replace(self.window, topic_marked=marked)
        write_guard.terminal(witness.dispatch_id)
        return True

    async def unmark(self):
        self.attempted.set()
        assert await topic_selection.set_topic_selection(
            self.window,
            False,
            backend=self.native,
            store=self.store,
            flush=lambda: None,
            endpoint_scope=make_witness().endpoint_scope,
        )
        self.bound_when_unmark_completed = self.router.get_window_for_thread(
            100, 42, -100999
        )

    async def run(self, admission, *, cancelled=False):
        reader = asyncio.create_task(admission)
        writer = None
        try:
            async with asyncio.timeout(3):
                await self.captured.wait()
                writer = asyncio.create_task(self.unmark())
                await self.attempted.wait()
                if cancelled:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                    result = None
                else:
                    self.release.set()
                    result = await reader
                await writer
        finally:
            self.release.set()
            for task in (reader, writer):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(t for t in (reader, writer) if t is not None), return_exceptions=True
            )
        assert self.window.topic_marked is False
        assert not self.store.topic_mark_operations
        assert self.bound_when_unmark_completed == (
            None if cancelled else self.window.window_id
        )
        return result
