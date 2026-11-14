"""A cancelled native marker operation must finish before its owner unlocks."""

import asyncio
from pathlib import Path
import stat
import threading
from types import SimpleNamespace
from typing import cast

import libtmux
import pytest

from ccgram.multiplexer.tmux import TmuxManager
from ccgram.topic_selection import _selection_lock


class NativeMarker:
    def __init__(self, endpoint: Path, outcome: str) -> None:
        self.endpoint = endpoint
        self.outcome = outcome
        self.token: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.worker_thread: int | None = None

    def cmd(self, command, *args, target=None):
        result = SimpleNamespace(returncode=0, stderr=[], stdout=[])
        if command == "display-message":
            result.stdout = [str(self.endpoint)]
        elif command == "list-windows":
            result.stdout = ["@1"]
        elif command == "show-options":
            result.stdout = [f"@ccgram_topic {self.token}"] if self.token else []
        elif command == "set-option":
            token = args[-1]
            if token.startswith("v1:1:"):
                self.worker_thread = threading.get_ident()
                self.entered.set()
                try:
                    if not self.release.wait(5):
                        raise TimeoutError("test did not release native mark-on")
                    if self.outcome == "api-error":
                        return SimpleNamespace(
                            returncode=1, stderr=["refused"], stdout=[]
                        )
                    if self.outcome == "os-error":
                        raise OSError("native connection failed")
                    if self.outcome == "worker-error":
                        raise RuntimeError("native worker failed")
                    self.token = token
                finally:
                    self.finished.set()
            else:
                self.token = token
        else:
            raise AssertionError(command)
        return result


@pytest.fixture
def native_marker(tmp_path, monkeypatch):
    endpoint = (tmp_path / "dummy-native.sock").resolve()
    real_stat = Path.stat

    def socket_stat(path, *args, **kwargs):
        if path == endpoint:
            return SimpleNamespace(
                st_mode=stat.S_IFSOCK, st_dev=1, st_ino=2, st_ctime_ns=3
            )
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", socket_stat)

    def build(outcome):
        native = NativeMarker(endpoint, outcome)
        mux = TmuxManager("test-cancellation")
        mux._server = cast(libtmux.Server, native)
        snapshot = mux._topic_inventory()
        assert snapshot is not None
        owner = snapshot[1]["@1"][1]
        assert owner is not None
        return mux, native, owner

    return build


async def checkpoint() -> None:
    future = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(future.set_result, None)
    await future


@pytest.mark.parametrize("cancellations", [1, 3])
@pytest.mark.parametrize(
    "outcome", ["success", "api-error", "os-error", "worker-error"]
)
async def test_cancelled_mark_finishes_before_waiting_owner_clear(
    native_marker, cancellations, outcome
) -> None:
    mux, native, owner = native_marker(outcome)
    lock = _selection_lock("@1", owner=owner)
    waiting = asyncio.Event()
    settled_off = asyncio.Event()
    clear_overtook_on = False

    async def mark_on():
        async with lock:
            return await mux.set_topic_mark("@1", True, owner=owner)

    async def detach():
        nonlocal clear_overtook_on
        waiting.set()
        async with lock:
            clear_overtook_on = not native.finished.is_set()
            verified = await mux.set_topic_mark("@1", False, owner=owner)
            if verified:
                settled_off.set()
            return verified

    on = asyncio.create_task(mark_on())
    off = None
    try:
        assert await asyncio.to_thread(native.entered.wait, 3)
        assert native.worker_thread != threading.get_ident()
        on.cancel("original cancellation")
        off = asyncio.create_task(detach())
        await waiting.wait()
        await checkpoint()
        for _ in range(cancellations - 1):
            on.cancel("repeated cancellation")
            await checkpoint()
        cancelled_before_quiescent = on.done()
        if cancelled_before_quiescent:
            async with asyncio.timeout(3):
                await settled_off.wait()
        off_settled_before_quiescent = settled_off.is_set()
        native.release.set()
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            await on
        assert await off
        assert await asyncio.to_thread(native.finished.wait, 3)
        assert not cancelled_before_quiescent
        assert not off_settled_before_quiescent
        assert not clear_overtook_on
        assert settled_off.is_set()
        assert native.token is None or native.token.startswith("v1:0:")
    finally:
        native.release.set()
        await asyncio.gather(
            *(task for task in (on, off) if task is not None), return_exceptions=True
        )
        assert await asyncio.to_thread(native.finished.wait, 3)


@pytest.mark.parametrize(
    "outcome", ["success", "api-error", "os-error", "worker-error"]
)
async def test_uncancelled_native_marker_preserves_result_or_error(
    native_marker, outcome
) -> None:
    mux, native, owner = native_marker(outcome)
    native.release.set()
    if outcome == "worker-error":
        with pytest.raises(RuntimeError, match="native worker failed"):
            await mux.set_topic_mark("@1", True, owner=owner)
    else:
        assert await mux.set_topic_mark("@1", True, owner=owner) is (
            outcome == "success"
        )
    assert native.finished.is_set()
