"""The production agterm runner cannot leave a flag process past cancellation."""

import asyncio
import json
import subprocess
import threading

import pytest

from ccgram.multiplexer.agterm import AgtermManager
from ccgram.topic_selection import _selection_lock

SESSION = "6FE911B4-B39A-47AB-970C-8AC13B14CB88"
WINDOW = "AC1E8A7F-4DD4-49FA-8DFE-E03A8519A6E8"
OWNER = f"agterm:{SESSION.lower()}"


class NativeFlagProcess:
    def __init__(self, original: bool, outcome: str) -> None:
        self.flagged = original
        self.outcome = outcome
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.first_write = True
        self.worker_thread: int | None = None

    def __call__(self, args, **kwargs):
        assert args[0] == "/owned-test/agtermctl"
        assert args[-3:] == ["--json", "--socket", "/owned-test/agterm.sock"]
        command = list(args[1:-3])
        result = {}
        if command == ["window", "list"]:
            result = {"windows": [{"id": WINDOW, "open": True}]}
        elif command == ["tree", "--window", WINDOW]:
            result = {
                "tree": {
                    "workspaces": [
                        {
                            "name": "test",
                            "sessions": [
                                {
                                    "id": SESSION,
                                    "name": "_owned-test",
                                    "cwd": "/test",
                                    "flagged": self.flagged,
                                    "foreground": ["pi"],
                                }
                            ],
                        }
                    ],
                }
            }
        elif command in (
            ["session", "flag", "on", "--target", SESSION],
            ["session", "flag", "off", "--target", SESSION],
        ):
            desired = command[2] == "on"
            if self.first_write:
                self.first_write = False
                self.worker_thread = threading.get_ident()
                self.entered.set()
                try:
                    if not self.release.wait(5):
                        raise TimeoutError("test did not release native flag process")
                    if self.outcome == "api-error":
                        return subprocess.CompletedProcess(args, 1, "", "refused")
                    if self.outcome == "os-error":
                        raise OSError("native process could not start")
                    if self.outcome == "timeout":
                        raise subprocess.TimeoutExpired(args, kwargs["timeout"])
                    if self.outcome == "worker-error":
                        raise RuntimeError("native process worker failed")
                    self.flagged = desired
                finally:
                    self.finished.set()
            else:
                self.flagged = desired
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(
            args, 0, json.dumps({"ok": True, "result": result}), ""
        )


@pytest.fixture
def native_flag(monkeypatch):
    def build(original, outcome):
        native = NativeFlagProcess(original, outcome)
        monkeypatch.setattr(subprocess, "run", native)
        mux = AgtermManager(
            socket_path="/owned-test/agterm.sock",
            binary="/owned-test/agtermctl",
            own_session_id="",
            workspaces=None,
        )
        return mux, native

    return build


async def checkpoint() -> None:
    future = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(future.set_result, None)
    await future


@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize("cancellations", [1, 3])
@pytest.mark.parametrize(
    "outcome", ["success", "api-error", "os-error", "timeout", "worker-error"]
)
async def test_cancelled_flag_process_quiesces_before_opposing_owner_mutation(
    native_flag, marked, cancellations, outcome
) -> None:
    mux, native = native_flag(not marked, outcome)
    captured = await mux.find_window_by_id(SESSION)
    assert captured is not None and captured.topic_mark_owner == OWNER
    lock = _selection_lock(SESSION, owner=captured.topic_mark_owner)
    waiting = asyncio.Event()
    settled = asyncio.Event()
    opposing_overtook = False

    async def first_mutation():
        async with lock:
            return await mux.set_topic_mark(SESSION, marked, owner=OWNER)

    async def opposing_mutation():
        nonlocal opposing_overtook
        waiting.set()
        async with lock:
            opposing_overtook = not native.finished.is_set()
            verified = await mux.set_topic_mark(SESSION, not marked, owner=OWNER)
            if verified:
                settled.set()
            return verified

    first = asyncio.create_task(first_mutation())
    second = None
    try:
        assert await asyncio.to_thread(native.entered.wait, 3)
        assert native.worker_thread != threading.get_ident()
        first.cancel("original cancellation")
        second = asyncio.create_task(opposing_mutation())
        await waiting.wait()
        await checkpoint()
        for _ in range(cancellations - 1):
            first.cancel("repeated cancellation")
            await checkpoint()
        cancelled_before_quiescent = first.done()
        if cancelled_before_quiescent:
            async with asyncio.timeout(3):
                await settled.wait()
        settled_before_quiescent = settled.is_set()
        native.release.set()
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            await first
        assert await second
        assert await asyncio.to_thread(native.finished.wait, 3)
        assert native.flagged is not marked
        assert not cancelled_before_quiescent
        assert not settled_before_quiescent
        assert not opposing_overtook
        assert settled.is_set()
    finally:
        native.release.set()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )
        assert await asyncio.to_thread(native.finished.wait, 3)


@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize(
    "outcome", ["success", "api-error", "os-error", "timeout", "worker-error"]
)
async def test_uncancelled_flag_process_keeps_result_and_error_semantics(
    native_flag, marked, outcome
) -> None:
    mux, native = native_flag(not marked, outcome)
    native.release.set()
    if outcome == "worker-error":
        with pytest.raises(RuntimeError, match="native process worker failed"):
            await mux.set_topic_mark(SESSION, marked, owner=OWNER)
    else:
        assert await mux.set_topic_mark(SESSION, marked, owner=OWNER) is (
            outcome == "success"
        )
    assert native.finished.is_set()
