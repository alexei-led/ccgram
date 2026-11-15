"""Cancellation keeps native marker ownership through a real socket response."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ccgram.multiplexer import herdr as herdr_module
from ccgram.multiplexer.herdr import (
    HerdrManager,
    HerdrSessionComposite,
    herdr_session_target_id,
)
from ccgram.multiplexer.herdr_socket import request
from ccgram.topic_selection import _selection_lock

_TARGET = herdr_session_target_id(
    HerdrSessionComposite("herdr:pi", "pi", "id", "cancel-session")
)


class DelayedHerdrSocket:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.original = "user label 中"
        self.label = self.original
        self.received = asyncio.Event()
        self.release = asyncio.Event()
        self.quiescent = asyncio.Event()
        self.handlers: set[asyncio.Task] = set()
        self.errors: list[Exception] = []

    def pane(self) -> dict[str, str]:
        return {"terminal_id": "term-cancel", "pane_id": "w1:p1", "label": self.label}

    async def rename(self, payload: dict) -> dict | None:
        label = payload["params"]["label"]
        if label != self.original:
            self.received.set()
            await self.release.wait()
            if self.outcome != "remote-error":
                self.label = label
            self.quiescent.set()
            if self.outcome == "remote-error":
                return {
                    "id": payload["id"],
                    "error": {"code": "refused", "message": "native write refused"},
                }
            if self.outcome == "disconnect":
                return None
        else:
            self.label = label
        return {"id": payload["id"], "result": {"pane": self.pane()}}

    async def response(self, payload: dict) -> dict | None:
        method = payload["method"]
        if method == "pane.rename":
            return await self.rename(payload)
        results = {
            "ping": {"type": "pong"},
            "pane.list": {"panes": [self.pane()]},
            "pane.get": {"pane": self.pane()},
            "workspace.list": {
                "workspaces": [{"workspace_id": "w1", "label": "workspace"}]
            },
            "tab.list": {"tabs": [{"tab_id": "w1:t1", "label": "tab"}]},
            "agent.list": {
                "agents": [
                    {
                        "terminal_id": "term-cancel",
                        "pane_id": "w1:p1",
                        "workspace_id": "w1",
                        "tab_id": "w1:t1",
                        "agent_session": {
                            "source": "herdr:pi",
                            "agent": "pi",
                            "kind": "id",
                            "value": "cancel-session",
                        },
                    }
                ]
            },
        }
        return {"id": payload["id"], "result": results[method]}

    async def serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.handlers.add(task)
        try:
            line = await reader.readline()
            if line:
                reply = await self.response(json.loads(line))
                if reply is not None:
                    writer.write(json.dumps(reply).encode() + b"\n")
                    await writer.drain()
        except ConnectionError, OSError:
            pass
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            self.errors.append(exc)
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()
            self.handlers.discard(task)


@pytest.fixture
async def native_marker(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    outcome = request.param
    native = DelayedHerdrSocket(outcome)
    production_request = herdr_module.socket_request

    async def boundary(*args, **kwargs):
        response = await production_request(*args, **kwargs)
        if (
            outcome == "worker-error"
            and args[1] == "pane.rename"
            and args[2]["label"] != native.original
        ):
            raise RuntimeError("unexpected RPC worker failure")
        return response

    def forbid_binary(*args, **kwargs):
        raise AssertionError("native binaries are forbidden in fake-socket tests")

    monkeypatch.setattr(herdr_module, "socket_request", boundary)
    monkeypatch.setattr(herdr_module.subprocess, "run", forbid_binary)
    with TemporaryDirectory(prefix="ccg-hcancel-", dir="/tmp") as directory:
        socket_path = Path(directory) / "rpc.sock"
        server = await asyncio.start_unix_server(native.serve, path=socket_path)
        manager = HerdrManager(socket_path=str(socket_path), binary="/forbidden-herdr")
        window = await manager.find_window_by_id(_TARGET)
        assert window is not None and window.topic_mark_owner
        try:
            yield manager, native, window.topic_mark_owner, str(socket_path)
        finally:
            native.release.set()
            server.close()
            await server.wait_closed()
            if native.handlers:
                await asyncio.gather(*tuple(native.handlers), return_exceptions=True)
            assert not native.errors


@pytest.mark.parametrize(
    "native_marker",
    ["success", "remote-error", "disconnect", "worker-error"],
    indirect=True,
)
@pytest.mark.parametrize("cancellations", [1, 3])
async def test_herdr_cancelled_marker_rpc_quiesces_before_opposing_clear(
    native_marker, cancellations: int
) -> None:
    manager, native, owner, socket_path = native_marker
    lock = _selection_lock(_TARGET, owner=owner)
    waiting = asyncio.Event()
    settled_off = asyncio.Event()
    clear_overtook_on = False

    async def mark_on():
        async with lock:
            return await manager.set_topic_mark(_TARGET, True, owner=owner)

    async def detach():
        nonlocal clear_overtook_on
        waiting.set()
        async with lock:
            clear_overtook_on = not native.quiescent.is_set()
            result = await manager.set_topic_mark(_TARGET, False, owner=owner)
            if result:
                settled_off.set()
            return result

    on = asyncio.create_task(mark_on())
    off = None
    try:
        async with asyncio.timeout(3):
            await native.received.wait()
        on.cancel("original cancellation")
        off = asyncio.create_task(detach())
        await waiting.wait()
        # A real independent RPC roundtrip advances transport cancellation,
        # without assuming an App enqueue order across socket connections.
        await request(socket_path, "ping", {})
        for _ in range(cancellations - 1):
            on.cancel("repeated cancellation")
            await request(socket_path, "ping", {})
        ended_early = on.done()
        if ended_early:
            async with asyncio.timeout(3):
                await settled_off.wait()
        settled_early = settled_off.is_set()
        native.release.set()
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            await on
        assert await off
        async with asyncio.timeout(3):
            await native.quiescent.wait()
        assert not ended_early
        assert not settled_early
        assert not clear_overtook_on
        assert native.label == native.original
        assert settled_off.is_set()
    finally:
        native.release.set()
        await asyncio.gather(
            *(task for task in (on, off) if task is not None), return_exceptions=True
        )


@pytest.mark.parametrize(
    "native_marker",
    ["success", "remote-error", "disconnect", "worker-error"],
    indirect=True,
)
async def test_herdr_uncancelled_marker_rpc_preserves_failure_or_verification(
    native_marker,
) -> None:
    manager, native, owner, _ = native_marker
    native.release.set()
    if native.outcome == "worker-error":
        with pytest.raises(RuntimeError, match="unexpected RPC worker failure"):
            await manager.set_topic_mark(_TARGET, True, owner=owner)
    else:
        result = await manager.set_topic_mark(_TARGET, True, owner=owner)
        assert result is (native.outcome == "success")
    assert native.quiescent.is_set()
    if native.outcome == "disconnect":
        assert native.label != native.original
        assert await manager.set_topic_mark(_TARGET, True, owner=owner)
