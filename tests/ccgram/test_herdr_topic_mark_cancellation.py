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
from _topic_mark_helpers import RecordingTopicMarkGuard
from test_herdr_topic_marks import native_factory as native_factory

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
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, native_factory
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
async def test_cancelled_herdr_write_returns_with_durable_ambiguity(
    native_marker, cancellations: int
):
    manager, native, owner, _socket_path = native_marker
    write_guard = RecordingTopicMarkGuard()
    task = asyncio.create_task(
        manager.set_topic_mark(_TARGET, True, owner=owner, write_guard=write_guard)
    )
    try:
        await native.received.wait()
        assert len(write_guard.armed) == 1
        task.cancel("original cancellation")
        await asyncio.sleep(0)
        for _ in range(cancellations - 1):
            task.cancel("repeated cancellation")
            await asyncio.sleep(0)
        async with asyncio.timeout(1):
            with pytest.raises(asyncio.CancelledError, match="original cancellation"):
                await task
        assert not native.quiescent.is_set()
        assert write_guard.terminal_ids == []
        native.release.set()
        await native.quiescent.wait()
        assert write_guard.terminal_ids == []
    finally:
        native.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "native_marker",
    ["success", "remote-error", "disconnect", "worker-error"],
    indirect=True,
)
async def test_uncancelled_herdr_notifies_only_valid_correlated_completion(
    native_marker,
):
    manager, native, owner, _socket_path = native_marker
    write_guard = RecordingTopicMarkGuard()
    native.release.set()
    if native.outcome == "worker-error":
        with pytest.raises(RuntimeError, match="unexpected RPC worker failure"):
            await manager.set_topic_mark(
                _TARGET, True, owner=owner, write_guard=write_guard
            )
    else:
        assert await manager.set_topic_mark(
            _TARGET, True, owner=owner, write_guard=write_guard
        ) is (native.outcome == "success")
    assert len(write_guard.armed) == 1
    expected = [write_guard.armed[0].dispatch_id] if native.outcome == "success" else []
    assert write_guard.terminal_ids == expected


@pytest.mark.parametrize("native_marker", ["success"], indirect=True)
async def test_failed_herdr_arm_sends_no_request_frame(native_marker):
    manager, native, owner, _socket_path = native_marker

    class FailedCheckpoint(RecordingTopicMarkGuard):
        def arm(self, witness):
            raise OSError("checkpoint failed")

    assert not await manager.set_topic_mark(
        _TARGET, True, owner=owner, write_guard=FailedCheckpoint()
    )
    assert not native.received.is_set()
    assert native.label == native.original


@pytest.mark.parametrize("native_marker", ["success"], indirect=True)
async def test_herdr_actual_write_requires_state_owner_guard(native_marker):
    manager, native, owner, _socket_path = native_marker

    assert not await manager.set_topic_mark(_TARGET, True, owner=owner)
    assert not native.received.is_set()
    assert native.label == native.original
