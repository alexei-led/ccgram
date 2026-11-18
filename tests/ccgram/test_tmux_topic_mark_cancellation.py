"""Native tmux completion requires a per-dispatch same-queue acknowledgement."""

import asyncio
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast

import libtmux
import pytest

from ccgram.multiplexer import native_process as native_process
from ccgram.multiplexer import tmux as tmux_module
from ccgram.multiplexer import topic_mark_transport as transport
from ccgram.multiplexer.tmux import TmuxManager
from _topic_mark_helpers import RecordingTopicMarkGuard


class NativeMarker:
    def __init__(self, endpoint: Path, outcome: str):
        self.endpoint = endpoint
        self.outcome = outcome
        self.token: str | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.remote: asyncio.Task | None = None
        self.writes = 0

    async def run(self, _binary: str, args: list[str]):
        assert args[0] == "-N"
        if "display-message" in args and "set-option" not in args:
            value = str(self.endpoint) if args[-1] == "#{socket_path}" else "123"
            return 0, value + "\n", ""
        if "list-windows" in args:
            return 0, "@1\n", ""
        if "show-options" in args:
            return 0, f"@ccgram_topic {self.token}\n" if self.token else "", ""
        assert "set-option" in args and args[args.index("-S") + 1] != str(self.endpoint)
        token = args[args.index("@ccgram_topic") + 1]
        completion = args[-1].replace("#{pid}", "123")
        self.writes += 1

        async def remote():
            self.entered.set()
            await self.release.wait()
            try:
                if self.outcome == "worker-error":
                    raise RuntimeError("native worker failed")
                if self.outcome in {"api-error", "os-error"}:
                    return 1, "", self.outcome
                self.token = token
                if self.outcome == "uncorrelated-success":
                    return 0, "", ""
                if self.outcome == "wrong-pid":
                    return 0, completion.removesuffix(":123") + ":124\n", ""
                return 0, completion + "\n", ""
            finally:
                self.finished.set()

        self.remote = asyncio.create_task(remote())
        return await asyncio.shield(self.remote)


@pytest.fixture
async def native_marker(monkeypatch):
    with TemporaryDirectory(prefix="ccg-tmark-", dir="/tmp") as directory:
        endpoint = Path(directory) / "rpc"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(endpoint))
            listener.listen(32)
            identity = native_process.NativeProcessIdentity(
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
                "darwin-local",
                123,
                "darwin:100:200",
                "darwin:tmux:3.8:cdhash:" + "a" * 40,
            )
            monkeypatch.setattr(native_process.sys, "platform", "darwin")
            monkeypatch.setattr(
                native_process, "capture_native_process", lambda *_a, **_k: identity
            )
            monkeypatch.setattr(transport.tempfile, "gettempdir", lambda: directory)
            monkeypatch.setattr(
                tmux_module.shutil, "which", lambda _: "/forbidden-tmux"
            )

            async def build(outcome, marked=True):
                native = NativeMarker(endpoint, outcome)
                monkeypatch.setattr(transport, "run_marker_cli", native.run)
                mux = TmuxManager("owned-test")
                mux._server = cast(
                    libtmux.Server,
                    SimpleNamespace(
                        tmux_bin="/forbidden-tmux",
                        socket_path=str(endpoint),
                        socket_name=None,
                    ),
                )
                snapshot = await mux._marker_inventory()
                assert snapshot is not None
                owner = snapshot[1]["@1"][1]
                assert owner is not None
                native.token = f"v1:{int(not marked)}:{owner.removeprefix('tmux:')}"
                return mux, native, owner

            yield build


@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize("cancellations", [1, 3])
async def test_cancelled_mark_keeps_fence_while_native_job_can_still_apply(
    native_marker, marked, cancellations
):
    mux, native, owner = await native_marker("success", marked)
    write_guard = RecordingTopicMarkGuard()
    task = asyncio.create_task(
        mux.set_topic_mark("@1", marked, owner=owner, write_guard=write_guard)
    )
    try:
        await native.entered.wait()
        assert len(write_guard.armed) == 1
        task.cancel("original cancellation")
        await asyncio.sleep(0)
        for _ in range(cancellations - 1):
            task.cancel("repeated cancellation")
            await asyncio.sleep(0)
        async with asyncio.timeout(1):
            with pytest.raises(asyncio.CancelledError, match="original cancellation"):
                await task
        assert not native.finished.is_set()
        assert write_guard.terminal_ids == []
        native.release.set()
        await native.finished.wait()
        assert native.token.startswith(f"v1:{int(marked)}:")
        assert write_guard.terminal_ids == []
    finally:
        native.release.set()
        await asyncio.gather(
            *(item for item in (task, native.remote) if item is not None),
            return_exceptions=True,
        )


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "api-error",
        "os-error",
        "worker-error",
        "uncorrelated-success",
        "wrong-pid",
    ],
)
async def test_tmux_terminal_callback_requires_correlated_same_queue_result(
    native_marker, outcome
):
    mux, native, owner = await native_marker(outcome)
    write_guard = RecordingTopicMarkGuard()
    native.release.set()
    if outcome == "worker-error":
        with pytest.raises(RuntimeError, match="native worker failed"):
            await mux.set_topic_mark("@1", True, owner=owner, write_guard=write_guard)
    else:
        assert await mux.set_topic_mark(
            "@1", True, owner=owner, write_guard=write_guard
        ) is (outcome == "success")
    assert native.writes == 1
    assert len(write_guard.armed) == 1
    expected = [write_guard.armed[0].dispatch_id] if outcome == "success" else []
    assert write_guard.terminal_ids == expected


async def test_tmux_actual_mutation_without_guard_is_refused(native_marker):
    mux, native, owner = await native_marker("success")

    assert not await mux.set_topic_mark("@1", True, owner=owner)
    assert native.writes == 0


async def test_tmux_endpoint_scope_matches_actual_pinned_guard_witness(native_marker):
    mux, native, owner = await native_marker("success")
    scope = await mux.topic_mark_endpoint_scope()
    native.release.set()
    write_guard = RecordingTopicMarkGuard()

    assert scope is not None and native.writes == 0
    assert await mux.set_topic_mark("@1", True, owner=owner, write_guard=write_guard)
    assert write_guard.armed[0].endpoint_scope == scope
