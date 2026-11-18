"""Socket pins own only one dispatch capability, never a canonical endpoint."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import asyncio
import threading
import socket
from tempfile import TemporaryDirectory

import pytest

from ccgram.multiplexer import native_process as native
from ccgram.multiplexer import topic_mark_transport as transport
from ccgram.topic_mark_types import TopicMarkWriteWitness

HOST = "11111111-1111-4111-8111-111111111111"
BOOT = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def endpoint(monkeypatch):
    with TemporaryDirectory(prefix="ccg-pin-", dir="/tmp") as directory:
        root = Path(directory).resolve()
        path = root / "native"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(16)
            monkeypatch.setattr(transport.tempfile, "gettempdir", lambda: str(root))
            monkeypatch.setattr(native.sys, "platform", "darwin")
            identity = native.NativeProcessIdentity(
                HOST,
                BOOT,
                "darwin-local",
                123,
                "darwin:100:200",
                "darwin:tmux:3.8:cdhash:" + "a" * 40,
            )
            monkeypatch.setattr(
                native, "capture_native_process", lambda *_args, **_kw: identity
            )
            yield path


def prepare(endpoint: Path):
    witness = transport.prepare_cli_write(
        backend="tmux", binary="/unused-native", socket_path=str(endpoint)
    )
    assert witness is not None and witness.pin_path is not None
    return witness


def pin_path(witness: TopicMarkWriteWitness) -> Path:
    assert witness.pin_path is not None
    return Path(witness.pin_path)


def test_pin_keeps_old_inode_across_canonical_replacement(endpoint):
    witness = prepare(endpoint)
    pin = pin_path(witness)
    inode = pin.stat().st_ino
    endpoint.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as replacement:
        replacement.bind(str(endpoint))

        assert endpoint.stat().st_ino != inode
        assert pin.stat().st_ino == inode
        assert transport.cleanup_pin(witness) is True
        assert endpoint.exists()
        assert not pin.parent.exists()


def test_link_metadata_does_not_change_process_scope(endpoint):
    before = endpoint.stat()
    first = prepare(endpoint)
    second = prepare(endpoint)

    assert transport.process_scope(first) == transport.process_scope(second)
    assert first.dispatch_id != second.dispatch_id
    assert first.pin_path != second.pin_path
    assert endpoint.stat().st_nlink == before.st_nlink + 2
    assert transport.cleanup_pin(first) and transport.cleanup_pin(second)


@pytest.mark.parametrize(
    "kind",
    ["canonical", "wrong-directory", "symlink", "replacement", "public", "extra"],
)
def test_cleanup_refuses_unproved_pin_ownership(endpoint, kind):
    witness = prepare(endpoint)
    pin = pin_path(witness)
    if kind == "canonical":
        witness = replace(witness, pin_path=str(endpoint))
    elif kind == "wrong-directory":
        witness = replace(witness, pin_path=str(pin.parent.parent / "arbitrary" / "s"))
    elif kind == "symlink":
        pin.unlink()
        pin.symlink_to(endpoint)
    elif kind == "replacement":
        pin.unlink()
    elif kind == "public":
        pin.parent.chmod(0o755)
    elif kind == "extra":
        (pin.parent / "foreign").write_text("keep", encoding="utf-8")

    @contextmanager
    def replacement_socket():
        if kind == "replacement":
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as other:
                other.bind(str(pin))
                yield
        else:
            yield

    with replacement_socket():
        assert transport.cleanup_pin(witness) is False
        assert endpoint.exists()


def test_unknown_peer_leaves_no_pin_or_dispatch(endpoint, monkeypatch):
    monkeypatch.setattr(native, "capture_native_process", lambda *_args, **_kw: None)

    assert (
        transport.prepare_cli_write(
            backend="tmux", binary="/unused-native", socket_path=str(endpoint)
        )
        is None
    )
    assert list(endpoint.parent.glob("ccgram-mark-*")) == []
    assert endpoint.exists()


@pytest.mark.parametrize(
    ("retired", "expected"), [(False, False), (None, None), (True, True)]
)
def test_recovery_cleans_only_after_verified_retirement(
    endpoint, monkeypatch, retired, expected
):
    witness = prepare(endpoint)
    monkeypatch.setattr(native, "process_retired", lambda _: retired)

    assert transport.recover_write(witness, backend="tmux") is expected
    assert pin_path(witness).exists() is (retired is not True)


def test_recovery_keeps_fence_when_cleanup_fails(endpoint, monkeypatch):
    witness = prepare(endpoint)
    pin_path(witness).unlink()
    pin_path(witness).symlink_to(endpoint)
    monkeypatch.setattr(native, "process_retired", lambda _: True)

    assert transport.recover_write(witness, backend="tmux") is None
    assert endpoint.exists()


def test_foreign_backend_cannot_recover_owned_pin(endpoint, monkeypatch):
    witness = prepare(endpoint)
    monkeypatch.setattr(native, "process_retired", lambda _: True)

    assert transport.recover_write(witness, backend="herdr") is None
    assert pin_path(witness).exists()


class RecordingGuard:
    def __init__(self):
        self.armed: list[TopicMarkWriteWitness] = []
        self.terminal_ids: list[str] = []
        self.owner_threads: list[int] = []

    def arm(self, witness: TopicMarkWriteWitness) -> None:
        self.armed.append(witness)
        self.owner_threads.append(threading.get_ident())

    def terminal(self, dispatch_id: str) -> None:
        assert not pin_path(self.armed[-1]).parent.exists()
        self.terminal_ids.append(dispatch_id)
        self.owner_threads.append(threading.get_ident())


async def test_guard_arms_on_state_owner_before_cli_and_cleans_before_terminal(
    endpoint, monkeypatch
):
    witness = prepare(endpoint)
    guard = RecordingGuard()

    async def cli(_binary, _args):
        assert guard.armed == [witness]
        assert pin_path(witness).exists()
        return 0, "success", ""

    monkeypatch.setattr(transport, "run_marker_cli", cli)

    assert await transport.guarded_cli_write(
        witness, guard, "/unused", [], success=lambda result: result[0] == 0
    ) == (0, "success", "")
    assert guard.terminal_ids == [witness.dispatch_id]
    assert guard.owner_threads == [threading.get_ident()] * 2


async def test_failed_arm_sends_nothing_and_removes_unarmed_pin(endpoint, monkeypatch):
    witness = prepare(endpoint)

    class FailingGuard(RecordingGuard):
        def arm(self, witness):
            raise OSError("checkpoint refused")

    async def forbidden(*_):
        raise AssertionError("dispatch after failed checkpoint")

    monkeypatch.setattr(transport, "run_marker_cli", forbidden)
    with pytest.raises(OSError, match="checkpoint refused"):
        await transport.guarded_cli_write(
            witness, FailingGuard(), "/unused", [], success=lambda _: True
        )
    assert not pin_path(witness).parent.exists()


@pytest.mark.parametrize("result", [None, (1, "", "refused"), (0, "invalid", "")])
async def test_uncertain_cli_retains_pin_and_fence(endpoint, monkeypatch, result):
    witness = prepare(endpoint)
    guard = RecordingGuard()

    async def cli(*_):
        return result

    monkeypatch.setattr(transport, "run_marker_cli", cli)
    assert (
        await transport.guarded_cli_write(
            witness,
            guard,
            "/unused",
            [],
            success=lambda value: value == (0, "success", ""),
        )
        is None
    )
    assert guard.armed == [witness]
    assert guard.terminal_ids == []
    assert pin_path(witness).exists()


async def test_cancelled_cli_does_not_erase_armed_dispatch(endpoint, monkeypatch):
    witness = prepare(endpoint)
    guard = RecordingGuard()
    entered = asyncio.Event()

    async def cli(*_):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(transport, "run_marker_cli", cli)
    task = asyncio.create_task(
        transport.guarded_cli_write(
            witness, guard, "/unused", [], success=lambda _: True
        )
    )
    await entered.wait()
    task.cancel("original cancellation")
    with pytest.raises(asyncio.CancelledError, match="original cancellation"):
        await task
    assert guard.armed == [witness] and guard.terminal_ids == []
    assert pin_path(witness).exists()


async def test_pin_cleanup_failure_cannot_notify_terminal(endpoint, monkeypatch):
    witness = prepare(endpoint)
    guard = RecordingGuard()
    pin_path(witness).parent.chmod(0o755)

    async def cli(*_):
        return 0, "", ""

    monkeypatch.setattr(transport, "run_marker_cli", cli)
    assert (
        await transport.guarded_cli_write(
            witness, guard, "/unused", [], success=lambda _: True
        )
        is None
    )
    assert guard.terminal_ids == []


async def test_native_cli_cancellation_kills_only_child_without_unbounded_wait(
    monkeypatch,
):
    entered = asyncio.Event()

    class Child:
        returncode = None
        killed = False

        async def communicate(self):
            entered.set()
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True

        async def wait(self):
            await asyncio.Event().wait()

    child = Child()

    async def spawn(*_args, **_kwargs):
        return child

    monkeypatch.setattr(transport.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(transport, "_KILL_WAIT_SECONDS", 0.01)
    task = asyncio.create_task(transport.run_marker_cli("/unused", []))
    await entered.wait()
    task.cancel("original cancellation")
    async with asyncio.timeout(1):
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            await task
    assert child.killed is True


def test_recovery_refuses_symlink_cycle_in_capability_parent(endpoint, monkeypatch):
    witness = prepare(endpoint)
    loop = endpoint.parent / "loop"
    loop.symlink_to(loop)
    invalid = replace(
        witness, pin_path=str(loop / ("ccgram-mark-" + witness.dispatch_id) / "s")
    )
    monkeypatch.setattr(native, "process_retired", lambda _: True)

    assert transport.cleanup_pin(invalid) is False
    assert transport.recover_write(invalid, backend="tmux") is None
    assert endpoint.exists()
