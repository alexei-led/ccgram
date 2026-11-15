"""Connection-local CLI authority cannot outlive its owner or erase native fences."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from _topic_mark_helpers import make_witness
import ccgram.state_persistence as persistence_module
import ccgram.topic_mark_ipc as ipc_module
from ccgram.multiplexer.base import Multiplexer, WindowRef
from ccgram.state_persistence import StatePersistence
from ccgram.state_writer_lock import StateWriterLease
from ccgram.topic_mark_ipc import (
    request_topic_mark,
    start_topic_mark_server,
    topic_mark_socket_path,
)
from ccgram.topic_mark_journal import read_topic_mark_operations
from ccgram.topic_mark_types import TopicMarkWriteGuard
from ccgram.window_state_store import WindowStateStore


class _Native:
    capabilities = SimpleNamespace(name="tmux")

    def __init__(
        self, state_path: Path, *, endpoint: str = "native-A", socket_ino: int = 456
    ):
        self.state_path = state_path
        self.endpoint = endpoint
        self.origin_witness = replace(make_witness("origin"), socket_ino=socket_ino)
        self.marked = False
        self.mutations: list[bool] = []
        self.uncertain = False
        self.before_arm: asyncio.Event | None = None
        self.release_arm = asyncio.Event()

    def window(self) -> WindowRef:
        return WindowRef(
            "@1",
            "native",
            "/project",
            topic_marked=self.marked,
            topic_mark_owner="owner",
        )

    async def find_window_by_id(self, window_id: str) -> WindowRef | None:
        return self.window() if window_id == "@1" else None

    async def list_windows_for_reconciliation(self) -> list[WindowRef]:
        return [self.window()]

    async def topic_mark_endpoint_scope(self) -> str | None:
        return self.origin_witness.endpoint_scope

    async def set_topic_mark(
        self, window_id: str, marked: bool, *, owner=None, write_guard=None
    ) -> bool:
        assert window_id == "@1" and owner == "owner"
        if self.marked == marked:
            return True
        guard = cast(TopicMarkWriteGuard, write_guard)
        assert guard is not None
        if self.before_arm is not None:
            self.before_arm.set()
            await self.release_arm.wait()
        witness = replace(
            self.origin_witness, dispatch_id=f"dispatch-{len(self.mutations) + 1}"
        )
        guard.arm(witness)
        (saved,) = read_topic_mark_operations(self.state_path)
        assert saved.attempt is not None
        assert saved.attempt.witness == witness
        assert saved.attempt.desired is marked
        self.mutations.append(marked)
        if self.uncertain:
            return False
        self.marked = marked
        guard.terminal(witness.dispatch_id)
        return True


@dataclass
class _Endpoint:
    state_path: Path
    store: WindowStateStore
    native: _Native
    flush: Callable[[], None]


@pytest.fixture
async def endpoint():
    with TemporaryDirectory(prefix="ccg-ipc-", dir="/tmp") as directory:
        state_path = Path(directory) / "state.json"
        store = WindowStateStore(
            schedule_save=lambda: None, on_hookless_provider_switch=lambda _wid: None
        )
        persistence = StatePersistence(
            state_path,
            lambda: {"unrelated": {"keep": True}, **store.topic_mark_journal_to_dict()},
        )

        def flush():
            persistence.flush(strict=True)

        flush()
        native = _Native(state_path)
        with StateWriterLease(state_path) as lease:
            server = await start_topic_mark_server(
                lease, cast(Multiplexer, native), store, flush
            )
            try:
                yield _Endpoint(state_path, store, native, flush)
            finally:
                await server.close()


async def _send(writer: asyncio.StreamWriter, frame: dict) -> None:
    writer.write(json.dumps(frame).encode() + b"\n")
    await writer.drain()


def _prepare_frame(endpoint: _Endpoint, desired: bool) -> dict:
    return {
        "version": 1,
        "phase": "prepare",
        "scope": str(endpoint.state_path.resolve()),
        "backend": "tmux",
        "endpoint_scope": endpoint.native.origin_witness.endpoint_scope,
        "window_id": "@1",
        "owner": "owner",
        "desired": desired,
    }


async def _prepare(endpoint: _Endpoint, desired: bool):
    reader, writer = await asyncio.open_unix_connection(
        topic_mark_socket_path(endpoint.state_path)
    )
    await _send(writer, _prepare_frame(endpoint, desired))
    response = json.loads(await reader.readline())
    assert response["phase"] == "prepared", response
    assert isinstance(response["token"], str) and response["token"]
    return reader, writer, response["token"]


async def _request(endpoint: _Endpoint, marked: bool) -> dict[str, object]:
    scope = await endpoint.native.topic_mark_endpoint_scope()
    assert scope is not None
    return await request_topic_mark(
        endpoint.state_path,
        "tmux",
        endpoint.native.window(),
        marked,
        endpoint_scope=scope,
    )


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(ConnectionError):
        await writer.wait_closed()


async def test_live_request_checkpoints_before_native_send_and_preserves_raw_state(
    endpoint,
):
    result = await _request(endpoint, True)

    assert result["target"] == "@1" and result["marked"] is True
    assert endpoint.native.mutations == [True]
    assert read_topic_mark_operations(endpoint.state_path) == ()
    assert json.loads(endpoint.state_path.read_text(encoding="utf-8"))["unrelated"] == {
        "keep": True
    }


async def test_disconnected_prepare_has_no_retryable_intent_or_native_effect(endpoint):
    original = endpoint.state_path.read_bytes()
    reader, writer, _token = await _prepare(endpoint, True)
    try:
        assert endpoint.store.topic_mark_journal_to_dict() == {}
        assert endpoint.state_path.read_bytes() == original
    finally:
        await _close(writer)

    result = await _request(endpoint, False)

    assert result["marked"] is False
    assert endpoint.native.mutations == []
    assert read_topic_mark_operations(endpoint.state_path) == ()
    remaining = await reader.read()
    assert not remaining or all(
        json.loads(line).get("phase") != "result" for line in remaining.splitlines()
    )


async def test_commit_token_cannot_be_replayed_on_a_second_connection(endpoint):
    _reader, original, token = await _prepare(endpoint, True)
    reader, replay = await asyncio.open_unix_connection(
        topic_mark_socket_path(endpoint.state_path)
    )
    try:
        await _send(replay, {"version": 1, "phase": "commit", "token": token})
        async with asyncio.timeout(2):
            response = await reader.readline()
        assert not response or json.loads(response).get("phase") != "result"
        assert endpoint.native.mutations == []
        assert read_topic_mark_operations(endpoint.state_path) == ()
    finally:
        await _close(replay)
        await _close(original)


async def test_duplicate_commit_frames_launch_at_most_one_native_attempt(endpoint):
    reader, writer, token = await _prepare(endpoint, True)
    frame = (
        json.dumps({"version": 1, "phase": "commit", "token": token}).encode() + b"\n"
    )
    try:
        writer.write(frame + frame)
        await writer.drain()
        response = json.loads(await reader.readline())
        assert response["phase"] == "result" and response["marked"] is True
        assert endpoint.native.mutations == [True]
        assert read_topic_mark_operations(endpoint.state_path) == ()
    finally:
        await _close(writer)


@pytest.mark.parametrize("desired", [False, True])
async def test_uncertain_commit_blocks_same_and_opposing_writes(endpoint, desired):
    endpoint.native.uncertain = True
    with pytest.raises(RuntimeError):
        await _request(endpoint, True)
    (armed,) = read_topic_mark_operations(endpoint.state_path)
    assert armed.attempt is not None and armed.attempt.desired is True
    assert endpoint.native.marked is False

    with pytest.raises(RuntimeError):
        await _request(endpoint, desired)

    assert endpoint.native.mutations == [True]
    assert read_topic_mark_operations(endpoint.state_path) == (armed,)


async def test_expired_original_commit_cannot_follow_successful_noop_off(
    endpoint, monkeypatch
):
    monkeypatch.setattr(ipc_module, "_FRAME_TIMEOUT", 0.2)
    reader, writer, token = await _prepare(endpoint, True)
    try:
        off = await _request(endpoint, False)
        assert off["marked"] is False
        with contextlib.suppress(ConnectionError, OSError):
            await _send(writer, {"version": 1, "phase": "commit", "token": token})
        try:
            response = await reader.readline()
        except ConnectionError, OSError:
            response = b""
        assert not response or json.loads(response).get("phase") != "result"
        assert endpoint.native.mutations == []
        assert endpoint.native.marked is False
        assert read_topic_mark_operations(endpoint.state_path) == ()
    finally:
        await _close(writer)


@pytest.mark.parametrize("checkpoint", ["arm", "terminal"])
async def test_failed_checkpoint_never_authorizes_another_native_write(
    endpoint, monkeypatch, checkpoint
):
    atomic_write = persistence_module.atomic_write_json

    def fail_checkpoint(path, state, *args, **kwargs):
        rows = state.get("topic_mark_operations", [])
        prior = read_topic_mark_operations(endpoint.state_path)
        arming = any(row["attempt"] is not None for row in rows)
        finishing = bool(prior and prior[0].attempt is not None) and not arming
        if (checkpoint == "arm" and arming) or (checkpoint == "terminal" and finishing):
            raise OSError("checkpoint unavailable")
        return atomic_write(path, state, *args, **kwargs)

    monkeypatch.setattr(persistence_module, "atomic_write_json", fail_checkpoint)

    with pytest.raises(RuntimeError):
        await _request(endpoint, True)

    assert endpoint.native.mutations == ([] if checkpoint == "arm" else [True])
    (current,) = endpoint.store.topic_mark_operations.values()
    assert current.attempt is not None
    if checkpoint == "terminal":
        assert read_topic_mark_operations(endpoint.state_path) == (current,)
    with pytest.raises(RuntimeError):
        await _request(endpoint, True)
    assert endpoint.native.mutations == ([] if checkpoint == "arm" else [True])


async def test_bot_off_queued_after_prepare_is_not_overwritten_by_old_commit(endpoint):
    reader, writer, token = await _prepare(endpoint, True)
    try:
        latest = endpoint.store.queue_topic_mark("@1", "owner", False)
        endpoint.flush()
        await _send(writer, {"version": 1, "phase": "commit", "token": token})
        response = await reader.readline()
        assert not response or json.loads(response).get("phase") != "result"
        assert endpoint.native.mutations == []
        assert read_topic_mark_operations(endpoint.state_path) == (latest,)
    finally:
        await _close(writer)


@pytest.mark.parametrize("origin_changed", [False, True])
async def test_arm_rechecks_latest_intent_after_awaited_native_prechecks(
    endpoint, origin_changed
):
    endpoint.native.before_arm = asyncio.Event()
    request = asyncio.create_task(_request(endpoint, True))
    try:
        async with asyncio.timeout(2):
            await endpoint.native.before_arm.wait()
        latest = endpoint.store.queue_topic_mark("@1", "owner", False)
        endpoint.flush()
        if origin_changed:
            endpoint.native.origin_witness = replace(
                endpoint.native.origin_witness, socket_ino=457
            )
        endpoint.native.release_arm.set()

        with pytest.raises(RuntimeError):
            await request

        assert endpoint.native.mutations == []
        assert read_topic_mark_operations(endpoint.state_path) == (latest,)
    finally:
        endpoint.native.release_arm.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.parametrize(
    "invalid",
    [
        {"scope": "/other/state.json"},
        {"backend": "herdr"},
        {"owner": "different-owner"},
        {"desired": 1},
        {"version": True},
        {"unexpected": "field"},
        {"endpoint_scope": None},
        {"endpoint_scope": False},
        {"endpoint_scope": ""},
        {"endpoint_scope": "a" * 63},
        {"endpoint_scope": "A" * 64},
    ],
)
async def test_invalid_prepare_has_no_commit_authority_or_state_effect(
    endpoint, invalid
):
    original = endpoint.state_path.read_bytes()
    reader, writer = await asyncio.open_unix_connection(
        topic_mark_socket_path(endpoint.state_path)
    )
    try:
        await _send(writer, _prepare_frame(endpoint, True) | invalid)
        async with asyncio.timeout(2):
            response = await reader.readline()
        assert not response or json.loads(response).get("phase") not in {
            "prepared",
            "result",
        }
        assert endpoint.native.mutations == []
        assert endpoint.state_path.read_bytes() == original
    finally:
        await _close(writer)


@pytest.mark.parametrize("initial_marked", [False, True], ids=["write", "noop"])
@pytest.mark.parametrize(
    "preexisting", [False, True], ids=["new-intent", "existing-intent"]
)
async def test_live_cli_wrong_native_origin_cannot_use_copied_owner(
    endpoint, monkeypatch, initial_marked, preexisting
):
    from ccgram.topic_mark_cmd import _topic_action

    monkeypatch.setenv("CCGRAM_DIR", str(endpoint.state_path.parent))
    endpoint.native.marked = initial_marked
    requested = _Native(endpoint.state_path, endpoint="native-B", socket_ino=457)
    requested.marked = initial_marked
    if preexisting:
        endpoint.store.queue_topic_mark("@1", "owner", True)
        endpoint.flush()
    original = endpoint.state_path.read_bytes()
    journal_before = endpoint.store.topic_mark_journal_to_dict()
    error = None
    try:
        await _topic_action(cast(Multiplexer, requested), "mark", "@1")
    except (RuntimeError, ValueError) as exc:
        error = exc

    assert endpoint.native.mutations == []
    assert requested.mutations == []
    assert endpoint.native.marked is initial_marked
    assert requested.marked is initial_marked
    assert endpoint.state_path.read_bytes() == original
    assert endpoint.store.topic_mark_journal_to_dict() == journal_before
    assert error is not None and str(error)


@pytest.mark.parametrize("configured_endpoint", ["native-A", "native-alias-A"])
async def test_live_cli_matching_native_origin_uses_daemon_writer(
    endpoint, monkeypatch, configured_endpoint
):
    from ccgram.topic_mark_cmd import _topic_action

    monkeypatch.setenv("CCGRAM_DIR", str(endpoint.state_path.parent))
    requested = _Native(endpoint.state_path, endpoint=configured_endpoint)

    result = await _topic_action(cast(Multiplexer, requested), "mark", "@1")

    assert result == {"target": "@1", "marked": True}
    assert endpoint.native.mutations == [True]
    assert requested.mutations == []
    assert read_topic_mark_operations(endpoint.state_path) == ()


async def test_committed_lookup_deadline_releases_undispatched_authority(
    endpoint, monkeypatch
):
    monkeypatch.setattr(ipc_module, "_RPC_TIMEOUT", 0.2)
    reader, writer, token = await _prepare(endpoint, True)
    started, release = asyncio.Event(), asyncio.Event()
    original_find = endpoint.native.find_window_by_id
    original_state = endpoint.state_path.read_bytes()

    async def blocked_lookup(_window_id):
        started.set()
        await release.wait()
        return None

    monkeypatch.setattr(endpoint.native, "find_window_by_id", blocked_lookup)
    try:
        await _send(writer, {"version": 1, "phase": "commit", "token": token})
        async with asyncio.timeout(1):
            await started.wait()
            response = await reader.readline()
        assert not response or json.loads(response).get("phase") != "result"
        assert endpoint.native.mutations == []
        assert endpoint.store.topic_mark_journal_to_dict() == {}
        assert endpoint.state_path.read_bytes() == original_state
        monkeypatch.setattr(endpoint.native, "find_window_by_id", original_find)
        result = await _request(endpoint, False)
        assert result["marked"] is False
        assert endpoint.native.mutations == []
    finally:
        release.set()
        await _close(writer)


async def test_committed_native_deadline_releases_lock_but_preserves_armed_fence(
    endpoint, monkeypatch
):
    monkeypatch.setattr(ipc_module, "_RPC_TIMEOUT", 0.2)
    reader, writer, token = await _prepare(endpoint, True)
    started, release, cancelling = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked_native(_window_id, marked, *, owner, write_guard):
        assert owner == "owner"
        write_guard.arm(make_witness("armed-timeout"))
        endpoint.native.mutations.append(marked)
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise
        return False

    monkeypatch.setattr(endpoint.native, "set_topic_mark", blocked_native)
    try:
        await _send(writer, {"version": 1, "phase": "commit", "token": token})
        async with asyncio.timeout(1):
            await started.wait()
            await cancelling.wait()
            (during_cleanup,) = read_topic_mark_operations(endpoint.state_path)
            assert during_cleanup.attempt is not None
            release.set()
            response = await reader.readline()
        assert not response or json.loads(response).get("phase") != "result"
        (armed,) = read_topic_mark_operations(endpoint.state_path)
        assert armed.attempt is not None
        assert armed.attempt.witness.dispatch_id == "armed-timeout"
        with pytest.raises(RuntimeError):
            await _request(endpoint, False)
        assert endpoint.native.mutations == [True]
        assert read_topic_mark_operations(endpoint.state_path) == (armed,)
    finally:
        release.set()
        await _close(writer)


@pytest.mark.parametrize("unknown_side", ["client", "daemon"])
async def test_unknown_native_origin_has_no_mutation_or_manual_authority(
    endpoint, monkeypatch, unknown_side
):
    from ccgram.topic_mark_cmd import _topic_action

    monkeypatch.setenv("CCGRAM_DIR", str(endpoint.state_path.parent))
    requested = _Native(endpoint.state_path)
    unknown = requested if unknown_side == "client" else endpoint.native
    monkeypatch.setattr(
        unknown, "topic_mark_endpoint_scope", AsyncMock(return_value=None)
    )
    original = endpoint.state_path.read_bytes()

    with pytest.raises((RuntimeError, ValueError)):
        await _topic_action(cast(Multiplexer, requested), "mark", "@1")

    assert endpoint.native.mutations == requested.mutations == []
    assert endpoint.store.topic_mark_journal_to_dict() == {}
    assert endpoint.state_path.read_bytes() == original


async def test_client_inventory_origin_change_is_rejected_before_daemon_native_read(
    endpoint, monkeypatch
):
    from ccgram.topic_mark_cmd import _topic_action

    monkeypatch.setenv("CCGRAM_DIR", str(endpoint.state_path.parent))
    requested = _Native(endpoint.state_path)
    daemon_scope = AsyncMock(wraps=endpoint.native.topic_mark_endpoint_scope)
    monkeypatch.setattr(endpoint.native, "topic_mark_endpoint_scope", daemon_scope)

    async def inventory():
        window = requested.window()
        requested.origin_witness = replace(requested.origin_witness, socket_ino=457)
        return [window]

    monkeypatch.setattr(requested, "list_windows_for_reconciliation", inventory)
    original = endpoint.state_path.read_bytes()

    with pytest.raises((RuntimeError, ValueError)):
        await _topic_action(cast(Multiplexer, requested), "mark", "@1")

    daemon_scope.assert_not_awaited()
    assert endpoint.native.mutations == requested.mutations == []
    assert endpoint.store.topic_mark_journal_to_dict() == {}
    assert endpoint.state_path.read_bytes() == original


async def test_origin_change_before_native_arm_leaves_no_new_retryable_on(endpoint):
    endpoint.native.before_arm = asyncio.Event()
    original = endpoint.state_path.read_bytes()
    request = asyncio.create_task(_request(endpoint, True))
    try:
        async with asyncio.timeout(2):
            await endpoint.native.before_arm.wait()
        endpoint.native.origin_witness = replace(
            endpoint.native.origin_witness, socket_ino=457
        )
        endpoint.native.release_arm.set()
        with pytest.raises(RuntimeError):
            await request

        assert endpoint.native.mutations == []
        assert endpoint.store.topic_mark_journal_to_dict() == {}
        assert endpoint.state_path.read_bytes() == original
    finally:
        endpoint.native.release_arm.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.parametrize("change", ["origin", "newer-off"])
async def test_noop_rechecks_native_origin_after_committed_target_lookup(
    endpoint, monkeypatch, change
):
    endpoint.native.marked = True
    reader, writer, token = await _prepare(endpoint, True)
    started, release = asyncio.Event(), asyncio.Event()

    async def lookup(_window_id):
        started.set()
        await release.wait()
        return endpoint.native.window()

    monkeypatch.setattr(endpoint.native, "find_window_by_id", lookup)
    try:
        await _send(writer, {"version": 1, "phase": "commit", "token": token})
        async with asyncio.timeout(1):
            await started.wait()
            expected_operations = ()
            if change == "origin":
                endpoint.native.origin_witness = replace(
                    endpoint.native.origin_witness, socket_ino=457
                )
            else:
                expected_operations = (
                    endpoint.store.queue_topic_mark("@1", "owner", False),
                )
                endpoint.flush()
            expected_state = endpoint.state_path.read_bytes()
            release.set()
            response = await reader.readline()
        assert not response or json.loads(response).get("phase") != "result"
        assert endpoint.native.mutations == []
        assert endpoint.native.marked is True
        assert read_topic_mark_operations(endpoint.state_path) == expected_operations
        assert endpoint.state_path.read_bytes() == expected_state
    finally:
        release.set()
        await _close(writer)


@pytest.mark.parametrize("changed_scope", ["different", "unknown"])
async def test_post_native_scope_refusal_preserves_completed_attempt_as_quarantine(
    endpoint, monkeypatch, changed_scope
):
    original_scope = endpoint.native.origin_witness.endpoint_scope
    native_write = endpoint.native.set_topic_mark

    async def write_then_change(*args, **kwargs):
        verified = await native_write(*args, **kwargs)
        if changed_scope == "different":
            endpoint.native.origin_witness = replace(
                endpoint.native.origin_witness, socket_ino=457
            )
        else:
            monkeypatch.setattr(
                endpoint.native,
                "topic_mark_endpoint_scope",
                AsyncMock(return_value=None),
            )
        return verified

    monkeypatch.setattr(endpoint.native, "set_topic_mark", write_then_change)

    with pytest.raises(RuntimeError):
        await _request(endpoint, True)

    assert endpoint.native.mutations == [True]
    (operation,) = read_topic_mark_operations(endpoint.state_path)
    assert operation.attempt is not None
    assert operation.attempt.witness.endpoint_scope == original_scope
    assert endpoint.store.topic_mark_operations[operation.key] == operation


async def test_missing_native_scope_cannot_use_unpublished_old_rpc_protocol(endpoint):
    reader, writer = await asyncio.open_unix_connection(
        topic_mark_socket_path(endpoint.state_path)
    )
    frame = _prepare_frame(endpoint, True)
    del frame["endpoint_scope"]
    original = endpoint.state_path.read_bytes()
    try:
        await _send(writer, frame)
        async with asyncio.timeout(1):
            response = await reader.readline()
        assert not response or json.loads(response).get("phase") not in {
            "prepared",
            "result",
        }
        assert endpoint.native.mutations == []
        assert endpoint.store.topic_mark_journal_to_dict() == {}
        assert endpoint.state_path.read_bytes() == original
    finally:
        await _close(writer)
