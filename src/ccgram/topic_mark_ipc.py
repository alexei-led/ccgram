"""Private credential-free marker RPC over one lock-held prepare/commit stream."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
from typing import TYPE_CHECKING
from uuid import uuid4

from .state_writer_lock import StateWriterLease
from .topic_mark_journal import has_topic_mark_conflict
from .topic_selection import (
    _selection_lock,
    is_topic_mark_endpoint_scope,
    require_topic_mark_endpoint_scope,
    set_topic_selection_locked,
)

if TYPE_CHECKING:
    from .multiplexer.base import Multiplexer, WindowRef
    from .window_state_store import WindowStateStore

logger = logging.getLogger(__name__)
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_SOCKET_MODE = 0o600
_MAX_FRAME = 4096
_MAX_SOCKET_BYTES = 103
_MAX_CLIENTS = 16
_FRAME_TIMEOUT = 3.0
_PREPARE_TIMEOUT = 15.0
_RPC_TIMEOUT = 45.0
_CLOSE_TIMEOUT = 2.0
_PROTOCOL_VERSION = 1


def topic_mark_socket_path(state_path: Path) -> Path:
    scope = hashlib.sha256(os.fsencode(state_path.resolve())).hexdigest()
    directory = Path("/tmp") / f"ccgram-topic-rpc-{os.getuid()}"
    path = directory / (scope[:32] + ".sock")
    if len(os.fsencode(path)) > _MAX_SOCKET_BYTES:
        raise RuntimeError("Configured state path is too long for private marker RPC")
    return path


def _private_directory(path: Path, *, create: bool) -> None:
    if create:
        path.mkdir(mode=_PRIVATE_DIRECTORY_MODE, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != _PRIVATE_DIRECTORY_MODE
    ):
        raise RuntimeError("Unsafe private marker RPC directory")


def _socket_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if (
        not stat.S_ISSOCK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != _PRIVATE_SOCKET_MODE
        or info.st_nlink != 1
    ):
        raise RuntimeError("Unsafe private marker RPC socket")
    return info.st_dev, info.st_ino


async def _read_frame(
    reader: asyncio.StreamReader, *, timeout: float | None = None
) -> dict[str, object]:
    async with asyncio.timeout(_FRAME_TIMEOUT if timeout is None else timeout):
        line = await reader.readline()
    if not line.endswith(b"\n") or len(line) > _MAX_FRAME:
        raise ValueError("Invalid marker RPC frame")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected marker RPC object")
    if value.get("phase") == "error":
        raise RuntimeError(str(value.get("message", "Marker RPC refused")))
    return value


async def _send_frame(writer: asyncio.StreamWriter, value: dict[str, object]) -> None:
    frame = json.dumps(value, ensure_ascii=True).encode("utf-8") + b"\n"
    if len(frame) > _MAX_FRAME:
        raise ValueError("Marker RPC frame is too large")
    writer.write(frame)
    async with asyncio.timeout(_FRAME_TIMEOUT):
        await writer.drain()


def _abort(writer: asyncio.StreamWriter) -> None:
    writer.close()


class TopicMarkServer:
    def __init__(
        self,
        lease: StateWriterLease,
        backend: Multiplexer,
        store: WindowStateStore,
        flush: Callable[[], None],
    ) -> None:
        self.lease = lease
        self.backend = backend
        self.store = store
        self.flush = flush
        self.socket_path = topic_mark_socket_path(lease.state_path)
        self.server: asyncio.Server | None = None
        self._identity: tuple[int, int] | None = None
        self._tasks: set[asyncio.Task[object]] = set()
        self._closing = False

    async def start(self) -> None:
        if not self.lease.held:
            raise RuntimeError("Marker RPC requires the state writer lease")
        _private_directory(self.socket_path.parent, create=True)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            _socket_identity(self.socket_path)
            self.socket_path.unlink()
        self.server = await asyncio.start_unix_server(
            self._handle,
            path=self.socket_path,
            limit=_MAX_FRAME,
            start_serving=False,
            cleanup_socket=False,
        )
        try:
            self.socket_path.chmod(_PRIVATE_SOCKET_MODE)
            self._identity = _socket_identity(self.socket_path)
            await self.server.start_serving()
        except BaseException:
            self.server.close()
            raise

    def _validate_prepare(
        self, request: dict[str, object]
    ) -> tuple[str, str, bool, str]:
        if set(request) != {
            "version",
            "phase",
            "scope",
            "backend",
            "window_id",
            "owner",
            "desired",
            "endpoint_scope",
        } or (
            type(request["version"]) is not int
            or request["version"] != _PROTOCOL_VERSION
            or request["phase"] != "prepare"
            or request["scope"] != str(self.lease.state_path)
            or request["backend"] != self.backend.capabilities.name
        ):
            raise ValueError("Marker RPC state/backend scope mismatch")
        window_id, owner, desired, endpoint_scope = (
            request["window_id"],
            request["owner"],
            request["desired"],
            request["endpoint_scope"],
        )
        if (
            not isinstance(window_id, str)
            or not window_id
            or not isinstance(owner, str)
            or not owner
            or not isinstance(desired, bool)
            or not isinstance(endpoint_scope, str)
            or not is_topic_mark_endpoint_scope(endpoint_scope)
        ):
            raise ValueError("Marker RPC needs an exact target and native owner")
        return window_id, owner, desired, endpoint_scope

    async def _prepare_window(
        self, window_id: str, owner: str, desired: bool, endpoint_scope: str
    ) -> WindowRef:
        if self._closing or not self.lease.held:
            raise RuntimeError("Marker RPC is stopping")
        await require_topic_mark_endpoint_scope(self.backend, endpoint_scope)
        window = await self.backend.find_window_by_id(window_id)
        await require_topic_mark_endpoint_scope(self.backend, endpoint_scope)
        if (
            window is None
            or not window.matches(window_id)
            or window.topic_mark_owner != owner
            or not window.topic_markable
        ):
            raise RuntimeError("Marker target changed; refresh topic status")
        if has_topic_mark_conflict(
            self.store.topic_mark_operations.values(), window_id, owner, desired
        ):
            raise RuntimeError("Pending native marker work owns this target")
        return window

    async def _handle_locked(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        window_id: str,
        owner: str,
        desired: bool,
        endpoint_scope: str,
    ) -> None:
        lock = _selection_lock(window_id, owner=owner)
        async with asyncio.timeout(_PREPARE_TIMEOUT):
            await lock.acquire()
        try:
            async with asyncio.timeout(_PREPARE_TIMEOUT):
                await self._prepare_window(window_id, owner, desired, endpoint_scope)
                token = uuid4().hex
                await _send_frame(writer, {"phase": "prepared", "token": token})
                commit = await _read_frame(reader)
            if (
                commit
                != {"version": _PROTOCOL_VERSION, "phase": "commit", "token": token}
                or type(commit.get("version")) is not int
            ):
                raise ValueError("Invalid connection-local marker commit")
            async with asyncio.timeout(_RPC_TIMEOUT):
                window = await self._prepare_window(
                    window_id, owner, desired, endpoint_scope
                )
                verified = await set_topic_selection_locked(
                    window,
                    desired,
                    backend=self.backend,
                    store=self.store,
                    flush=self.flush,
                    endpoint_scope=endpoint_scope,
                )
                if not verified:
                    raise RuntimeError(
                        "Marker update is unconfirmed; inspect topic status"
                    )
                await _send_frame(
                    writer,
                    {"phase": "result", "target": window.window_id, "marked": desired},
                )
        except Exception as exc:
            logger.exception("Marker RPC execution failed")
            await _send_frame(writer, {"phase": "error", "message": str(exc)})
        finally:
            # No parser or token survives release of the native-owner lock.
            _abort(writer)
            lock.release()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is None or self._closing or len(self._tasks) >= _MAX_CLIENTS:
            _abort(writer)
            return
        self._tasks.add(task)
        try:
            request = await _read_frame(reader)
            window_id, owner, desired, endpoint_scope = self._validate_prepare(request)
            await self._handle_locked(
                reader, writer, window_id, owner, desired, endpoint_scope
            )
        except Exception as exc:
            logger.exception("Marker RPC request failed")
            with contextlib.suppress(OSError, RuntimeError, TimeoutError, ValueError):
                await _send_frame(writer, {"phase": "error", "message": str(exc)})
        finally:
            _abort(writer)
            self._tasks.discard(task)

    async def close(self) -> None:
        self._closing = True
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        tasks = set(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=_CLOSE_TIMEOUT)
        if self._identity is not None:
            with contextlib.suppress(FileNotFoundError):
                if _socket_identity(self.socket_path) != self._identity:
                    raise RuntimeError("Marker RPC endpoint changed; refusing cleanup")
                self.socket_path.unlink()


async def start_topic_mark_server(
    lease: StateWriterLease,
    backend: Multiplexer,
    store: WindowStateStore,
    flush: Callable[[], None],
) -> TopicMarkServer:
    server = TopicMarkServer(lease, backend, store, flush)
    await server.start()
    return server


async def request_topic_mark(
    state_path: Path,
    backend_name: str,
    window: WindowRef,
    marked: bool,
    *,
    endpoint_scope: str,
) -> dict[str, object]:
    if not is_topic_mark_endpoint_scope(endpoint_scope):
        raise ValueError("Native marker endpoint is unconfirmed")
    path = topic_mark_socket_path(state_path)
    _private_directory(path.parent, create=False)
    _socket_identity(path)
    async with asyncio.timeout(_RPC_TIMEOUT):
        reader, writer = await asyncio.open_unix_connection(path, limit=_MAX_FRAME)
        try:
            await _send_frame(
                writer,
                {
                    "version": _PROTOCOL_VERSION,
                    "phase": "prepare",
                    "scope": str(state_path.resolve()),
                    "backend": backend_name,
                    "window_id": window.window_id,
                    "owner": window.topic_mark_owner,
                    "desired": marked,
                    "endpoint_scope": endpoint_scope,
                },
            )
            prepared = await _read_frame(reader, timeout=_PREPARE_TIMEOUT)
            token = prepared.get("token")
            if (
                set(prepared) != {"phase", "token"}
                or prepared["phase"] != "prepared"
                or not isinstance(token, str)
                or not token
            ):
                raise ValueError("Invalid marker RPC preparation")
            await _send_frame(
                writer,
                {"version": _PROTOCOL_VERSION, "phase": "commit", "token": token},
            )
            result = await _read_frame(reader, timeout=_RPC_TIMEOUT)
            if (
                result
                != {
                    "phase": "result",
                    "target": window.window_id,
                    "marked": marked,
                }
                or type(result.get("marked")) is not bool
            ):
                raise ValueError("Unconfirmed marker RPC result")
            return {"target": window.window_id, "marked": marked}
        finally:
            _abort(writer)
