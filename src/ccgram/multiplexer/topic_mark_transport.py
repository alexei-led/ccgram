"""Owned socket pins bind marker CLIs to one proved native executor."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import os
from pathlib import Path
import socket
import stat
import tempfile
from uuid import uuid4

from ..topic_mark_types import TopicMarkWriteGuard, TopicMarkWriteWitness
from . import native_process

_PIN_DIRECTORY_PREFIX = "ccgram-mark-"
_PIN_SOCKET_NAME = "s"
_MAX_SOCKET_PATH_BYTES = 103
_CLI_TIMEOUT_SECONDS = 10
_KILL_WAIT_SECONDS = 0.5
_PRIVATE_MODE = 0o700


def witness_for_connection(
    connection: socket.socket,
    *,
    backend: str,
    binary: str,
    socket_path: str,
    dispatch_id: str | None = None,
    pin_path: str | None = None,
) -> TopicMarkWriteWitness | None:
    identity = native_process.capture_native_process(
        connection, backend=backend, binary=binary
    )
    if identity is None:
        return None
    try:
        info = Path(socket_path).lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            return None
        witness = identity.witness(
            backend, info, dispatch_id=dispatch_id or uuid4().hex, pin_path=pin_path
        )
        return witness if native_process.valid_saved_witness(witness) else None
    except OSError, ValueError:
        return None


def process_scope(witness: TopicMarkWriteWitness) -> str:
    return native_process.NativeProcessIdentity(
        witness.host_id,
        witness.boot_id,
        witness.namespace_id,
        witness.native_pid,
        witness.native_start,
        witness.code_id,
    ).scope


def _pin_path_is_owned_capability(witness: TopicMarkWriteWitness) -> bool:
    if not native_process.valid_saved_witness(witness):
        return False
    if witness.pin_path is None:
        return True
    try:
        path = Path(witness.pin_path)
        return (
            path.name == _PIN_SOCKET_NAME
            and path.parent.name == _PIN_DIRECTORY_PREFIX + witness.dispatch_id
            and path.is_absolute()
            and str(path) == witness.pin_path
            and len(os.fsencode(path)) <= _MAX_SOCKET_PATH_BYTES
            and path.parent.parent.resolve() == path.parent.parent
        )
    except OSError, ValueError, RuntimeError:
        return False


def cleanup_pin(witness: TopicMarkWriteWitness) -> bool:
    """Delete only the recorded nonce capability, never a native/canonical path."""
    if not _pin_path_is_owned_capability(witness):
        return False
    if witness.pin_path is None:
        return True
    path = Path(witness.pin_path)
    try:
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        _clean_pin_directory(fd, path, witness)
        return True
    except OSError, ValueError:
        return False
    finally:
        os.close(fd)


def _clean_pin_directory(fd: int, path: Path, witness: TopicMarkWriteWitness) -> None:
    directory = os.fstat(fd)
    if (
        directory.st_uid != os.getuid()
        or stat.S_IMODE(directory.st_mode) != _PRIVATE_MODE
    ):
        raise ValueError("Unproved pin directory ownership")
    if set(os.listdir(fd)) - {_PIN_SOCKET_NAME}:
        raise ValueError("Foreign pin directory contents")
    try:
        current = os.stat(_PIN_SOCKET_NAME, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        current = None
    if current is not None:
        if (
            not stat.S_ISSOCK(current.st_mode)
            or current.st_uid != os.getuid()
            or (current.st_dev, current.st_ino)
            != (witness.socket_dev, witness.socket_ino)
        ):
            raise ValueError("Pin socket identity changed")
        os.unlink(_PIN_SOCKET_NAME, dir_fd=fd)
    parent = path.parent.lstat()
    if (parent.st_dev, parent.st_ino) != (directory.st_dev, directory.st_ino):
        raise ValueError("Pin directory identity changed")
    os.rmdir(path.parent)


def _create_pin(endpoint: Path, dispatch_id: str) -> Path | None:
    original = endpoint.lstat()
    if not stat.S_ISSOCK(original.st_mode) or original.st_uid != os.getuid():
        return None
    for base in (Path(tempfile.gettempdir()).resolve(), endpoint.parent):
        directory = base / (_PIN_DIRECTORY_PREFIX + dispatch_id)
        pin = directory / _PIN_SOCKET_NAME
        if (
            base.stat().st_dev != original.st_dev
            or len(os.fsencode(pin)) > _MAX_SOCKET_PATH_BYTES
        ):
            continue
        try:
            directory.mkdir(mode=_PRIVATE_MODE)
        except FileExistsError:
            return None
        linked = False
        prepared = False
        try:
            os.link(endpoint, pin, follow_symlinks=False)
            linked = True
            current = pin.lstat()
            if (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino):
                prepared = True
                return pin
        except OSError:
            pass
        finally:
            if not prepared:
                with contextlib.suppress(OSError):
                    if linked:
                        pin.unlink()
                    directory.rmdir()
    return None


def prepare_cli_write(
    *, backend: str, binary: str, socket_path: str
) -> TopicMarkWriteWitness | None:
    """Prepare only; the state-owner guard must still arm before spawning."""
    pin = None
    try:
        endpoint = Path(socket_path).resolve(strict=True)
        dispatch_id = uuid4().hex
        pin = _create_pin(endpoint, dispatch_id)
        if pin is None:
            return None
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            connection.connect(str(pin))
            witness = witness_for_connection(
                connection,
                backend=backend,
                binary=binary,
                socket_path=str(pin),
                dispatch_id=dispatch_id,
                pin_path=str(pin),
            )
        if witness is not None:
            return witness
    except OSError, ValueError:
        pass
    if pin is not None:
        # This path was exclusively created in this call and has never dispatched.
        with contextlib.suppress(OSError):
            pin.unlink()
            pin.parent.rmdir()
    return None


def recover_write(witness: TopicMarkWriteWitness, *, backend: str) -> bool | None:
    if witness.backend != backend or not _pin_path_is_owned_capability(witness):
        return None
    retired = native_process.process_retired(witness)
    if retired is not True:
        return retired
    return True if cleanup_pin(witness) else None


async def run_marker_cli(binary: str, args: list[str]) -> tuple[int, str, str] | None:
    """Bounded child lifetime; killing this child is not remote completion proof."""
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            binary,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(_CLI_TIMEOUT_SECONDS):
            output, errors = await process.communicate()
        return process.returncode or 0, output.decode(), errors.decode()
    except OSError, TimeoutError, UnicodeError:
        return None
    finally:
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(OSError, TimeoutError, asyncio.CancelledError):
                async with asyncio.timeout(_KILL_WAIT_SECONDS):
                    await process.wait()


async def guarded_cli_write(
    witness: TopicMarkWriteWitness,
    guard: TopicMarkWriteGuard,
    binary: str,
    args: list[str],
    *,
    success: Callable[[tuple[int, str, str]], bool],
    runner: Callable[[str, list[str]], Awaitable[tuple[int, str, str] | None]]
    | None = None,
) -> tuple[int, str, str] | None:
    armed = False
    try:
        guard.arm(witness)
        armed = True
        result = await (runner or run_marker_cli)(binary, args)
        if result is None or not success(result) or not cleanup_pin(witness):
            return None
        guard.terminal(witness.dispatch_id)
        return result
    finally:
        if not armed:
            cleanup_pin(witness)
