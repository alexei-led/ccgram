"""Bounded request/response transport for the Herdr Unix socket."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import time
import uuid
from collections.abc import Mapping

__all__ = ["MAX_FRAME_BYTES", "HerdrSocketError", "request"]

MAX_FRAME_BYTES = 16 * 1024 * 1024
_CLOSE_TIMEOUT_SECONDS = 0.5


class HerdrSocketError(RuntimeError):
    """A Herdr socket request failed or returned an invalid response."""

    def __init__(self, message: str, code: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def request_sync(
    socket_path: str,
    method: str,
    params: Mapping[str, object],
    *,
    timeout: float = 5.0,
) -> dict:
    """Make the same one-shot request from a synchronous provider hook."""
    request_id = uuid.uuid4().hex
    line = _encode_request(request_id, method, params)
    deadline = time.monotonic() + timeout
    response = bytearray()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            connection.settimeout(remaining)
            connection.connect(socket_path)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            connection.settimeout(remaining)
            connection.sendall(line)
            while b"\n" not in response:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                connection.settimeout(remaining)
                chunk = connection.recv(min(65536, MAX_FRAME_BYTES + 1 - len(response)))
                if not chunk:
                    break
                response.extend(chunk)
                frame_end = response.find(b"\n")
                if frame_end > MAX_FRAME_BYTES or (
                    frame_end < 0 and len(response) > MAX_FRAME_BYTES
                ):
                    raise HerdrSocketError(
                        "herdr socket response exceeded the maximum frame size"
                    )
    except TimeoutError as exc:
        raise HerdrSocketError("herdr socket request timed out") from exc
    except OSError as exc:
        raise HerdrSocketError(f"herdr socket transport failed: {exc}") from exc
    return _parse_response(bytes(response).split(b"\n", 1)[0], request_id)


async def request(
    socket_path: str,
    method: str,
    params: Mapping[str, object],
    *,
    timeout: float = 8.0,
) -> dict:
    """Send one Herdr request and return its correlated response envelope."""
    request_id = uuid.uuid4().hex
    request_line = _encode_request(request_id, method, params)
    response_line = await _exchange(socket_path, request_line, timeout)
    return _parse_response(response_line, request_id)


def _encode_request(
    request_id: str, method: str, params: Mapping[str, object]
) -> bytes:
    try:
        request_line = (
            json.dumps(
                {"id": request_id, "method": method, "params": dict(params)},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise HerdrSocketError("herdr socket request is not JSON serializable") from exc
    if len(request_line) - 1 > MAX_FRAME_BYTES:
        raise HerdrSocketError("herdr socket request exceeds the maximum frame size")
    return request_line


async def _exchange(socket_path: str, request_line: bytes, timeout: float) -> bytes:
    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(timeout):
            reader, writer = await asyncio.open_unix_connection(
                socket_path,
                limit=MAX_FRAME_BYTES,
            )
            writer.write(request_line)
            await writer.drain()
            try:
                return await reader.readline()
            except ValueError as exc:
                raise HerdrSocketError(
                    "herdr socket response exceeded the maximum frame size"
                ) from exc
    except HerdrSocketError:
        raise
    except TimeoutError as exc:
        raise HerdrSocketError("herdr socket request timed out") from exc
    except (OSError, asyncio.IncompleteReadError) as exc:
        raise HerdrSocketError(f"herdr socket transport failed: {exc}") from exc
    finally:
        if writer is not None:
            await _close_writer(writer)


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
    except OSError, RuntimeError:
        return
    try:
        async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
            await writer.wait_closed()
    except TimeoutError:
        _abort_writer(writer)
    except asyncio.CancelledError:
        _abort_writer(writer)
        raise
    except OSError, RuntimeError:
        _abort_writer(writer)


def _abort_writer(writer: asyncio.StreamWriter) -> None:
    transport = getattr(writer, "transport", None)
    if transport is not None:
        with contextlib.suppress(Exception):
            transport.abort()


def _parse_response(response_line: bytes, request_id: str) -> dict:
    if not response_line:
        raise HerdrSocketError("herdr socket closed before sending a response")
    envelope = _decode_response(response_line)
    if not isinstance(envelope, dict):
        raise HerdrSocketError("herdr socket response must be a JSON object")
    _validate_response_id(envelope, request_id)
    if "error" in envelope:
        _raise_remote_error(envelope["error"])
    _validate_result(envelope)
    return envelope


def _decode_response(response_line: bytes) -> object:
    try:
        response_text = response_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HerdrSocketError("herdr socket response is not valid UTF-8") from exc
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise HerdrSocketError("herdr socket response is not valid JSON") from exc


def _validate_response_id(envelope: Mapping[str, object], request_id: str) -> None:
    response_id = envelope.get("id")
    if response_id is None:
        raise HerdrSocketError("herdr socket response is missing its id")
    if not isinstance(response_id, str):
        raise HerdrSocketError("herdr socket response id must be a string")
    if response_id != request_id:
        raise HerdrSocketError("herdr socket response id does not match the request")


def _validate_result(envelope: Mapping[str, object]) -> None:
    if "result" not in envelope:
        raise HerdrSocketError("herdr socket response is missing its result")
    result = envelope["result"]
    if not isinstance(result, dict):
        raise HerdrSocketError("herdr socket response result must be a JSON object")


def _raise_remote_error(error: object) -> None:
    if not isinstance(error, Mapping):
        raise HerdrSocketError("herdr socket response error must be an object")
    message = error.get("message")
    if not isinstance(message, str):
        raise HerdrSocketError(
            "herdr socket response error is missing a string message"
        )
    raise HerdrSocketError(message, error.get("code"))
