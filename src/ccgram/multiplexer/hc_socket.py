"""Bounded JSON-RPC 2.0 client for the herdr-core (``hc``) Unix socket.

herdr-core is the terminal runtime behind the Herdr desktop app. It speaks
line-delimited JSON-RPC 2.0: one request object per line, one response per
line, with ``id`` correlation. A connection may also carry notifications
(``"method": "event"`` / ``"heartbeat"``, no ``id``) once it has subscribed;
a request/response connection never subscribes, but the reader still skips any
line without the matching ``id`` so a stray notification cannot be mistaken
for the answer.

Errors carry a stable ``data.kind`` (``not_found``, ``finished``, ``blocked``,
…), surfaced on ``HcRpcError.kind`` so the adapter can tell "this terminal is
gone" from "the socket could not be reached".

The persistent ``events.subscribe`` reader lives in ``hc_events``; this module
is strictly one connection per request, like ``herdr_socket``.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

__all__ = [
    "DEFAULT_SOCKET_RELATIVE",
    "MAX_FRAME_BYTES",
    "HcRpcError",
    "HcTransport",
    "default_socket_path",
    "encode_request",
    "request",
]

MAX_FRAME_BYTES = 16 * 1024 * 1024
_CLOSE_TIMEOUT_SECONDS = 0.5

# herdr-core's default socket, relative to the home directory. Every Herdr
# terminal also exports the live path as ``HC_TERMINAL_SOCKET``.
DEFAULT_SOCKET_RELATIVE = ".local/state/herdr-core/default/default.sock"

# Transport contract: ``(method, params, timeout) -> result``. Raises
# ``HcRpcError`` on any failure. Injectable so unit tests need no socket.
HcTransport = Callable[[str, Mapping[str, object], float], Awaitable[dict]]

_request_ids = itertools.count(1)


class HcRpcError(RuntimeError):
    """A herdr-core request failed: transport, framing, or a JSON-RPC error.

    ``kind`` is herdr-core's stable ``error.data.kind`` (``not_found``,
    ``finished``, ``blocked``, …) for a remote error, and ``"transport"`` /
    ``"timeout"`` / ``"protocol"`` for failures on this side.
    """

    def __init__(self, message: str, *, kind: str, code: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.code = code


def default_socket_path(env: Mapping[str, str] | None = None) -> str:
    """Return ``$HC_TERMINAL_SOCKET`` or herdr-core's default socket path.

    Pure path arithmetic, no filesystem access.
    """
    source = os.environ if env is None else env
    configured = source.get("HC_TERMINAL_SOCKET", "")
    if configured:
        return configured
    return str(Path.home() / DEFAULT_SOCKET_RELATIVE)


def encode_request(request_id: int, method: str, params: Mapping[str, object]) -> bytes:
    """Encode one JSON-RPC 2.0 request line."""
    try:
        line = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": dict(params),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise HcRpcError(
            "hc request is not JSON serializable", kind="protocol"
        ) from exc
    if len(line) - 1 > MAX_FRAME_BYTES:
        raise HcRpcError("hc request exceeds the maximum frame size", kind="protocol")
    return line


async def request(
    socket_path: str,
    method: str,
    params: Mapping[str, object],
    timeout: float = 8.0,
) -> dict:
    """Send one request on a fresh connection and return its ``result`` object.

    The whole exchange (connect, write, read) is bounded by *timeout*. A
    result that is not a JSON object (herdr-core's results all are) is a
    protocol error.
    """
    request_id = next(_request_ids)
    line = encode_request(request_id, method, params)
    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(timeout):
            reader, writer = await asyncio.open_unix_connection(
                socket_path, limit=MAX_FRAME_BYTES
            )
            writer.write(line)
            await writer.drain()
            while True:
                try:
                    raw = await reader.readline()
                except ValueError as exc:
                    raise HcRpcError(
                        "hc response exceeded the maximum frame size", kind="protocol"
                    ) from exc
                if not raw:
                    raise HcRpcError(
                        "hc socket closed before sending a response", kind="transport"
                    )
                envelope = _decode(raw)
                if envelope is None or envelope.get("id") != request_id:
                    # A notification or an unrelated line; keep reading.
                    continue
                return _result_of(envelope)
    except HcRpcError:
        raise
    except TimeoutError as exc:
        raise HcRpcError(f"hc {method} timed out", kind="timeout") from exc
    except (OSError, asyncio.IncompleteReadError) as exc:
        raise HcRpcError(
            f"hc socket transport failed: {exc}", kind="transport"
        ) from exc
    finally:
        if writer is not None:
            await _close_writer(writer)


def _decode(raw: bytes) -> dict | None:
    text = raw.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except UnicodeDecodeError, ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _result_of(envelope: Mapping[str, object]) -> dict:
    """Return the result object of a correlated response, or raise its error."""
    error = envelope.get("error")
    if error is not None:
        if not isinstance(error, Mapping):
            raise HcRpcError("hc response error must be an object", kind="protocol")
        data = error.get("data")
        kind = data.get("kind") if isinstance(data, Mapping) else None
        message = error.get("message")
        raise HcRpcError(
            message if isinstance(message, str) else "hc request failed",
            kind=kind if isinstance(kind, str) and kind else "error",
            code=error.get("code"),
        )
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise HcRpcError("hc response result must be an object", kind="protocol")
    return result


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
