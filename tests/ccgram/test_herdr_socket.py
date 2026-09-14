from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ccgram.multiplexer import herdr_socket
from ccgram.multiplexer.herdr_socket import HerdrSocketError, request


async def test_synchronous_hook_request_uses_public_envelope(unix_server):
    socket_path, start = unix_server
    received = []

    async def handler(reader, writer):
        payload = await _read_request(reader)
        received.append(payload)
        data = (
            json.dumps(
                {"id": payload["id"], "result": {"agents": []}, "future": True}
            ).encode()
            + b"\n"
        )
        writer.write(data[:10])
        await writer.drain()
        writer.write(data[10:])
        await writer.drain()

    await start(handler)
    reply = await asyncio.to_thread(
        herdr_socket.request_sync, str(socket_path), "agent.list", {}
    )
    assert reply["result"] == {"agents": []}
    assert received[0]["method"] == "agent.list"
    assert set(received[0]) == {"id", "method", "params"}
    assert len(received) == 1


async def test_synchronous_hook_accepts_exact_frame_limit(unix_server, monkeypatch):
    socket_path, start = unix_server
    limit = 256
    monkeypatch.setattr(herdr_socket, "MAX_FRAME_BYTES", limit)

    async def handler(reader, writer):
        payload = await _read_request(reader)
        response = {"id": payload["id"], "result": {}, "future": ""}
        for size in range(limit):
            response["future"] = "x" * size
            encoded = json.dumps(response, separators=(",", ":")).encode()
            if len(encoded) == limit:
                writer.write(encoded + b"\n")
                await writer.drain()
                return
        raise AssertionError("unable to construct an exact frame")

    await start(handler)
    reply = await asyncio.to_thread(
        herdr_socket.request_sync, str(socket_path), "agent.list", {}
    )
    assert reply["result"] == {}


async def test_synchronous_hook_error_is_not_replayed(unix_server):
    socket_path, start = unix_server
    received = []

    async def handler(reader, writer):
        payload = await _read_request(reader)
        received.append(payload)
        writer.write(
            json.dumps(
                {
                    "id": payload["id"],
                    "error": {"code": "unsupported", "message": "unsupported method"},
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()

    await start(handler)
    with pytest.raises(HerdrSocketError, match="unsupported method"):
        await asyncio.to_thread(
            herdr_socket.request_sync, str(socket_path), "agent.list", {}
        )
    assert len(received) == 1


async def test_synchronous_hook_timeout_closes_connection(unix_server):
    socket_path, start = unix_server
    closed = asyncio.Event()

    async def handler(reader, writer):
        await _read_request(reader)
        assert await reader.read() == b""
        closed.set()

    await start(handler)
    with pytest.raises(HerdrSocketError, match="timed out"):
        await asyncio.to_thread(
            herdr_socket.request_sync, str(socket_path), "agent.list", {}, timeout=0.05
        )
    await asyncio.wait_for(closed.wait(), 1)


@pytest.fixture
async def unix_server():
    socket_dir = TemporaryDirectory(prefix="ccgram-socket-", dir="/tmp")
    socket_path = Path(socket_dir.name) / "herdr.sock"
    servers = []
    tasks: set[asyncio.Task] = set()

    async def start(handler):
        async def tracked(reader, writer):
            task = asyncio.current_task()
            if task is not None:
                tasks.add(task)
            try:
                await handler(reader, writer)
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
                if task is not None:
                    tasks.discard(task)

        server = await asyncio.start_unix_server(tracked, path=socket_path)
        servers.append(server)
        return server

    yield socket_path, start
    for server in servers:
        server.close()
        await server.wait_closed()
    pending = list(tasks)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    socket_path.unlink(missing_ok=True)
    socket_dir.cleanup()


async def _read_request(reader: asyncio.StreamReader) -> dict:
    line = await reader.readline()
    assert line
    payload = json.loads(line)
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    ("method", "params"),
    [("rawping", {}), ("agent.list", {"include_hidden": False})],
)
async def test_envelope(unix_server, method: str, params: dict[str, object]) -> None:
    socket_path, start = unix_server
    received = []

    async def handler(reader, writer):
        payload = await _read_request(reader)
        received.append(payload)
        response = {
            "id": payload["id"],
            "result": {"ok": True},
            "protocol": 22,
            "unknown": {"ignored": True},
        }
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()

    await start(handler)
    response = await request(str(socket_path), method, params)

    assert response["result"] == {"ok": True}
    assert response["protocol"] == 22
    assert len(received) == 1
    assert received[0]["method"] == method
    assert received[0]["params"] == params
    assert set(received[0]) == {"id", "method", "params"}
    assert isinstance(received[0]["id"], str)
    assert received[0]["id"]


async def test_error(unix_server) -> None:
    socket_path, start = unix_server
    received = []

    async def handler(reader, writer):
        payload = await _read_request(reader)
        received.append(payload)
        writer.write(
            json.dumps(
                {
                    "id": payload["id"],
                    "error": {"code": -32001, "message": "denied", "data": 7},
                    "extra": True,
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()

    await start(handler)
    with pytest.raises(HerdrSocketError) as caught:
        await request(str(socket_path), "agent.list", {})

    assert str(caught.value) == "denied"
    assert caught.value.message == "denied"
    assert caught.value.code == -32001
    assert len(received) == 1


@pytest.mark.parametrize(
    ("mode", "message"),
    [("eof", "closed before"), ("nonjson", "not valid JSON")],
)
async def test_transport_invalid(unix_server, mode, message) -> None:
    socket_path, start = unix_server

    async def handler(reader, writer):
        await _read_request(reader)
        if mode == "nonjson":
            writer.write(b"not json\n")
            await writer.drain()

    await start(handler)
    with pytest.raises(HerdrSocketError, match=message):
        await request(str(socket_path), "rawping", {})


@pytest.mark.parametrize(
    "response_kind",
    ["nonobject", "missing_id", "bad_id", "missing_result", "bad_result", "bad_error"],
)
async def test_envelope_invalid(unix_server, response_kind) -> None:
    socket_path, start = unix_server

    async def handler(reader, writer):
        payload = await _read_request(reader)
        if response_kind == "nonobject":
            response = []
        elif response_kind == "missing_id":
            response = {"result": {}}
        elif response_kind == "bad_id":
            response = {"id": 1, "result": {}}
        elif response_kind == "missing_result":
            response = {"id": payload["id"]}
        elif response_kind == "bad_result":
            response = {"id": payload["id"], "result": []}
        else:
            response = {"id": payload["id"], "error": {"code": 1}}
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()

    await start(handler)
    with pytest.raises(HerdrSocketError):
        await request(str(socket_path), "rawping", {})


async def test_timeout(unix_server) -> None:
    socket_path, start = unix_server
    received = asyncio.Event()
    release = asyncio.Event()

    async def handler(reader, writer):
        await _read_request(reader)
        received.set()
        await release.wait()

    await start(handler)
    pending = asyncio.create_task(
        request(str(socket_path), "rawping", {}, timeout=0.05)
    )
    await asyncio.wait_for(received.wait(), 1)
    with pytest.raises(HerdrSocketError, match="timed out"):
        await pending
    release.set()


async def test_cancellation(unix_server) -> None:
    socket_path, start = unix_server
    received = asyncio.Event()
    closed = asyncio.Event()

    async def handler(reader, writer):
        await _read_request(reader)
        received.set()
        if not await reader.read():
            closed.set()

    await start(handler)
    pending = asyncio.create_task(
        request(str(socket_path), "agent.list", {}, timeout=10)
    )
    await asyncio.wait_for(received.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await asyncio.wait_for(closed.wait(), 1)


async def test_oversized(unix_server, monkeypatch) -> None:
    socket_path, start = unix_server
    monkeypatch.setattr(herdr_socket, "MAX_FRAME_BYTES", 128)

    async def handler(reader, writer):
        payload = await _read_request(reader)
        response = {"id": payload["id"], "result": {"value": "x" * 100}}
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()

    await start(handler)
    with pytest.raises(HerdrSocketError, match="maximum frame"):
        await request(str(socket_path), "rawping", {})
