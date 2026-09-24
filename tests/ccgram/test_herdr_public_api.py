from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ccgram.multiplexer import herdr as herdr_module
from ccgram.multiplexer.herdr import (
    HerdrError,
    HerdrManager,
    HerdrSessionComposite,
    herdr_session_target_id,
)

Handler = Callable[[Mapping[str, object]], Awaitable[object | None]]


@pytest.fixture
async def herdr_server():
    socket_dir = TemporaryDirectory(prefix="ccgram-api-", dir="/tmp")
    socket_path = Path(socket_dir.name) / "herdr.sock"
    requests: list[dict[str, object]] = []
    handlers: set[asyncio.Task] = set()
    servers: list[asyncio.AbstractServer] = []

    async def start(handler: Handler):
        async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            task = asyncio.current_task()
            if task is not None:
                handlers.add(task)
            try:
                line = await reader.readline()
                if not line:
                    return
                payload = json.loads(line)
                assert isinstance(payload, dict)
                requests.append(payload)
                response = await handler(payload)
                if response is not None:
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
                if task is not None:
                    handlers.discard(task)

        server = await asyncio.start_unix_server(serve, path=socket_path)
        servers.append(server)
        return server

    yield socket_path, requests, start
    for server in servers:
        server.close()
        await server.wait_closed()
    pending = list(handlers)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    socket_path.unlink(missing_ok=True)
    socket_dir.cleanup()


def _response(payload: Mapping[str, object], result: object) -> dict[str, object]:
    return {
        "id": payload["id"],
        "result": result,
        "future_field": {"accepted": True},
    }


def _agent(*, pane_id: str = "w2:p1") -> dict[str, object]:
    return {
        "terminal_id": "term-a",
        "pane_id": pane_id,
        "tab_id": "w2:t1",
        "workspace_id": "w2",
        "cwd": "/repo",
        "agent_session": {
            "source": "herdr",
            "agent": "claude",
            "kind": "id",
            "value": "session-a",
        },
        "new_field": "ignored",
    }


def _server_result(
    payload: Mapping[str, object], *, protocol: int = 20
) -> dict[str, object]:
    method = payload["method"]
    if method == "ping":
        return _response(
            payload,
            {
                "type": "pong",
                "version": "0.8.0",
                "protocol": protocol,
                "future": {"field": True},
            },
        )
    if method == "agent.list":
        return _response(payload, {"agents": [_agent()]})
    if method == "workspace.list":
        return _response(
            payload,
            {"workspaces": [{"workspace_id": "w2", "label": "Workspace"}]},
        )
    if method == "tab.list":
        return _response(payload, {"tabs": [{"tab_id": "w2:t1", "label": "Tab"}]})
    raise AssertionError(f"unexpected method: {method}")


@pytest.mark.parametrize(
    ("protocol", "binary"),
    [(20, "/missing/herdr-binary"), (22, "wrong-herdr"), (99, "wrong-herdr")],
)
async def test_explicit_socket_uses_public_ping_without_cli(
    herdr_server, monkeypatch: pytest.MonkeyPatch, protocol: int, binary: str
) -> None:
    socket_path, requests, start = herdr_server

    async def handler(payload):
        return _server_result(payload, protocol=protocol)

    await start(handler)

    def fail(*_args, **_kwargs):
        raise AssertionError("the Herdr CLI must not run")

    monkeypatch.setattr(subprocess, "run", fail)
    manager = HerdrManager(socket_path=str(socket_path), binary=binary)
    await manager.ensure_session()

    assert [payload["method"] for payload in requests] == ["ping"]
    assert requests[0]["params"] == {}


async def test_socket_listing_projects_stable_session_target(herdr_server) -> None:
    socket_path, requests, start = herdr_server

    async def handler(payload):
        return _server_result(payload)

    await start(handler)
    manager = HerdrManager(socket_path=str(socket_path), binary="wrong-herdr")

    windows = await manager.list_windows_for_reconciliation()

    assert windows is not None
    assert len(windows) == 1
    expected = herdr_session_target_id(
        HerdrSessionComposite("herdr", "claude", "id", "session-a")
    )
    assert windows[0].window_id == expected
    assert windows[0].window_name == "Claude ▸ Workspace ▸ Tab ▸ p1"
    assert windows[0].topic_eligible is True
    assert [payload["method"] for payload in requests] == [
        "agent.list",
        "workspace.list",
        "tab.list",
    ]


async def test_capture_and_guarded_send_use_socket_payloads(
    herdr_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path, requests, start = herdr_server
    target = herdr_session_target_id(
        HerdrSessionComposite("herdr", "claude", "id", "session-a")
    )

    async def handler(payload):
        method = payload["method"]
        if method == "agent.list":
            return _response(payload, {"agents": [_agent()]})
        if method == "pane.read":
            text = (
                "plain text\n"
                if payload["params"]["format"] == "text"
                else "\u001b[31mred\u001b[0m\n"
            )
            return _response(payload, {"read": {"text": text}})
        if method == "pane.send_text":
            assert payload["params"]["text"] == "first\nsecond"
            return _response(payload, {"type": "ok"})
        raise AssertionError(f"unexpected method: {method}")

    await start(handler)

    def fail(*_args, **_kwargs):
        raise AssertionError("the Herdr CLI must not run")

    monkeypatch.setattr(subprocess, "run", fail)
    manager = HerdrManager(socket_path=str(socket_path), binary="wrong-herdr")

    assert await manager.capture_pane(target) == "plain text"
    assert (
        await manager.capture_pane(target, with_ansi=True) == "\u001b[31mred\u001b[0m"
    )
    assert await manager.send(target, "first\nsecond", enter=False)
    assert [payload["method"] for payload in requests] == [
        "agent.list",
        "pane.read",
        "agent.list",
        "pane.read",
        "agent.list",
        "pane.send_text",
    ]
    assert requests[1]["params"] == {
        "pane_id": "w2:p1",
        "source": "visible",
        "format": "text",
        "strip_ansi": True,
    }
    assert requests[3]["params"] == {
        "pane_id": "w2:p1",
        "source": "visible",
        "format": "ansi",
        "strip_ansi": False,
    }


@pytest.mark.parametrize("agents", [None, {"unsupported": True}])
async def test_malformed_agent_list_keeps_reconciliation_unknown(
    herdr_server, agents
) -> None:
    socket_path, requests, start = herdr_server

    async def handler(payload):
        assert payload["method"] == "agent.list"
        return _response(payload, {"agents": agents})

    await start(handler)
    manager = HerdrManager(socket_path=str(socket_path), binary="wrong-herdr")

    assert await manager.list_windows_for_reconciliation() is None
    assert [payload["method"] for payload in requests] == ["agent.list"]
    assert not any(
        payload["method"] in {"pane.close", "tab.close", "workspace.close"}
        for payload in requests
    )


async def test_missing_session_identity_keeps_reconciliation_unknown(
    herdr_server,
) -> None:
    socket_path, requests, start = herdr_server
    record = _agent()
    session = {
        "source": "herdr",
        "agent": "claude",
        "kind": "id",
    }
    record["agent_session"] = session

    async def handler(payload):
        return _response(payload, {"agents": [record]})

    await start(handler)
    manager = HerdrManager(socket_path=str(socket_path), binary="wrong-herdr")

    assert await manager.list_windows_for_reconciliation() is None
    assert [payload["method"] for payload in requests] == ["agent.list"]


async def test_empty_socket_path_discovers_once_then_uses_public_socket(
    herdr_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path, requests, start = herdr_server
    discovery_calls = []

    async def handler(payload):
        return _server_result(payload)

    await start(handler)
    monkeypatch.delenv("HERDR_SOCKET_PATH", raising=False)
    manager = HerdrManager(socket_path="", binary="wrong-herdr")

    async def discover(args):
        discovery_calls.append(tuple(args))
        return (
            0,
            json.dumps(
                {
                    "server": {
                        "running": True,
                        "compatible": False,
                        "socket": str(socket_path),
                    }
                }
            ),
            "",
        )

    monkeypatch.setattr(manager, "_subprocess_run", discover)
    await manager.ensure_session()
    windows = await manager.list_windows()

    assert windows
    assert discovery_calls == [("status", "--json")]
    assert [payload["method"] for payload in requests] == [
        "ping",
        "agent.list",
        "workspace.list",
        "tab.list",
    ]


async def test_watch_events_surfaces_socket_discovery_failure(
    herdr_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HERDR_SOCKET_PATH", raising=False)
    monkeypatch.setattr(herdr_module, "_STREAM_BACKOFF_BASE", 0.001)
    manager = HerdrManager(socket_path="", binary="wrong-herdr")
    discovery_calls = []

    async def discover(args):
        discovery_calls.append(tuple(args))
        return 1, "", "status failed"

    monkeypatch.setattr(manager, "_subprocess_run", discover)
    stream = manager.watch_events([])
    try:
        # A failed discovery is deterministic, not transient: the stream
        # surfaces HerdrError instead of retrying forever inside the
        # generator, so the monitor's supervisor owns the restart cadence.
        with pytest.raises(HerdrError, match="Cannot discover"):
            await asyncio.wait_for(anext(stream), 1.0)
    finally:
        await stream.aclose()

    assert discovery_calls
    assert set(discovery_calls) == {("status", "--json")}
