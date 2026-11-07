"""hc event translation, ``watch_events`` and the JSON-RPC client.

Event lines in ``fixtures/hc/events.jsonl`` were recorded live from
herdr-core 0.1.18 (``created`` / ``foreground`` / ``finished`` / ``status`` /
``removed`` / ``heartbeat``); the trailing ``lost``, ``title`` and null-status
lines are synthetic, shaped like the recorded ones, for types the schema names
but the probe did not trigger.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

import pytest

from ccgram.multiplexer import hc as hc_module
from ccgram.multiplexer.base import AgentStatus, MuxEvent
from ccgram.multiplexer.hc import HcManager
from ccgram.multiplexer.hc_events import (
    RESYNC,
    agent_name_for,
    is_resync,
    is_subscribed,
    open_event_stream,
    parse_status,
    snapshot_events,
    subscribed_sentinel,
    translate_event,
)
from ccgram.multiplexer.hc_socket import HcRpcError, default_socket_path, request

FIXTURES = Path(__file__).parent / "fixtures" / "hc"


def _events() -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURES / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _snapshot_terminals() -> list[dict]:
    listing = json.loads((FIXTURES / "terminal_list.json").read_text())
    return listing["result"]["terminals"]


# ── Status parsing ──────────────────────────────────────────────────────


def test_parse_status_maps_app_and_msg() -> None:
    assert parse_status(
        {"app": "claude-code", "state": "working", "msg": "Reading"}
    ) == AgentStatus(state="working", agent="claude", custom_status="Reading")
    assert parse_status({"app": "codex", "state": "blocked", "kind": "permission"}) == (
        AgentStatus(state="blocked", agent="codex")
    )
    assert parse_status(None) is None
    assert parse_status({"app": "claude-code"}) is None
    assert parse_status({"state": "  "}) is None


def test_agent_name_for_unknown_apps_passes_through() -> None:
    assert agent_name_for("claude-code") == "claude"
    assert agent_name_for("Gemini-CLI") == "gemini"
    assert agent_name_for("my-tool") == "my-tool"


# ── Translation ─────────────────────────────────────────────────────────


def test_translate_recorded_and_synthetic_events() -> None:
    watched = {"t2b": "hc-t2b", "t16": "hc-t16", "t5": "hc-t5", "t3": "hc-t3"}
    translated = [event for obj in _events() for event in translate_event(obj, watched)]
    assert translated == [
        # created and foreground are ignored; finished is death.
        MuxEvent(kind="window_died", window_id="hc-t2b"),
        MuxEvent(
            kind="agent_status",
            window_id="hc-t16",
            status=AgentStatus(state="working", agent="claude"),
        ),
        # removed after finished: a second death report, deduplicated downstream.
        MuxEvent(kind="window_died", window_id="hc-t2b"),
        # heartbeat → nothing
        MuxEvent(
            kind="agent_status",
            window_id="hc-t5",
            status=AgentStatus(
                state="done", agent="claude", custom_status="Wrapped up"
            ),
        ),
        # The program stopped reporting (agent exited): not "working" any more.
        MuxEvent(kind="agent_status", window_id="hc-t5", status=AgentStatus("unknown")),
        MuxEvent(kind="window_died", window_id="hc-t3"),
        # title → nothing
    ]


def test_translate_ignores_unwatched_terminals_and_noise() -> None:
    for obj in _events():
        assert translate_event(obj, {"t999": "hc-t999"}) == ()
    assert (
        translate_event({"method": "heartbeat", "params": {"seq": 1}}, {"t5": "x"})
        == ()
    )
    assert translate_event({"method": "event", "params": "junk"}, {"t5": "x"}) == ()
    assert (
        translate_event(
            {"method": "event", "params": {"type": "brand_new", "terminal": "t5"}},
            {"t5": "hc-t5"},
        )
        == ()
    )
    # A response line (an id and a result) is not a notification.
    assert translate_event({"id": 3, "result": {}}, {"t5": "hc-t5"}) == ()


def test_snapshot_events_reprime_watched_live_terminals() -> None:
    watched = {"t5": "hc-t5", "t1y": "hc-t1y", "t9": "hc-t9", "t404": "hc-t404"}
    events = snapshot_events(_snapshot_terminals(), watched)
    assert events == (
        MuxEvent(
            kind="agent_status",
            window_id="hc-t5",
            status=AgentStatus("working", "claude", "Reading the plan"),
        ),
        MuxEvent(
            kind="agent_status",
            window_id="hc-t1y",
            status=AgentStatus("blocked", "codex"),
        ),
    )


def test_sentinels() -> None:
    sentinel = subscribed_sentinel([{"address": "t5"}])
    assert is_subscribed(sentinel) and not is_resync(sentinel)
    assert is_resync(RESYNC) and not is_subscribed(RESYNC)
    assert not is_subscribed(_events()[0])


# ── watch_events ────────────────────────────────────────────────────────


def _opener(*sessions: list[dict | BaseException]) -> tuple[Callable, list[int]]:
    """Stream opener replaying one canned list per connection."""
    opened: list[int] = []
    queue = list(sessions)

    async def open_stream() -> AsyncGenerator[dict, None]:
        opened.append(len(opened) + 1)
        if not queue:
            await asyncio.Event().wait()  # idle forever once the script runs out
        for item in queue.pop(0):
            if isinstance(item, BaseException):
                raise item
            yield item

    return open_stream, opened


async def _collect(manager: HcManager, ids: list[str], count: int) -> list[MuxEvent]:
    out: list[MuxEvent] = []
    stream = manager.watch_events(ids)
    try:
        async with asyncio.timeout(5):
            async for event in stream:
                out.append(event)
                if len(out) == count:
                    break
    finally:
        await stream.aclose()
    return out


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc_module, "_STREAM_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(hc_module, "_STREAM_BACKOFF_MAX", 0.0)


@pytest.mark.usefixtures("no_backoff")
async def test_watch_events_reprimes_then_translates() -> None:
    opener, opened = _opener(
        [subscribed_sentinel(_snapshot_terminals()), *_events()],
    )
    manager = HcManager(transport=None, stream_opener=opener, own_terminal_id="")
    events = await _collect(manager, ["hc-t5", "hc-t16", "@1", "hc-name"], 4)
    assert events == [
        MuxEvent(
            kind="agent_status",
            window_id="hc-t5",
            status=AgentStatus("working", "claude", "Reading the plan"),
        ),
        MuxEvent(
            kind="agent_status",
            window_id="hc-t16",
            status=AgentStatus("working", "claude"),
        ),
        MuxEvent(
            kind="agent_status",
            window_id="hc-t16",
            status=AgentStatus("working", "claude"),
        ),
        MuxEvent(
            kind="agent_status",
            window_id="hc-t5",
            status=AgentStatus("done", "claude", "Wrapped up"),
        ),
    ]
    assert opened == [1]


@pytest.mark.usefixtures("no_backoff")
async def test_watch_events_yields_the_callers_window_id_spelling() -> None:
    opener, _opened = _opener([subscribed_sentinel([]), *_events()])
    manager = HcManager(transport=None, stream_opener=opener, own_terminal_id="")
    (event,) = await _collect(manager, ["HC-T3"], 1)
    assert event == MuxEvent(kind="window_died", window_id="HC-T3")


@pytest.mark.usefixtures("no_backoff")
async def test_watch_events_reconnects_after_a_drop_and_reprimes() -> None:
    snapshot = [t for t in _snapshot_terminals() if t["address"] == "t5"]
    opener, opened = _opener(
        [subscribed_sentinel([]), OSError("connection reset")],
        [TimeoutError()],  # silent connection: no heartbeat within the idle limit
        [subscribed_sentinel(snapshot)],
    )
    manager = HcManager(transport=None, stream_opener=opener, own_terminal_id="")
    (event,) = await _collect(manager, ["hc-t5"], 1)
    assert event.kind == "agent_status" and event.window_id == "hc-t5"
    assert opened == [1, 2, 3]


async def test_watch_events_resubscribes_at_once_on_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(hc_module.asyncio, "sleep", fake_sleep)
    snapshot = [t for t in _snapshot_terminals() if t["address"] == "t5"]
    opener, opened = _opener(
        [subscribed_sentinel([]), RESYNC],
        [subscribed_sentinel(snapshot)],
    )
    manager = HcManager(transport=None, stream_opener=opener, own_terminal_id="")
    (event,) = await _collect(manager, ["hc-t5"], 1)
    assert event.window_id == "hc-t5"
    assert opened == [1, 2]
    assert sleeps == []  # a resync is not a failure: no backoff


async def test_watch_events_backs_off_exponentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(hc_module.asyncio, "sleep", fake_sleep)
    snapshot = [t for t in _snapshot_terminals() if t["address"] == "t5"]
    opener, _opened = _opener(
        [OSError("refused")],
        [OSError("refused")],
        [OSError("refused")],
        [subscribed_sentinel(snapshot)],
    )
    manager = HcManager(transport=None, stream_opener=opener, own_terminal_id="")
    await _collect(manager, ["hc-t5"], 1)
    assert sleeps == [1.0, 2.0, 4.0]


# ── Live-socket plumbing against a local fake server ────────────────────


class _FakeServer:
    """A unix-socket JSON-RPC server scripted per request line."""

    def __init__(self, handler: Callable[[dict], list[bytes]]) -> None:
        self.handler = handler
        self.dir = tempfile.mkdtemp(prefix="hc-", dir="/tmp")
        self.path = os.path.join(self.dir, "s.sock")
        self.server: asyncio.Server | None = None

    async def __aenter__(self) -> _FakeServer:
        async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            try:
                while line := await reader.readline():
                    for out in self.handler(json.loads(line)):
                        writer.write(out)
                        await writer.drain()
            finally:
                writer.close()

        self.server = await asyncio.start_unix_server(serve, path=self.path)
        return self

    async def __aexit__(self, *_exc) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()
        with contextlib.suppress(OSError):
            os.unlink(self.path)
            os.rmdir(self.dir)


def _line(obj: dict) -> bytes:
    return json.dumps(obj).encode() + b"\n"


async def test_request_correlates_ids_and_skips_notifications() -> None:
    def handler(req: dict) -> list[bytes]:
        assert req["jsonrpc"] == "2.0" and req["method"] == "terminal.get"
        return [
            _line({"jsonrpc": "2.0", "method": "heartbeat", "params": {"seq": 1}}),
            _line({"jsonrpc": "2.0", "id": -1, "result": {"address": "wrong"}}),
            _line({"jsonrpc": "2.0", "id": req["id"], "result": {"address": "t5"}}),
        ]

    async with _FakeServer(handler) as server:
        result = await request(server.path, "terminal.get", {"terminal": "t5"})
    assert result == {"address": "t5"}


async def test_request_surfaces_error_kind() -> None:
    def handler(req: dict) -> list[bytes]:
        return [
            _line(
                {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "error": {
                        "code": -32000,
                        "message": 'no terminal "t9"',
                        "data": {"kind": "not_found", "hint": "see `hc ls --all`"},
                    },
                }
            )
        ]

    async with _FakeServer(handler) as server:
        with pytest.raises(HcRpcError) as info:
            await request(server.path, "terminal.get", {"terminal": "t9"})
    assert info.value.kind == "not_found"
    assert info.value.code == -32000


async def test_request_times_out() -> None:
    async with _FakeServer(lambda _req: []) as server:
        with pytest.raises(HcRpcError) as info:
            await request(server.path, "terminal.read", {"terminal": "t5"}, 0.2)
    assert info.value.kind == "timeout"


async def test_request_transport_failure() -> None:
    with pytest.raises(HcRpcError) as info:
        await request("/tmp/definitely-not-an-hc-socket.sock", "hello", {}, 0.5)
    assert info.value.kind == "transport"


def test_default_socket_path() -> None:
    assert default_socket_path({"HC_TERMINAL_SOCKET": "/x.sock"}) == "/x.sock"
    assert default_socket_path({}).endswith(
        ".local/state/herdr-core/default/default.sock"
    )


async def test_open_event_stream_yields_snapshot_then_notifications() -> None:
    def handler(req: dict) -> list[bytes]:
        assert req["method"] == "events.subscribe"
        return [
            _line(
                {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {
                        "channel": None,
                        "seq": 7,
                        "terminals": [{"address": "t5"}],
                    },
                }
            ),
            b"not json\n",
            _line({"jsonrpc": "2.0", "method": "heartbeat", "params": {"seq": 7}}),
            _line({"jsonrpc": "2.0", "method": "missed", "params": {"seq": 9}}),
            _line(_events()[2]),
        ]

    async with _FakeServer(handler) as server:
        items = []
        async with asyncio.timeout(5):
            async for item in open_event_stream(server.path, idle_timeout=0.5):
                items.append(item)
                if len(items) == 4:
                    break
    assert is_subscribed(items[0]) and items[0]["terminals"] == [{"address": "t5"}]
    assert items[1]["method"] == "heartbeat"
    assert is_resync(items[2])
    assert items[3]["params"]["type"] == "finished"


async def test_open_event_stream_times_out_when_silent() -> None:
    def handler(req: dict) -> list[bytes]:
        return [
            _line(
                {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {"seq": 1, "terminals": []},
                }
            )
        ]

    async with _FakeServer(handler) as server:
        stream = open_event_stream(server.path, idle_timeout=0.2)
        assert is_subscribed(await anext(stream))
        with pytest.raises(TimeoutError):
            await anext(stream)
        await stream.aclose()


async def test_open_event_stream_ends_on_a_rejected_subscription() -> None:
    def handler(req: dict) -> list[bytes]:
        return [
            _line(
                {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "error": {
                        "code": -32000,
                        "message": "no",
                        "data": {"kind": "not_found"},
                    },
                }
            )
        ]

    async with _FakeServer(handler) as server:
        items = [item async for item in open_event_stream(server.path, idle_timeout=1)]
    assert items == []
