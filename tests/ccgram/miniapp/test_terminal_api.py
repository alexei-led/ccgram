import asyncio
import json

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from ccgram.miniapp import build_app, sign_token
from ccgram.miniapp.api.terminal import (
    DEFAULT_POLL_INTERVAL,
    MAX_FRAME_BYTES,
    register_terminal_routes,
)

from ._helpers import make_init_data

BOT = "1234:abcdef"
WINDOW_ID = "ccgram:@7"


def _init_headers(*, user_id: int = 42) -> dict[str, str]:
    return {"X-Telegram-Init-Data": make_init_data(bot_token=BOT, user_id=user_id)}


async def _ws_authenticate(ws, *, user_id: int = 42) -> None:
    await ws.send_json({"init_data": make_init_data(bot_token=BOT, user_id=user_id)})


class FakePane:
    """Capture stub returning a queue of frames; blocks once exhausted."""

    def __init__(self, frames: list[str | None]):
        self.frames = list(frames)
        self.calls: list[str] = []

    async def __call__(self, window_id: str) -> str | None:
        self.calls.append(window_id)
        if self.frames:
            return self.frames.pop(0)
        # Idle forever — keeps the websocket alive between assertions.
        await asyncio.sleep(0.5)
        return None


def _make_app(capture, *, interval: float = 0.05) -> web.Application:
    app = web.Application()
    register_terminal_routes(
        app, bot_token=BOT, capture=capture, poll_interval=interval
    )
    return app


@pytest.fixture
async def app_client():
    capture = FakePane(["screen one", "screen one", "screen two"])
    app = _make_app(capture, interval=0.05)
    async with TestClient(TestServer(app)) as c:
        yield c, capture


async def _read_one(ws, *, timeout: float = 1.0) -> dict:
    msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
    assert msg.type == WSMsgType.TEXT, f"unexpected ws msg type {msg.type}"
    return json.loads(msg.data)


async def test_websocket_rejects_invalid_token(app_client):
    c, _ = app_client
    resp = await c.get("/ws/terminal/garbage")
    assert resp.status == 403


async def test_websocket_rejects_token_for_other_bot(app_client):
    c, _ = app_client
    tok = sign_token(bot_token="9999:other", window_id=WINDOW_ID, user_id=1)
    resp = await c.get(f"/ws/terminal/{tok}")
    assert resp.status == 403


async def test_websocket_closes_when_first_frame_missing_init_data(app_client):
    c, _ = app_client
    tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
    async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
        await ws.send_json({"not_init_data": "anything"})
        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
        assert msg.type == WSMsgType.CLOSE
        assert msg.data == 4003


async def test_websocket_closes_when_first_frame_user_mismatch(app_client):
    c, _ = app_client
    tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
    async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
        # Wrong user id → server closes with auth-failed code.
        await _ws_authenticate(ws, user_id=999)
        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
        assert msg.type == WSMsgType.CLOSE
        assert msg.data == 4003


async def test_websocket_closes_on_auth_timeout(app_client):
    # Server gives clients _WS_AUTH_TIMEOUT (5s) to send the auth frame; we
    # poke the timeout knob to keep the test snappy.
    from ccgram.miniapp.api import terminal as term_mod

    original = term_mod._WS_AUTH_TIMEOUT
    term_mod._WS_AUTH_TIMEOUT = 0.1
    try:
        c, _ = app_client
        tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
        async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
            msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
            assert msg.type == WSMsgType.CLOSE
            assert msg.data == 4001
    finally:
        term_mod._WS_AUTH_TIMEOUT = original


async def test_websocket_streams_hello_then_frame(app_client):
    c, capture = app_client
    tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
    async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
        await _ws_authenticate(ws)
        hello = await _read_one(ws)
        assert hello["type"] == "hello"
        assert hello["window_id"] == WINDOW_ID
        assert hello["interval"] == pytest.approx(0.05)

        frame = await _read_one(ws)
        assert frame["type"] == "frame"
        assert frame["text"] == "screen one"
        assert frame["hash"]
        await ws.close()

    # First frame plus possibly a couple of poll calls.
    assert capture.calls and capture.calls[0] == WINDOW_ID


async def test_websocket_dedupes_unchanged_frames(app_client):
    c, _ = app_client
    tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
    async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
        await _ws_authenticate(ws)
        hello = await _read_one(ws)
        assert hello["type"] == "hello"

        frame_a = await _read_one(ws)
        assert frame_a["type"] == "frame"
        assert frame_a["text"] == "screen one"

        # The next pane capture returns identical text — no new frame should
        # arrive within a short window. The third capture flips text and
        # produces "screen two".
        frame_b = await _read_one(ws, timeout=2.0)
        assert frame_b["type"] == "frame"
        assert frame_b["text"] == "screen two"
        assert frame_b["hash"] != frame_a["hash"]
        await ws.close()


async def test_websocket_disconnect_stops_capture():
    capture = FakePane(["a", "b", "c", "d", "e", "f"])
    app = _make_app(capture, interval=0.05)
    async with TestClient(TestServer(app)) as c:
        tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
        async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
            await _ws_authenticate(ws)
            await _read_one(ws)  # hello
            await _read_one(ws)  # first frame
            await ws.close()
        # Drain in-flight tick, then snapshot call count.
        await asyncio.sleep(0.05)
        baseline = len(capture.calls)
        # No further captures should fire after the socket is closed.
        await asyncio.sleep(0.1)
        assert len(capture.calls) == baseline


async def test_websocket_capture_failure_emits_error_then_continues():
    class ExplodingThenFine:
        def __init__(self):
            self.calls = 0

        async def __call__(self, _window_id: str) -> str | None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return "ok"

    capture = ExplodingThenFine()
    app = _make_app(capture, interval=0.05)
    async with TestClient(TestServer(app)) as c:
        tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
        async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
            await _ws_authenticate(ws)
            await _read_one(ws)  # hello
            err = await _read_one(ws, timeout=1.0)
            assert err["type"] == "error"
            assert "capture failed" in err["message"]
            frame = await _read_one(ws, timeout=1.0)
            assert frame["type"] == "frame"
            assert frame["text"] == "ok"
            await ws.close()


async def test_websocket_truncates_oversized_frame():
    huge = "x" * (MAX_FRAME_BYTES + 1024)
    capture = FakePane([huge])
    app = _make_app(capture, interval=0.05)
    async with TestClient(TestServer(app)) as c:
        tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
        async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
            await _ws_authenticate(ws)
            await _read_one(ws)  # hello
            frame = await _read_one(ws)
            assert frame["type"] == "frame"
            assert len(frame["text"].encode("utf-8")) <= MAX_FRAME_BYTES
            await ws.close()


async def test_websocket_handles_none_capture_as_empty():
    capture = FakePane([None])
    app = _make_app(capture, interval=0.05)
    async with TestClient(TestServer(app)) as c:
        tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
        async with c.ws_connect(f"/ws/terminal/{tok}") as ws:
            await _ws_authenticate(ws)
            await _read_one(ws)  # hello
            frame = await _read_one(ws)
            assert frame["type"] == "frame"
            assert frame["text"] == ""
            await ws.close()


def test_register_clamps_low_poll_interval():
    app = _make_app(FakePane([]), interval=0.0001)
    # Stash key uses module-private AppKey but we can still iterate values.
    intervals = [v for k, v in app.items() if isinstance(v, float)]
    assert intervals
    assert min(intervals) >= 0.05


def test_default_poll_interval_constant():
    assert DEFAULT_POLL_INTERVAL == 0.2


async def test_build_app_includes_terminal_route():
    app = build_app(bot_token=BOT, terminal_capture=FakePane(["hello"]))
    routes = {
        getattr(r.resource, "canonical", str(r.resource)) for r in app.router.routes()
    }
    assert any("/ws/terminal" in r for r in routes)


async def test_default_capture_delegates_to_tmux_singleton(monkeypatch):
    from ccgram.miniapp.api import terminal as term_mod
    from ccgram.multiplexer import tmux as tmux_mod

    captured: list[tuple[str, bool]] = []

    async def fake_capture(window_id: str, *, with_ansi: bool = False) -> str:
        captured.append((window_id, with_ansi))
        return "tmux-out"

    monkeypatch.setattr(tmux_mod.tmux_manager, "capture_pane", fake_capture)
    out = await term_mod._default_capture("ccgram:@1")
    assert out == "tmux-out"
    assert captured == [("ccgram:@1", True)]


async def test_default_pane_capture_delegates_to_tmux_singleton(monkeypatch):
    from ccgram.miniapp.api import terminal as term_mod
    from ccgram.multiplexer import tmux as tmux_mod

    seen: list[tuple[str, str, bool]] = []

    async def fake_pane(
        pane_id: str, *, with_ansi: bool = False, window_id: str = ""
    ) -> str:
        seen.append((pane_id, window_id, with_ansi))
        return "pane-out"

    monkeypatch.setattr(tmux_mod.tmux_manager, "capture_pane_by_id", fake_pane)
    out = await term_mod._default_pane_capture("ccgram:@1", "%5")
    assert out == "pane-out"
    assert seen == [("%5", "ccgram:@1", True)]


async def test_default_pane_list_merges_window_state(monkeypatch):
    from ccgram.miniapp.api import terminal as term_mod
    from ccgram.multiplexer import tmux as tmux_mod
    from ccgram.window_state_store import (
        WindowState,
        PaneInfo as StatePaneInfo,
        window_store,
    )
    from ccgram.multiplexer.base import PaneInfo as TmuxPaneInfo

    tmux_panes = [
        TmuxPaneInfo(
            pane_id="%5",
            index=0,
            active=True,
            command="claude",
            path="/tmp",
            width=80,
            height=24,
        ),
        TmuxPaneInfo(
            pane_id="%6",
            index=1,
            active=False,
            command="bash",
            path="/tmp",
            width=80,
            height=24,
        ),
    ]

    async def fake_list(window_id: str):
        return tmux_panes

    monkeypatch.setattr(tmux_mod.tmux_manager, "list_panes", fake_list)

    state = WindowState()
    state.panes["%5"] = StatePaneInfo(
        pane_id="%5",
        name="api",
        provider="claude",
        last_active_ts=0.0,
        state="active",
        subscribed=True,
    )
    monkeypatch.setitem(window_store.window_states, "ccgram:@9", state)
    try:
        out = await term_mod._default_pane_list("ccgram:@9")
    finally:
        window_store.window_states.pop("ccgram:@9", None)

    by_id = {entry["pane_id"]: entry for entry in out}
    assert by_id["%5"]["name"] == "api"
    assert by_id["%5"]["subscribed"] is True
    assert by_id["%5"]["state"] == "active"
    assert by_id["%6"]["name"] is None
    assert by_id["%6"]["subscribed"] is False
    assert by_id["%6"]["state"] == "idle"


async def test_default_pane_list_handles_missing_window(monkeypatch):
    from ccgram.miniapp.api import terminal as term_mod
    from ccgram.multiplexer import tmux as tmux_mod
    from ccgram.multiplexer.base import PaneInfo as TmuxPaneInfo

    async def fake_list(window_id: str):
        return [
            TmuxPaneInfo(
                pane_id="%1",
                index=0,
                active=True,
                command="zsh",
                path="/",
                width=80,
                height=24,
            )
        ]

    monkeypatch.setattr(tmux_mod.tmux_manager, "list_panes", fake_list)
    out = await term_mod._default_pane_list("ccgram:@nonexistent")
    assert out and out[0]["name"] is None
    assert out[0]["state"] == "active"


async def test_panes_endpoint_accepts_init_data_header():
    """HTTP /api/panes/* must read initData from the header, not the URL."""

    async def pane_list(_window_id: str) -> list[dict]:
        return [{"pane_id": "%1", "active": True, "name": "x"}]

    app = web.Application()
    register_terminal_routes(
        app, bot_token=BOT, capture=FakePane([]), pane_list=pane_list
    )
    async with TestClient(TestServer(app)) as c:
        tok = sign_token(bot_token=BOT, window_id=WINDOW_ID, user_id=42)
        # Without header → 403.
        resp = await c.get(f"/api/panes/{tok}")
        assert resp.status == 403
        # With valid header → 200.
        resp = await c.get(f"/api/panes/{tok}", headers=_init_headers())
        assert resp.status == 200
        body = await resp.json()
        assert body["window_id"] == WINDOW_ID
        assert body["panes"][0]["pane_id"] == "%1"
