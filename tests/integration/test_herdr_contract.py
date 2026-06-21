"""F2 herdr leg — the Multiplexer contract against a live herdr server.

Marked ``herdr`` (and ``integration``); auto-skips when ``$HERDR_SOCKET_PATH``
is unset or the server is unreachable, so it never runs in ``make test``. Run
locally with a herdr server up::

    uv run pytest tests/integration/ -m "herdr" -v

Drives one real agent-less shell pane through the contract: create → send →
capture round-trips text, ``list_panes``/``pane_dims``/``foreground`` return
sane shapes, and ``kill_window`` removes it (design "Module test
specifications").
"""

from __future__ import annotations

import asyncio
import os

import pytest

from ccgram.multiplexer.herdr import HerdrError, HerdrManager

pytestmark = [pytest.mark.integration, pytest.mark.herdr]


def _socket_or_skip() -> str:
    socket = os.environ.get("HERDR_SOCKET_PATH", "")
    if not socket or not os.path.exists(socket):
        pytest.skip("herdr socket not available ($HERDR_SOCKET_PATH unset/missing)")
    return socket


@pytest.fixture
async def herdr() -> HerdrManager:
    socket = _socket_or_skip()
    mgr = HerdrManager(socket_path=socket)
    try:
        await mgr.ensure_session()
    except HerdrError as exc:
        pytest.skip(f"herdr server unavailable: {exc}")
    return mgr


async def _capture_until(
    mgr: HerdrManager, window_id: str, needle: str, *, timeout: float = 8.0
) -> str:
    """Poll ``capture`` until *needle* appears, or fail after *timeout*."""
    deadline = asyncio.get_event_loop().time() + timeout
    last = ""
    while asyncio.get_event_loop().time() < deadline:
        result = await mgr.capture(window_id)
        last = result.text if result else ""
        if needle in last:
            return last
        await asyncio.sleep(0.3)
    raise AssertionError(f"never saw {needle!r}; last capture:\n{last}")


async def test_create_send_capture_kill_roundtrip(
    herdr: HerdrManager, tmp_path
) -> None:
    ok, _msg, _name, window_id = await herdr.create_window(
        str(tmp_path), window_name="ccgram-itest", start_agent=False
    )
    assert ok is True
    assert window_id  # a herdr pane id like "wN:pM"

    try:
        # send → capture round-trips text through the real pane.
        marker = "ccgram_herdr_marker_42"
        assert await herdr.send(window_id, f"echo {marker}") is True
        text = await _capture_until(herdr, window_id, marker)
        assert marker in text

        # list_panes: a herdr window is one pane.
        panes = await herdr.list_panes(window_id)
        assert len(panes) == 1
        assert panes[0].pane_id == window_id

        # pane_dims: positive cols/rows.
        dims = await herdr.pane_dims(window_id)
        assert dims is not None and dims.width > 0 and dims.height > 0

        # foreground: a real pid and argv, no tty (capability says so).
        fg = await herdr.foreground(window_id)
        assert fg is not None
        assert fg.pid > 0
        assert fg.argv
        assert fg.tty == ""
    finally:
        assert await herdr.kill_window(window_id) is True

    # Window is gone after kill.
    await asyncio.sleep(0.3)
    assert await herdr.find_window(window_id) is None


async def test_scrollback_clamps_to_capability(herdr: HerdrManager, tmp_path) -> None:
    ok, _msg, _name, window_id = await herdr.create_window(
        str(tmp_path), window_name="ccgram-itest-scroll", start_agent=False
    )
    assert ok is True
    try:
        # Asking past the 1000-line cap reports truncation.
        result = await herdr.capture_scrollback(window_id, lines=5000)
        if result is not None:
            assert result.truncated is True
    finally:
        await herdr.kill_window(window_id)
