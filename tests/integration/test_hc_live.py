"""Live checks against the running herdr-core socket (Herdr desktop).

Opt-in: these run only when the marker expression selects ``hc``
(``uv run pytest tests/integration/test_hc_live.py -m hc``); every other run,
including ``make test-integration``, skips them.

They never touch an existing terminal. Every write goes to one throwaway
terminal the test starts itself with ``terminal.run``: hidden (no Herdr tab),
labelled ``ccgram.probe=1``, and killed and removed in a ``finally`` block.
Reads elsewhere are limited to ``hello`` and ``terminal.list``.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator

import pytest

from ccgram.multiplexer.base import MuxEvent
from ccgram.multiplexer.hc import HcManager, window_id_for_address
from ccgram.multiplexer.hc_socket import HcRpcError, default_socket_path, request

pytestmark = [pytest.mark.integration, pytest.mark.hc]

# Captured at import: the shared autouse fixture scrubs HC_* variables per test.
_SOCKET = default_socket_path()
_PROBE_LABELS = {"ccgram.probe": "1"}
# Raw mode without echo, so keys arrive as cat -v renders them (ESC as ^[).
_PROBE_ARGV = ["/bin/sh", "-c", "stty raw -isig -echo; cat -v"]


def _hc_selected(config: pytest.Config) -> bool:
    expr = str(config.getoption("markexpr") or "")
    return bool(re.search(r"(?<!not )\bhc\b", expr))


@pytest.fixture(autouse=True)
def _opt_in(request: pytest.FixtureRequest) -> None:
    if not _hc_selected(request.config):
        pytest.skip("live herdr-core test: run with -m hc")
    if not os.path.exists(_SOCKET):
        pytest.skip(f"herdr-core socket not found at {_SOCKET}")


@pytest.fixture
async def probe_terminal() -> AsyncIterator[str]:
    """Start a hidden throwaway terminal; always kill and remove it."""
    result = await request(
        _SOCKET,
        "terminal.run",
        {
            "argv": _PROBE_ARGV,
            "cwd": "/tmp",
            "labels": _PROBE_LABELS,
            "no_notify": True,
            "no_parent": True,
        },
        10.0,
    )
    address = result["address"]
    assert result.get("labels", {}).get("ccgram.probe") == "1"
    try:
        await asyncio.sleep(0.5)  # let stty switch the terminal to raw mode
        yield address
    finally:
        try:
            await request(_SOCKET, "terminal.kill", {"terminal": address}, 10.0)
        except HcRpcError as exc:
            if exc.kind not in {"not_found", "finished"}:
                raise
        for _attempt in range(20):
            try:
                await request(_SOCKET, "terminal.rm", {"terminal": address}, 5.0)
                break
            except HcRpcError as exc:
                if exc.kind == "not_found":
                    break
                if exc.kind != "live":
                    raise
                await asyncio.sleep(0.25)


@pytest.fixture
def manager() -> HcManager:
    return HcManager(socket_path=_SOCKET, own_terminal_id="")


async def _screen_contains(manager: HcManager, window_id: str, needle: str) -> bool:
    for _attempt in range(20):
        text = await manager.capture_pane(window_id) or ""
        if needle in text:
            return True
        await asyncio.sleep(0.2)
    return False


async def test_handshake(manager: HcManager) -> None:
    await manager.ensure_session()


async def test_probe_terminal_is_listed_but_never_adopted(
    manager: HcManager, probe_terminal: str
) -> None:
    window_id = window_id_for_address(probe_terminal)
    complete = await manager.list_windows_for_reconciliation()
    assert complete is not None
    probe = next(w for w in complete if w.window_id == window_id)
    assert probe.topic_eligible is False
    assert window_id not in {w.window_id for w in await manager.list_windows()}
    assert await manager.window_exists(window_id) is True
    found = await manager.find_window_by_id(window_id)
    assert found is not None and found.cwd in {"/tmp", "/private/tmp"}
    dims = await manager.pane_dims(window_id)
    assert dims is not None and dims.width > 0 and dims.height > 0
    foreground = await manager.foreground(window_id)
    assert foreground is not None and foreground.pid > 0 and foreground.pgid == 0
    assert await manager.agent_status(window_id) is None


async def test_reads_keys_and_send(manager: HcManager, probe_terminal: str) -> None:
    window_id = window_id_for_address(probe_terminal)
    assert await manager.send(window_id, "Escape", enter=False, literal=False)
    assert await _screen_contains(manager, window_id, "^[")
    assert await manager.send(window_id, "C-y", enter=False, literal=False)
    assert await _screen_contains(manager, window_id, "^Y")
    assert await manager.send(window_id, "Ab9", enter=False)
    assert await _screen_contains(manager, window_id, "Ab9")
    assert await manager.send(window_id, "hello from ccgram")
    assert await _screen_contains(manager, window_id, "hello from ccgram")
    loop = asyncio.get_running_loop()
    started = loop.time()
    capture = await manager.capture_scrollback(window_id, lines=50)
    assert capture is not None and "hello from ccgram" in capture.text
    assert loop.time() - started < 5


async def test_watch_events_reports_the_probe_dying(
    manager: HcManager, probe_terminal: str
) -> None:
    window_id = window_id_for_address(probe_terminal)
    stream = manager.watch_events([window_id])
    first = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(1.0)  # let the subscription go live before the kill
    await request(_SOCKET, "terminal.kill", {"terminal": probe_terminal}, 10.0)
    try:
        async with asyncio.timeout(10):
            event: MuxEvent = await first
            while event.kind != "window_died":
                event = await anext(stream)
    finally:
        if not first.done():
            first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        await stream.aclose()
    assert event == MuxEvent(kind="window_died", window_id=window_id)
    assert await manager.window_exists(window_id) is False
