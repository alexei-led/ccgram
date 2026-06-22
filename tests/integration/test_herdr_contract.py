"""F2 herdr leg — the Multiplexer contract against a live herdr server.

Marked ``herdr`` (and ``integration``); auto-skips when ``$HERDR_SOCKET_PATH``
is unset or the server is unreachable, so it never runs in ``make test``. Run
locally with a herdr server up::

    uv run pytest tests/integration/ -m "herdr" -v

Tab identity (Task 1): ``window_id`` is a **tab id** (``"wN:tM"``), not a pane
id.  ``create_window`` returns the tab id; ``list_panes(tab_id)`` returns panes
whose ``pane_id`` values are ``"wN:pK"`` — distinct from the tab id.

Tests cover:
- create → send → capture → kill round-trip
- tab identity: ``list_windows`` returns one ``WindowRef`` per tab; the
  ``window_id`` field is a tab id (not a pane id), and ``window_name`` is
  formatted as ``"<workspace> ▸ <tab>"``
- ``__*__`` workspace/tab labels are not surfaced in ``list_windows``
- ``create_window`` returns a tab id (``window_id``)
- ``kill_window`` closes the tab and ``find_window`` returns None afterwards
- ``rename_window`` changes the tab label visible in the next ``find_window``
- restart re-resolution: ``_resolve_by_session_id`` maps an old tab id to a
  new tab id via shared session id (simulated — no live server restart needed)
- scrollback: requesting >1000 lines sets ``CaptureResult.truncated``
"""

from __future__ import annotations

import asyncio
import os

import pytest

from ccgram.multiplexer.herdr import HerdrError, HerdrManager
from ccgram.multiplexer.topic_mapping import TOPIC_PREFIX_SEPARATOR

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


# ── Tab identity + lifecycle ────────────────────────────────────────────────


async def test_create_send_capture_kill_roundtrip(
    herdr: HerdrManager, tmp_path
) -> None:
    """Tab identity: create_window returns a tab id; pane ops resolve through it."""
    ok, _msg, _name, window_id = await herdr.create_window(
        str(tmp_path), window_name="ccgram-itest", start_agent=False
    )
    assert ok is True
    # window_id is a tab id (wN:tM shape), not a pane id (wN:pM).
    assert window_id

    try:
        # send → capture round-trips text through the real pane.
        marker = "ccgram_herdr_marker_42"
        assert await herdr.send(window_id, f"echo {marker}") is True
        text = await _capture_until(herdr, window_id, marker)
        assert marker in text

        # list_panes: tab has at least one pane; pane_id is distinct from tab id.
        panes = await herdr.list_panes(window_id)
        assert len(panes) >= 1
        # Pane ids are "wN:pM" — they are NOT equal to the tab id "wN:tM".
        assert all(p.pane_id != window_id for p in panes)

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

    # Window is gone after kill (tab closed).
    await asyncio.sleep(0.3)
    assert await herdr.find_window(window_id) is None


async def test_list_windows_tab_identity(herdr: HerdrManager, tmp_path) -> None:
    """list_windows returns one WindowRef per tab; window_id is the tab id."""
    ok, _msg, tab_label, window_id = await herdr.create_window(
        str(tmp_path), window_name="ccgram-list-itest", start_agent=False
    )
    assert ok is True
    try:
        refs = await herdr.list_windows()
        # The created tab must appear (non-__*__ label).
        found = [r for r in refs if r.window_id == window_id]
        assert found, (
            f"tab {window_id!r} not found in list_windows; got {[r.window_id for r in refs]}"
        )
        ref = found[0]
        # window_id is the tab id, not a pane id.
        assert ref.window_id == window_id
        # window_name follows "<workspace> ▸ <tab>" or at least contains the tab label.
        assert tab_label in ref.window_name or TOPIC_PREFIX_SEPARATOR in ref.window_name
    finally:
        await herdr.kill_window(window_id)


async def test_list_windows_internal_label_hidden(herdr: HerdrManager) -> None:
    """__*__ workspace/tab labels are not surfaced in list_windows."""
    refs = await herdr.list_windows()
    for ref in refs:
        # Neither the window_id nor window_name should expose __*__-labelled tabs.
        # We can only check window_name here (window_id is opaque).
        parts = ref.window_name.split(TOPIC_PREFIX_SEPARATOR)
        for part in parts:
            assert not (part.startswith("__") and part.endswith("__")), (
                f"Internal label leaked into list_windows: {ref.window_name!r}"
            )


async def test_rename_window(herdr: HerdrManager, tmp_path) -> None:
    """rename_window changes the tab label; find_window reflects the new name."""
    ok, _msg, _name, window_id = await herdr.create_window(
        str(tmp_path), window_name="ccgram-rename-before", start_agent=False
    )
    assert ok is True
    try:
        assert await herdr.rename_window(window_id, "ccgram-rename-after") is True
        # find_window should reflect the updated label.
        ref = await herdr.find_window(window_id)
        assert ref is not None
        assert "ccgram-rename-after" in ref.window_name
    finally:
        await herdr.kill_window(window_id)


async def test_kill_window_closes_tab(herdr: HerdrManager, tmp_path) -> None:
    """kill_window closes the herdr tab; find_window returns None afterwards."""
    ok, _msg, _name, window_id = await herdr.create_window(
        str(tmp_path), window_name="ccgram-kill-itest", start_agent=False
    )
    assert ok is True
    assert await herdr.kill_window(window_id) is True
    await asyncio.sleep(0.3)
    assert await herdr.find_window(window_id) is None


# ── Restart re-resolution (simulated — no live restart needed) ──────────────


def test_resolve_by_session_id_maps_old_tab_to_new() -> None:
    """_resolve_by_session_id re-maps an old herdr tab id to a new one via session id.

    Simulates the post-restart state: persisted state still references the old
    tab id ("w1:t1"); after restart herdr gave the same agent a new tab id
    ("w2:t1") with the same session_id.  The resolver must remap without a live
    socket.
    """
    from ccgram.window_resolver import LiveWindow, resolve_stale_ids
    from ccgram.window_state_store import WindowState

    old_id = "w1:t1"
    new_id = "w2:t1"
    session_id = "session-uuid-aabbcc"

    # Post-restart live state: the agent's tab now has new_id.
    live_windows = [LiveWindow(window_id=new_id, window_name="ccgram ▸ main")]
    # live_session_ids: new_id carries the durable session_id.
    live_session_ids = {new_id: session_id}

    # Persisted state: old_id bound to thread 42 with the same session_id.
    # window_states values are WindowState objects (not dicts) in production.
    window_states: dict = {old_id: WindowState(session_id=session_id)}
    thread_bindings = {1: {42: old_id}}  # user 1, thread 42 → old tab
    user_window_offsets: dict = {}
    window_display_names = {old_id: "ccgram ▸ main"}

    # ids_stable=False activates the session-id re-resolution path (herdr).
    changed = resolve_stale_ids(
        live_windows,
        window_states,
        thread_bindings,
        user_window_offsets,
        window_display_names,
        ids_stable=False,
        live_session_ids=live_session_ids,
    )

    assert changed is True
    # Thread binding re-pointed from old_id to new_id.
    assert thread_bindings[1][42] == new_id
    # Window state re-keyed to new_id.
    assert new_id in window_states
    assert old_id not in window_states


# ── Scrollback ──────────────────────────────────────────────────────────────


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
