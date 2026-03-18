"""Integration tests for TmuxManager with a real tmux server."""

import asyncio
import shutil

import pytest

from ccgram.tmux_manager import TmuxManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed"),
]

TEST_SESSION = "ccgram-test-integration"


@pytest.fixture()
async def tmux(tmp_path):
    mgr = TmuxManager(session_name=TEST_SESSION)
    mgr.get_or_create_session()
    yield mgr
    session = mgr.get_session()
    if session:
        session.kill()


async def test_create_and_list_windows(tmux, tmp_path) -> None:
    ok, _msg, name, window_id = await tmux.create_window(
        str(tmp_path), window_name="test-win", start_agent=False
    )
    assert ok
    assert name == "test-win"
    assert window_id.startswith("@")

    windows = await tmux.list_windows()
    ids = [w.window_id for w in windows]
    assert window_id in ids

    match = next(w for w in windows if w.window_id == window_id)
    assert match.window_name == "test-win"


async def test_find_window_by_id(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="find-me", start_agent=False
    )
    assert ok

    found = await tmux.find_window_by_id(window_id)
    assert found is not None
    assert found.window_name == "find-me"

    missing = await tmux.find_window_by_id("@99999")
    assert missing is None


async def test_send_keys_and_capture_pane(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="echo-win", start_agent=False
    )
    assert ok

    await tmux.send_keys(window_id, "echo hello-integration")

    await asyncio.sleep(0.5)

    output = await tmux.capture_pane(window_id)
    assert output is not None
    assert "hello-integration" in output


async def test_kill_window(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="kill-me", start_agent=False
    )
    assert ok

    killed = await tmux.kill_window(window_id)
    assert killed is True

    windows = await tmux.list_windows()
    ids = [w.window_id for w in windows]
    assert window_id not in ids


async def test_reset_server_reconnects(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="reset-test", start_agent=False
    )
    assert ok

    tmux._reset_server()

    windows = await tmux.list_windows()
    ids = [w.window_id for w in windows]
    assert window_id in ids


async def test_capture_pane_raw_returns_tuple(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="raw-test", start_agent=False
    )
    assert ok

    # Send something so pane has content (empty panes return None)
    await tmux.send_keys(window_id, "echo raw-test-output")

    await asyncio.sleep(0.5)

    result = await tmux.capture_pane_raw(window_id)
    assert result is not None
    content, cols, rows = result
    assert isinstance(content, str)
    assert "raw-test-output" in content
    assert cols > 0
    assert rows > 0


async def test_get_pane_title(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="title-test", start_agent=False
    )
    assert ok

    title = await tmux.get_pane_title(window_id)
    assert isinstance(title, str)


# ── Pane-level operations ────────────────────────────────────────────


async def test_list_panes_single(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="pane-list", start_agent=False
    )
    assert ok

    panes = await tmux.list_panes(window_id)
    assert len(panes) == 1
    assert panes[0].active is True
    assert panes[0].pane_id.startswith("%")
    assert panes[0].width > 0
    assert panes[0].height > 0


async def test_list_panes_multiple(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="multi-pane", start_agent=False
    )
    assert ok

    # Split the window to create a second pane
    session = tmux.get_session()
    assert session
    window = session.windows.get(window_id=window_id)
    window.split()

    panes = await tmux.list_panes(window_id)
    assert len(panes) == 2
    pane_ids = [p.pane_id for p in panes]
    assert len(set(pane_ids)) == 2  # IDs are unique
    active_count = sum(1 for p in panes if p.active)
    assert active_count == 1


async def test_list_panes_missing_window(tmux) -> None:
    panes = await tmux.list_panes("@99999")
    assert panes == []


async def test_capture_pane_by_id(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="cap-pane", start_agent=False
    )
    assert ok

    panes = await tmux.list_panes(window_id)
    pane_id = panes[0].pane_id

    await tmux.send_keys(window_id, "echo pane-capture-test")
    await asyncio.sleep(0.5)

    output = await tmux.capture_pane_by_id(pane_id)
    assert output is not None
    assert "pane-capture-test" in output


async def test_capture_pane_by_id_missing(tmux) -> None:
    output = await tmux.capture_pane_by_id("%99999")
    assert output is None


async def test_send_keys_to_pane(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="send-pane", start_agent=False
    )
    assert ok

    # Split to create two panes
    session = tmux.get_session()
    assert session
    window = session.windows.get(window_id=window_id)
    window.split()

    panes = await tmux.list_panes(window_id)
    assert len(panes) == 2

    # Send to the non-active pane
    inactive = next(p for p in panes if not p.active)
    sent = await tmux.send_keys_to_pane(inactive.pane_id, "echo pane-target-test")
    assert sent is True
    await asyncio.sleep(0.5)

    output = await tmux.capture_pane_by_id(inactive.pane_id)
    assert output is not None
    assert "pane-target-test" in output


async def test_send_keys_to_pane_missing(tmux) -> None:
    sent = await tmux.send_keys_to_pane("%99999", "hello")
    assert sent is False
