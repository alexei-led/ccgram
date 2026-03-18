"""Integration tests for tmux auto-detection with a real tmux server."""

import asyncio
import shutil
import subprocess

import pytest

from ccgram.tmux_manager import TmuxManager
from ccgram.utils import check_duplicate_ccgram, detect_tmux_context

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed"),
]

TEST_SESSION = "ccgram-autodetect-test"


@pytest.fixture()
def tmux():
    mgr = TmuxManager(session_name=TEST_SESSION)
    mgr.get_or_create_session()
    yield mgr
    session = mgr.get_session()
    if session:
        session.kill()


def test_detect_tmux_context_real(monkeypatch):
    """detect_tmux_context returns values when inside tmux."""
    monkeypatch.setenv("TMUX", "/tmp/tmux-test/default,99999,0")
    session, _window_id = detect_tmux_context()
    if session is not None:
        assert isinstance(session, str)
        assert len(session) > 0


async def test_check_duplicate_no_ccgram_running(tmux, monkeypatch):
    """No duplicate detected when no ccgram process is running."""
    monkeypatch.setenv("TMUX_PANE", "%99999")
    result = check_duplicate_ccgram(TEST_SESSION)
    assert result is None


async def test_list_windows_skips_own_window(tmux, tmp_path, monkeypatch):
    """list_windows excludes our own window when own_window_id is set."""
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="agent-win", start_agent=False
    )
    assert ok

    ok2, _msg2, _name2, own_id = await tmux.create_window(
        str(tmp_path), window_name="ccgram-self", start_agent=False
    )
    assert ok2

    windows_before = await tmux.list_windows()
    all_ids = [w.window_id for w in windows_before]
    assert window_id in all_ids
    assert own_id in all_ids

    from ccgram.config import config

    monkeypatch.setattr(config, "own_window_id", own_id)
    windows_after = await tmux.list_windows()
    filtered_ids = [w.window_id for w in windows_after]
    assert window_id in filtered_ids
    assert own_id not in filtered_ids


async def test_check_duplicate_with_ccgram_process(tmux, tmp_path, monkeypatch):
    """Duplicate detection finds ccgram process in another pane."""
    monkeypatch.setenv("TMUX_PANE", "%99999")

    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="fake-ccgram", start_agent=False
    )
    assert ok

    session = tmux.get_session()
    assert session
    window = session.windows.get(window_id=window_id, default=None)
    assert window
    pane = window.active_pane
    assert pane

    pane.send_keys("exec bash -c 'exec -a ccgram sleep 60'", enter=True)
    await asyncio.sleep(0.5)

    result = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-s",
            "-t",
            TEST_SESSION,
            "-F",
            "#{pane_current_command}",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    has_ccgram = any(
        line.strip() == "ccgram" for line in result.stdout.strip().splitlines()
    )
    if has_ccgram:
        dup = check_duplicate_ccgram(TEST_SESSION)
        assert dup is not None
        assert "Another ccgram instance" in dup
