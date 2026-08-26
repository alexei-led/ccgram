"""Integration tests for tmux auto-detection with a real tmux server."""

import asyncio
import os
import shutil
import subprocess

import pytest

from ccgram.multiplexer.tmux import TmuxManager
from ccgram.utils import check_duplicate_ccgram, detect_tmux_context

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed"),
]

TEST_SESSION = f"ccgram-autodetect-test-{os.environ.get('PYTEST_XDIST_WORKER', 'main')}"


@pytest.fixture()
def tmux():
    mgr = TmuxManager(session_name=TEST_SESSION)
    mgr.get_or_create_session()
    yield mgr
    session = mgr.get_session()
    if session:
        session.kill()


async def _make_window(tmux: TmuxManager, tmp_path, name: str) -> tuple[str, str]:
    """Create a window and return (window_id, its active pane_id)."""
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name=name, start_agent=False
    )
    assert ok
    panes = await tmux.list_panes(window_id)
    assert panes
    return window_id, panes[0].pane_id


def _inside_tmux_env(monkeypatch, pane_id: str) -> None:
    """Set $TMUX/$TMUX_PANE the way a process running inside this server sees them."""
    probe = subprocess.run(
        ["tmux", "display-message", "-t", TEST_SESSION, "-p", "#{socket_path}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert probe.returncode == 0, probe.stderr
    monkeypatch.setenv("TMUX", f"{probe.stdout.strip()},0,0")
    monkeypatch.setenv("TMUX_PANE", pane_id)


async def _run_as_ccgram(tmux: TmuxManager, window_id: str) -> bool:
    """Rename a pane's process to ``ccgram``; False if the shell never exec'd it."""
    session = tmux.get_session()
    assert session
    window = session.windows.get(window_id=window_id, default=None)
    assert window
    pane = window.active_pane
    assert pane
    pane.send_keys("exec bash -c 'exec -a ccgram sleep 60'", enter=True)
    for _ in range(8):
        await asyncio.sleep(0.25)
        if _session_has_ccgram_pane():
            return True
    return False


class TestDetectTmuxContext:
    def test_returns_nothing_outside_tmux(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        assert detect_tmux_context() == (None, None)

    async def test_resolves_session_and_window_from_own_pane(
        self, tmux, tmp_path, monkeypatch
    ):
        window_id, pane_id = await _make_window(tmux, tmp_path, "ctx-probe")
        _inside_tmux_env(monkeypatch, pane_id)

        assert detect_tmux_context() == (TEST_SESSION, window_id)

    async def test_unknown_pane_resolves_to_nothing(self, tmux, monkeypatch):
        _inside_tmux_env(monkeypatch, "%99999")

        assert detect_tmux_context() == (None, None)


class TestDuplicateInstanceCheck:
    async def test_no_duplicate_when_no_ccgram_running(self, tmux, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%99999")
        assert check_duplicate_ccgram(TEST_SESSION) is None

    async def test_detects_ccgram_process_in_another_pane(
        self, tmux, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TMUX_PANE", "%99999")
        window_id, _pane_id = await _make_window(tmux, tmp_path, "fake-ccgram")

        # Some sandboxed environments never execute piped keystrokes; without
        # the renamed process there is nothing for the check to find, so skip
        # rather than assert on an environment that cannot reproduce the case.
        if not await _run_as_ccgram(tmux, window_id):
            pytest.skip("tmux pane shell did not exec the renamed process here")

        dup = check_duplicate_ccgram(TEST_SESSION)
        assert dup is not None
        assert "Another ccgram instance" in dup
        assert window_id in dup

    async def test_own_pane_is_never_reported_as_duplicate(
        self, tmux, tmp_path, monkeypatch
    ):
        window_id, pane_id = await _make_window(tmux, tmp_path, "self-ccgram")

        if not await _run_as_ccgram(tmux, window_id):
            pytest.skip("tmux pane shell did not exec the renamed process here")

        monkeypatch.setenv("TMUX_PANE", pane_id)
        assert check_duplicate_ccgram(TEST_SESSION) is None


def _session_has_ccgram_pane() -> bool:
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
    return any(line.strip() == "ccgram" for line in result.stdout.strip().splitlines())


async def test_list_windows_skips_own_window(tmux, tmp_path, monkeypatch):
    """list_windows excludes our own window when own_window_id is set."""
    agent_id, _ = await _make_window(tmux, tmp_path, "agent-win")
    own_id, _ = await _make_window(tmux, tmp_path, "ccgram-self")

    all_ids = [w.window_id for w in await tmux.list_windows()]
    assert agent_id in all_ids
    assert own_id in all_ids

    from ccgram.config import config

    monkeypatch.setattr(config, "own_window_id", own_id)
    filtered_ids = [w.window_id for w in await tmux.list_windows()]
    assert agent_id in filtered_ids
    assert own_id not in filtered_ids
