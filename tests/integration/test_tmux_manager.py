"""Integration tests for TmuxManager with a real tmux server."""

import asyncio
import shutil

import pytest

from ccgram.multiplexer.tmux import TmuxManager

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


async def test_capture_pane_by_id_missing(tmux) -> None:
    output = await tmux.capture_pane_by_id("%99999")
    assert output is None


# ── ANSI capture ───────────────────────────────────────────────────────


async def test_capture_pane_with_ansi(tmux, tmp_path) -> None:
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="ansi-test", start_agent=False
    )
    assert ok

    # raw=True: plain shell window — use the direct send path, not the
    # Claude-TUI path (vim-mode probe + 500ms Enter delay). The trailing
    # `touch` of a sentinel file proves the shell actually executed the line,
    # independent of pane capture: the command is always *echoed* on screen
    # (so "red"/"normal"/the ESC sequence appear whether or not it ran), making
    # the captured text alone an unreliable execution signal.
    done = tmp_path / ".ansi_executed"
    await tmux.send_keys(
        window_id,
        f'printf "\\033[31mred\\033[0m normal"; touch "{done}"',
        raw=True,
    )

    ansi = ""
    for _ in range(10):
        await asyncio.sleep(0.3)
        ansi = await tmux.capture_pane(window_id, with_ansi=True) or ""
        if done.exists():
            break

    # Some sandboxed CI environments run the pane under a shell that never
    # executes piped keystrokes — skip there. But if the shell DID execute
    # (sentinel present) yet ANSI capture lost the escape, that is a real
    # capture regression and must fail, not skip.
    if not done.exists():
        pytest.skip("tmux pane shell did not execute the command in this environment")

    plain = await tmux.capture_pane(window_id, with_ansi=False)
    assert plain is not None
    assert "red" in plain
    assert "normal" in plain
    assert "\x1b[31m" in ansi
    assert "red" in ansi


# ── YOLO bypass prompt detection ────────────────────────────────────────


async def test_accept_yolo_confirmation_detects_prompt(tmux, tmp_path) -> None:
    from unittest.mock import patch

    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="yolo-test", start_agent=False
    )
    assert ok

    await tmux.send_keys(
        window_id,
        'echo "WARNING: Claude Code running in Bypass Permissions mode"',
    )
    await asyncio.sleep(0.5)

    with patch("ccgram.handlers.topics.directory_callbacks.tmux_manager", tmux):
        from ccgram.handlers.topics.directory_callbacks import _accept_yolo_confirmation

        result = await _accept_yolo_confirmation(window_id, timeout=3.0)

    assert result is True


async def test_create_window_with_special_char_launch_command(tmux, tmp_path) -> None:
    """Launch command containing = and / is sent literally (regression for Gemini bug).

    Gemini's hardened launch command is `env VAR=/path/to/file gemini`.
    Without literal=True in pane.send_keys, the = and / are misinterpreted
    as tmux key sequences and the command is corrupted.
    """
    marker = "CCGRAM_LAUNCH_TEST_MARKER"
    launch_cmd = (
        f"env {marker}=special/path/value sh -c 'echo ${{CCGRAM_LAUNCH_TEST_MARKER}}'"
    )
    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path),
        window_name="literal-launch",
        start_agent=True,
        launch_command=launch_cmd,
    )
    assert ok

    await asyncio.sleep(1.0)
    output = await tmux.capture_pane(window_id)
    assert output is not None
    assert "special/path/value" in output, (
        f"Launch command with = and / was not sent literally. Pane output: {output!r}"
    )


async def test_accept_yolo_confirmation_timeout_on_no_prompt(tmux, tmp_path) -> None:
    from unittest.mock import patch

    ok, _msg, _name, window_id = await tmux.create_window(
        str(tmp_path), window_name="yolo-nope", start_agent=False
    )
    assert ok

    await tmux.send_keys(window_id, "echo hello world")
    await asyncio.sleep(0.3)

    with patch("ccgram.handlers.topics.directory_callbacks.tmux_manager", tmux):
        from ccgram.handlers.topics.directory_callbacks import _accept_yolo_confirmation

        result = await _accept_yolo_confirmation(window_id, timeout=1.0)

    assert result is False
