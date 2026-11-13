"""Real native marks on a private tmux socket, never the user's server."""

from pathlib import Path
import shutil
import tempfile

import libtmux
import pytest

from ccgram.multiplexer.tmux import TmuxManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed"),
]


@pytest.fixture
def isolated_tmux(tmp_path: Path):
    # macOS Unix sockets have a short path limit; pytest tmp_path can exceed it.
    socket_dir = tempfile.TemporaryDirectory(prefix="ccg-mark-", dir="/tmp")
    server = libtmux.Server(
        socket_path=str(Path(socket_dir.name) / "mux.sock"), config_file="/dev/null"
    )
    session = server.new_session(session_name="marks", start_directory=str(tmp_path))
    session.set_option("default-shell", "/bin/sh")
    mgr = TmuxManager(session_name="marks")
    mgr._server = server
    try:
        yield mgr, session
    finally:
        server.kill()
        socket_dir.cleanup()


async def test_local_marks_ignore_global_defaults_and_survive_reconnect(
    isolated_tmux,
) -> None:
    mgr, session = isolated_tmux
    server = session.server
    server.cmd("set-option", "-gw", "@ccgram_topic", "1")
    session.set_option("@ccgram_topic", "1")
    first = session.new_window(window_name="project")
    second = session.new_window(window_name="project")
    assert first.window_id is not None and second.window_id is not None
    before = await mgr.list_windows()
    assert {window.window_id for window in before} >= {
        first.window_id,
        second.window_id,
    }
    assert all(window.topic_marked is False for window in before)
    assert await mgr.set_topic_mark(first.window_id, True)
    assert await mgr.set_topic_mark(first.window_id, True)

    reconnected = TmuxManager(session_name="marks")
    reconnected._server = libtmux.Server(socket_path=server.socket_path)
    windows = await reconnected.list_windows_for_reconciliation()
    assert windows is not None
    marks = {window.window_id: window.topic_marked for window in windows}
    assert marks[first.window_id] is True
    assert marks[second.window_id] is False

    first.rename_window("_hidden")
    hidden = await reconnected.find_window_by_id(first.window_id)
    assert hidden is not None and hidden.topic_marked is True
    assert not hidden.topic_eligible
    assert await reconnected.set_topic_mark(first.window_id, False)
    assert first in session.windows
    assert server.cmd("show-options", "-gwv", "@ccgram_topic").stdout == ["1"]
    assert session.show_option("@ccgram_topic") == 1


async def test_new_window_and_respawn_do_not_copy_or_lose_selection(
    isolated_tmux, tmp_path: Path
) -> None:
    mgr, session = isolated_tmux
    first = session.new_window(window_name="project")
    assert first.window_id is not None
    assert await mgr.set_topic_mark(first.window_id, True)
    result = first.cmd("respawn-pane", "-k", "/bin/sh")
    assert result.returncode == 0
    after_respawn = await mgr.find_window_by_id(first.window_id)
    assert after_respawn is not None and after_respawn.topic_marked is True
    ok, _, _, new_id = await mgr.create_window(
        str(tmp_path), window_name="project", start_agent=False
    )
    assert ok
    created = await mgr.find_window_by_id(new_id)
    assert created is not None and created.topic_marked is False


async def test_server_destruction_does_not_restore_marks_by_reused_id(
    isolated_tmux,
) -> None:
    mgr, session = isolated_tmux
    original = session.new_window(window_name="project")
    assert original.window_id is not None
    assert await mgr.set_topic_mark(original.window_id, True)
    session.server.kill()
    replacement = session.server.new_session(session_name="marks")
    replacement.new_window(window_name="project")
    windows = await mgr.list_windows_for_reconciliation()
    assert windows is not None
    assert original.window_id in {window.window_id for window in windows}
    assert all(window.topic_marked is False for window in windows)
