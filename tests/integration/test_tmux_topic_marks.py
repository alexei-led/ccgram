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


async def test_expected_owner_is_stable_through_unmark_and_native_respawn(
    isolated_tmux,
) -> None:
    mgr, session = isolated_tmux
    window = session.new_window(window_name="project")
    wid = window.window_id
    assert wid is not None
    row = await mgr.find_window_by_id(wid)
    assert row is not None and row.topic_mark_owner
    owner = row.topic_mark_owner
    assert await mgr.set_topic_mark(wid, True, owner=owner)
    token = window.cmd("show-options", "-wv", "@ccgram_topic").stdout[0]
    assert token.startswith("v1:1:")
    assert await mgr.set_topic_mark(wid, True, owner=owner)
    assert window.cmd("show-options", "-wv", "@ccgram_topic").stdout == [token]
    assert window.cmd("respawn-pane", "-k", "/bin/sh").returncode == 0
    assert await mgr.set_topic_mark("@999", False, owner=owner)
    cleared = await mgr.find_window_by_id(wid)
    assert cleared is not None and cleared.topic_mark_owner == owner
    assert cleared.topic_marked is False
    assert window.cmd("show-options", "-wv", "@ccgram_topic").stdout == [
        token.replace("v1:1:", "v1:0:")
    ]


async def test_explicit_metadata_restore_uses_owner_not_old_routing_id(
    isolated_tmux,
) -> None:
    mgr, session = isolated_tmux
    original = session.new_window(window_name="project")
    old_id = original.window_id
    assert old_id is not None
    row = await mgr.find_window_by_id(old_id)
    assert row is not None and row.topic_mark_owner
    owner = row.topic_mark_owner
    assert await mgr.set_topic_mark(old_id, True, owner=owner)
    token = original.cmd("show-options", "-wv", "@ccgram_topic").stdout[0]
    original.kill()
    restored = session.new_window(window_name="project")
    assert restored.window_id != old_id
    restored.set_option("@ccgram_topic", token)
    assert await mgr.set_topic_mark(old_id, False, owner=owner)
    found = await mgr.find_window_by_id(restored.window_id)
    assert found is not None and found.topic_mark_owner == owner
    assert found.topic_marked is False


async def test_copied_owners_are_quarantined_without_deleting_liveness(
    isolated_tmux,
) -> None:
    mgr, session = isolated_tmux
    first = session.new_window(window_name="project")
    assert first.window_id is not None
    row = await mgr.find_window_by_id(first.window_id)
    assert row is not None and row.topic_mark_owner
    assert await mgr.set_topic_mark(first.window_id, True, owner=row.topic_mark_owner)
    token = first.cmd("show-options", "-wv", "@ccgram_topic").stdout[0]
    second = session.new_window(window_name="project")
    second.set_option("@ccgram_topic", token)
    windows = await mgr.list_windows_for_reconciliation()
    assert windows is not None
    copies = [
        window
        for window in windows
        if window.window_id in {first.window_id, second.window_id}
    ]
    assert len(copies) == 2 and all(window.topic_eligible for window in copies)
    assert all(
        window.topic_marked is None and window.topic_mark_owner is None
        for window in copies
    )
    assert not await mgr.set_topic_mark(
        first.window_id, False, owner=row.topic_mark_owner
    )
    assert first.cmd("show-options", "-wv", "@ccgram_topic").stdout == [token]
    assert second.cmd("show-options", "-wv", "@ccgram_topic").stdout == [token]


async def test_reused_id_cannot_authorize_old_owner_after_server_destruction(
    isolated_tmux,
) -> None:
    mgr, session = isolated_tmux
    original = session.new_window(window_name="project")
    wid = original.window_id
    assert wid is not None
    old = await mgr.find_window_by_id(wid)
    assert old is not None and old.topic_mark_owner
    assert await mgr.set_topic_mark(wid, True, owner=old.topic_mark_owner)
    session.server.kill()
    replacement_session = session.server.new_session(session_name="marks")
    replacement = replacement_session.new_window(window_name="project")
    assert replacement.window_id == wid
    new = await mgr.find_window_by_id(wid)
    assert new is not None and new.topic_mark_owner != old.topic_mark_owner
    assert not await mgr.set_topic_mark(wid, True, owner=old.topic_mark_owner)
    assert await mgr.set_topic_mark(wid, True, owner=new.topic_mark_owner)
    token = replacement.cmd("show-options", "-wv", "@ccgram_topic").stdout
    assert await mgr.set_topic_mark(wid, False, owner=old.topic_mark_owner)
    assert replacement.cmd("show-options", "-wv", "@ccgram_topic").stdout == token


@pytest.mark.parametrize("marked", [True, False])
async def test_native_write_ack_loss_recovers_by_saved_owner(
    isolated_tmux, monkeypatch, marked: bool
) -> None:
    from types import SimpleNamespace

    mgr, session = isolated_tmux
    window = session.new_window(window_name="project")
    wid = window.window_id
    assert wid is not None
    row = await mgr.find_window_by_id(wid)
    assert row is not None and row.topic_mark_owner
    assert await mgr.set_topic_mark(wid, not marked, owner=row.topic_mark_owner)
    native = session.server.cmd
    writes = []

    def lost_ack(command, *args, **kwargs):
        result = native(command, *args, **kwargs)
        if command == "set-option":
            writes.append(args)
            return SimpleNamespace(returncode=1, stderr=["reply lost"], stdout=[])
        return result

    monkeypatch.setattr(session.server, "cmd", lost_ack)
    assert not await mgr.set_topic_mark(wid, marked, owner=row.topic_mark_owner)
    assert await mgr.set_topic_mark("@999", marked, owner=row.topic_mark_owner)
    assert len(writes) == 1


@pytest.mark.parametrize(
    "failed_command", ["show-options", "list-windows", "display-message"]
)
async def test_marker_read_failure_retains_liveness_and_never_settles_clear(
    isolated_tmux, monkeypatch, failed_command
) -> None:
    from types import SimpleNamespace

    mgr, session = isolated_tmux
    window = session.new_window(window_name="project")
    wid = window.window_id
    assert wid is not None
    row = await mgr.find_window_by_id(wid)
    assert row is not None and row.topic_mark_owner
    native = session.server.cmd

    def fail(command, *args, **kwargs):
        if command == failed_command and (command != "list-windows" or "-a" in args):
            return SimpleNamespace(returncode=1, stderr=["unavailable"], stdout=[])
        return native(command, *args, **kwargs)

    monkeypatch.setattr(session.server, "cmd", fail)
    rows = await mgr.list_windows_for_reconciliation()
    assert rows is not None
    live = next(window for window in rows if window.window_id == wid)
    assert (
        live.topic_eligible
        and live.topic_marked is None
        and live.topic_mark_owner is None
    )
    assert not await mgr.set_topic_mark(wid, False, owner=row.topic_mark_owner)


async def test_foreign_backend_target_or_owner_cannot_mutate(isolated_tmux) -> None:
    mgr, session = isolated_tmux
    window = session.new_window(window_name="project")
    wid = window.window_id
    assert wid is not None
    row = await mgr.find_window_by_id(wid)
    assert row is not None and row.topic_mark_owner
    assert not await mgr.set_topic_mark(
        "157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4", True, owner=row.topic_mark_owner
    )
    assert not await mgr.set_topic_mark(
        wid, True, owner="agterm:157b4c8c-efae-40c2-ba54-9a5d7fd8b5e4"
    )
    assert not any(
        line.startswith("@ccgram_topic ")
        for line in window.cmd("show-options", "-w").stdout
    )


async def test_malformed_owned_marker_prevents_false_absence_clear(
    isolated_tmux,
) -> None:
    mgr, session = isolated_tmux
    window = session.new_window(window_name="project")
    wid = window.window_id
    assert wid is not None
    row = await mgr.find_window_by_id(wid)
    assert row is not None and row.topic_mark_owner
    assert await mgr.set_topic_mark(wid, True, owner=row.topic_mark_owner)
    broken = session.new_window(window_name="broken")
    broken.set_option("@ccgram_topic", "v1:1:broken")
    assert not await mgr.set_topic_mark(wid, False, owner=row.topic_mark_owner)
    unchanged = await mgr.find_window_by_id(wid)
    assert unchanged is not None and unchanged.topic_marked is True
    unknown = await mgr.find_window_by_id(broken.window_id)
    assert (
        unknown is not None
        and unknown.topic_marked is None
        and unknown.topic_mark_owner is None
    )


async def test_socket_incarnation_change_during_scan_never_authorizes_write(
    isolated_tmux, monkeypatch
) -> None:
    import os

    mgr, session = isolated_tmux
    window = session.new_window(window_name="project")
    wid = window.window_id
    assert wid is not None
    row = await mgr.find_window_by_id(wid)
    assert row is not None and row.topic_mark_owner
    native = session.server.cmd
    writes = []

    def changed_socket(command, *args, **kwargs):
        result = native(command, *args, **kwargs)
        if command == "list-windows" and "-a" in args:
            os.utime(session.server.socket_path, None)
        if command == "set-option":
            writes.append(args)
        return result

    monkeypatch.setattr(session.server, "cmd", changed_socket)
    assert not await mgr.set_topic_mark(wid, True, owner=row.topic_mark_owner)
    rows = await mgr.list_windows_for_reconciliation()
    assert rows is not None
    uncertain = next(window for window in rows if window.window_id == wid)
    assert uncertain.topic_mark_owner is None and uncertain.topic_marked is None
    assert writes == []


async def test_unreadable_owner_during_write_readback_is_not_verified(
    isolated_tmux, monkeypatch
) -> None:
    from types import SimpleNamespace

    mgr, session = isolated_tmux
    target = session.new_window(window_name="project")
    other = session.new_window(window_name="other")
    wid = target.window_id
    assert wid is not None
    row = await mgr.find_window_by_id(wid)
    assert row is not None and row.topic_mark_owner
    native = session.server.cmd
    written = False

    def fail_readback(command, *args, **kwargs):
        nonlocal written
        if (
            command == "show-options"
            and written
            and kwargs.get("target") == other.window_id
        ):
            return SimpleNamespace(returncode=1, stderr=["readback failed"], stdout=[])
        result = native(command, *args, **kwargs)
        if command == "set-option":
            written = True
        return result

    monkeypatch.setattr(session.server, "cmd", fail_readback)
    assert not await mgr.set_topic_mark(wid, True, owner=row.topic_mark_owner)
    assert (
        target.cmd("show-options", "-wv", "@ccgram_topic").stdout[0].startswith("v1:1:")
    )
