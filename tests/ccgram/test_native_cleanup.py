"""Native fixture cleanup retains resource ownership, never default context."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import libtmux

import pytest

from ccgram.multiplexer.herdr import HerdrManager
from ccgram.multiplexer.tmux import TmuxManager


def _root_cleanup_fixture(request):
    path = Path(__file__).resolve().parents[1] / "conftest.py"
    plugin = request.config.pluginmanager.get_plugin(str(path))
    assert plugin is not None
    return plugin._cleanup_created_windows.__wrapped__


@pytest.fixture
def native_socket():
    import socket
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ccg-clean-", dir="/tmp") as directory:
        path = Path(directory) / "native.sock"
        handle = socket.socket(socket.AF_UNIX)
        handle.bind(str(path))
        yield path
        handle.close()


def test_private_tmux_cleanup_never_targets_default_server(
    request, monkeypatch, native_socket
):
    result = SimpleNamespace(returncode=0, stderr=[], stdout=[str(native_socket)])
    server = SimpleNamespace(cmd=Mock(return_value=result))
    manager = TmuxManager(session_name="disposable")
    manager._server = cast(libtmux.Server, server)

    async def create(_self, *args, **kwargs):
        return True, "created", "disposable", "@77"

    monkeypatch.setattr(TmuxManager, "create_window", create)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr("subprocess.run", run)
    fixture = _root_cleanup_fixture(request)
    context = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda _: True))

    with contextmanager(fixture)(context, monkeypatch):
        assert asyncio.run(manager.create_window("/tmp"))[0]

    assert run.call_count == 1
    assert run.call_args.args[0] == [
        "tmux",
        "-S",
        str(native_socket.resolve()),
        "kill-window",
        "-t",
        "@77",
    ]


def test_herdr_cleanup_closes_only_guarded_created_session(request, monkeypatch):
    target = "herdr-session-v1-" + "a" * 64
    manager = HerdrManager()
    close = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "kill_window", close)

    async def create(_self, *args, **kwargs):
        return True, "created", "disposable", target

    monkeypatch.setattr(HerdrManager, "create_window", create)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr("subprocess.run", run)
    fixture = _root_cleanup_fixture(request)
    context = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda _: True))

    with contextmanager(fixture)(context, monkeypatch):
        assert asyncio.run(manager.create_window("/tmp"))[0]

    close.assert_awaited_once_with(target)
    run.assert_not_called()


def test_tmux_cleanup_rejects_missing_or_non_socket_endpoint(tmp_path, monkeypatch):
    from _native_cleanup import tmux_window_cleanup

    endpoint = tmp_path / "not-a-socket"
    endpoint.touch()
    server = SimpleNamespace(
        cmd=lambda *_args: SimpleNamespace(
            returncode=0, stderr=[], stdout=[str(endpoint)]
        )
    )
    run = Mock()
    monkeypatch.setattr("subprocess.run", run)

    assert tmux_window_cleanup(server, "@1") is None
    run.assert_not_called()


def test_tmux_cleanup_does_not_follow_replaced_socket(native_socket, monkeypatch):
    import socket

    from _native_cleanup import tmux_window_cleanup

    server = SimpleNamespace(
        cmd=lambda *_args: SimpleNamespace(
            returncode=0, stderr=[], stdout=[str(native_socket)]
        )
    )
    close = tmux_window_cleanup(server, "@1")
    assert close is not None
    native_socket.unlink()
    replacement = socket.socket(socket.AF_UNIX)
    replacement.bind(str(native_socket))
    run = Mock()
    monkeypatch.setattr("subprocess.run", run)
    try:
        close()
        run.assert_not_called()
    finally:
        replacement.close()


def test_herdr_cleanup_rejects_locator_ids(monkeypatch):
    from _native_cleanup import herdr_window_cleanup

    manager = SimpleNamespace(kill_window=AsyncMock())
    assert herdr_window_cleanup(manager, "w1:t1") is None
    manager.kill_window.assert_not_called()


def test_namespace_cleanup_never_follows_socket_symlink(tmp_path, monkeypatch):
    import os
    import socket

    from _native_cleanup import tmux_test_namespace

    root, close = tmux_test_namespace()
    directory = root / f"tmux-{os.getuid()}"
    directory.mkdir()
    outside = tmp_path / "outside.sock"
    handle = socket.socket(socket.AF_UNIX)
    # Keep the path short on macOS.
    outside = root.parent / (root.name + "-outside.sock")
    handle.bind(str(outside))
    (directory / "default").symlink_to(outside)
    run = Mock()
    monkeypatch.setattr("subprocess.run", run)
    try:
        close()
        assert outside.exists()
        assert not root.exists()
        run.assert_not_called()
    finally:
        handle.close()
        outside.unlink()
