"""Native fixture cleanup retains resource ownership, never default context."""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
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


def test_root_conftest_isolates_namespace_before_first_application_import(tmp_path):
    foreign = tmp_path / "synthetic-ambient"
    foreign.mkdir(mode=0o700)
    sentinel = foreign / "not-a-server"
    sentinel.write_text("synthetic foreign data", encoding="utf-8")
    child_tmp = tmp_path / "child-tmp"
    child_tmp.mkdir()
    conftest = Path(__file__).resolve().parents[1] / "conftest.py"
    script = dedent(
        """
        import atexit
        import importlib.abc
        import importlib.util
        import json
        import os
        from pathlib import Path
        import shutil
        import socket
        import stat
        import subprocess
        import sys
        import traceback

        def main():
            foreign = Path(os.environ["TMUX_TMPDIR"])
            created, callbacks, attempts, observed, imports = set(), [], [], [], []
            mkdir = os.mkdir
            rmtree = shutil.rmtree

            def record_directory(path, *args, **kwargs):
                result = mkdir(path, *args, **kwargs)
                created.add(os.path.abspath(os.fspath(path)))
                return result

            def owned_rmtree(path, *args, **kwargs):
                assert os.path.abspath(os.fspath(path)) in created, "foreign deletion"
                return rmtree(path, *args, **kwargs)

            def deny_native(*args, **kwargs):
                attempts.append(repr(args))
                raise AssertionError("native execution forbidden: " + repr(args))

            def record_exit(callback, *args, **kwargs):
                callbacks.append((callback, args, kwargs))
                return callback

            os.mkdir = record_directory
            shutil.rmtree = owned_rmtree
            atexit.register = record_exit
            for name in ("Popen", "run", "call", "check_call", "check_output"):
                setattr(subprocess, name, deny_native)
            for name in dir(os):
                if name in ("system", "popen", "fork") or name.startswith(("exec", "spawn", "posix_spawn")):
                    setattr(os, name, deny_native)
            for name in ("connect", "connect_ex", "send", "sendall", "sendto", "sendmsg"):
                if hasattr(socket.socket, name):
                    setattr(socket.socket, name, deny_native)
            socket.create_connection = deny_native

            class FirstApplicationImport(BaseException):
                pass

            class ImportGate(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "ccgram" or fullname.startswith("ccgram."):
                        imports.append(fullname)
                        assert len(imports) == 1
                        assert not observed
                        assert "TMUX" not in os.environ, "ambient TMUX survived"
                        assert "TMUX_PANE" not in os.environ, "ambient TMUX_PANE survived"
                        root = Path(os.environ["TMUX_TMPDIR"])
                        assert root != foreign, "ambient TMUX_TMPDIR survived"
                        assert str(root) in created, "namespace was not created here"
                        assert root.name.startswith("ccg-pytest-")
                        assert root.parent == Path("/tmp")
                        info = root.lstat()
                        assert stat.S_ISDIR(info.st_mode)
                        assert stat.S_IMODE(info.st_mode) == 0o700
                        assert info.st_uid == os.getuid()
                        observed.append((root, stat.S_IMODE(info.st_mode)))
                        raise FirstApplicationImport

            assert not any(name == "ccgram" or name.startswith("ccgram.") for name in sys.modules)
            sys.meta_path.insert(0, ImportGate())
            sys.path.insert(0, sys.argv[2])
            spec = importlib.util.spec_from_file_location("namespace_probe_conftest", sys.argv[1])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                try:
                    spec.loader.exec_module(module)
                    # Current conftest imports the app only when fixtures execute.
                    if not imports:
                        importlib.import_module("ccgram")
                except FirstApplicationImport:
                    pass
                assert len(imports) == len(observed) == 1, "first import was not isolated"
            finally:
                for callback, args, kwargs in reversed(callbacks):
                    callback(*args, **kwargs)
            root, mode = observed[0]
            assert not root.exists(), "owned namespace leaked at exit"
            assert not attempts
            print(json.dumps({"namespace": str(root), "mode": mode, "first_import": imports[0], "checked": True, "cleaned": True}))

        try:
            main()
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        sys.stdout.flush()
        os._exit(0)
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(conftest),
            str(conftest.parent / "ccgram"),
        ],
        env={
            **os.environ,
            "TMPDIR": str(child_tmp),
            "TMUX_TMPDIR": str(foreign),
            "TMUX": f"{sentinel},424242,0",
            "TMUX_PANE": "%424242",
            "AGTERM_SOCKET": str(foreign / "agterm.sock"),
            "HERDR_SOCKET_PATH": str(foreign / "herdr.sock"),
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    proof = json.loads(result.stdout)
    assert proof["checked"] and proof["cleaned"]
    assert proof["first_import"] == "ccgram"
    assert proof["mode"] == 0o700
    assert Path(proof["namespace"]).name.startswith("ccg-pytest-")
    assert not Path(proof["namespace"]).exists()
    assert sentinel.read_text(encoding="utf-8") == "synthetic foreign data"
