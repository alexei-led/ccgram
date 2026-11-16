"""Real credential-free CLI processes and transport; only peer identity is synthetic."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest

import ccgram
from ccgram.multiplexer.herdr import (
    HerdrManager,
    HerdrSessionComposite,
    herdr_session_target_id,
)
from ccgram.topic_mark_journal import read_topic_mark_operations

pytestmark = pytest.mark.integration
assert ccgram.__file__ is not None
_SOURCE = Path(ccgram.__file__).resolve().parents[1]
_TARGET = herdr_session_target_id(
    HerdrSessionComposite("herdr:pi", "pi", "id", "cli-session")
)
_CHILD = """
import os
import subprocess
import sys
from ccgram.multiplexer import herdr
import ccgram.multiplexer.native_process as native_process

assert herdr._CALL_TIMEOUT_SECONDS == 8.0

def capture(connection, *, backend, binary):
    assert backend == 'herdr'
    host, boot, namespace = native_process._local_scope()
    parent_pid = os.getppid()
    start = native_process._process_start(parent_pid)
    assert start is not None
    code = (
        'linux:herdr:0.9.5:sha256:'+'a'*64 if sys.platform == 'linux'
        else 'darwin:herdr:0.9.5:cdhash:'+'a'*40
    )
    return native_process.NativeProcessIdentity(
        host_id=host, boot_id=boot, namespace_id=namespace,
        pid=parent_pid, start=start, code_id=code)

native_process.capture_native_process = capture

def forbidden(*args, **kwargs):
    raise AssertionError('native subprocesses forbidden in CLI transport regression')

subprocess.run = subprocess.Popen = forbidden
assert 'TELEGRAM_BOT_TOKEN' not in os.environ
assert 'ALLOWED_USERS' not in os.environ
from ccgram.cli import cli
cli(['topic', sys.argv[1], sys.argv[2]], prog_name='ccgram')
"""


class _DelayedNative:
    def __init__(self, failure: str):
        self.failure = failure
        self.original = "owned label"
        self.label = self.original
        self.received = asyncio.Event()
        self.release = asyncio.Event()
        self.applied = asyncio.Event()
        self.handlers: set[asyncio.Task] = set()
        self.errors: list[Exception] = []
        self.renames: list[str] = []

    def pane(self) -> dict[str, str]:
        return {"terminal_id": "term-cli", "pane_id": "w1:p1", "label": self.label}

    async def response(self, payload):
        method = payload["method"]
        if method == "pane.rename":
            self.renames.append(payload["params"]["label"])
            self.received.set()
            await self.release.wait()
            self.label = payload["params"]["label"]
            self.applied.set()
            return {"pane": self.pane()}
        return {
            "ping": {"type": "pong"},
            "pane.list": {"panes": [self.pane()]},
            "pane.get": {"pane": self.pane()},
            "workspace.list": {
                "workspaces": [{"workspace_id": "w1", "label": "owned"}]
            },
            "tab.list": {"tabs": [{"tab_id": "w1:t1", "label": "owned"}]},
            "agent.list": {
                "agents": [
                    {
                        "terminal_id": "term-cli",
                        "pane_id": "w1:p1",
                        "workspace_id": "w1",
                        "tab_id": "w1:t1",
                        "agent_session": {
                            "source": "herdr:pi",
                            "agent": "pi",
                            "kind": "id",
                            "value": "cli-session",
                        },
                    }
                ]
            },
        }[method]

    async def serve(self, reader, writer):
        task = asyncio.current_task()
        assert task is not None
        self.handlers.add(task)
        try:
            line = await reader.readline()
            if not line:
                return
            payload = json.loads(line)
            if payload["method"] == "pane.rename" and self.failure == "eof":
                writer.close()
                await writer.wait_closed()
            result = await self.response(payload)
            if not writer.is_closing():
                writer.write(
                    json.dumps({"id": payload["id"], "result": result}).encode() + b"\n"
                )
                await writer.drain()
        except ConnectionError, OSError:
            pass
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            self.errors.append(exc)
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()
            self.handlers.discard(task)


@pytest.mark.parametrize("failure", ["timeout", "eof"])
async def test_separate_cli_cannot_confirm_off_while_old_on_can_still_apply(failure):
    with TemporaryDirectory(prefix="ccg-cli-", dir="/tmp") as directory:
        private = Path(directory)
        home = private / "home"
        home.mkdir()
        state_dir = private / "state"
        state_dir.mkdir()
        state_path = state_dir / "state.json"
        state_path.write_text('{"unrelated":{"keep":true}}\n', encoding="utf-8")
        socket_path = private / "native.sock"
        native = _DelayedNative(failure)
        server = await asyncio.start_unix_server(native.serve, path=socket_path)
        socket_path.chmod(0o600)
        children: list[asyncio.subprocess.Process] = []
        pending: list[asyncio.Task] = []
        env = {
            "PATH": os.defpath,
            "HOME": str(home),
            "CCGRAM_DIR": str(state_dir),
            "CCGRAM_MULTIPLEXER": "herdr",
            "CCGRAM_TOPIC_DISCOVERY": "marked",
            "HERDR_SOCKET_PATH": str(socket_path),
            "PYTHONPATH": str(_SOURCE),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        async def cli(action):
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _CHILD,
                action,
                _TARGET,
                cwd=private,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            children.append(process)
            output, error = await process.communicate()
            return process.returncode, output.decode(), error.decode()

        try:
            async with asyncio.timeout(25):
                mark = asyncio.create_task(cli("mark"))
                pending.append(mark)
                await native.received.wait()
                rc, _output, error = await mark
                assert rc != 0, error
                assert not native.applied.is_set()
                (armed,) = read_topic_mark_operations(state_path)
                assert armed.attempt is not None and armed.attempt.desired is True
                assert armed.attempt.witness.native_pid == os.getpid()
                manager = HerdrManager(
                    socket_path=str(socket_path), binary="/forbidden-native-herdr"
                )
                assert (
                    await manager.recover_topic_mark_write(armed.attempt.witness)
                    is False
                )
                assert json.loads(state_path.read_text(encoding="utf-8"))[
                    "unrelated"
                ] == {"keep": True}
                checkpoint = state_path.read_bytes()

                rc, _output, error = await cli("unmark")
                assert rc != 0, error
                assert not native.applied.is_set()
                assert state_path.read_bytes() == checkpoint

                native.release.set()
                await native.applied.wait()
                rc, output, error = await cli("status")
                assert rc == 0, error
                status = json.loads(output)
                assert status["targets"][0]["marked"] is True
                assert status["pending_operations"] == [armed.to_dict()]
                assert read_topic_mark_operations(state_path) == (armed,)
                assert state_path.read_bytes() == checkpoint
                assert len(native.renames) == 1
        finally:
            native.release.set()
            for process in children:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            await asyncio.gather(*pending, return_exceptions=True)
            server.close()
            await server.wait_closed()
            if native.handlers:
                await asyncio.gather(*tuple(native.handlers), return_exceptions=True)
            assert not native.errors
