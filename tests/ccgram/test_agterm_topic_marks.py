"""Native flags select primary sessions without changing visibility or identity."""

import json
from pathlib import Path
import socket as unix_socket
from tempfile import TemporaryDirectory

import pytest

from ccgram.multiplexer.agterm import AgtermManager
from ccgram.multiplexer import native_process
from _topic_mark_helpers import RecordingTopicMarkGuard

SESSION = "157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4"


class FlagSocket:
    def __init__(self) -> None:
        self.sessions = [
            {
                "id": SESSION,
                "name": "project",
                "cwd": "/project",
                "foreground": ["pi"],
                "flagged": False,
                "hasSplit": True,
                "split": False,
                "splitForeground": ["claude"],
            }
        ]
        self.writes: list[list[str]] = []
        self.failure = ""

    async def __call__(self, args, stdin=None):
        args = list(args)
        if args[:2] == ["window", "list"]:
            result = {"windows": [{"id": "window", "open": True}]}
        elif args[0] == "tree":
            if self.failure == "read" or (self.writes and self.failure == "readback"):
                return 1, "", "unreachable"
            result = {
                "tree": {"workspaces": [{"name": "code", "sessions": self.sessions}]}
            }
        elif args[:2] == ["session", "text"]:
            target = args[args.index("--target") + 1]
            if not any(
                session["id"].casefold() == target.casefold()
                for session in self.sessions
            ):
                return (
                    1,
                    json.dumps({"ok": False, "error": f"no such session: {target}"}),
                    "",
                )
            result = {"text": ""}
        elif args[:2] == ["session", "flag"]:
            self.writes.append(args)
            if self.failure == "write":
                return 1, "", "refused"
            if self.failure == "gone":
                self.sessions.clear()
            elif self.failure != "unchanged":
                self.sessions[0]["flagged"] = args[2] == "on"
            result = {"id": SESSION}
        else:
            raise AssertionError(args)
        return 0, json.dumps({"ok": True, "result": result}), ""


def manager(socket: FlagSocket) -> AgtermManager:
    return AgtermManager(runner=socket, own_session_id="", workspaces=None)


@pytest.mark.parametrize("flag", [True, False, None, "yes", 1])
async def test_flags_never_select_a_hidden_split(flag: object) -> None:
    socket = FlagSocket()
    socket.sessions[0]["flagged"] = flag
    mux = manager(socket)
    windows = await mux.list_windows_for_reconciliation()
    assert windows is not None and len(windows) == 2
    primary, split = windows
    expected = flag if isinstance(flag, bool) else None
    assert primary.topic_marked is expected
    assert primary.topic_markable is True
    assert split.topic_marked is False
    assert split.topic_markable is False
    assert all(window.topic_eligible for window in windows)
    assert [window.window_id for window in await mux.list_windows()] == [
        primary.window_id,
        split.window_id,
    ]
    assert not await mux.set_topic_mark(
        split.window_id, True, write_guard=RecordingTopicMarkGuard()
    )
    assert not await mux.set_topic_mark(
        split.window_id, False, write_guard=RecordingTopicMarkGuard()
    )
    assert socket.writes == []


async def test_flag_write_is_verified_idempotent_and_primary_only() -> None:
    socket = FlagSocket()
    mux = manager(socket)
    assert await mux.set_topic_mark(
        SESSION.lower(), True, write_guard=RecordingTopicMarkGuard()
    )
    assert len(socket.writes) == 1
    assert socket.writes[0][:5] == ["session", "flag", "on", "--target", SESSION]
    assert await mux.set_topic_mark(
        SESSION, True, write_guard=RecordingTopicMarkGuard()
    )
    assert len(socket.writes) == 1
    assert await mux.set_topic_mark(
        SESSION, False, write_guard=RecordingTopicMarkGuard()
    )
    assert len(socket.writes) == 2
    assert await mux.set_topic_mark(
        SESSION, False, write_guard=RecordingTopicMarkGuard()
    )
    assert len(socket.writes) == 2
    assert not await mux.set_topic_mark(
        "missing", True, write_guard=RecordingTopicMarkGuard()
    )
    assert len(socket.writes) == 2


@pytest.mark.parametrize("failure", ["read", "write", "readback", "unchanged", "gone"])
async def test_unverified_flag_operation_fails(failure: str) -> None:
    socket = FlagSocket()
    socket.failure = failure
    assert not await manager(socket).set_topic_mark(
        SESSION, True, write_guard=RecordingTopicMarkGuard()
    )


async def test_missing_native_flag_is_unknown_not_selected() -> None:
    socket = FlagSocket()
    socket.sessions[0].pop("flagged")
    mux = manager(socket)
    window = await mux.find_window_by_id(SESSION)
    assert window is not None and window.topic_marked is None
    assert not await mux.set_topic_mark(
        SESSION, False, write_guard=RecordingTopicMarkGuard()
    )
    assert socket.writes == []


@pytest.mark.parametrize("foreground", [[], ["zsh"], ["claude"], ["codex"]])
async def test_selection_survives_agent_replacement_and_shells(
    foreground: list[str],
) -> None:
    socket = FlagSocket()
    mux = manager(socket)
    assert await mux.set_topic_mark(
        SESSION, True, write_guard=RecordingTopicMarkGuard()
    )
    socket.sessions[0]["foreground"] = foreground
    window = await mux.find_window_by_id(SESSION)
    assert window is not None and window.topic_marked is True
    assert window.topic_eligible is (foreground in (["claude"], ["codex"]))
    assert window.window_id == SESSION


async def test_restored_uuid_keeps_flag_but_same_named_duplicate_is_unmarked() -> None:
    socket = FlagSocket()
    assert await manager(socket).set_topic_mark(
        SESSION, True, write_guard=RecordingTopicMarkGuard()
    )
    duplicate = dict(socket.sessions[0], id="new-id", flagged=False)
    socket.sessions.append(duplicate)
    restored = manager(socket)
    windows = await restored.list_windows_for_reconciliation()
    assert windows is not None
    primary = [window for window in windows if window.topic_markable]
    assert [(window.window_id, window.topic_marked) for window in primary] == [
        (SESSION, True),
        ("new-id", False),
    ]
    socket.sessions.pop(0)
    assert await restored.find_window_by_id(SESSION) is None
    recreated = await restored.find_window_by_id("new-id")
    assert recreated is not None and recreated.topic_marked is False


@pytest.fixture(autouse=True)
def _native_marker_endpoint(monkeypatch):
    with TemporaryDirectory(prefix="ccg-agmark-", dir="/tmp") as directory:
        endpoint = Path(directory) / "rpc"
        with unix_socket.socket(
            unix_socket.AF_UNIX, unix_socket.SOCK_STREAM
        ) as listener:
            listener.bind(str(endpoint))
            listener.listen(32)
            monkeypatch.setenv("AGTERM_SOCKET", str(endpoint))
            monkeypatch.setattr(native_process.sys, "platform", "darwin")
            identity = native_process.NativeProcessIdentity(
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
                "darwin-local",
                123,
                "darwin:100:200",
                "darwin:agterm:0.42.3:cdhash:" + "a" * 40,
            )
            monkeypatch.setattr(
                native_process, "capture_native_process", lambda *_a, **_k: identity
            )
            monkeypatch.setattr(
                "ccgram.multiplexer.topic_mark_transport.tempfile.gettempdir",
                lambda: directory,
            )
            yield


async def test_actual_flag_write_without_state_guard_is_refused():
    native = FlagSocket()
    mux = manager(native)

    assert not await mux.set_topic_mark(SESSION, True)
    assert native.writes == []
