"""Two agents in one agterm session must never share a topic identity."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ccgram.hook import hook_main
from ccgram.multiplexer.agterm import AgtermManager

SESSION = "157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4"
CLAUDE = "834d8e17-a40c-477b-9359-9d75ddd78d3b"
PI = "01a0cf3e-fdab-71f3-ae2b-8c97eff39382"


def peer_session() -> dict:
    return {
        "id": SESSION,
        "name": "peer-chat",
        "cwd": "/project/pi",
        "foreground": ["pi", ""],
        "hasSplit": True,
        "split": False,
        "splitForeground": ["claude", "--session-id", CLAUDE],
        "splitCwd": "/project/claude",
        "status": "active",
        "statusPane": "right",
    }


class PaneSocket:
    def __init__(self) -> None:
        self.session = peer_session()
        self.calls: list[tuple[list[str], str | None]] = []
        self.unavailable = False

    def tree(self) -> dict:
        return {
            "workspaces": [
                {"id": "workspace", "name": "code", "sessions": [self.session]}
            ]
        }

    async def __call__(self, args, stdin=None):
        args = list(args)
        self.calls.append((args, stdin))
        if self.unavailable:
            return 1, "", "socket unavailable"
        if args[:2] == ["window", "list"]:
            result = {"windows": [{"id": "window", "open": True}]}
        elif args[0] == "tree":
            result = {"tree": self.tree()}
        elif args[:2] == ["session", "text"]:
            result = {
                "text": "Claude peer reply"
                if args[args.index("--pane") + 1] == "right"
                else "Pi reply"
            }
        else:
            result = {}
        return 0, json.dumps({"ok": True, "result": result}), ""


def manager(socket: PaneSocket) -> AgtermManager:
    return AgtermManager(runner=socket, own_session_id="", workspaces=None)


async def split_target(mux: AgtermManager) -> str:
    windows = await mux.list_windows_for_reconciliation()
    assert windows is not None and len(windows) == 2
    return next(window.window_id for window in windows if window.window_id != SESSION)


async def test_hidden_split_has_independent_topic_and_provider() -> None:
    mux = manager(PaneSocket())
    windows = await mux.list_windows_for_reconciliation()
    assert windows is not None and len(windows) == 2
    primary, peer = windows
    assert primary.window_id == SESSION
    assert primary.pane_current_command == "pi"
    assert peer.window_id != primary.window_id
    assert peer.pane_current_command == "claude"
    assert peer.cwd == "/project/claude"
    assert peer.window_name != primary.window_name
    assert all(window.topic_eligible for window in windows)
    foreground = await mux.foreground(peer.window_id)
    assert (
        foreground is not None and foreground.argv == peer_session()["splitForeground"]
    )
    assert await mux.agent_status(primary.window_id) is None
    assert await mux.agent_status(peer.window_id) is not None


async def test_split_read_and_write_use_right_without_changing_primary() -> None:
    socket = PaneSocket()
    mux = manager(socket)
    target = await split_target(mux)
    assert await mux.capture_pane(target) == "Claude peer reply"
    assert await mux.send(target, "peer message", enter=False)
    assert await mux.send(SESSION, "main message", enter=False)
    writes = [
        (args[args.index("--pane") + 1], args[args.index("--target") + 1], text)
        for args, text in socket.calls
        if args[:2] == ["session", "type"]
    ]
    assert writes == [
        ("right", SESSION, "peer message"),
        ("left", SESSION, "main message"),
    ]


@pytest.mark.parametrize("change", ["closed", "promoted", "replaced", "unavailable"])
async def test_stale_split_target_never_reads_or_types_into_replacement(
    change: str,
) -> None:
    socket = PaneSocket()
    mux = manager(socket)
    target = await split_target(mux)
    if change in {"closed", "promoted"}:
        socket.session["hasSplit"] = False
        socket.session.pop("splitForeground")
        if change == "promoted":
            socket.session["foreground"] = ["claude", "--session-id", CLAUDE]
    elif change == "replaced":
        socket.session["splitForeground"] = [
            "claude",
            "--session-id",
            "different-session",
        ]
    else:
        socket.unavailable = True
    socket.calls.clear()
    assert not await mux.send(target, "must not leak", enter=False)
    assert await mux.capture_pane(target) is None
    assert await mux.find_window_by_id(target) is None
    assert await mux.window_exists(target) is (
        None if change == "unavailable" else False
    )
    assert not any(
        args[:2] in (["session", "type"], ["session", "text"])
        for args, _ in socket.calls
    )


async def test_split_topic_cannot_close_or_rename_both_agents() -> None:
    socket = PaneSocket()
    mux = manager(socket)
    target = await split_target(mux)
    socket.calls.clear()
    assert not await mux.kill_window(target)
    assert not await mux.rename_window(target, "wrong shared name")
    assert not any(
        args[:2] in (["session", "close"], ["session", "rename"])
        for args, _ in socket.calls
    )


@pytest.mark.parametrize("hidden", [True, False])
async def test_workspace_and_self_exclusions_apply_to_both_panes(hidden: bool) -> None:
    socket = PaneSocket()
    if hidden:
        socket.session["name"] = "__service"
    mux = AgtermManager(
        runner=socket, own_session_id="" if hidden else SESSION, workspaces=None
    )
    windows = await mux.list_windows_for_reconciliation()
    assert windows is not None and len(windows) == 2
    assert not any(window.topic_eligible for window in windows)
    assert await mux.list_windows() == []


@pytest.mark.parametrize("promoted", [False, True])
@pytest.mark.parametrize("event", ["SessionStart", "Stop"])
def test_hooks_for_peer_chat_resolve_distinct_live_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, promoted: bool, event: str
) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    monkeypatch.setenv("AGTERM_SESSION_ID", SESSION)
    monkeypatch.setenv("AGTERM_WINDOW_ID", "window")
    for key in (
        "TMUX_PANE",
        "HERDR_PANE_ID",
        "HERDR_WORKSPACE_ID",
        "PI_SUBAGENT_CHILD",
    ):
        monkeypatch.delenv(key, raising=False)
    snapshot = PaneSocket()
    if promoted:
        snapshot.session["foreground"] = snapshot.session.pop("splitForeground")
        snapshot.session["hasSplit"] = False

    def command(args, **kwargs):
        assert "agtermctl" in str(args[0])
        return subprocess.CompletedProcess(
            args, 0, json.dumps({"ok": True, "result": {"tree": snapshot.tree()}}), ""
        )

    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setenv("AGTERM_PANE", "right")
    payload = {
        "session_id": CLAUDE,
        "hook_event_name": event,
        "cwd": "/project/claude",
        "transcript_path": str(tmp_path / ".claude" / f"{CLAUDE}.jsonl"),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    hook_main(provider_name="claude")
    state = json.loads((tmp_path / "session_map.json").read_text())
    key = next(iter(state))
    assert (key == f"agterm:{SESSION}") is promoted
    assert state[key]["provider_name"] == "claude"
    if not promoted:
        monkeypatch.setenv("AGTERM_PANE", "left")
        payload.update(
            session_id=PI, transcript_path=str(tmp_path / ".pi" / f"{PI}.jsonl")
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        hook_main(provider_name="pi")
        state = json.loads((tmp_path / "session_map.json").read_text())
        assert len(state) == 2
        assert state[f"agterm:{SESSION}"]["provider_name"] == "pi"
        assert state[key]["session_id"] == CLAUDE


def test_real_pi_hook_runner_without_bash_marker_or_pane_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("AGTERM_SESSION_ID", SESSION)
    monkeypatch.setenv("PI_HOOK_TIMEOUT_SEC", "10")
    for key in ("PI_CODING_AGENT", "AGTERM_PANE", "TMUX_PANE", "HERDR_PANE_ID"):
        monkeypatch.delenv(key, raising=False)
    folder = tmp_path / ".pi/agent/sessions/--project-pi--"
    folder.mkdir(parents=True)
    transcript = folder / f"2026-09-23_{PI}.jsonl"
    transcript.write_text(
        json.dumps({"type": "session", "id": PI, "cwd": "/project/pi"}) + "\n"
    )
    socket = PaneSocket()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, json.dumps({"ok": True, "result": {"tree": socket.tree()}}), ""
        ),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "hook_event_name": "Stop",
                    "session_id": PI,
                    "cwd": "/project/pi",
                }
            )
        ),
    )
    hook_main()
    state = json.loads((tmp_path / "state/session_map.json").read_text())
    assert state[f"agterm:{SESSION}"]["provider_name"] == "pi"
    assert state[f"agterm:{SESSION}"]["transcript_path"] == str(transcript)


async def test_peer_replaced_between_text_and_enter_does_not_receive_enter() -> None:
    socket = PaneSocket()

    async def runner(args, stdin=None):
        result = await socket(args, stdin)
        if list(args[:2]) == ["session", "type"]:
            socket.session["splitForeground"] = [
                "claude",
                "--session-id",
                "replacement",
            ]
        return result

    mux = AgtermManager(runner=runner, own_session_id="", workspaces=None)
    target = await split_target(mux)
    assert not await mux.send(target, "original peer only")
    assert [text for args, text in socket.calls if args[:2] == ["session", "type"]] == [
        "original peer only"
    ]


@pytest.mark.parametrize(
    "case",
    ["ambiguous", "wrong_session", "scratch", "bad_json", "bad_shape", "offline"],
)
def test_unresolved_hook_does_not_create_or_overwrite_a_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    monkeypatch.setenv("AGTERM_SESSION_ID", SESSION)
    monkeypatch.setenv("AGTERM_PANE", "scratch" if case == "scratch" else "right")
    for key in ("TMUX_PANE", "HERDR_PANE_ID", "HERDR_WORKSPACE_ID"):
        monkeypatch.delenv(key, raising=False)
    socket = PaneSocket()
    if case == "ambiguous":
        socket.session["foreground"] = ["claude"]
        socket.session["splitForeground"] = ["claude"]
    elif case == "wrong_session":
        socket.session["splitForeground"] = ["claude", "--session-id", "other"]
    output = json.dumps({"ok": True, "result": {"tree": socket.tree()}})
    if case == "bad_json":
        output = "invalid"
    elif case == "bad_shape":
        output = "[]"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1 if case == "offline" else 0, output, ""
        ),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "hook_event_name": "SessionStart",
                    "session_id": CLAUDE,
                    "cwd": "/project/claude",
                    "transcript_path": f"/transcripts/{CLAUDE}.jsonl",
                }
            )
        ),
    )
    hook_main(provider_name="claude")
    assert not (tmp_path / "session_map.json").exists()
    assert not (tmp_path / "events.jsonl").exists()
