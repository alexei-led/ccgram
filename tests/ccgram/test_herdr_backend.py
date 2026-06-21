"""Task 7 unit + boundary tests for the herdr backend.

The backend shells out to the ``herdr`` CLI; here the command runner is
replaced by ``FakeHerdr`` so every test feeds real captured JSON fixtures
(``pane get`` / ``pane list`` / ``process-info`` / ``layout`` / ``tab create``)
with no socket. Boundary tests cover socket-down, bad id, scrollback
truncation, and protocol-version refusal (design "herdr backend (unit,
boundary)").

Fixtures are trimmed from live herdr 0.7.0 output.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from ccgram.multiplexer.base import (
    CaptureResult,
    ForegroundInfo,
    PaneDims,
    PaneInfo,
    WindowRef,
)
from ccgram.multiplexer.herdr import (
    HERDR_PROTOCOL_VERSION,
    HerdrError,
    HerdrManager,
    HerdrProtocolError,
)

# ── Captured JSON fixtures (live herdr 0.7.0) ──────────────────────────

PANE_GET = json.dumps(
    {
        "id": "cli:pane:get",
        "result": {
            "pane": {
                "agent": "claude",
                "agent_status": "idle",
                "cwd": "/Users/alexei/Workspace/ccgram",
                "focused": True,
                "foreground_cwd": "/Users/alexei/Workspace/ccgram",
                "pane_id": "w2:p1",
                "tab_id": "w2:t1",
                "terminal_id": "term_abc",
                "title": "ccgram:claude",
                "workspace_id": "w2",
            },
            "type": "pane_info",
        },
    }
)

PANE_LIST = json.dumps(
    {
        "id": "cli:pane:list",
        "result": {
            "panes": [
                {
                    "agent": "claude",
                    "agent_status": "working",
                    "cwd": "/Users/alexei/Workspace/archfit",
                    "focused": True,
                    "pane_id": "w1:p1",
                    "tab_id": "w1:t1",
                    "workspace_id": "w1",
                },
                {
                    "agent_status": "unknown",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "focused": False,
                    "pane_id": "w2:p2",
                    "tab_id": "w2:t2",
                    "workspace_id": "w2",
                },
            ],
            "type": "pane_list",
        },
    }
)

TAB_LIST = json.dumps(
    {
        "id": "cli:tab:list",
        "result": {
            "tabs": [
                {"label": "archfit", "tab_id": "w1:t1", "workspace_id": "w1"},
                {"label": "ralphex", "tab_id": "w2:t2", "workspace_id": "w2"},
            ],
            "type": "tab_list",
        },
    }
)

TAB_GET = json.dumps(
    {
        "id": "cli:tab:get",
        "result": {
            "tab": {"label": "herdr-support", "tab_id": "w2:t1", "workspace_id": "w2"},
            "type": "tab_info",
        },
    }
)

WORKSPACE_LIST = json.dumps(
    {
        "id": "cli:workspace:list",
        "result": {
            "workspaces": [
                {
                    "workspace_id": "w1",
                    "label": "archfit",
                    "cwd": "/Users/alexei/Workspace/archfit",
                },
                {
                    "workspace_id": "w2",
                    "label": "ccgram",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                },
            ],
            "type": "workspace_list",
        },
    }
)

PROCESS_INFO = json.dumps(
    {
        "id": "cli:pane:process_info",
        "result": {
            "process_info": {
                "foreground_process_group_id": 40777,
                "foreground_processes": [
                    {
                        "argv": ["python", "-m", "agent"],
                        "cwd": "/Users/alexei/Workspace/ccgram",
                        "name": "python",
                        "pid": 40777,
                    }
                ],
                "pane_id": "w2:p1",
                "shell_pid": 38702,
            },
            "type": "pane_process_info",
        },
    }
)

LAYOUT = json.dumps(
    {
        "id": "cli:pane:layout",
        "result": {
            "layout": {
                "area": {"height": 63, "width": 199, "x": 26, "y": 1},
                "focused_pane_id": "w2:p1",
                "panes": [
                    {
                        "focused": True,
                        "pane_id": "w2:p1",
                        "rect": {"height": 50, "width": 120, "x": 0, "y": 0},
                    }
                ],
                "splits": [],
                "tab_id": "w2:t1",
                "workspace_id": "w2",
                "zoomed": False,
            },
            "type": "pane_layout",
        },
    }
)

TAB_CREATE = json.dumps(
    {
        "id": "cli:tab:create",
        "result": {
            "root_pane": {
                "cwd": "/tmp/work",
                "pane_id": "w2:p9",
                "tab_id": "w2:t9",
                "workspace_id": "w2",
            },
            "tab": {"label": "work", "tab_id": "w2:t9", "workspace_id": "w2"},
            "type": "tab_created",
        },
    }
)

OK = json.dumps({"id": "cli:ok", "result": {"type": "ok"}})

PANE_READ_TEXT = "line one\nline two\n"

ERROR_NOT_FOUND = json.dumps(
    {"error": {"code": "pane_not_found", "message": "pane w9:p9 not found"}, "id": "x"}
)


def _status_json(protocol: int = HERDR_PROTOCOL_VERSION, running: bool = True) -> str:
    return json.dumps(
        {
            "client": {"version": "0.7.0", "protocol": protocol},
            "server": {
                "status": "running" if running else "stopped",
                "running": running,
                "version": "0.7.0",
                "protocol": protocol,
                "compatible": True,
            },
            "update": {"restart_needed": False},
        }
    )


# ── Fake CLI runner ────────────────────────────────────────────────────


class FakeHerdr:
    """Injectable runner returning canned ``(rc, stdout, stderr)`` per command.

    ``on(*prefix, out=...)`` registers a response for any call whose leading
    args match ``prefix`` (longest match wins). ``calls`` records every
    invocation for assertions.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._responses: dict[tuple[str, ...], tuple[int, str, str]] = {}
        self.default: tuple[int, str, str] = (1, "", "no canned response")

    def on(self, *prefix: str, rc: int = 0, out: str = "", err: str = "") -> FakeHerdr:
        self._responses[tuple(prefix)] = (rc, out, err)
        return self

    async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
        args = list(args)
        self.calls.append(args)
        best: tuple[tuple[str, ...], tuple[int, str, str]] | None = None
        for key, resp in self._responses.items():
            if list(key) == args[: len(key)] and (
                best is None or len(key) > len(best[0])
            ):
                best = (key, resp)
        return best[1] if best else self.default

    def sent(self, *prefix: str) -> list[str] | None:
        """Return the first recorded call matching *prefix*, or None."""
        for call in self.calls:
            if call[: len(prefix)] == list(prefix):
                return call
        return None


def _manager(fake: FakeHerdr) -> HerdrManager:
    return HerdrManager(socket_path="/tmp/herdr.sock", runner=fake)


# ── Capabilities ───────────────────────────────────────────────────────


def test_capabilities_are_pinned() -> None:
    caps = HerdrManager().capabilities
    assert caps.name == "herdr"
    assert caps.ids_stable_across_restart is False
    assert caps.exposes_pane_tty is False
    assert caps.native_agent_status is True
    assert caps.read_max_lines == 1000
    assert caps.self_identify_env == "HERDR_PANE_ID"
    assert caps.supports_event_stream is True


def test_constructor_does_no_io() -> None:
    # Construction must touch no socket: the injected runner records zero calls.
    fake = FakeHerdr()
    HerdrManager(socket_path="/tmp/herdr.sock", runner=fake)
    assert fake.calls == []


# ── Value-type mapping (wN:pN → window_id) ─────────────────────────────


async def test_find_window_parses_pane_get() -> None:
    fake = FakeHerdr().on("pane", "get", out=PANE_GET).on("tab", "get", out=TAB_GET)
    win = await _manager(fake).find_window("w2:p1")
    assert win == WindowRef(
        window_id="w2:p1",
        window_name="herdr-support",
        cwd="/Users/alexei/Workspace/ccgram",
        pane_current_command="claude",
    )


async def test_list_windows_renders_adaptive_labels() -> None:
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST)
        .on("tab", "list", out=TAB_LIST)
        .on("workspace", "list", out=WORKSPACE_LIST)
    )
    wins = await _manager(fake).list_windows()
    ids = {w.window_id: w for w in wins}
    assert set(ids) == {"w1:p1", "w2:p2"}
    # Single agent pane in its tab → "<workspace> ▸ <agent>", no "/tab".
    assert ids["w1:p1"].window_name == "archfit ▸ claude"
    assert ids["w1:p1"].pane_current_command == "claude"
    # A pane with no agent degrades to the workspace label, no stray separator.
    assert ids["w2:p2"].window_name == "ccgram"


_SPLIT_PANES = json.dumps(
    {
        "result": {
            "panes": [
                {
                    "agent": "claude",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "pane_id": "w2:p1",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                },
                {
                    "agent": "codex",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "pane_id": "w2:p2",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                },
            ],
            "type": "pane_list",
        }
    }
)
_SPLIT_TABS = json.dumps(
    {
        "result": {
            "tabs": [{"label": "feature", "tab_id": "w2:t1", "workspace_id": "w2"}],
            "type": "tab_list",
        }
    }
)


async def test_list_windows_appends_tab_label_on_split() -> None:
    # Two agent panes share one tab (an agent team) → each label carries the
    # tab name so the two topics stay distinguishable.
    fake = (
        FakeHerdr()
        .on("pane", "list", out=_SPLIT_PANES)
        .on("tab", "list", out=_SPLIT_TABS)
        .on("workspace", "list", out=WORKSPACE_LIST)
    )
    wins = {w.window_id: w.window_name for w in await _manager(fake).list_windows()}
    assert wins["w2:p1"] == "ccgram ▸ claude/feature"
    assert wins["w2:p2"] == "ccgram ▸ codex/feature"


async def test_workspace_rename_relabels_without_changing_pane_id() -> None:
    # Renaming a workspace re-labels the topic on the next poll; the pane id —
    # the live handle the binding is anchored to — is unchanged (design
    # "Binding key = agent session id … renaming re-labels, never rebinds").
    before = {
        w.window_id: w.window_name
        for w in await _manager(
            FakeHerdr()
            .on("pane", "list", out=PANE_LIST)
            .on("tab", "list", out=TAB_LIST)
            .on("workspace", "list", out=WORKSPACE_LIST)
        ).list_windows()
    }
    renamed = json.dumps(
        {
            "result": {
                "workspaces": [
                    {"workspace_id": "w1", "label": "archfit-v2", "cwd": "/a"},
                    {"workspace_id": "w2", "label": "ccgram", "cwd": "/b"},
                ],
                "type": "workspace_list",
            }
        }
    )
    after = {
        w.window_id: w.window_name
        for w in await _manager(
            FakeHerdr()
            .on("pane", "list", out=PANE_LIST)
            .on("tab", "list", out=TAB_LIST)
            .on("workspace", "list", out=renamed)
        ).list_windows()
    }
    assert before["w1:p1"] == "archfit ▸ claude"
    assert after["w1:p1"] == "archfit-v2 ▸ claude"
    assert set(after) == set(before)  # same pane ids → no rebind


async def test_foreground_from_process_info() -> None:
    fake = FakeHerdr().on("pane", "process-info", out=PROCESS_INFO)
    fg = await _manager(fake).foreground("w2:p1")
    assert fg == ForegroundInfo(
        pid=40777,
        pgid=40777,
        argv=["python", "-m", "agent"],
        cwd="/Users/alexei/Workspace/ccgram",
        tty="",
    )


async def test_pane_dims_from_layout() -> None:
    fake = FakeHerdr().on("pane", "layout", out=LAYOUT)
    dims = await _manager(fake).pane_dims("w2:p1")
    assert dims == PaneDims(width=120, height=50)


async def test_list_panes_returns_single_pane() -> None:
    fake = FakeHerdr().on("pane", "get", out=PANE_GET).on("pane", "layout", out=LAYOUT)
    panes = await _manager(fake).list_panes("w2:p1")
    assert panes == [
        PaneInfo(
            pane_id="w2:p1",
            index=1,
            active=True,
            command="claude",
            path="/Users/alexei/Workspace/ccgram",
            width=120,
            height=50,
        )
    ]


# ── Capture / scrollback ───────────────────────────────────────────────


async def test_capture_returns_text() -> None:
    fake = FakeHerdr().on("pane", "read", out=PANE_READ_TEXT)
    res = await _manager(fake).capture("w2:p1")
    assert res == CaptureResult(text="line one\nline two", truncated=False)
    call = fake.sent("pane", "read")
    assert call is not None
    assert "--source" in call and "visible" in call
    assert "text" in call


async def test_capture_ansi_requests_ansi_format() -> None:
    fake = FakeHerdr().on("pane", "read", out=PANE_READ_TEXT)
    await _manager(fake).capture("w2:p1", ansi=True)
    call = fake.sent("pane", "read")
    assert call is not None and "ansi" in call


async def test_scrollback_clamps_to_read_max_lines_and_flags_truncated() -> None:
    fake = FakeHerdr().on("pane", "read", out=PANE_READ_TEXT)
    res = await _manager(fake).capture_scrollback("w2:p1", lines=5000)
    assert res is not None
    assert res.truncated is True
    call = fake.sent("pane", "read")
    assert call is not None
    # Requested 5000 but herdr caps at 1000 → request clamped.
    assert "1000" in call
    assert "5000" not in call


async def test_scrollback_under_cap_is_not_truncated() -> None:
    fake = FakeHerdr().on("pane", "read", out=PANE_READ_TEXT)
    res = await _manager(fake).capture_scrollback("w2:p1", lines=200)
    assert res is not None and res.truncated is False
    call = fake.sent("pane", "read")
    assert call is not None and "200" in call


# ── Send paths ─────────────────────────────────────────────────────────


async def test_send_literal_enter_uses_pane_run() -> None:
    fake = FakeHerdr().on("pane", "run", out=OK)
    assert await _manager(fake).send("w2:p1", "hello world") is True
    assert fake.sent("pane", "run") == ["pane", "run", "w2:p1", "hello world"]


async def test_send_no_enter_uses_send_text() -> None:
    fake = FakeHerdr().on("pane", "send-text", out=OK)
    assert await _manager(fake).send("w2:p1", "draft", enter=False) is True
    assert fake.sent("pane", "send-text") == ["pane", "send-text", "w2:p1", "draft"]


async def test_send_special_keys_uses_send_keys() -> None:
    fake = FakeHerdr().on("pane", "send-keys", out=OK)
    assert (
        await _manager(fake).send("w2:p1", "Down", enter=False, literal=False) is True
    )
    assert fake.sent("pane", "send-keys") == ["pane", "send-keys", "w2:p1", "Down"]


async def test_send_keys_appends_enter_when_requested() -> None:
    fake = FakeHerdr().on("pane", "send-keys", out=OK)
    await _manager(fake).send("w2:p1", "", enter=True, literal=False)
    assert fake.sent("pane", "send-keys") == ["pane", "send-keys", "w2:p1", "Enter"]


async def test_kill_and_rename() -> None:
    fake = FakeHerdr().on("pane", "close", out=OK).on("pane", "rename", out=OK)
    mgr = _manager(fake)
    assert await mgr.kill_window("w2:p1") is True
    assert await mgr.rename_window("w2:p1", "newname") is True
    assert fake.sent("pane", "close") == ["pane", "close", "w2:p1"]
    assert fake.sent("pane", "rename") == ["pane", "rename", "w2:p1", "newname"]


# ── create_window ──────────────────────────────────────────────────────


async def test_create_window_returns_pane_id_and_launches(tmp_path) -> None:
    fake = FakeHerdr().on("tab", "create", out=TAB_CREATE).on("pane", "run", out=OK)
    ok, msg, name, win_id = await _manager(fake).create_window(
        str(tmp_path),
        window_name="work",
        launch_command="claude",
        agent_args="--continue",
    )
    assert ok is True
    assert win_id == "w2:p9"
    assert name == "work"
    assert str(tmp_path) in msg
    # The launch command + args ran in the new pane.
    assert fake.sent("pane", "run") == ["pane", "run", "w2:p9", "claude --continue"]


async def test_create_window_rejects_missing_directory() -> None:
    fake = FakeHerdr()
    ok, msg, name, win_id = await _manager(fake).create_window("/no/such/dir")
    assert ok is False
    assert "does not exist" in msg
    assert win_id == ""
    assert fake.calls == []  # bailed before touching herdr


async def test_create_window_reuses_matching_workspace(tmp_path) -> None:
    # A workspace already rooted at the chosen cwd → reuse it (no create), and
    # scope the new tab to it via --workspace (design "cwd → workspace").
    ws_list = json.dumps(
        {
            "result": {
                "workspaces": [
                    {"workspace_id": "w5", "label": "repo", "cwd": str(tmp_path)}
                ],
                "type": "workspace_list",
            }
        }
    )
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=ws_list)
        .on("tab", "create", out=TAB_CREATE)
        .on("pane", "run", out=OK)
    )
    ok, *_ = await _manager(fake).create_window(str(tmp_path), launch_command="claude")
    assert ok is True
    assert fake.sent("workspace", "create") is None  # reused, not created
    tab_call = fake.sent("tab", "create")
    assert tab_call is not None
    assert "--workspace" in tab_call and "w5" in tab_call


async def test_create_window_creates_workspace_when_absent(tmp_path) -> None:
    # No workspace matches the cwd → create one, then scope the tab to it.
    ws_list = json.dumps({"result": {"workspaces": [], "type": "workspace_list"}})
    ws_create = json.dumps(
        {
            "result": {
                "workspace": {"workspace_id": "w7", "cwd": str(tmp_path)},
                "type": "workspace_created",
            }
        }
    )
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=ws_list)
        .on("workspace", "create", out=ws_create)
        .on("tab", "create", out=TAB_CREATE)
        .on("pane", "run", out=OK)
    )
    ok, *_ = await _manager(fake).create_window(str(tmp_path), launch_command="claude")
    assert ok is True
    create_call = fake.sent("workspace", "create")
    assert create_call is not None and "--cwd" in create_call
    tab_call = fake.sent("tab", "create")
    assert tab_call is not None
    assert "--workspace" in tab_call and "w7" in tab_call


async def test_create_window_falls_back_when_no_workspace_support(tmp_path) -> None:
    # An older herdr without workspace addressing → tab create lands in the
    # active workspace, no --workspace flag, behavior unchanged (Task 7).
    fake = FakeHerdr().on("tab", "create", out=TAB_CREATE).on("pane", "run", out=OK)
    ok, _msg, _name, win_id = await _manager(fake).create_window(
        str(tmp_path), launch_command="claude"
    )
    assert ok is True
    assert win_id == "w2:p9"
    tab_call = fake.sent("tab", "create")
    assert tab_call is not None
    assert "--workspace" not in tab_call


# ── Boundary: socket down, bad id, protocol ────────────────────────────


async def test_socket_down_returns_none_not_crash() -> None:
    # rc=127 simulates the herdr binary/socket being unavailable.
    fake = FakeHerdr().on("pane", "get", rc=127, err="connection refused")
    mgr = _manager(fake)
    assert await mgr.find_window("w2:p1") is None
    fake.on("pane", "read", rc=127, err="connection refused")
    assert await mgr.capture("w2:p1") is None


async def test_bad_id_error_payload_returns_none() -> None:
    fake = FakeHerdr().on("pane", "get", rc=1, out=ERROR_NOT_FOUND)
    assert await _manager(fake).find_window("w9:p9") is None


async def test_foreground_missing_process_returns_none() -> None:
    empty = json.dumps(
        {
            "result": {
                "process_info": {
                    "foreground_process_group_id": 0,
                    "foreground_processes": [],
                    "pane_id": "w2:p1",
                },
                "type": "pane_process_info",
            }
        }
    )
    fake = FakeHerdr().on("pane", "process-info", out=empty)
    assert await _manager(fake).foreground("w2:p1") is None


async def test_ensure_session_accepts_pinned_protocol() -> None:
    fake = FakeHerdr().on("status", out=_status_json())
    await _manager(fake).ensure_session()  # no raise
    # The protocol check must actually probe the server, not no-op.
    assert fake.sent("status") is not None


async def test_ensure_session_raises_on_non_json_status() -> None:
    fake = FakeHerdr().on("status", out="not json {{{")
    with pytest.raises(HerdrError, match="non-JSON"):
        await _manager(fake).ensure_session()


async def test_ensure_session_raises_on_non_object_json_status() -> None:
    # Valid JSON of the wrong shape (a list) must not crash with AttributeError.
    fake = FakeHerdr().on("status", out="[]")
    with pytest.raises(HerdrError, match="non-object JSON"):
        await _manager(fake).ensure_session()


async def test_ensure_session_refuses_protocol_mismatch() -> None:
    fake = FakeHerdr().on("status", out=_status_json(protocol=99))
    with pytest.raises(HerdrProtocolError, match="99"):
        await _manager(fake).ensure_session()


async def test_ensure_session_raises_when_socket_down() -> None:
    fake = FakeHerdr().on("status", rc=127, err="connection refused")
    with pytest.raises(HerdrError, match="status failed"):
        await _manager(fake).ensure_session()


async def test_ensure_session_raises_when_server_not_running() -> None:
    fake = FakeHerdr().on("status", out=_status_json(running=False))
    with pytest.raises(HerdrError, match="not running"):
        await _manager(fake).ensure_session()
