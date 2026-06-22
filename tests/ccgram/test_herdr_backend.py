"""Task 1 unit + boundary tests for the herdr backend (tab identity).

The backend shells out to the ``herdr`` CLI; here the command runner is
replaced by ``FakeHerdr`` so every test feeds real captured JSON fixtures
(``tab get`` / ``tab list`` / ``pane list`` / ``workspace list`` /
``process-info`` / ``layout`` / ``tab create``) with no socket.

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

# TAB_GET for find_window("w2:t1") — tab identity (Task 1).
TAB_GET = json.dumps(
    {
        "id": "cli:tab:get",
        "result": {
            "tab": {
                "label": "herdr-support",
                "tab_id": "w2:t1",
                "workspace_id": "w2",
                "cwd": "/Users/alexei/Workspace/ccgram",
            },
            "type": "tab_info",
        },
    }
)

# PANE_LIST used by find_window to resolve representative agent/cwd.
PANE_LIST_FOR_FIND = json.dumps(
    {
        "id": "cli:pane:list",
        "result": {
            "panes": [
                {
                    "agent": "claude",
                    "agent_status": "idle",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "focused": True,
                    "pane_id": "w2:p1",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                },
            ],
            "type": "pane_list",
        },
    }
)

# TAB_LIST + PANE_LIST + WORKSPACE_LIST for list_windows.
TAB_LIST = json.dumps(
    {
        "id": "cli:tab:list",
        "result": {
            "tabs": [
                {
                    "label": "archfit",
                    "tab_id": "w1:t1",
                    "workspace_id": "w1",
                    "cwd": "/Users/alexei/Workspace/archfit",
                },
                {
                    "label": "ralphex",
                    "tab_id": "w2:t2",
                    "workspace_id": "w2",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                },
            ],
            "type": "tab_list",
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

# tab create returns tab_id in result["tab"]["tab_id"] (Task 1).
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


# ── find_window: tab identity (window_id = tab_id) ─────────────────────


async def test_find_window_uses_tab_get_and_returns_tab_id() -> None:
    # find_window(tab_id) → tab get + workspace list + pane list → WindowRef
    # with the same full "<workspace> ▸ <tab>" label as list_windows.
    fake = (
        FakeHerdr()
        .on("tab", "get", out=TAB_GET)
        .on("workspace", "list", out=WORKSPACE_LIST)
        .on("pane", "list", out=PANE_LIST_FOR_FIND)
    )
    win = await _manager(fake).find_window("w2:t1")
    assert win == WindowRef(
        window_id="w2:t1",
        window_name="ccgram ▸ herdr-support",
        cwd="/Users/alexei/Workspace/ccgram",
        pane_current_command="claude",
    )
    # Must use tab get, not pane get.
    assert fake.sent("tab", "get") is not None
    assert fake.sent("pane", "get") is None


async def test_find_window_returns_none_when_tab_gone() -> None:
    fake = FakeHerdr().on("tab", "get", rc=1, out=ERROR_NOT_FOUND)
    assert await _manager(fake).find_window("w9:t9") is None


async def test_find_window_bypasses_internal_label_filter() -> None:
    # __*__ tabs are filtered in list_windows but find_window always resolves.
    # workspace_id "w3" is absent from workspace list → workspace label "" →
    # format_agent_topic_prefix("", "__main__") == "__main__" (no stray separator).
    internal_tab = json.dumps(
        {
            "result": {
                "tab": {
                    "label": "__main__",
                    "tab_id": "w3:t1",
                    "workspace_id": "w3",
                    "cwd": "/tmp",
                },
                "type": "tab_info",
            }
        }
    )
    pane_list_empty = json.dumps({"result": {"panes": [], "type": "pane_list"}})
    workspace_list_empty = json.dumps(
        {"result": {"workspaces": [], "type": "workspace_list"}}
    )
    fake = (
        FakeHerdr()
        .on("tab", "get", out=internal_tab)
        .on("workspace", "list", out=workspace_list_empty)
        .on("pane", "list", out=pane_list_empty)
    )
    win = await _manager(fake).find_window("w3:t1")
    assert win is not None
    assert win.window_id == "w3:t1"
    assert win.window_name == "__main__"


# ── list_windows: one WindowRef per tab ────────────────────────────────


async def test_list_windows_returns_one_ref_per_tab() -> None:
    # Two tabs → two WindowRefs with tab_id as window_id, not pane ids.
    fake = (
        FakeHerdr()
        .on("tab", "list", out=TAB_LIST)
        .on("pane", "list", out=PANE_LIST)
        .on("workspace", "list", out=WORKSPACE_LIST)
    )
    wins = await _manager(fake).list_windows()
    ids = {w.window_id: w for w in wins}
    # window_ids are tab ids, not pane ids.
    assert set(ids) == {"w1:t1", "w2:t2"}
    # Agent tab → representative pane's agent in pane_current_command.
    assert ids["w1:t1"].pane_current_command == "claude"
    # window_name is "<workspace> ▸ <tab label>", not agent name.
    assert ids["w1:t1"].window_name == "archfit ▸ archfit"
    # Tab with no agent → empty pane_current_command; name uses tab label.
    assert ids["w2:t2"].pane_current_command == ""
    assert ids["w2:t2"].window_name == "ccgram ▸ ralphex"


async def test_list_windows_uses_focused_pane_as_representative() -> None:
    # When a tab has multiple panes, the focused one's agent is used.
    split_tab_list = json.dumps(
        {
            "result": {
                "tabs": [
                    {
                        "label": "feature",
                        "tab_id": "w2:t1",
                        "workspace_id": "w2",
                        "cwd": "/Users/alexei/Workspace/ccgram",
                    }
                ],
                "type": "tab_list",
            }
        }
    )
    split_pane_list = json.dumps(
        {
            "result": {
                "panes": [
                    {
                        "agent": "codex",
                        "cwd": "/Users/alexei/Workspace/ccgram",
                        "pane_id": "w2:p1",
                        "tab_id": "w2:t1",
                        "workspace_id": "w2",
                        "focused": False,
                    },
                    {
                        "agent": "claude",
                        "cwd": "/Users/alexei/Workspace/ccgram",
                        "pane_id": "w2:p2",
                        "tab_id": "w2:t1",
                        "workspace_id": "w2",
                        "focused": True,
                    },
                ],
                "type": "pane_list",
            }
        }
    )
    fake = (
        FakeHerdr()
        .on("tab", "list", out=split_tab_list)
        .on("pane", "list", out=split_pane_list)
        .on("workspace", "list", out=WORKSPACE_LIST)
    )
    wins = await _manager(fake).list_windows()
    assert len(wins) == 1
    win = wins[0]
    assert win.window_id == "w2:t1"
    # Focused pane is claude (not codex) — pane_current_command is agent-based.
    assert win.pane_current_command == "claude"
    # window_name is "<workspace> ▸ <tab label>"; agent name does not appear.
    assert win.window_name == "ccgram ▸ feature"


async def test_list_windows_filters_internal_workspace_label() -> None:
    # Tabs in a __*__ workspace must not appear in list_windows.
    tab_list = json.dumps(
        {
            "result": {
                "tabs": [
                    {
                        "label": "normal",
                        "tab_id": "w1:t1",
                        "workspace_id": "w1",
                        "cwd": "/a",
                    },
                    {
                        "label": "agent",
                        "tab_id": "w2:t1",
                        "workspace_id": "w2",
                        "cwd": "/b",
                    },
                ],
                "type": "tab_list",
            }
        }
    )
    workspace_list = json.dumps(
        {
            "result": {
                "workspaces": [
                    {"workspace_id": "w1", "label": "myproject", "cwd": "/a"},
                    {"workspace_id": "w2", "label": "__main__", "cwd": "/b"},
                ],
                "type": "workspace_list",
            }
        }
    )
    pane_list_empty = json.dumps({"result": {"panes": [], "type": "pane_list"}})
    fake = (
        FakeHerdr()
        .on("tab", "list", out=tab_list)
        .on("pane", "list", out=pane_list_empty)
        .on("workspace", "list", out=workspace_list)
    )
    wins = await _manager(fake).list_windows()
    ids = {w.window_id for w in wins}
    assert "w1:t1" in ids  # normal workspace — included
    assert "w2:t1" not in ids  # __main__ workspace — filtered


async def test_list_windows_filters_internal_tab_label() -> None:
    # Tabs whose own label is __*__ must not appear in list_windows.
    tab_list = json.dumps(
        {
            "result": {
                "tabs": [
                    {
                        "label": "normal",
                        "tab_id": "w1:t1",
                        "workspace_id": "w1",
                        "cwd": "/a",
                    },
                    {
                        "label": "__internal__",
                        "tab_id": "w1:t2",
                        "workspace_id": "w1",
                        "cwd": "/a",
                    },
                ],
                "type": "tab_list",
            }
        }
    )
    workspace_list = json.dumps(
        {
            "result": {
                "workspaces": [
                    {"workspace_id": "w1", "label": "myproject", "cwd": "/a"},
                ],
                "type": "workspace_list",
            }
        }
    )
    pane_list_empty = json.dumps({"result": {"panes": [], "type": "pane_list"}})
    fake = (
        FakeHerdr()
        .on("tab", "list", out=tab_list)
        .on("pane", "list", out=pane_list_empty)
        .on("workspace", "list", out=workspace_list)
    )
    wins = await _manager(fake).list_windows()
    ids = {w.window_id for w in wins}
    assert "w1:t1" in ids  # normal tab — included
    assert "w1:t2" not in ids  # __internal__ tab — filtered


async def test_list_windows_renders_adaptive_labels() -> None:
    fake = (
        FakeHerdr()
        .on("tab", "list", out=TAB_LIST)
        .on("pane", "list", out=PANE_LIST)
        .on("workspace", "list", out=WORKSPACE_LIST)
    )
    wins = await _manager(fake).list_windows()
    ids = {w.window_id: w for w in wins}
    # Tab label is primary → "<workspace> ▸ <tab>"; agent name does not appear.
    assert ids["w1:t1"].window_name == "archfit ▸ archfit"
    assert ids["w1:t1"].pane_current_command == "claude"
    # Tab with no agent still gets a label from workspace ▸ tab.
    assert ids["w2:t2"].window_name == "ccgram ▸ ralphex"


_SPLIT_TABS = json.dumps(
    {
        "result": {
            "tabs": [
                {
                    "label": "feature",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                }
            ],
            "type": "tab_list",
        }
    }
)

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
                    "focused": True,
                },
                {
                    "agent": "codex",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "pane_id": "w2:p2",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                    "focused": False,
                },
            ],
            "type": "pane_list",
        }
    }
)


async def test_list_windows_split_tab_produces_one_ref_with_tab_suffix() -> None:
    # A split tab (two panes) → ONE WindowRef with tab label suffix.
    fake = (
        FakeHerdr()
        .on("tab", "list", out=_SPLIT_TABS)
        .on("pane", "list", out=_SPLIT_PANES)
        .on("workspace", "list", out=WORKSPACE_LIST)
    )
    wins = await _manager(fake).list_windows()
    assert len(wins) == 1
    win = wins[0]
    assert win.window_id == "w2:t1"
    # Tab label is primary → "<workspace> ▸ <tab>"; agent name not in label.
    assert win.window_name == "ccgram ▸ feature"


async def test_workspace_rename_relabels_without_changing_tab_id() -> None:
    # Renaming a workspace re-labels the topic on the next poll; the tab id
    # (the binding handle) is unchanged.
    before = {
        w.window_id: w.window_name
        for w in await _manager(
            FakeHerdr()
            .on("tab", "list", out=TAB_LIST)
            .on("pane", "list", out=PANE_LIST)
            .on("workspace", "list", out=WORKSPACE_LIST)
        ).list_windows()
    }
    renamed_ws = json.dumps(
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
            .on("tab", "list", out=TAB_LIST)
            .on("pane", "list", out=PANE_LIST)
            .on("workspace", "list", out=renamed_ws)
        ).list_windows()
    }
    assert before["w1:t1"] == "archfit ▸ archfit"
    assert after["w1:t1"] == "archfit-v2 ▸ archfit"
    assert set(after) == set(before)  # same tab ids → no rebind


# ── CRUD: kill/rename use tab commands ────────────────────────────────


async def test_kill_window_uses_tab_close() -> None:
    fake = FakeHerdr().on("tab", "close", out=OK)
    assert await _manager(fake).kill_window("w2:t1") is True
    assert fake.sent("tab", "close") == ["tab", "close", "w2:t1"]
    assert fake.sent("pane", "close") is None  # must NOT use pane close


async def test_rename_window_uses_tab_rename() -> None:
    fake = FakeHerdr().on("tab", "rename", out=OK)
    assert await _manager(fake).rename_window("w2:t1", "newname") is True
    assert fake.sent("tab", "rename") == ["tab", "rename", "w2:t1", "newname"]
    assert fake.sent("pane", "rename") is None  # must NOT use pane rename


# ── create_window: returns tab_id ──────────────────────────────────────


async def test_create_window_returns_tab_id_and_launches(tmp_path) -> None:
    fake = FakeHerdr().on("tab", "create", out=TAB_CREATE).on("pane", "run", out=OK)
    ok, msg, name, win_id = await _manager(fake).create_window(
        str(tmp_path),
        window_name="work",
        launch_command="claude",
        agent_args="--continue",
    )
    assert ok is True
    # window_id must be the tab id, not the pane id.
    assert win_id == "w2:t9"
    assert name == "work"
    assert str(tmp_path) in msg
    # The launch command still targets the root pane id.
    assert fake.sent("pane", "run") == ["pane", "run", "w2:p9", "claude --continue"]


async def test_create_window_rejects_missing_directory() -> None:
    fake = FakeHerdr()
    ok, msg, _name, win_id = await _manager(fake).create_window("/no/such/dir")
    assert ok is False
    assert "does not exist" in msg
    assert win_id == ""
    assert fake.calls == []  # bailed before touching herdr


async def test_create_window_reuses_matching_workspace(tmp_path) -> None:
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
    ok, _msg, _name, win_id = await _manager(fake).create_window(
        str(tmp_path), launch_command="claude"
    )
    assert ok is True
    assert win_id == "w2:t9"
    assert fake.sent("workspace", "create") is None  # reused, not created
    tab_call = fake.sent("tab", "create")
    assert tab_call is not None
    assert "--workspace" in tab_call and "w5" in tab_call


async def test_create_window_creates_workspace_when_absent(tmp_path) -> None:
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
    ok, _msg, _name, win_id = await _manager(fake).create_window(
        str(tmp_path), launch_command="claude"
    )
    assert ok is True
    assert win_id == "w2:t9"
    create_call = fake.sent("workspace", "create")
    assert create_call is not None and "--cwd" in create_call
    tab_call = fake.sent("tab", "create")
    assert tab_call is not None
    assert "--workspace" in tab_call and "w7" in tab_call


async def test_create_window_falls_back_when_no_workspace_support(tmp_path) -> None:
    # An older herdr without workspace addressing → tab create in active workspace.
    fake = FakeHerdr().on("tab", "create", out=TAB_CREATE).on("pane", "run", out=OK)
    ok, _msg, _name, win_id = await _manager(fake).create_window(
        str(tmp_path), launch_command="claude"
    )
    assert ok is True
    assert win_id == "w2:t9"
    tab_call = fake.sent("tab", "create")
    assert tab_call is not None
    assert "--workspace" not in tab_call


# ── Task 4 fixtures: tab→pane resolution ──────────────────────────────

# Single-pane tab: w2:t1 → pane w2:p1 (focused).
PANE_LIST_SINGLE = json.dumps(
    {
        "id": "cli:pane:list",
        "result": {
            "panes": [
                {
                    "agent": "claude",
                    "agent_status": "idle",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "focused": True,
                    "pane_id": "w2:p1",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                },
            ],
            "type": "pane_list",
        },
    }
)

# Split tab: w2:t1 → panes w2:p1 (focused) + w2:p2 (not focused).
PANE_LIST_SPLIT = json.dumps(
    {
        "id": "cli:pane:list",
        "result": {
            "panes": [
                {
                    "agent": "claude",
                    "agent_status": "working",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "focused": True,
                    "pane_id": "w2:p1",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                },
                {
                    "agent": "codex",
                    "agent_status": "idle",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "focused": False,
                    "pane_id": "w2:p2",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                },
            ],
            "type": "pane_list",
        },
    }
)

# Split tab with no focused pane: _active_pane falls back to first.
PANE_LIST_SPLIT_NO_FOCUS = json.dumps(
    {
        "id": "cli:pane:list",
        "result": {
            "panes": [
                {
                    "agent": "claude",
                    "agent_status": "idle",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "focused": False,
                    "pane_id": "w2:p1",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                },
                {
                    "agent": "codex",
                    "agent_status": "idle",
                    "cwd": "/Users/alexei/Workspace/ccgram",
                    "focused": False,
                    "pane_id": "w2:p2",
                    "tab_id": "w2:t1",
                    "workspace_id": "w2",
                },
            ],
            "type": "pane_list",
        },
    }
)

# Layout for a split tab (two pane rects).
LAYOUT_SPLIT = json.dumps(
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
                    },
                    {
                        "focused": False,
                        "pane_id": "w2:p2",
                        "rect": {"height": 50, "width": 79, "x": 120, "y": 0},
                    },
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


# ── _active_pane resolution ────────────────────────────────────────────


async def test_active_pane_returns_focused_pane() -> None:
    fake = FakeHerdr().on("pane", "list", out=PANE_LIST_SPLIT)
    pane_id = await _manager(fake)._active_pane("w2:t1")
    assert pane_id == "w2:p1"


async def test_active_pane_returns_first_when_no_focused() -> None:
    fake = FakeHerdr().on("pane", "list", out=PANE_LIST_SPLIT_NO_FOCUS)
    pane_id = await _manager(fake)._active_pane("w2:t1")
    assert pane_id == "w2:p1"  # first in list


async def test_active_pane_returns_none_for_empty_tab() -> None:
    empty = json.dumps({"result": {"panes": [], "type": "pane_list"}})
    fake = FakeHerdr().on("pane", "list", out=empty)
    assert await _manager(fake)._active_pane("w2:t1") is None


async def test_active_pane_single_tab_returns_that_pane() -> None:
    fake = FakeHerdr().on("pane", "list", out=PANE_LIST_SINGLE)
    pane_id = await _manager(fake)._active_pane("w2:t1")
    assert pane_id == "w2:p1"


# ── Pane ops: tab→pane resolution (Task 4) ─────────────────────────────


async def test_foreground_from_process_info() -> None:
    # foreground(tab_id) resolves tab→active pane, then reads process-info.
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "process-info", out=PROCESS_INFO)
    )
    fg = await _manager(fake).foreground("w2:t1")
    assert fg == ForegroundInfo(
        pid=40777,
        pgid=40777,
        argv=["python", "-m", "agent"],
        cwd="/Users/alexei/Workspace/ccgram",
        tty="",
    )
    # Must resolve through pane list first.
    assert fake.sent("pane", "list") is not None
    # process-info must target the resolved pane id, not the tab id.
    pi_call = fake.sent("pane", "process-info")
    assert pi_call is not None and "w2:p1" in pi_call


async def test_foreground_split_tab_uses_focused_pane() -> None:
    # Split tab: focused is w2:p1; process-info must target w2:p1, not w2:p2.
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SPLIT)
        .on("pane", "process-info", out=PROCESS_INFO)
    )
    fg = await _manager(fake).foreground("w2:t1")
    assert fg is not None
    pi_call = fake.sent("pane", "process-info")
    assert pi_call is not None and "w2:p1" in pi_call


async def test_pane_dims_from_layout() -> None:
    # pane_dims(tab_id) resolves tab→active pane, then fetches layout.
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "layout", out=LAYOUT)
    )
    dims = await _manager(fake).pane_dims("w2:t1")
    assert dims == PaneDims(width=120, height=50)
    layout_call = fake.sent("pane", "layout")
    assert layout_call is not None and "w2:p1" in layout_call


async def test_list_panes_returns_single_pane() -> None:
    # list_panes(tab_id) returns the one pane in a single-pane tab.
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "layout", out=LAYOUT)
    )
    panes = await _manager(fake).list_panes("w2:t1")
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


async def test_list_panes_returns_all_panes_in_split_tab() -> None:
    # list_panes(tab_id) returns ALL panes in a split tab (team awareness).
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SPLIT)
        .on("pane", "layout", out=LAYOUT_SPLIT)
    )
    panes = await _manager(fake).list_panes("w2:t1")
    assert len(panes) == 2
    pane_ids = {p.pane_id for p in panes}
    assert pane_ids == {"w2:p1", "w2:p2"}
    # Active (focused) pane.
    p1 = next(p for p in panes if p.pane_id == "w2:p1")
    assert p1.active is True
    assert p1.command == "claude"
    assert p1.width == 120
    # Non-focused pane.
    p2 = next(p for p in panes if p.pane_id == "w2:p2")
    assert p2.active is False
    assert p2.command == "codex"
    assert p2.width == 79


async def test_list_panes_returns_empty_for_empty_tab() -> None:
    empty = json.dumps({"result": {"panes": [], "type": "pane_list"}})
    fake = FakeHerdr().on("pane", "list", out=empty)
    assert await _manager(fake).list_panes("w2:t1") == []


# ── Capture / scrollback ───────────────────────────────────────────────


async def test_capture_returns_text() -> None:
    # capture(tab_id) resolves tab→active pane, then reads visible text.
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "read", out=PANE_READ_TEXT)
    )
    res = await _manager(fake).capture("w2:t1")
    assert res == CaptureResult(text="line one\nline two", truncated=False)
    call = fake.sent("pane", "read")
    assert call is not None
    assert "--source" in call and "visible" in call
    assert "text" in call
    # Must target the resolved pane id, not the tab id.
    assert "w2:p1" in call


async def test_capture_ansi_requests_ansi_format() -> None:
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "read", out=PANE_READ_TEXT)
    )
    await _manager(fake).capture("w2:t1", ansi=True)
    call = fake.sent("pane", "read")
    assert call is not None and "ansi" in call


async def test_capture_returns_none_when_tab_has_no_panes() -> None:
    empty = json.dumps({"result": {"panes": [], "type": "pane_list"}})
    fake = FakeHerdr().on("pane", "list", out=empty)
    assert await _manager(fake).capture("w2:t1") is None


async def test_scrollback_clamps_to_read_max_lines_and_flags_truncated() -> None:
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "read", out=PANE_READ_TEXT)
    )
    res = await _manager(fake).capture_scrollback("w2:t1", lines=5000)
    assert res is not None
    assert res.truncated is True
    call = fake.sent("pane", "read")
    assert call is not None
    assert "1000" in call
    assert "5000" not in call
    assert "w2:p1" in call


async def test_scrollback_under_cap_is_not_truncated() -> None:
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "read", out=PANE_READ_TEXT)
    )
    res = await _manager(fake).capture_scrollback("w2:t1", lines=200)
    assert res is not None and res.truncated is False
    call = fake.sent("pane", "read")
    assert call is not None and "200" in call


# ── Send paths ─────────────────────────────────────────────────────────


async def test_send_literal_enter_uses_pane_run() -> None:
    # send(tab_id, ...) resolves tab→active pane, then runs pane run.
    fake = (
        FakeHerdr().on("pane", "list", out=PANE_LIST_SINGLE).on("pane", "run", out=OK)
    )
    assert await _manager(fake).send("w2:t1", "hello world") is True
    assert fake.sent("pane", "run") == ["pane", "run", "w2:p1", "hello world"]


async def test_send_no_enter_uses_send_text() -> None:
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "send-text", out=OK)
    )
    assert await _manager(fake).send("w2:t1", "draft", enter=False) is True
    assert fake.sent("pane", "send-text") == ["pane", "send-text", "w2:p1", "draft"]


async def test_send_special_keys_uses_send_keys() -> None:
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "send-keys", out=OK)
    )
    assert (
        await _manager(fake).send("w2:t1", "Down", enter=False, literal=False) is True
    )
    assert fake.sent("pane", "send-keys") == ["pane", "send-keys", "w2:p1", "Down"]


async def test_send_keys_appends_enter_when_requested() -> None:
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "send-keys", out=OK)
    )
    await _manager(fake).send("w2:t1", "", enter=True, literal=False)
    assert fake.sent("pane", "send-keys") == ["pane", "send-keys", "w2:p1", "Enter"]


async def test_send_returns_false_when_tab_has_no_panes() -> None:
    empty = json.dumps({"result": {"panes": [], "type": "pane_list"}})
    fake = FakeHerdr().on("pane", "list", out=empty)
    assert await _manager(fake).send("w2:t1", "hello") is False


async def test_send_split_tab_targets_focused_pane() -> None:
    # Split tab: send must go to the focused pane (w2:p1), not w2:p2.
    fake = FakeHerdr().on("pane", "list", out=PANE_LIST_SPLIT).on("pane", "run", out=OK)
    assert await _manager(fake).send("w2:t1", "go") is True
    assert fake.sent("pane", "run") == ["pane", "run", "w2:p1", "go"]


async def test_send_to_pane_bypasses_tab_resolution() -> None:
    # send_to_pane(pane_id, ...) sends directly to the pane id — no pane list call.
    fake = FakeHerdr().on("pane", "run", out=OK)
    assert await _manager(fake).send_to_pane("w2:p2", "msg") is True
    assert fake.sent("pane", "run") == ["pane", "run", "w2:p2", "msg"]
    assert fake.sent("pane", "list") is None  # no resolution


# ── Boundary: socket down, bad id, protocol ────────────────────────────


async def test_socket_down_returns_none_not_crash() -> None:
    fake = FakeHerdr().on("tab", "get", rc=127, err="connection refused")
    mgr = _manager(fake)
    assert await mgr.find_window("w2:t1") is None
    # capture with no pane list response → pane list fails → None.
    assert await mgr.capture("w2:t1") is None


async def test_bad_id_error_payload_returns_none() -> None:
    fake = FakeHerdr().on("tab", "get", rc=1, out=ERROR_NOT_FOUND)
    assert await _manager(fake).find_window("w9:t9") is None


async def test_foreground_missing_process_returns_none() -> None:
    empty_proc = json.dumps(
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
    fake = (
        FakeHerdr()
        .on("pane", "list", out=PANE_LIST_SINGLE)
        .on("pane", "process-info", out=empty_proc)
    )
    assert await _manager(fake).foreground("w2:t1") is None


async def test_ensure_session_accepts_pinned_protocol() -> None:
    fake = FakeHerdr().on("status", out=_status_json())
    await _manager(fake).ensure_session()  # no raise
    assert fake.sent("status") is not None


async def test_ensure_session_raises_on_non_json_status() -> None:
    fake = FakeHerdr().on("status", out="not json {{{")
    with pytest.raises(HerdrError, match="non-JSON"):
        await _manager(fake).ensure_session()


async def test_ensure_session_raises_on_non_object_json_status() -> None:
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
