"""Herdr adapter tests for the guarded session-target contract.

Every target-facing test supplies ``agent list`` records.  Tab and pane IDs are
intentionally asserted only as one-shot herdr dispatch locators; no test binds
a topic to either locator.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from ccgram.multiplexer.base import AgentStatus, ForegroundInfo, PaneDims, PaneInfo
from ccgram.multiplexer.herdr import (
    HERDR_PROTOCOL_VERSION,
    HerdrAgentListError,
    HerdrAmbiguousTargetError,
    HerdrError,
    HerdrMalformedRecordError,
    HerdrManager,
    HerdrSessionComposite,
    HerdrUnresolvedTargetError,
    canonical_session_bytes,
    herdr_session_target_id,
)


class FakeHerdr:
    """Prefix-matching canned Herdr command runner."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: dict[tuple[str, ...], tuple[int, str, str]] = {}
        self.default = (1, "", "no canned response")

    def on(self, *prefix: str, rc: int = 0, out: str = "", err: str = "") -> FakeHerdr:
        self.responses[prefix] = (rc, out, err)
        return self

    async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
        call = list(args)
        self.calls.append(call)
        matching = [key for key in self.responses if call[: len(key)] == list(key)]
        return self.responses[max(matching, key=len)] if matching else self.default


def _manager(fake: FakeHerdr) -> HerdrManager:
    return HerdrManager(socket_path="/tmp/herdr.sock", runner=fake)


def _result(**result: object) -> str:
    return json.dumps({"result": result})


def _agent(
    *,
    pane_id: str = "w2:p1",
    tab_id: str = "w2:t1",
    workspace_id: str = "w2",
    value: str = "session-a",
    agent: str = "claude",
    **extra: object,
) -> dict[str, object]:
    return {
        "terminal_id": "term-a",
        "pane_id": pane_id,
        "tab_id": tab_id,
        "workspace_id": workspace_id,
        "agent_session": {
            "source": "herdr",
            "agent": agent,
            "kind": "id",
            "value": value,
        },
        **extra,
    }


def _agents(*records: Mapping[str, object]) -> str:
    return _result(agents=list(records))


def _target(value: str = "session-a", agent: str = "claude") -> str:
    return herdr_session_target_id(HerdrSessionComposite("herdr", agent, "id", value))


def _live_fake(*records: Mapping[str, object]) -> FakeHerdr:
    return FakeHerdr().on("agent", "list", out=_agents(*records))


# ── session identity and discovery ─────────────────────────────────────


def test_capabilities_are_pinned() -> None:
    caps = HerdrManager().capabilities
    assert caps.name == "herdr"
    assert caps.ids_stable_across_restart is False
    assert caps.exposes_pane_tty is False
    assert caps.native_agent_status is True
    assert caps.read_max_lines == 1000
    assert caps.self_identify_env == "HERDR_PANE_ID"
    assert caps.supports_event_stream is True
    assert caps.native_worktrees is True


def test_session_target_digest_is_deterministic_and_private() -> None:
    composite = HerdrSessionComposite("herdr", "claude", "id", "opaque-value")
    target = herdr_session_target_id(composite)
    assert target == herdr_session_target_id(composite)
    assert target.startswith("herdr-session-v1-")
    assert "opaque-value" not in target
    assert canonical_session_bytes(composite).startswith(b'{"source":"herdr"')


async def test_list_windows_exposes_only_session_targets() -> None:
    live = _agent(pane_id="w2:p1", tab_id="w2:t9", value="one")
    sessionless = {
        "pane_id": "w2:p2",
        "tab_id": "w2:t9",
        "workspace_id": "w2",
        "agent": "claude",
    }
    windows = await _manager(_live_fake(live, sessionless)).list_windows()
    assert [
        (win.window_id, win.window_name, win.pane_current_command) for win in windows
    ] == [(_target("one"), "claude", "claude")]
    assert all("w2:" not in win.window_id for win in windows)


async def test_find_window_requires_a_fresh_matching_session_target() -> None:
    fake = _live_fake(_agent(value="one"))
    found = await _manager(fake).find_window_by_id(_target("one"))
    assert found is not None
    assert found.window_id == _target("one")
    assert found.window_name == "claude"
    assert await _manager(fake).find_window_by_id("w2:t1") is None
    assert fake.calls == [["agent", "list"], ["agent", "list"]]


async def test_reconciliation_distinguishes_empty_snapshot_from_agent_list_failure() -> (
    None
):
    assert await _manager(_live_fake()).list_windows_for_reconciliation() == []
    assert await _manager(FakeHerdr()).list_windows_for_reconciliation() is None


@pytest.mark.parametrize(
    "record",
    [
        {"pane_id": "w2:p1", "agent_session": {}},
        _agent(terminal_id="", pane_id="w2:p1"),
        _agent(pane_id="", tab_id="w2:t1"),
    ],
)
async def test_guard_rejects_malformed_live_records(record: dict[str, object]) -> None:
    with pytest.raises(HerdrMalformedRecordError):
        await _manager(_live_fake(record)).guard_session_target(_target())


async def test_guard_reports_unresolved_ambiguous_and_transport_failures() -> None:
    with pytest.raises(HerdrUnresolvedTargetError):
        await _manager(_live_fake(_agent(value="other"))).guard_session_target(
            _target()
        )
    with pytest.raises(HerdrAmbiguousTargetError):
        await _manager(_live_fake(_agent(), _agent())).guard_session_target(_target())
    with pytest.raises(HerdrAgentListError):
        await _manager(FakeHerdr()).guard_session_target(_target())


# ── target actions: fresh guard then locator dispatch ───────────────────


async def test_capture_and_scrollback_guard_then_read_the_matched_pane() -> None:
    fake = _live_fake(_agent(pane_id="w7:p4")).on(
        "pane", "read", rc=0, out="screen text"
    )
    mux = _manager(fake)
    assert await mux.capture_pane(_target()) == "screen text"
    scrollback = await mux.capture_scrollback(_target(), lines=1200)
    assert scrollback is not None and scrollback.text == "screen text"
    assert scrollback.truncated is True
    assert fake.calls == [
        ["agent", "list"],
        ["pane", "read", "w7:p4", "--source", "visible", "--format", "text"],
        ["agent", "list"],
        [
            "pane",
            "read",
            "w7:p4",
            "--source",
            "recent",
            "--lines",
            "1000",
            "--format",
            "text",
        ],
    ]


async def test_send_variants_guard_then_dispatch_to_live_pane() -> None:
    fake = (
        _live_fake(_agent(pane_id="w7:p4"))
        .on("pane", "run", out=_result(type="ok"))
        .on("pane", "send-text", out=_result(type="ok"))
        .on("pane", "send-keys", out=_result(type="ok"))
    )
    mux = _manager(fake)
    assert await mux.send(_target(), "hello")
    assert await mux.send(_target(), "partial", enter=False)
    assert await mux.send(_target(), "C-c Up", literal=False)
    assert fake.calls == [
        ["agent", "list"],
        ["pane", "run", "w7:p4", "hello"],
        ["agent", "list"],
        ["pane", "send-text", "w7:p4", "partial"],
        ["agent", "list"],
        ["pane", "send-keys", "w7:p4", "C-c", "Up", "Enter"],
    ]


async def test_every_mutating_tab_action_uses_live_record_locator() -> None:
    fake = (
        _live_fake(_agent(pane_id="w7:p4", tab_id="w7:t3"))
        .on("tab", "close", out=_result(type="ok"))
        .on("tab", "rename", out=_result(type="ok"))
        .on("pane", "report-metadata", out=_result(type="ok"))
    )
    mux = _manager(fake)
    assert await mux.kill_window(_target())
    assert await mux.rename_window(_target(), "renamed")
    await mux.stamp_pane_title(_target(), "codex")
    assert fake.calls == [
        ["agent", "list"],
        ["tab", "close", "w7:t3"],
        ["agent", "list"],
        ["tab", "rename", "w7:t3", "renamed"],
        ["agent", "list"],
        [
            "pane",
            "report-metadata",
            "w7:p4",
            "--source",
            "ccgram",
            "--title",
            "ccgram:codex",
        ],
    ]


async def test_status_panes_dims_foreground_and_title_are_guarded() -> None:
    pane = {
        "pane_id": "w7:p4",
        "agent_status": "working",
        "custom_status": "doing",
        "title": "ccgram:claude",
    }
    layout = {
        "layout": {"panes": [{"pane_id": "w7:p4", "rect": {"width": 99, "height": 42}}]}
    }
    process = {
        "process_info": {
            "foreground_process_group_id": 12,
            "foreground_processes": [
                {"pid": 12, "argv": ["claude"], "cwd": "/project"}
            ],
        }
    }
    fake = (
        _live_fake(_agent(pane_id="w7:p4"))
        .on("pane", "get", out=_result(pane=pane))
        .on("pane", "layout", out=_result(**layout))
        .on("pane", "process-info", out=_result(**process))
    )
    mux = _manager(fake)
    assert await mux.agent_status(_target()) == AgentStatus(
        "working", "claude", "doing"
    )
    assert await mux.list_panes(_target()) == [
        PaneInfo(_target(), 4, True, "claude", "", 99, 42)
    ]
    assert await mux.pane_dims(_target()) == PaneDims(99, 42)
    assert await mux.foreground(_target()) == ForegroundInfo(
        12, 12, ["claude"], "/project", ""
    )
    assert await mux.get_pane_title(_target()) == "ccgram:claude"
    assert [call for call in fake.calls if call == ["agent", "list"]] == [
        ["agent", "list"]
    ] * 5


async def test_split_and_watch_resolution_use_target_guard() -> None:
    fake = _live_fake(_agent(pane_id="w7:p4")).on(
        "pane", "split", out=_result(pane={"pane_id": "w7:p5"})
    )
    mux = _manager(fake)
    assert await mux.split_window(_target()) == "w7:p5"
    assert await mux._resolve_panes([_target(), "w7:p4"]) == {"w7:p4": _target()}


async def test_raw_pane_helpers_cannot_bypass_target_guard() -> None:
    fake = _live_fake(_agent())
    mux = _manager(fake)
    assert not await mux.send_to_pane("w2:p1", "unsafe", window_id=_target())
    assert await mux.capture_pane_by_id("w2:p1", window_id=_target()) is None
    assert not await mux.send_keys_to_pane("w2:p1", "unsafe", window_id=_target())
    assert fake.calls == []


async def test_action_error_refreshes_guard_without_retargeting() -> None:
    fake = _live_fake(_agent(pane_id="w2:p1")).on("pane", "run", rc=1, err="closed")
    assert not await _manager(fake).send(_target(), "hello")
    assert fake.calls == [
        ["agent", "list"],
        ["pane", "run", "w2:p1", "hello"],
        ["agent", "list"],
    ]


async def test_post_guard_dispatch_race_never_retargets_another_pane() -> None:
    class RacingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self.agent_reads = 0

        async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
            self.calls.append(list(args))
            if args == ["agent", "list"]:
                self.agent_reads += 1
                # First read authorizes p1; refresh observes only replacement p2.
                record = (
                    _agent(pane_id="w2:p1")
                    if self.agent_reads == 1
                    else _agent(pane_id="w2:p2", value="replacement")
                )
                return 0, _agents(record), ""
            return 1, "", "pane disappeared"

    runner = RacingRunner()
    assert not await HerdrManager(runner=runner).send(_target(), "hello")
    assert runner.calls == [
        ["agent", "list"],
        ["pane", "run", "w2:p1", "hello"],
        ["agent", "list"],
    ]
    assert not any(
        "w2:p2" in call for call in runner.calls if call[:2] != ["agent", "list"]
    )


# ── selected-workspace creation and rollback ───────────────────────────


def _workspace(workspace_id: str, cwd: Path) -> str:
    return _result(
        workspaces=[
            {"workspace_id": workspace_id, "label": "selected", "cwd": str(cwd)}
        ]
    )


def _created(tab_id: str = "w9:t1", pane_id: str = "w9:p1") -> str:
    return _result(
        tab={"tab_id": tab_id, "label": "new"}, root_pane={"pane_id": pane_id}
    )


async def test_create_topic_target_uses_selected_workspace_and_returns_session_target(
    tmp_path: Path,
) -> None:
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("pane", "run", out=_result(type="ok"))
        .on(
            "agent",
            "list",
            out=_agents(
                _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="selected")
            ),
        )
    )
    target = await _manager(fake).create_topic_target(
        str(tmp_path),
        launch_command="claude",
        workspace_id="selected",
        agent_args="--dangerously-skip-permissions",
    )
    assert target.target_id == _target()
    assert target.window_id == "w9:t1"
    assert target.pane_id == "w9:p1"
    assert fake.calls == [
        ["workspace", "list"],
        [
            "tab",
            "create",
            "--cwd",
            str(tmp_path),
            "--no-focus",
            "--workspace",
            "selected",
        ],
        ["pane", "run", "w9:p1", "claude --dangerously-skip-permissions"],
        ["agent", "list"],
    ]


async def test_create_topic_target_rejects_missing_selected_workspace(
    tmp_path: Path,
) -> None:
    fake = FakeHerdr().on("workspace", "list", out=_workspace("other", tmp_path))
    with pytest.raises(HerdrError, match="Selected Herdr workspace"):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command="claude", workspace_id="selected"
        )
    assert fake.calls == [["workspace", "list"]]


async def test_create_topic_target_rolls_back_malformed_root_pane_response(
    tmp_path: Path,
) -> None:
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_result(tab={"tab_id": "w9:t1", "label": "new"}))
        .on("tab", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrError, match="no root pane"):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command="claude", workspace_id="selected"
        )
    assert fake.calls[-1] == ["tab", "close", "w9:t1"]


async def test_create_topic_target_rolls_back_only_its_new_tab_when_launch_fails(
    tmp_path: Path,
) -> None:
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("pane", "run", rc=1, err="launch failed")
        .on("tab", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrError, match="Failed to start"):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command="claude", workspace_id="selected"
        )
    assert ["tab", "close", "w9:t1"] in fake.calls
    assert not any(call[:2] == ["workspace", "close"] for call in fake.calls)


async def test_create_topic_target_rolls_back_on_duplicate_or_missing_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = _agents(
        _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="selected"),
        _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="selected"),
    )
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("agent", "list", out=duplicate)
        .on("tab", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrAmbiguousTargetError):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command=None, workspace_id="selected"
        )
    assert ["tab", "close", "w9:t1"] in fake.calls

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    missing = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("agent", "list", out=_agents())
        .on("tab", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrUnresolvedTargetError):
        await _manager(missing).create_topic_target(
            str(tmp_path), launch_command=None, workspace_id="selected"
        )
    assert ["tab", "close", "w9:t1"] in missing.calls


async def test_native_worktree_returns_session_target_or_fails_unbound(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path / "worktree"
    created = _result(
        tab={"tab_id": "w10:t1", "label": "worktree"},
        root_pane={"pane_id": "w10:p1"},
        workspace={"workspace_id": "worktree-ws"},
    )
    fake = (
        FakeHerdr()
        .on("worktree", "create", out=created)
        .on("pane", "run", out=_result(type="ok"))
        .on(
            "agent",
            "list",
            out=_agents(
                _agent(pane_id="w10:p1", tab_id="w10:t1", workspace_id="worktree-ws")
            ),
        )
    )
    ok, _message, _label, target = await _manager(fake).create_worktree_window(
        str(repo), str(worktree), "ccg/topic", launch_command="claude"
    )
    assert ok and target == _target()
    assert ["pane", "run", "w10:p1", "claude"] in fake.calls

    malformed = (
        FakeHerdr()
        .on("worktree", "create", out=_result(tab={"tab_id": "w10:t1"}))
        .on("tab", "close", out=_result(type="ok"))
    )
    ok, message, _label, target = await _manager(malformed).create_worktree_window(
        str(repo), str(worktree), "ccg/topic", launch_command="claude"
    )
    assert not ok and target == "" and "root pane" in message
    assert malformed.calls[-1] == ["tab", "close", "w10:t1"]


# ── non-target transport/protocol behavior ─────────────────────────────


async def test_subprocess_run_maps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["herdr", "status"], timeout=5)

    monkeypatch.setattr("ccgram.multiplexer.herdr.asyncio.to_thread", fail)
    assert await HerdrManager()._subprocess_run(["status"]) == (
        124,
        "",
        "herdr call timed out",
    )


async def test_ensure_session_accepts_protocol_and_rejects_unavailable_server() -> None:
    good = json.dumps(
        {
            "server": {
                "running": True,
                "protocol": HERDR_PROTOCOL_VERSION,
                "compatible": True,
            }
        }
    )
    await _manager(FakeHerdr().on("status", out=good)).ensure_session()
    with pytest.raises(HerdrError):
        await _manager(FakeHerdr().on("status", out="not json")).ensure_session()
