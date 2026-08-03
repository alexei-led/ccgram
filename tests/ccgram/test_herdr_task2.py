"""Focused Task 2 guarded-session target tests."""
from __future__ import annotations

import json
from collections.abc import Sequence

from ccgram.multiplexer.herdr import (
    HerdrManager,
    HerdrSessionComposite,
    herdr_session_target_id,
)


def _envelope(result: dict) -> str:
    return json.dumps({"result": result})


def _agent(*, pane: str = "w1:p1", tab: str = "w1:t1", workspace: str = "w1") -> dict:
    return {"terminal_id": "term-1", "pane_id": pane, "tab_id": tab, "workspace_id": workspace,
            "agent_session": {"source": "herdr", "agent": "claude", "kind": "id", "value": "opaque"}}


async def test_send_guards_target_and_uses_matched_pane() -> None:
    target = herdr_session_target_id(HerdrSessionComposite("herdr", "claude", "id", "opaque"))
    calls: list[list[str]] = []

    async def runner(args: Sequence[str]) -> tuple[int, str, str]:
        calls.append(list(args))
        if args == ["agent", "list"]:
            return 0, _envelope({"agents": [_agent(pane="w7:p4")]}), ""
        return 0, "", ""

    assert await HerdrManager(runner=runner).send(target, "hello")
    assert calls == [["agent", "list"], ["pane", "run", "w7:p4", "hello"]]


async def test_raw_pane_operations_are_rejected_without_dispatch() -> None:
    calls: list[list[str]] = []

    async def runner(args: Sequence[str]) -> tuple[int, str, str]:
        calls.append(list(args))
        return 0, "", ""

    mux = HerdrManager(runner=runner)
    assert not await mux.send_to_pane("w1:p1", "unsafe", window_id="target")
    assert await mux.capture_pane_by_id("w1:p1", window_id="target") is None
    assert calls == []


async def test_create_target_uses_selected_workspace_and_rolls_back(tmp_path) -> None:
    calls: list[list[str]] = []

    async def runner(args: Sequence[str]) -> tuple[int, str, str]:
        calls.append(list(args))
        if args == ["workspace", "list"]:
            return 0, _envelope({"workspaces": [{"workspace_id": "selected", "label": "selected", "cwd": str(tmp_path)}]}), ""
        if args[:2] == ["tab", "create"]:
            return 0, _envelope({"tab": {"tab_id": "w2:t1", "label": "new"}, "root_pane": {"pane_id": "w2:p1"}}), ""
        if args == ["agent", "list"]:
            return 0, _envelope({"agents": [_agent(pane="w2:p1", tab="w2:t1", workspace="selected")]}), ""
        return 0, "", ""

    result = await HerdrManager(runner=runner).create_topic_target(str(tmp_path), launch_command="claude", workspace_id="selected")
    assert result.target_id.startswith("herdr-session-v1-")
    assert ["tab", "create", "--cwd", str(tmp_path), "--no-focus", "--workspace", "selected"] in calls
    assert ["pane", "run", "w2:p1", "claude"] in calls
    assert not any(call[:2] == ["tab", "close"] for call in calls)


async def test_replaced_target_blocks_action() -> None:
    target = herdr_session_target_id(HerdrSessionComposite("herdr", "claude", "id", "old"))

    async def runner(args: Sequence[str]) -> tuple[int, str, str]:
        if args == ["agent", "list"]:
            return 0, _envelope({"agents": [_agent()]}), ""
        return 0, "", ""

    mux = HerdrManager(runner=runner)
    assert not await mux.send(target, "must not dispatch")
