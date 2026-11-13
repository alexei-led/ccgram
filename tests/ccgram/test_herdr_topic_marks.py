"""Behavior contracts for native Herdr manual-label selection storage."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from ccgram.multiplexer.herdr import (
    HerdrManager,
    HerdrSessionComposite,
    herdr_session_target_id,
)


def _target(session: str = "one") -> str:
    return herdr_session_target_id(
        HerdrSessionComposite("herdr:pi", "pi", "id", session)
    )


class NativeMarkHerdr:
    def __init__(self, label: str | None = None) -> None:
        self.panes: list[dict[str, object]] = [
            {"terminal_id": "term-one", "pane_id": "w1:p1", "label": label}
        ]
        self.agents: list[dict[str, object]] = [self.agent("term-one", "w1:p1", "one")]
        self.writes: list[list[str]] = []
        self.unavailable = False
        self.ignore_write = False
        self.lose_ack = False
        self.manager = HerdrManager(socket_path="/tmp/herdr-test.sock", runner=self)

    @staticmethod
    def agent(terminal: str, pane: str, session: str) -> dict[str, object]:
        return {
            "terminal_id": terminal,
            "pane_id": pane,
            "workspace_id": "w1",
            "tab_id": "w1:t1",
            "agent_session": {
                "source": "herdr:pi",
                "agent": "pi",
                "kind": "id",
                "value": session,
            },
        }

    async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
        command = tuple(args[:2])
        if command == ("agent", "list"):
            result = {"agents": self.agents}
        elif command == ("workspace", "list"):
            result = {"workspaces": [{"workspace_id": "w1", "label": "workspace"}]}
        elif command == ("tab", "list"):
            result = {"tabs": [{"tab_id": "w1:t1", "label": "tab"}]}
        elif command == ("pane", "list"):
            if self.unavailable:
                return 1, "", "inventory unavailable"
            result = {"panes": self.panes}
        elif command == ("pane", "get"):
            matches = [p for p in self.panes if p.get("pane_id") == args[2]]
            if len(matches) != 1:
                return 1, "", "pane not found"
            result = {"pane": matches[0]}
        elif command == ("pane", "rename"):
            return self.rename(args)
        else:
            return 1, "", "unexpected command"
        return 0, json.dumps({"result": result}), ""

    def rename(self, args: Sequence[str]) -> tuple[int, str, str]:
        self.writes.append(list(args))
        matches = [p for p in self.panes if p.get("pane_id") == args[2]]
        if len(matches) != 1:
            return 1, "", "pane not found"
        if not self.ignore_write:
            matches[0]["label"] = (
                None if list(args[3:]) == ["--clear"] else " ".join(args[3:]).strip()
            )
        if self.lose_ack:
            self.lose_ack = False
            return 1, "", "response lost after write"
        return 0, json.dumps({"result": {"pane": matches[0]}}), ""

    async def window(self):
        windows = await self.manager.list_windows_for_reconciliation()
        assert windows is not None
        return windows[0]


@pytest.mark.parametrize("label", [None, "editor", "build 中 שלום", "--clear"])
async def test_herdr_topic_mark_preserves_user_label_and_is_idempotent(label) -> None:
    native = NativeMarkHerdr(label)
    assert await native.manager.set_topic_mark(_target(), True)
    selected = await native.window()
    assert selected.topic_marked is True
    assert selected.topic_mark_owner
    assert native.panes[0]["label"] != label
    assert await native.manager.set_topic_mark(
        _target(), True, owner=selected.topic_mark_owner
    )
    assert len(native.writes) == 1
    assert await native.manager.set_topic_mark(
        _target(), False, owner=selected.topic_mark_owner
    )
    assert native.panes[0]["label"] == label
    assert (await native.window()).topic_marked is False
    assert await native.manager.set_topic_mark(
        _target(), False, owner=selected.topic_mark_owner
    )
    assert len(native.writes) == 2


async def test_herdr_mark_owner_is_stable_native_witness_not_conversation_or_label() -> (
    None
):
    native = NativeMarkHerdr("manager")
    first = await native.window()
    assert first.topic_marked is False
    assert first.topic_mark_owner
    native.agents = [native.agent("term-one", "w1:p1", "two")]
    native.panes[0]["label"] = "worker"
    second = await native.window()
    assert first.window_id != second.window_id
    assert first.topic_mark_owner == second.topic_mark_owner
    native.panes[0]["terminal_id"] = "new-terminal"
    native.agents = [native.agent("new-terminal", "w1:p1", "two")]
    assert (await native.window()).topic_mark_owner != first.topic_mark_owner


async def test_herdr_mark_owner_is_connection_namespaced() -> None:
    native = NativeMarkHerdr()
    first = await native.window()
    other = HerdrManager(socket_path="/tmp/another-herdr.sock", runner=native)
    assert (await other.list_windows())[0].topic_mark_owner != first.topic_mark_owner


async def test_herdr_mark_owner_clears_after_rekey_bare_shell_and_native_restore() -> (
    None
):
    native = NativeMarkHerdr("original")
    original_owner = (await native.window()).topic_mark_owner
    assert await native.manager.set_topic_mark(_target(), True, owner=original_owner)
    native.agents = []
    native.panes[0].update(terminal_id="restored-terminal", pane_id="w9:p7")
    assert await native.manager.set_topic_mark(_target(), False, owner=original_owner)
    assert native.panes[0]["label"] == "original"
    assert await native.manager.set_topic_mark(_target(), False, owner=original_owner)
    assert not await native.manager.set_topic_mark(
        _target(), True, owner=original_owner
    )


async def test_herdr_marker_survives_native_restore_without_reusing_runtime_witness() -> (
    None
):
    native = NativeMarkHerdr()
    owner = (await native.window()).topic_mark_owner
    assert await native.manager.set_topic_mark(_target(), True, owner=owner)
    native.panes[0]["terminal_id"] = "restored"
    native.agents = [native.agent("restored", "w1:p1", "two")]
    restored = await native.window()
    assert restored.topic_marked is True
    assert restored.topic_mark_owner == owner
    native.panes.append(
        {"terminal_id": "new-worker", "pane_id": "w1:p2", "label": None}
    )
    native.agents.append(native.agent("new-worker", "w1:p2", "worker"))
    worker = (await native.manager.list_windows())[1]
    assert worker.topic_marked is False
    assert worker.topic_mark_owner != owner


async def test_herdr_mark_owner_recovers_ack_loss_without_old_agent_guard() -> None:
    native = NativeMarkHerdr("keep me")
    owner = (await native.window()).topic_mark_owner
    native.lose_ack = True
    assert not await native.manager.set_topic_mark(_target(), True, owner=owner)
    native.agents = []
    assert await native.manager.set_topic_mark(_target(), True, owner=owner)
    assert await native.manager.set_topic_mark(_target(), False, owner=owner)
    assert native.panes[0]["label"] == "keep me"
    assert len(native.writes) == 2


async def test_herdr_mark_owner_rejects_duplicate_copied_markers_without_touching_panes() -> (
    None
):
    native = NativeMarkHerdr("source")
    owner = (await native.window()).topic_mark_owner
    assert await native.manager.set_topic_mark(_target(), True, owner=owner)
    native.panes.append(
        {"terminal_id": "copied", "pane_id": "w1:p2", "label": native.panes[0]["label"]}
    )
    native.agents.append(native.agent("copied", "w1:p2", "copy"))
    windows = await native.manager.list_windows()
    assert len(windows) == 2
    assert all(w.topic_marked is None and w.topic_mark_owner is None for w in windows)
    assert not await native.manager.set_topic_mark(_target(), False, owner=owner)
    assert len(native.writes) == 1


@pytest.mark.parametrize(
    "label",
    [
        "prefix [ccgram-topic-v1:bad]",
        "[ccgram-topic-v2:0123456789abcdef0123456789abcdef]",
        "[ccgram-topic-v1:0123456789abcdef0123456789abcdef] trailing",
        "[ccgram-topic-v1:0123456789abcdef0123456789abcdef] [ccgram-topic-v1:0123456789abcdef0123456789abcdef]",
        " outer whitespace ",
        "\ud800",
        17,
    ],
)
async def test_herdr_marker_invalid_label_is_unknown_and_never_overwritten(
    label,
) -> None:
    native = NativeMarkHerdr()
    native.panes[0]["label"] = label
    window = await native.window()
    assert window.topic_marked is None
    assert window.topic_mark_owner is None
    assert not await native.manager.set_topic_mark(_target(), True)
    assert native.writes == []


async def test_herdr_marker_queries_do_not_filter_liveness_or_change_eligibility() -> (
    None
):
    native = NativeMarkHerdr()
    native.unavailable = True
    window = await native.window()
    assert window.topic_eligible is True
    assert window.topic_marked is None
    assert window.topic_mark_owner is None
    assert await native.manager.find_window_by_id(_target())
    assert not await native.manager.set_topic_mark(_target(), False, owner="0" * 32)


async def test_herdr_marker_uses_raw_manual_label_not_agent_title() -> None:
    native = NativeMarkHerdr()
    native.panes[0]["terminal_title"] = "[ccgram-topic-v1:" + "a" * 32 + "]"
    native.agents[0]["title"] = native.panes[0]["terminal_title"]
    assert (await native.window()).topic_marked is False


async def test_herdr_topic_mark_rejects_oversize_serialized_label_before_mutation() -> (
    None
):
    native = NativeMarkHerdr('"' * (1024 * 1024))
    assert not await native.manager.set_topic_mark(_target(), True)
    assert native.writes == []


async def test_herdr_topic_mark_requires_verified_write_and_expected_owner() -> None:
    native = NativeMarkHerdr()
    owner = (await native.window()).topic_mark_owner
    assert not await native.manager.set_topic_mark(_target(), True, owner="a" * 32)
    native.ignore_write = True
    assert not await native.manager.set_topic_mark(_target(), True, owner=owner)
    assert (await native.window()).topic_marked is False


async def test_herdr_topic_mark_rejects_foreign_backend_targets_and_stale_witness() -> (
    None
):
    native = NativeMarkHerdr()
    owner = (await native.window()).topic_mark_owner
    assert not await native.manager.set_topic_mark("@1", True, owner=owner)
    assert not await native.manager.set_topic_mark("w1:p1", False, owner=owner)
    native.panes[0]["terminal_id"] = "replacement"
    native.agents = [native.agent("replacement", "w1:p1", "two")]
    assert not await native.manager.set_topic_mark(_target(), True, owner=owner)
    assert await native.manager.set_topic_mark(_target(), False, owner=owner)
    assert native.writes == []
