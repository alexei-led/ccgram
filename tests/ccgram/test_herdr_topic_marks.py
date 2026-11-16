"""Behavior contracts for native Herdr manual-label selection storage."""

from __future__ import annotations

import json
import asyncio
import contextlib
from contextlib import AsyncExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from collections.abc import Sequence

import pytest

from ccgram.multiplexer import native_process
from _topic_mark_helpers import RecordingTopicMarkGuard

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
    def __init__(self, label: str | None = None, *, socket_path: str) -> None:
        self.panes: list[dict[str, object]] = [
            {"terminal_id": "term-one", "pane_id": "w1:p1", "label": label}
        ]
        self.agents: list[dict[str, object]] = [self.agent("term-one", "w1:p1", "one")]
        self.writes: list[list[str]] = []
        self.unavailable = False
        self.ignore_write = False
        self.lose_ack = False
        self.manager = HerdrManager(
            socket_path=socket_path, runner=self, binary="/forbidden-herdr"
        )

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


@pytest.fixture
async def native_factory(monkeypatch):
    platform = sys.platform
    namespace = "darwin-local" if platform == "darwin" else "linux:1:2:3:4"
    start = "darwin:100:200" if platform == "darwin" else "linux:100"
    kind, width = ("cdhash", 40) if platform == "darwin" else ("sha256", 64)
    identity = native_process.NativeProcessIdentity(
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        namespace,
        123,
        start,
        f"{platform}:herdr:0.9.5:{kind}:" + "a" * width,
    )
    monkeypatch.setattr(
        native_process, "capture_native_process", lambda *_a, **_k: identity
    )

    def forbidden(*_a, **_k):
        raise AssertionError("native subprocess forbidden")

    monkeypatch.setattr("subprocess.run", forbidden)
    async with AsyncExitStack() as stack:
        directory = stack.enter_context(
            TemporaryDirectory(prefix="ccg-hmark-", dir="/tmp")
        )
        count = 0

        async def build(label=None):
            nonlocal count
            count += 1
            path = Path(directory) / str(count)
            native = NativeMarkHerdr(label, socket_path=str(path))
            handlers = set()

            async def serve(reader, writer):
                task = asyncio.current_task()
                handlers.add(task)
                try:
                    line = await reader.readline()
                    if not line:
                        return
                    request = json.loads(line)
                    assert request["method"] == "pane.rename"
                    params = request["params"]
                    label = params["label"]
                    args = ["pane", "rename", params["pane_id"]]
                    args += ["--clear"] if label is None else [label]
                    rc, text, _error = native.rename(args)
                    if rc == 0:
                        response = json.loads(text) | {"id": request["id"]}
                        pane = response["result"]["pane"]
                        if pane.get("label") is None:
                            pane.pop("label", None)
                        writer.write(json.dumps(response).encode() + b"\n")
                        await writer.drain()
                finally:
                    writer.close()
                    with contextlib.suppress(ConnectionError, OSError):
                        await writer.wait_closed()
                    handlers.discard(task)

            server = await asyncio.start_unix_server(serve, path=path)

            async def close():
                server.close()
                await server.wait_closed()
                if handlers:
                    await asyncio.gather(*handlers, return_exceptions=True)

            stack.push_async_callback(close)
            return native

        yield build


@pytest.mark.parametrize("label", [None, "", "editor", "build 中 שלום", "--clear"])
async def test_herdr_topic_mark_preserves_user_label_and_is_idempotent(
    native_factory, label
) -> None:
    native = await native_factory(label)
    assert await native.manager.set_topic_mark(
        _target(), True, write_guard=RecordingTopicMarkGuard()
    )
    selected = await native.window()
    assert selected.topic_marked is True
    assert selected.topic_mark_owner
    assert native.panes[0]["label"] != label
    assert await native.manager.set_topic_mark(
        _target(),
        True,
        owner=selected.topic_mark_owner,
        write_guard=RecordingTopicMarkGuard(),
    )
    assert len(native.writes) == 1
    assert await native.manager.set_topic_mark(
        _target(),
        False,
        owner=selected.topic_mark_owner,
        write_guard=RecordingTopicMarkGuard(),
    )
    prefix = f"{label} " if label else ""
    assert (
        native.panes[0]["label"]
        == f"{prefix}[ccgram-topic-v1:off:{selected.topic_mark_owner}]"
    )
    unselected = await native.window()
    assert unselected.topic_marked is False
    assert unselected.topic_mark_owner == selected.topic_mark_owner
    assert await native.manager.set_topic_mark(
        _target(),
        False,
        owner=selected.topic_mark_owner,
        write_guard=RecordingTopicMarkGuard(),
    )
    assert len(native.writes) == 2


async def test_herdr_mark_owner_is_stable_native_witness_not_conversation_or_label(
    native_factory,
) -> None:
    native = await native_factory("manager")
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


async def test_herdr_mark_owner_is_connection_namespaced(native_factory) -> None:
    native = await native_factory()
    first = await native.window()
    other = HerdrManager(socket_path="/tmp/another-herdr.sock", runner=native)
    assert (await other.list_windows())[0].topic_mark_owner != first.topic_mark_owner


async def test_herdr_mark_owner_clears_after_rekey_bare_shell_and_native_restore(
    native_factory,
) -> None:
    native = await native_factory("original")
    original_owner = (await native.window()).topic_mark_owner
    assert await native.manager.set_topic_mark(
        _target(), True, owner=original_owner, write_guard=RecordingTopicMarkGuard()
    )
    native.agents = []
    native.panes[0].update(terminal_id="restored-terminal", pane_id="w9:p7")
    assert await native.manager.set_topic_mark(
        _target(), False, owner=original_owner, write_guard=RecordingTopicMarkGuard()
    )
    assert (
        native.panes[0]["label"] == f"original [ccgram-topic-v1:off:{original_owner}]"
    )
    snapshot = await native.manager._topic_mark_snapshot()
    assert snapshot is not None and snapshot[0].owner == original_owner
    assert snapshot[0].marked is False
    assert await native.manager.set_topic_mark(
        _target(), False, owner=original_owner, write_guard=RecordingTopicMarkGuard()
    )
    native.panes[0].update(terminal_id="restored-again", pane_id="w10:p3")
    assert await native.manager.set_topic_mark(
        _target(), True, owner=original_owner, write_guard=RecordingTopicMarkGuard()
    )

    assert native.panes[0]["label"] == f"original [ccgram-topic-v1:{original_owner}]"


@pytest.mark.parametrize("marked", [True, False])
async def test_herdr_marker_survives_native_restore_without_reusing_runtime_witness(
    native_factory,
    marked,
) -> None:
    native = await native_factory()
    owner = (await native.window()).topic_mark_owner
    assert await native.manager.set_topic_mark(
        _target(), True, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    if not marked:
        assert await native.manager.set_topic_mark(
            _target(), False, owner=owner, write_guard=RecordingTopicMarkGuard()
        )
    native.panes[0]["terminal_id"] = "restored"
    native.agents = [native.agent("restored", "w1:p1", "two")]
    restored = await native.window()
    assert restored.topic_marked is marked
    assert restored.topic_mark_owner == owner
    native.panes.append(
        {"terminal_id": "new-worker", "pane_id": "w1:p2", "label": None}
    )
    native.agents.append(native.agent("new-worker", "w1:p2", "worker"))
    worker = (await native.manager.list_windows())[1]
    assert worker.topic_marked is False
    assert worker.topic_mark_owner != owner


async def test_herdr_mark_owner_recovers_ack_loss_without_old_agent_guard(
    native_factory,
) -> None:
    native = await native_factory("keep me")
    owner = (await native.window()).topic_mark_owner
    native.lose_ack = True
    assert not await native.manager.set_topic_mark(
        _target(), True, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    native.agents = []
    assert await native.manager.set_topic_mark(
        _target(), True, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    assert await native.manager.set_topic_mark(
        _target(), False, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    assert native.panes[0]["label"] == f"keep me [ccgram-topic-v1:off:{owner}]"
    assert len(native.writes) == 2


@pytest.mark.parametrize("source_marked", [True, False])
@pytest.mark.parametrize("copy_marked", [True, False])
async def test_herdr_mark_owner_rejects_duplicate_copied_markers_without_touching_panes(
    native_factory,
    source_marked,
    copy_marked,
) -> None:
    native = await native_factory("source")
    owner = (await native.window()).topic_mark_owner
    assert await native.manager.set_topic_mark(
        _target(), True, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    if not source_marked:
        assert await native.manager.set_topic_mark(
            _target(), False, owner=owner, write_guard=RecordingTopicMarkGuard()
        )
    copy_state = "" if copy_marked else "off:"
    copied_label = f"source [ccgram-topic-v1:{copy_state}{owner}]"
    native.panes.append(
        {"terminal_id": "copied", "pane_id": "w1:p2", "label": copied_label}
    )
    native.agents.append(native.agent("copied", "w1:p2", "copy"))
    windows = await native.manager.list_windows()
    assert len(windows) == 2
    assert all(w.topic_marked is None and w.topic_mark_owner is None for w in windows)
    writes = len(native.writes)
    labels = [pane["label"] for pane in native.panes]
    for desired in (True, False):
        assert not await native.manager.set_topic_mark(
            _target(), desired, owner=owner, write_guard=RecordingTopicMarkGuard()
        )
    assert [pane["label"] for pane in native.panes] == labels
    assert len(native.writes) == writes


@pytest.mark.parametrize(
    "label",
    [
        "prefix [ccgram-topic-v1:bad]",
        "[ccgram-topic-v1:off:bad]",
        "[ccgram-topic-v1:OFF:0123456789abcdef0123456789abcdef]",
        "[ccgram-topic-v1:on:0123456789abcdef0123456789abcdef]",
        "[ccgram-topic-v1:off:off:0123456789abcdef0123456789abcdef]",
        "[ccgram-topic-v1:off:0123456789abcdef0123456789abcdef] trailing",
        "[ccgram-topic-v1:0123456789abcdef0123456789abcdef] [ccgram-topic-v1:off:0123456789abcdef0123456789abcdef]",
        "[ccgram-topic-v2:0123456789abcdef0123456789abcdef]",
        "[ccgram-topic-v1:0123456789abcdef0123456789abcdef] trailing",
        "[ccgram-topic-v1:0123456789abcdef0123456789abcdef] [ccgram-topic-v1:0123456789abcdef0123456789abcdef]",
        " outer whitespace ",
        "\ud800",
        17,
    ],
)
async def test_herdr_marker_invalid_label_is_unknown_and_never_overwritten(
    native_factory,
    label,
) -> None:
    native = await native_factory()
    native.panes[0]["label"] = label
    window = await native.window()
    assert window.topic_marked is None
    assert window.topic_mark_owner is None
    assert not await native.manager.set_topic_mark(
        _target(), True, write_guard=RecordingTopicMarkGuard()
    )
    assert native.writes == []


async def test_herdr_marker_queries_do_not_filter_liveness_or_change_eligibility(
    native_factory,
) -> None:
    native = await native_factory()
    native.unavailable = True
    window = await native.window()
    assert window.topic_eligible is True
    assert window.topic_marked is None
    assert window.topic_mark_owner is None
    assert await native.manager.find_window_by_id(_target())
    assert not await native.manager.set_topic_mark(
        _target(), False, owner="0" * 32, write_guard=RecordingTopicMarkGuard()
    )


async def test_herdr_marker_uses_raw_manual_label_not_agent_title(
    native_factory,
) -> None:
    native = await native_factory()
    native.panes[0]["terminal_title"] = "[ccgram-topic-v1:" + "a" * 32 + "]"
    native.agents[0]["title"] = native.panes[0]["terminal_title"]
    assert (await native.window()).topic_marked is False


async def test_herdr_topic_mark_rejects_oversize_serialized_label_before_mutation(
    native_factory,
) -> None:
    native = await native_factory('"' * (1024 * 1024))
    assert not await native.manager.set_topic_mark(
        _target(), True, write_guard=RecordingTopicMarkGuard()
    )
    assert native.writes == []


async def test_herdr_topic_mark_requires_verified_write_and_expected_owner(
    native_factory,
) -> None:
    native = await native_factory()
    owner = (await native.window()).topic_mark_owner
    assert not await native.manager.set_topic_mark(
        _target(), True, owner="a" * 32, write_guard=RecordingTopicMarkGuard()
    )
    native.ignore_write = True
    assert not await native.manager.set_topic_mark(
        _target(), True, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    assert (await native.window()).topic_marked is False


async def test_herdr_topic_mark_rejects_foreign_backend_targets_and_stale_witness(
    native_factory,
) -> None:
    native = await native_factory()
    owner = (await native.window()).topic_mark_owner
    assert not await native.manager.set_topic_mark(
        "@1", True, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    assert not await native.manager.set_topic_mark(
        "w1:p1", False, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    native.panes[0]["terminal_id"] = "replacement"
    native.agents = [native.agent("replacement", "w1:p1", "two")]
    assert not await native.manager.set_topic_mark(
        _target(), True, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    assert await native.manager.set_topic_mark(
        _target(), False, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    assert native.writes == []


async def test_herdr_endpoint_scope_matches_actual_guard_witness(native_factory):
    native = await native_factory()
    scope = await native.manager.topic_mark_endpoint_scope()
    write_guard = RecordingTopicMarkGuard()

    assert scope is not None and native.writes == []
    assert await native.manager.set_topic_mark(_target(), True, write_guard=write_guard)
    assert write_guard.armed[0].endpoint_scope == scope


async def test_herdr_off_suffix_respects_full_envelope_limit_without_truncation(
    native_factory,
):
    native = await native_factory()
    owner = (await native.window()).topic_mark_owner
    assert owner is not None
    suffix = f" [ccgram-topic-v1:{owner}]"
    request = {
        "id": "0" * 32,
        "method": "pane.rename",
        "params": {"pane_id": "w1:p1", "label": suffix},
    }
    overhead = (
        len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()) + 1
    )
    original = "x" * (1024 * 1024 - overhead)
    selected_label = original + suffix
    native.panes[0]["label"] = selected_label
    write_guard = RecordingTopicMarkGuard()

    assert not await native.manager.set_topic_mark(
        _target(), False, owner=owner, write_guard=write_guard
    )
    assert native.panes[0]["label"] == selected_label
    assert native.writes == [] and write_guard.armed == []
