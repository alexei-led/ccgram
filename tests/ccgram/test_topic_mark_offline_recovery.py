"""Offline retries reuse actual retirement verification, never an acknowledgement flag."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from ccgram.multiplexer import native_process, topic_mark_transport
from ccgram.multiplexer.base import Multiplexer, WindowRef
from ccgram.topic_mark_cmd import _topic_action
from ccgram.topic_mark_journal import (
    TOPIC_MARK_OPERATIONS_KEY,
    TopicMarkAttempt,
    TopicMarkOperation,
    read_topic_mark_operations,
)
from ccgram.topic_mark_types import TopicMarkWriteGuard, TopicMarkWriteWitness
from ccgram.utils import atomic_write_json


class _RestoredNative:
    capabilities = SimpleNamespace(name="herdr")

    def __init__(
        self, path: Path, old: TopicMarkWriteWitness, current: TopicMarkWriteWitness
    ):
        self.path = path
        self.old = old
        self.current = current
        self.marked = False
        self.mutations: list[bool] = []

    def window(self) -> WindowRef:
        return WindowRef(
            "@1",
            "restored",
            "/owned",
            topic_marked=self.marked,
            topic_mark_owner="owner",
        )

    async def list_windows_for_reconciliation(self) -> list[WindowRef]:
        return [self.window()]

    async def find_window_by_id(self, target: str) -> WindowRef | None:
        return self.window() if target == "@1" else None

    async def topic_mark_endpoint_scope(self) -> str | None:
        return self.current.endpoint_scope

    async def recover_topic_mark_write(
        self, witness: TopicMarkWriteWitness
    ) -> bool | None:
        assert witness == self.old
        return topic_mark_transport.recover_write(witness, backend="herdr")

    async def set_topic_mark(
        self,
        target: str,
        marked: bool,
        *,
        owner=None,
        write_guard: TopicMarkWriteGuard | None = None,
    ) -> bool:
        assert target == "@1" and owner == "owner"
        (ready,) = read_topic_mark_operations(self.path)
        assert ready.attempt is None
        assert write_guard is not None
        witness = replace(self.current, dispatch_id=uuid4().hex)
        assert witness.dispatch_id != self.old.dispatch_id
        write_guard.arm(witness)
        (saved,) = read_topic_mark_operations(self.path)
        assert saved.attempt is not None
        assert saved.attempt.witness == witness
        self.mutations.append(marked)
        self.marked = marked
        write_guard.terminal(witness.dispatch_id)
        return True


@pytest.mark.parametrize(
    "retired,latest_desired,action,owner,success",
    [
        pytest.param(False, True, "mark", "owner", False, id="same-executor-fenced"),
        pytest.param(None, True, "mark", "owner", False, id="unproved-exit-fenced"),
        pytest.param(True, True, "mark", "owner", True, id="retired-exact-A-fresh-B"),
        pytest.param(True, False, "mark", "owner", False, id="newer-off-preserved"),
        pytest.param(True, True, "unmark", "owner", False, id="opposing-noop-refused"),
        pytest.param(
            True, True, "mark", None, False, id="unknown-owner-not-substituted"
        ),
    ],
)
async def test_offline_cli_recovers_only_verified_matching_attempt(
    tmp_path, monkeypatch, retired, latest_desired, action, owner, success
):
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    host, boot, namespace = native_process._local_scope()
    old_start, current_start, code_id = (
        ("darwin:100:200", "darwin:100:201", "darwin:herdr:0.9.5:cdhash:" + "a" * 40)
        if sys.platform == "darwin"
        else ("linux:100", "linux:101", "linux:herdr:0.9.5:sha256:" + "a" * 64)
    )
    old = TopicMarkWriteWitness(
        uuid4().hex,
        "herdr",
        host,
        boot,
        namespace,
        os.getpid(),
        old_start,
        code_id,
        0,
        456,
        None,
    )
    current = replace(
        old, dispatch_id=uuid4().hex, native_start=current_start, socket_ino=457
    )
    operation = TopicMarkOperation(
        "@1", owner, latest_desired, TopicMarkAttempt(True, old)
    )
    path = tmp_path / "state.json"
    unrelated = {
        "window_states": {"@keep": {"session_id": "preserve"}},
        "other": [1, 2],
    }
    atomic_write_json(
        path, unrelated | {TOPIC_MARK_OPERATIONS_KEY: [operation.to_dict()]}
    )
    original = path.read_bytes()
    probes = []

    def process_start(pid):
        assert pid == os.getpid()
        probes.append(pid)
        if retired is None:
            raise PermissionError("process generation is unconfirmed")
        return current_start if retired else old_start

    monkeypatch.setattr(native_process, "_process_start", process_start)
    native = _RestoredNative(path, old, current)

    if success:
        result = await _topic_action(cast(Multiplexer, native), action, "@1")
        assert result == {"target": "@1", "marked": True}
    else:
        with pytest.raises((RuntimeError, ValueError)):
            await _topic_action(cast(Multiplexer, native), action, "@1")

    saved = read_topic_mark_operations(path)
    assert native.mutations == ([True] if success else [])
    if owner is None:
        assert not probes
        assert saved == (operation,)
        assert path.read_bytes() == original
    elif retired is not True:
        assert probes
        assert saved == (operation,)
        assert path.read_bytes() == original
    elif success:
        assert probes and saved == ()
    else:
        assert probes and saved == (replace(operation, attempt=None),)
    import json

    contents = json.loads(path.read_text(encoding="utf-8"))
    assert {key: contents[key] for key in unrelated} == unrelated
