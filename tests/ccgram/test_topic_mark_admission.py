"""Real quarantine state gates discovery and final admission, not existing bindings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from _topic_mark_helpers import make_witness
from ccgram import topic_selection
from ccgram.config import config
from ccgram.handlers.callback_data import CB_WIN_BIND
from ccgram.handlers.topics import (
    topic_orchestration,
    window_callbacks,
    window_launch_service,
)
from ccgram.handlers.topics.directory_browser import UNBOUND_WINDOWS_KEY
from ccgram.handlers.user_state import PENDING_THREAD_ID
from ccgram.multiplexer.base import Multiplexer, WindowRef
from ccgram.providers import _ensure_registered
from ccgram.session import SessionManager
from ccgram.state_writer_lock import StateWriterLease
from ccgram.session_map import get_session_map_sync, install_session_map_sync
from ccgram.session_monitor import SessionMonitor
from ccgram.thread_router import get_thread_router, install_thread_router
from ccgram.topic_mark_journal import read_topic_mark_operations
from ccgram.user_preferences import get_user_preferences, install_user_preferences
from ccgram.window_state_store import get_window_store, install_window_store
from ccgram.window_state_ports import lifecycle_state


@dataclass
class _State:
    manager: SessionManager

    def arm(self):
        store = get_window_store()
        operation = store.queue_topic_mark("@1", "owner", True)
        store.begin_topic_mark_attempt(operation, make_witness("A"))
        self.manager.flush_state()


@pytest.fixture
async def state(tmp_path, monkeypatch):
    previous_store = get_window_store()
    previous_router = get_thread_router()
    previous_preferences = get_user_preferences()
    previous_sync = get_session_map_sync()
    monkeypatch.setattr(config, "state_file", tmp_path / "state.json")
    monkeypatch.setattr(config, "allowed_users", {100})
    monkeypatch.setattr(config, "group_id", -100500)
    manager = SessionManager()
    monkeypatch.setattr(topic_orchestration, "session_manager", manager)
    monkeypatch.setattr("ccgram.session.session_manager", manager)
    router = get_thread_router()
    router.bind_thread(100, 77, "@bound", chat_id=-100501)
    get_window_store().get_window_state("@bound").session_id = "existing-session"
    try:
        yield _State(manager)
    finally:
        manager.flush_state()
        install_window_store(previous_store)
        install_thread_router(previous_router)
        install_user_preferences(previous_preferences)
        install_session_map_sync(previous_sync)


def _window(window_id="@1", owner="owner"):
    return WindowRef(
        window_id, "owned", "/project", topic_marked=True, topic_mark_owner=owner
    )


@pytest.mark.parametrize("mode", ["auto", "marked"])
async def test_monitor_quarantine_prefilter_keeps_bound_and_independent_owners(
    state, tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(config, "topic_discovery_mode", mode)
    state.arm()
    observed = []

    async def on_window(event):
        observed.append(event.window_id)

    monitor = SessionMonitor(
        projects_path=tmp_path / "projects", state_file=tmp_path / "monitor.json"
    )
    monitor.set_new_window_callback(on_window)
    await monitor._emit_unbound_window_events(
        [_window(), _window("@free", "independent"), _window("@bound", "bound-owner")],
        set(),
    )

    assert observed == ["@free"]
    assert get_thread_router().get_window_for_chat_thread(-100501, 77) == "@bound"
    assert (
        get_window_store().get_window_state("@bound").session_id == "existing-session"
    )
    assert len(get_window_store().topic_mark_operations) == 1


@pytest.mark.parametrize("mode", ["auto", "marked"])
@pytest.mark.parametrize("phase", ["before-create", "during-create"])
async def test_quarantine_before_or_during_remote_creation_prevents_final_binding(
    state, monkeypatch, mode, phase
):
    monkeypatch.setattr(config, "topic_discovery_mode", mode)
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(return_value=_window()),
        list_windows_for_reconciliation=AsyncMock(return_value=[_window()]),
        set_topic_mark=AsyncMock(
            side_effect=AssertionError("automatic admission must not mark")
        ),
    )
    monkeypatch.setattr(topic_orchestration, "tmux_manager", native)
    monkeypatch.setattr("ccgram.multiplexer.multiplexer", native)
    client = AsyncMock()
    client.get_chat.return_value = SimpleNamespace(is_forum=True)

    async def create(**_kwargs):
        if phase == "during-create":
            state.arm()
        return SimpleNamespace(message_thread_id=501)

    client.create_forum_topic.side_effect = create
    if phase == "before-create":
        state.arm()

    result = await topic_orchestration.create_topic_in_chat(
        client, -100500, "@1", "owned", user_id=100, automatic=True
    )

    assert result is False
    router = get_thread_router()
    assert router.get_window_for_chat_thread(-100500, 501) is None
    assert router.get_window_for_chat_thread(-100501, 77) == "@bound"
    assert (
        get_window_store().get_window_state("@bound").session_id == "existing-session"
    )
    (armed,) = get_window_store().topic_mark_operations.values()
    assert armed.attempt is not None and armed.attempt.witness.dispatch_id == "A"
    native.set_topic_mark.assert_not_awaited()
    if phase == "before-create":
        client.create_forum_topic.assert_not_awaited()
        assert router.iter_topic_provisionings() == []
    else:
        client.create_forum_topic.assert_awaited_once()
        (claim,) = router.iter_topic_provisionings()
        assert claim.thread_id == 501 and claim.topic_mark_owner == "owner"


@pytest.mark.parametrize("retired", [False, None, True])
@pytest.mark.parametrize("latest_desired", [False, True])
async def test_recovery_needs_positive_retirement_then_fresh_exact_attempt(
    state, monkeypatch, retired, latest_desired
):
    state.arm()
    store = get_window_store()
    latest = store.queue_topic_mark("@1", "owner", latest_desired)
    state.manager.flush_state()
    path = config.state_file
    original = path.read_bytes()
    (armed,) = read_topic_mark_operations(path)
    assert armed.attempt is not None
    native_sends = []

    async def write(window_id, marked, *, owner, write_guard):
        assert window_id == "@1" and owner == "owner"
        assert marked is latest_desired
        write_guard.arm(make_witness("B"))
        (before_send,) = read_topic_mark_operations(path)
        assert before_send.attempt is not None
        assert before_send.attempt.witness.dispatch_id == "B"
        native_sends.append(marked)
        write_guard.terminal("B")
        return True

    native = SimpleNamespace(
        capabilities=SimpleNamespace(name="tmux"),
        recover_topic_mark_write=AsyncMock(return_value=retired),
        set_topic_mark=AsyncMock(side_effect=write),
        find_window_by_id=AsyncMock(return_value=_window()),
    )

    await topic_selection.recover_topic_selections(mux=cast(Multiplexer, native))

    native.recover_topic_mark_write.assert_awaited_once_with(armed.attempt.witness)
    assert get_thread_router().get_window_for_chat_thread(-100501, 77) == "@bound"
    if retired is True:
        assert native_sends == [latest_desired]
        assert read_topic_mark_operations(path) == ()
    else:
        assert native_sends == []
        native.set_topic_mark.assert_not_awaited()
        assert read_topic_mark_operations(path) == (latest,)
        assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["auto", "marked"])
async def test_picker_cannot_bind_after_quarantine_arrives_during_native_read(
    state, monkeypatch, mode
):
    monkeypatch.setattr(config, "topic_discovery_mode", mode)
    read_started, release_read = asyncio.Event(), asyncio.Event()
    selection_verified = False

    async def find(window_id):
        assert window_id == "@1"
        if selection_verified:
            read_started.set()
            await release_read.wait()
        return _window()

    async def verified_noop(*_args, **_kwargs):
        nonlocal selection_verified
        selection_verified = True
        return True

    native = SimpleNamespace(
        capabilities=SimpleNamespace(name="tmux"),
        find_window_by_id=AsyncMock(side_effect=find),
        set_topic_mark=AsyncMock(side_effect=verified_noop),
    )
    monkeypatch.setattr(window_callbacks, "session_manager", state.manager)
    monkeypatch.setattr(window_callbacks, "tmux_manager", native)
    monkeypatch.setattr("ccgram.multiplexer.multiplexer", native)
    query = AsyncMock()
    message = MagicMock()
    message.message_thread_id = 42
    message.chat.id = -100500
    message.chat.type = "supergroup"
    update = MagicMock()
    update.message = None
    update.callback_query.message = message
    context = MagicMock()
    context.user_data = {UNBOUND_WINDOWS_KEY: ["@1"], PENDING_THREAD_ID: 42}
    context.bot_data = {}
    task = asyncio.create_task(
        window_callbacks.handle_window_callback(
            query, 100, f"{CB_WIN_BIND}0", update, context
        )
    )
    try:
        async with asyncio.timeout(2):
            await read_started.wait()
        state.arm()
        release_read.set()
        await task
    finally:
        release_read.set()
        await asyncio.gather(task, return_exceptions=True)

    router = get_thread_router()
    assert router.get_window_for_chat_thread(-100500, 42) is None
    assert router.get_window_for_chat_thread(-100501, 77) == "@bound"
    (armed,) = read_topic_mark_operations(config.state_file)
    assert armed.attempt is not None and armed.attempt.witness.dispatch_id == "A"


@pytest.mark.parametrize("mode", ["auto", "marked"])
async def test_launch_rechecks_quarantine_after_awaited_native_title_write(
    state, tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(config, "topic_discovery_mode", mode)
    _ensure_registered()
    started, release = asyncio.Event(), asyncio.Event()

    async def stamp(*_args):
        started.set()
        await release.wait()

    native = SimpleNamespace(
        capabilities=SimpleNamespace(
            name="tmux",
            native_worktrees=False,
            native_agent_status=True,
            native_topic_targets=True,
        ),
        find_window_by_id=AsyncMock(return_value=_window()),
        set_topic_mark=AsyncMock(return_value=True),
        stamp_pane_title=AsyncMock(side_effect=stamp),
    )
    adapter = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(target_id="@1", title="owned")),
        close=AsyncMock(return_value=(True, "closed")),
        resolve=AsyncMock(
            return_value=(
                SimpleNamespace(
                    foreground_argv=("claude",), provider="claude", available=True
                ),
                None,
            )
        ),
    )
    monkeypatch.setattr(window_launch_service, "tmux_manager", native)
    monkeypatch.setattr(window_launch_service, "session_manager", state.manager)
    monkeypatch.setattr(window_launch_service, "adapter_for", lambda _data: adapter)
    monkeypatch.setattr("ccgram.multiplexer.multiplexer", native)
    monkeypatch.setattr(
        get_session_map_sync(),
        "wait_for_session_map_entry",
        AsyncMock(return_value=True),
    )
    query = AsyncMock()
    context = MagicMock()
    context.user_data = {PENDING_THREAD_ID: 42}
    context.bot_data = {}
    request = window_launch_service.WindowLaunchRequest(
        user_id=100,
        thread_id=42,
        provider_name="claude",
        cwd=str(tmp_path),
        mode="normal",
        pending_text=None,
        chat_id=-100500,
    )
    task = asyncio.create_task(
        window_launch_service.launch_window(query, context, request)
    )
    try:
        async with asyncio.timeout(2):
            await started.wait()
        state.arm()
        release.set()
        result = await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert result.success is False
    router = get_thread_router()
    assert router.get_window_for_chat_thread(-100500, 42) is None
    assert router.get_window_for_chat_thread(-100501, 77) == "@bound"
    (claim,) = router.iter_topic_provisionings()
    assert claim.target_id == "@1" and claim.topic_mark_owner == "owner"
    assert claim.uncertain
    adapter.close.assert_not_awaited()
    (armed,) = read_topic_mark_operations(config.state_file)
    assert armed.attempt is not None and armed.attempt.witness.dispatch_id == "A"


@pytest.mark.parametrize("consumer", ["binding", "claim"])
async def test_manual_unmark_survives_maintenance_with_existing_consumer(
    state, consumer
):
    router, store = get_thread_router(), get_window_store()
    lifecycle_state.record_topic_mark_owner("@1", "owner")
    if consumer == "binding":
        router.bind_thread(100, 42, "@1", chat_id=-100500)
    else:
        lifecycle_state.record_topic_detached("@1", "owner")
        router.begin_topic_provisioning(
            100,
            -100500,
            thread_id=42,
            target_id="@1",
            kind="target_for_topic",
            topic_mark_owner="owner",
        )
    state.manager.flush_state()
    selected = {"marked": True, "retired": False}
    writes = []

    def witness(dispatch_id):
        value = make_witness(dispatch_id)
        return (
            replace(
                value, native_pid=124, native_start="darwin:101:200", socket_ino=457
            )
            if selected["retired"]
            else value
        )

    async def write(_target, marked, *, owner, write_guard):
        assert owner == "owner"
        dispatch_id = "A" if not writes else "B"
        write_guard.arm(witness(dispatch_id))
        writes.append(marked)
        if dispatch_id == "A":
            raise OSError("ambiguous native OFF before replacement")
        (checkpoint,) = read_topic_mark_operations(config.state_file)
        assert checkpoint.attempt is not None
        assert checkpoint.attempt.witness.dispatch_id == "B"
        assert checkpoint.detach_clear is False
        selected["marked"] = marked
        write_guard.terminal(dispatch_id)
        return True

    async def retired(saved):
        assert saved.dispatch_id == "A"
        selected["retired"] = True
        return True

    backend = SimpleNamespace(
        capabilities=SimpleNamespace(name="tmux"),
        topic_mark_endpoint_scope=AsyncMock(
            side_effect=lambda: witness("scope").endpoint_scope
        ),
        set_topic_mark=AsyncMock(side_effect=write),
        recover_topic_mark_write=AsyncMock(side_effect=retired),
    )
    with StateWriterLease(config.state_file):
        with pytest.raises(OSError, match="ambiguous"):
            await topic_selection.set_topic_selection(
                _window(),
                False,
                backend=cast(Multiplexer, backend),
                store=store,
                flush=state.manager.flush_state,
                endpoint_scope=await backend.topic_mark_endpoint_scope(),
            )
        (armed,) = read_topic_mark_operations(config.state_file)
        assert armed.attempt is not None
        await topic_selection.recover_topic_selections(mux=cast(Multiplexer, backend))

    assert selected["marked"] is False
    assert writes == [False, False]
    assert read_topic_mark_operations(config.state_file) == ()
    assert armed.detach_clear is False
    if consumer == "binding":
        assert router.get_window_for_chat_thread(-100500, 42) == "@1"
    else:
        (claim,) = router.iter_topic_provisionings()
        assert claim.target_id == "@1" and claim.topic_mark_owner == "owner"
        assert lifecycle_state.is_topic_detached("@1")
