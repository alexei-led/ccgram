from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram import topic_selection
from ccgram.handlers import cleanup
from ccgram.handlers.polling.polling_state import terminal_poll_state
from ccgram.handlers.topics import topic_lifecycle
from ccgram.thread_router import ThreadRouter
from ccgram.window_state_ports import legacy_state, lifecycle_state
from ccgram.window_state_store import WindowStateStore


@pytest.fixture
def detached_env(monkeypatch):
    store = WindowStateStore(
        schedule_save=lambda: None,
        on_hookless_provider_switch=lambda _wid: None,
    )
    store.set_window_origin("@1", "ccgram_created")
    router = ThreadRouter(schedule_save=lambda: None, has_window_state=store.has_window)
    router.bind_thread(1, 42, "@1", chat_id=-100)
    mode = SimpleNamespace(
        topic_discovery_mode="marked",
        is_user_allowed=lambda _user: True,
        unbound_window_ttl_minutes=1,
    )
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(window_id="@1", topic_markable=True)
        ),
        set_topic_mark=AsyncMock(return_value=False),
    )
    monkeypatch.setattr(lifecycle_state, "window_store", store)
    monkeypatch.setattr(legacy_state, "window_store", store)
    monkeypatch.setattr("ccgram.config.config", mode)
    monkeypatch.setattr("ccgram.thread_router.thread_router", router)
    monkeypatch.setattr("ccgram.multiplexer.multiplexer", native)
    monkeypatch.setattr(
        "ccgram.session.session_manager", SimpleNamespace(flush_state=lambda: None)
    )
    clear = AsyncMock()
    for module in (cleanup, topic_lifecycle):
        monkeypatch.setattr(module, "config", mode)
        monkeypatch.setattr(module, "thread_router", router)
        monkeypatch.setattr(module, "clear_topic_state", clear)
    monkeypatch.setattr(cleanup, "enqueue_status_update", AsyncMock())
    reply = AsyncMock()
    monkeypatch.setattr(cleanup, "safe_reply", reply)
    update = MagicMock()
    update.effective_user.id = 1
    update.effective_chat.id = -100
    update.message.message_thread_id = 42
    terminal_poll_state.clear_state("@1")
    yield SimpleNamespace(
        store=store,
        router=router,
        native=native,
        mode=mode,
        update=update,
        context=MagicMock(),
        clear=clear,
        reply=reply,
    )
    terminal_poll_state.clear_state("@1")


@pytest.mark.parametrize(
    "operation", [cleanup.unbind_command, topic_lifecycle.topic_closed_handler]
)
async def test_intentional_last_detach_keeps_terminal_and_retries_native_clear(
    detached_env, operation
):
    env = detached_env
    await operation(env.update, env.context)
    assert env.router.get_window_for_thread(1, 42, -100) is None
    assert lifecycle_state.is_topic_detached("@1")
    assert lifecycle_state.is_topic_detach_pending("@1")
    assert lifecycle_state.get_origin("@1") == "ccgram_created"
    env.native.set_topic_mark.assert_awaited_once_with("@1", False)
    assert env.clear.call_args.kwargs["window_dead"] is False
    if operation is cleanup.unbind_command:
        assert "pending" in env.reply.call_args.args[1]
    env.native.set_topic_mark.return_value = True
    await topic_selection.recover_topic_selections(mux=env.native)
    assert lifecycle_state.is_topic_detached("@1")
    assert not lifecycle_state.is_topic_detach_pending("@1")


@pytest.mark.parametrize(
    "operation", [cleanup.unbind_command, topic_lifecycle.topic_closed_handler]
)
async def test_cleanup_cancellation_does_not_lose_detach_intent(
    detached_env, operation
):
    env = detached_env
    env.clear.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await operation(env.update, env.context)
    assert env.router.get_window_for_thread(1, 42, -100) is None
    assert lifecycle_state.is_topic_detach_pending("@1")


@pytest.mark.parametrize(
    "mode,detached,expected_kill",
    [
        ("marked", True, False),
        ("marked", False, True),
        ("auto", True, True),
        ("auto", False, True),
    ],
)
async def test_detached_ttl_exemption_is_marked_only_and_never_for_failed_creates(
    detached_env, monkeypatch, mode, detached, expected_kill
):
    env = detached_env
    env.router.unbind_thread(1, 42, -100)
    env.mode.topic_discovery_mode = mode
    env.store.set_topic_selection("@1", detached=detached, pending=None)
    terminal_poll_state.set_unbound_timer("@1", time.monotonic() - 100)
    monkeypatch.setattr(
        topic_lifecycle.window_query,
        "view_window",
        lambda _wid: SimpleNamespace(origin="ccgram_created"),
    )
    adapter = MagicMock(close=AsyncMock(return_value=(True, "closed")))
    window = MagicMock(window_id="@1", topic_marked=False)
    await topic_lifecycle.check_unbound_window_ttl([window], adapter=adapter)
    assert adapter.close.await_count == int(expected_kill)
    if not expected_kill:
        assert terminal_poll_state.get_state("@1").unbound_timer is None
