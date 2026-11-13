from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ccgram import topic_selection
from ccgram.thread_router import ThreadRouter
from ccgram.window_state_ports import legacy_state, lifecycle_state
from ccgram.window_state_store import WindowStateStore


@pytest.fixture
def selection(monkeypatch):
    store = WindowStateStore(
        schedule_save=lambda: None,
        on_hookless_provider_switch=lambda _wid: None,
    )
    router = ThreadRouter(schedule_save=lambda: None, has_window_state=store.has_window)
    saved = []
    monkeypatch.setattr(lifecycle_state, "window_store", store)
    monkeypatch.setattr(legacy_state, "window_store", store)
    monkeypatch.setattr(
        "ccgram.session.session_manager",
        SimpleNamespace(
            flush_state=lambda: saved.append((store.to_dict(), router.to_dict()))
        ),
    )
    monkeypatch.setattr(
        "ccgram.config.config", SimpleNamespace(topic_discovery_mode="marked")
    )
    mux = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(window_id="@1", topic_markable=True)
        ),
        set_topic_mark=AsyncMock(return_value=True),
    )
    return SimpleNamespace(store=store, router=router, saved=saved, mux=mux)


@pytest.mark.parametrize("marked", [False, True])
async def test_native_mark_operation_is_verified_and_does_not_touch_lifecycle(
    selection, marked
):
    assert await topic_selection.set_topic_selection(
        "@1", marked, backend=selection.mux
    )
    selection.mux.set_topic_mark.assert_awaited_once_with("@1", marked)
    assert selection.store.to_dict() == {}
    assert selection.saved == []


@pytest.mark.parametrize("window", [None, SimpleNamespace(topic_markable=False)])
async def test_native_mark_rejects_absent_or_unmarkable_target(selection, window):
    selection.mux.find_window_by_id.return_value = window
    assert not await topic_selection.set_topic_selection(
        "@1", True, backend=selection.mux
    )
    selection.mux.set_topic_mark.assert_not_awaited()


@pytest.mark.parametrize("result", [False, None])
async def test_explicit_selection_preserves_unfinished_mark_on_failure(
    selection, result
):
    selection.mux.set_topic_mark.return_value = result
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    assert lifecycle_state.get_pending_topic_mark("@1") is True
    assert not lifecycle_state.is_topic_detached("@1")
    assert selection.saved[0][0]["@1"]["topic_mark_pending"] is True


async def test_explicit_selection_and_detach_of_split_never_mutate_primary(selection):
    selection.mux.find_window_by_id.return_value = SimpleNamespace(
        window_id="primary:split", topic_markable=False
    )
    assert await topic_selection.select_topic_target("primary:split", mux=selection.mux)
    selection.router.bind_thread(1, 42, "primary:split", chat_id=-100)
    assert await topic_selection.detach_topic_target(
        1, 42, "primary:split", chat_id=-100, router=selection.router, mux=selection.mux
    )
    selection.mux.set_topic_mark.assert_not_awaited()
    assert lifecycle_state.is_topic_detached("primary:split")
    assert not lifecycle_state.is_topic_detach_pending("primary:split")


async def test_last_binding_detach_persists_before_native_clear(selection):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.store.set_window_origin("@1", "ccgram_created")

    async def clear_mark(window_id, marked):
        assert window_id == "@1"
        assert marked is False
        assert selection.router.get_window_for_thread(1, 42, -100) is None
        assert selection.saved[-1][0]["@1"]["topic_mark_pending"] is False
        assert selection.saved[-1][0]["@1"]["topic_detached"] is True
        return True

    selection.mux.set_topic_mark.side_effect = clear_mark
    assert await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    assert not lifecycle_state.is_topic_detach_pending("@1")
    assert lifecycle_state.is_topic_detached("@1")
    assert lifecycle_state.get_origin("@1") == "ccgram_created"


@pytest.mark.parametrize("other_user,other_chat", [(1, -200), (2, -100)])
async def test_remaining_binding_keeps_native_mark(selection, other_user, other_chat):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.router.bind_thread(other_user, 43, "@1", chat_id=other_chat)
    assert await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    selection.mux.set_topic_mark.assert_not_awaited()
    assert not lifecycle_state.is_topic_detached("@1")
    assert selection.router.get_window_for_thread(other_user, 43, other_chat) == "@1"


async def test_clear_failure_survives_restart_then_retries_and_allows_reselection(
    selection,
):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.mux.set_topic_mark.return_value = False
    assert not await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    snapshot = selection.store.to_dict()
    selection.store.reset()
    selection.store.from_dict(snapshot)
    assert lifecycle_state.is_topic_detach_pending("@1")
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    selection.mux.set_topic_mark.return_value = True
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert not lifecycle_state.is_topic_detach_pending("@1")
    assert lifecycle_state.is_topic_detached("@1")
    assert await topic_selection.select_topic_target("@1", mux=selection.mux)
    assert not lifecycle_state.is_topic_detached("@1")
    assert lifecycle_state.get_pending_topic_mark("@1") is None


@pytest.mark.parametrize("failure", [asyncio.CancelledError, OSError])
async def test_interrupted_clear_retains_durable_operation(selection, failure):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.mux.set_topic_mark.side_effect = failure
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await topic_selection.detach_topic_target(
                1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
            )
    else:
        assert not await topic_selection.detach_topic_target(
            1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
        )
    assert lifecycle_state.is_topic_detach_pending("@1")
    assert lifecycle_state.is_topic_detached("@1")
    assert selection.router.get_window_for_thread(1, 42, -100) is None
    selection.mux.set_topic_mark.side_effect = None
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert not lifecycle_state.is_topic_detach_pending("@1")


async def test_auto_mode_detach_preserves_old_lifecycle(selection, monkeypatch):
    monkeypatch.setattr(
        "ccgram.config.config", SimpleNamespace(topic_discovery_mode="auto")
    )
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.store.set_window_origin("@1", "ccgram_created")
    assert await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    assert not lifecycle_state.is_topic_detached("@1")
    selection.mux.find_window_by_id.assert_not_awaited()
    selection.mux.set_topic_mark.assert_not_awaited()


async def test_stale_detach_never_removes_replacement_binding(selection):
    selection.router.bind_thread(1, 42, "@2", chat_id=-100)
    assert await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    assert selection.router.get_window_for_thread(1, 42, -100) == "@2"
    selection.mux.set_topic_mark.assert_not_awaited()


async def test_settlement_save_failure_keeps_clear_retry_in_memory(
    selection, monkeypatch
):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    calls = 0

    def save():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")

    monkeypatch.setattr(
        "ccgram.session.session_manager", SimpleNamespace(flush_state=save)
    )
    with pytest.raises(OSError, match="disk full"):
        await topic_selection.detach_topic_target(
            1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
        )
    assert lifecycle_state.is_topic_detach_pending("@1")
    assert lifecycle_state.is_topic_detached("@1")


async def test_detach_persists_intent_before_waiting_for_inflight_mark(
    selection, monkeypatch
):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    mark_started = asyncio.Event()
    finish_mark = asyncio.Event()
    detach_persisted = asyncio.Event()

    def flush():
        if lifecycle_state.is_topic_detach_pending("@1"):
            detach_persisted.set()

    async def native_mark(_window_id, desired):
        if desired:
            mark_started.set()
            await finish_mark.wait()
        return True

    monkeypatch.setattr(
        "ccgram.session.session_manager", SimpleNamespace(flush_state=flush)
    )
    selection.mux.set_topic_mark.side_effect = native_mark
    selected = asyncio.create_task(
        topic_selection.select_topic_target("@1", mux=selection.mux)
    )
    await mark_started.wait()
    detached = asyncio.create_task(
        topic_selection.detach_topic_target(
            1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
        )
    )
    try:
        async with asyncio.timeout(1):
            await detach_persisted.wait()
        assert selection.router.get_window_for_thread(1, 42, -100) is None
        assert lifecycle_state.is_topic_detach_pending("@1")
    finally:
        finish_mark.set()
        results = await asyncio.gather(selected, detached, return_exceptions=True)
    assert results == [False, True]
    assert lifecycle_state.is_topic_detached("@1")
    assert not lifecycle_state.is_topic_detach_pending("@1")


async def test_concurrent_explicit_binds_keep_mark_until_the_last_detach(selection):
    async def bind(user_id, thread_id, chat_id):
        assert await topic_selection.select_topic_target("@1", mux=selection.mux)
        selection.router.bind_thread(user_id, thread_id, "@1", chat_id=chat_id)

    await asyncio.gather(bind(1, 42, -100), bind(2, 43, -200))
    assert selection.router.get_window_for_thread(1, 42, -100) == "@1"
    assert selection.router.get_window_for_thread(2, 43, -200) == "@1"
    selection.mux.set_topic_mark.reset_mock()
    await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    selection.mux.set_topic_mark.assert_not_awaited()
    await topic_selection.detach_topic_target(
        2, 43, "@1", chat_id=-200, router=selection.router, mux=selection.mux
    )
    selection.mux.set_topic_mark.assert_awaited_once_with("@1", False)


async def test_failed_mark_on_recovers_after_restart_without_detaching_topics(
    selection,
):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.mux.set_topic_mark.return_value = False
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    snapshot = selection.store.to_dict()
    selection.store.reset()
    selection.store.from_dict(snapshot)
    selection.mux.set_topic_mark.return_value = True
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert selection.router.get_window_for_thread(1, 42, -100) == "@1"
    assert lifecycle_state.get_pending_topic_mark("@1") is None
    assert not lifecycle_state.is_topic_detached("@1")
    selection.mux.set_topic_mark.assert_awaited_with("@1", True)


async def test_double_detach_does_not_queue_a_second_marker_write(selection):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    await asyncio.gather(
        *(
            topic_selection.detach_topic_target(
                1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
            )
            for _ in range(2)
        )
    )
    selection.mux.set_topic_mark.assert_awaited_once_with("@1", False)
