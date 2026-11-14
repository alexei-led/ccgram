from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ccgram import topic_selection
from ccgram.multiplexer.base import WindowRef
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
    monkeypatch.setattr("ccgram.thread_router.thread_router", router)

    def snapshot():
        return {"window_states": store.to_dict(), **store.topic_mark_journal_to_dict()}

    def restore(value):
        store.reset()
        store.from_dict(value["window_states"])
        store.load_topic_mark_journal(value)

    monkeypatch.setattr(
        "ccgram.session.session_manager",
        SimpleNamespace(
            flush_state=lambda: saved.append((snapshot(), router.to_dict()))
        ),
    )
    monkeypatch.setattr(
        "ccgram.config.config", SimpleNamespace(topic_discovery_mode="marked")
    )
    mux = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(
                window_id="@1",
                topic_markable=True,
                topic_mark_owner="owner:" + "@1",
                topic_marked=True,
            )
        ),
        set_topic_mark=AsyncMock(return_value=True),
    )
    return SimpleNamespace(
        store=store,
        router=router,
        saved=saved,
        mux=mux,
        snapshot=snapshot,
        restore=restore,
    )


@pytest.mark.parametrize("marked", [False, True])
async def test_native_mark_operation_is_verified_and_does_not_touch_lifecycle(
    selection, marked
):
    captured = WindowRef(
        "@1", "captured", "/tmp", topic_markable=True, topic_mark_owner="owner:@1"
    )
    selection.mux.find_window_by_id.return_value = WindowRef(
        "@1", "replacement", "/tmp", topic_mark_owner="replacement-owner"
    )
    assert await topic_selection.set_topic_selection(
        captured, marked, backend=selection.mux
    )
    selection.mux.find_window_by_id.assert_not_awaited()
    selection.mux.set_topic_mark.assert_awaited_once_with(
        "@1", marked, owner="owner:" + "@1"
    )
    assert selection.store.to_dict() == {}
    assert selection.saved == []


@pytest.mark.parametrize(
    "window",
    [
        WindowRef("split", "split", "/tmp", topic_markable=False),
        WindowRef("@1", "unknown-owner", "/tmp", topic_mark_owner=None),
    ],
)
async def test_native_mark_rejects_unmarkable_or_unknown_snapshot(selection, window):
    assert not await topic_selection.set_topic_selection(
        window, True, backend=selection.mux
    )
    selection.mux.find_window_by_id.assert_not_awaited()
    selection.mux.set_topic_mark.assert_not_awaited()


@pytest.mark.parametrize("result", [False, None])
async def test_explicit_selection_preserves_unfinished_mark_on_failure(
    selection, result
):
    selection.mux.set_topic_mark.return_value = result
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    assert lifecycle_state.get_pending_topic_mark("@1") is True
    assert not lifecycle_state.is_topic_detached("@1")
    assert selection.saved[0][0]["topic_mark_operations"][0]["desired"] is True


async def test_explicit_selection_and_detach_of_split_never_mutate_primary(selection):
    selection.mux.find_window_by_id.return_value = SimpleNamespace(
        window_id="primary:split",
        topic_markable=False,
        topic_mark_owner=None,
        topic_marked=True,
    )
    assert await topic_selection.select_topic_target("primary:split", mux=selection.mux)
    selection.router.bind_thread(1, 42, "primary:split", chat_id=-100)
    assert await topic_selection.detach_topic_target(
        1, 42, "primary:split", chat_id=-100, router=selection.router, mux=selection.mux
    )
    selection.mux.set_topic_mark.assert_not_awaited()
    assert lifecycle_state.is_topic_detached("primary:split")
    assert not lifecycle_state.is_topic_detach_pending("primary:split", owner=None)


async def test_last_binding_detach_persists_before_native_clear(selection):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.store.set_window_origin("@1", "ccgram_created")

    async def clear_mark(window_id, marked, *, owner):
        assert window_id == "@1"
        assert marked is False
        assert owner == "owner:@1"
        assert selection.router.get_window_for_thread(1, 42, -100) is None
        assert selection.saved[-1][0]["topic_mark_operations"][0]["desired"] is False
        assert selection.saved[-1][0]["window_states"]["@1"]["topic_detached"] is True
        return True

    selection.mux.set_topic_mark.side_effect = clear_mark
    assert await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    assert not lifecycle_state.is_topic_detach_pending("@1", owner=None)
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
    snapshot = selection.snapshot()
    selection.store.reset()
    selection.restore(snapshot)
    assert lifecycle_state.is_topic_detach_pending("@1", owner=None)
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    selection.mux.set_topic_mark.return_value = True
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert not lifecycle_state.is_topic_detach_pending("@1", owner=None)
    assert lifecycle_state.is_topic_detached("@1")
    assert await topic_selection.select_topic_target("@1", mux=selection.mux)
    assert lifecycle_state.is_topic_detached("@1")
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    assert lifecycle_state.complete_topic_admission("@1", "owner:@1")
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
    assert lifecycle_state.is_topic_detach_pending("@1", owner=None)
    assert lifecycle_state.is_topic_detached("@1")
    assert selection.router.get_window_for_thread(1, 42, -100) is None
    selection.mux.set_topic_mark.side_effect = None
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert not lifecycle_state.is_topic_detach_pending("@1", owner=None)


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
        if calls == 3:
            raise OSError("disk full")

    monkeypatch.setattr(
        "ccgram.session.session_manager", SimpleNamespace(flush_state=save)
    )
    with pytest.raises(OSError, match="disk full"):
        await topic_selection.detach_topic_target(
            1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
        )
    assert lifecycle_state.is_topic_detach_pending("@1", owner=None)
    assert lifecycle_state.is_topic_detached("@1")


async def test_detach_persists_intent_before_waiting_for_inflight_mark(
    selection, monkeypatch
):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    mark_started = asyncio.Event()
    finish_mark = asyncio.Event()
    detach_persisted = asyncio.Event()

    def flush():
        if lifecycle_state.is_topic_detach_pending("@1", owner=None):
            detach_persisted.set()

    async def native_mark(_window_id, desired, *, owner):
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
        assert lifecycle_state.is_topic_detach_pending("@1", owner=None)
    finally:
        finish_mark.set()
        results = await asyncio.gather(selected, detached, return_exceptions=True)
    assert results == [False, True]
    assert lifecycle_state.is_topic_detached("@1")
    assert not lifecycle_state.is_topic_detach_pending("@1", owner=None)


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
    selection.mux.set_topic_mark.assert_awaited_once_with(
        "@1", False, owner="owner:" + "@1"
    )


async def test_failed_mark_on_recovers_after_restart_without_detaching_topics(
    selection,
):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.mux.set_topic_mark.return_value = False
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    snapshot = selection.snapshot()
    selection.store.reset()
    selection.restore(snapshot)
    selection.mux.set_topic_mark.return_value = True
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert selection.router.get_window_for_thread(1, 42, -100) == "@1"
    assert lifecycle_state.get_pending_topic_mark("@1") is None
    assert not lifecycle_state.is_topic_detached("@1")
    selection.mux.set_topic_mark.assert_awaited_with("@1", True, owner="owner:" + "@1")


async def test_owner_is_checkpointed_before_mark_ack_loss_and_rekey(selection):
    async def mark(_wid, desired, *, owner):
        assert owner == "owner:@1" and desired is True
        assert (
            selection.saved[-1][0]["window_states"]["@1"]["topic_mark_owner"] == owner
        )
        raise OSError("acknowledgement lost")

    selection.mux.set_topic_mark.side_effect = mark
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    snapshot = selection.snapshot()
    selection.restore(snapshot)
    selection.mux.find_window_by_id.reset_mock()
    selection.mux.find_window_by_id.return_value = None
    selection.mux.set_topic_mark.side_effect = None
    await topic_selection.recover_topic_selections(mux=selection.mux)
    selection.mux.find_window_by_id.assert_not_awaited()
    selection.mux.set_topic_mark.assert_awaited_with("@1", True, owner="owner:@1")
    assert lifecycle_state.get_pending_topic_mark("@1") is None


async def test_pending_clear_survives_rekey_and_bare_shell_restart(selection):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.mux.set_topic_mark.return_value = False
    assert not await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    snapshot = selection.snapshot()
    selection.store.reset()
    selection.restore(snapshot)
    assert lifecycle_state.is_topic_detach_pending("new-conversation", owner="owner:@1")
    assert not lifecycle_state.is_topic_detach_pending(
        "independent", owner="owner:other"
    )
    selection.mux.find_window_by_id.reset_mock()
    selection.mux.find_window_by_id.return_value = None
    selection.mux.set_topic_mark.return_value = True
    await topic_selection.recover_topic_selections(mux=selection.mux)
    selection.mux.find_window_by_id.assert_not_awaited()
    selection.mux.set_topic_mark.assert_awaited_with("@1", False, owner="owner:@1")
    assert not lifecycle_state.is_topic_detach_pending(
        "new-conversation", owner="owner:@1"
    )
    assert lifecycle_state.is_topic_detached("@1")


async def test_unknown_owner_never_generates_a_candidate_or_mutates(selection):
    selection.mux.find_window_by_id.return_value = SimpleNamespace(
        window_id="@1", topic_markable=True, topic_mark_owner=None
    )
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    assert lifecycle_state.get_pending_topic_mark("@1") is True
    assert lifecycle_state.get_topic_mark_owner("@1") is None
    selection.mux.set_topic_mark.assert_not_awaited()


@pytest.mark.parametrize(
    "saved_owner,live_owner,expected_owner",
    [
        pytest.param("saved-O", None, "saved-O", id="unavailable-live-preserves-saved"),
        pytest.param(None, None, None, id="neither-known-stays-quarantined"),
        pytest.param(None, "live-P", "live-P", id="confirmed-live-is-captured"),
        pytest.param("same-O", "same-O", "same-O", id="confirmed-witnesses-agree"),
    ],
)
async def test_detach_retains_best_confirmed_owner_through_restart_and_retry(
    selection, saved_owner, live_owner, expected_owner
):
    if saved_owner is not None:
        lifecycle_state.record_topic_mark_owner("@1", saved_owner)
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.mux.find_window_by_id.return_value = WindowRef(
        "@1",
        "live",
        "/tmp",
        topic_mark_owner=live_owner,
        topic_marked=True if live_owner is not None else None,
    )
    selection.mux.set_topic_mark.return_value = False

    assert not await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    (operation,) = lifecycle_state.pending_topic_selections()
    assert operation.owner == expected_owner
    assert operation.desired is False
    selection.restore(selection.snapshot())
    assert lifecycle_state.pending_topic_selections() == [operation]

    selection.mux.set_topic_mark.reset_mock()
    selection.mux.find_window_by_id.reset_mock()
    selection.mux.find_window_by_id.return_value = WindowRef(
        "@1",
        "recovered or replaced",
        "/tmp",
        topic_mark_owner="later-P",
        topic_marked=True,
    )
    selection.mux.set_topic_mark.return_value = True
    await topic_selection.recover_topic_selections(mux=selection.mux)

    if expected_owner is None:
        selection.mux.set_topic_mark.assert_not_awaited()
        assert lifecycle_state.pending_topic_selections() == [operation]
    else:
        selection.mux.find_window_by_id.assert_not_awaited()
        selection.mux.set_topic_mark.assert_awaited_once_with(
            "@1", False, owner=expected_owner
        )
        assert not lifecycle_state.pending_topic_selections()
    assert lifecycle_state.is_topic_detached("@1")


async def test_ownerless_historical_detach_stays_unfinished_without_guessing(selection):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.mux.find_window_by_id.return_value = None
    assert not await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    assert lifecycle_state.is_topic_detach_pending("@1", owner=None)
    assert not lifecycle_state.is_topic_detach_pending("unrelated", owner="new-owner")
    assert lifecycle_state.get_topic_mark_owner("@1") is None
    selection.mux.set_topic_mark.assert_not_awaited()


async def test_late_external_owner_is_read_while_binding_still_intact(selection):
    async def current(wid):
        assert selection.router.get_window_for_thread(1, 42, -100) == "@1"
        return SimpleNamespace(
            window_id=wid, topic_markable=True, topic_mark_owner="late-owner"
        )

    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.mux.find_window_by_id.side_effect = current
    assert await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    selection.mux.set_topic_mark.assert_awaited_once_with(
        "@1", False, owner="late-owner"
    )
    assert (
        selection.saved[0][0]["window_states"]["@1"]["topic_mark_owner"] == "late-owner"
    )


async def test_pending_owner_blocks_another_routing_target_without_creating_state(
    selection,
):
    lifecycle_state.set_topic_selection(
        "@old", detached=True, pending=False, owner="owner:@1"
    )
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    assert "@1" not in selection.store.window_states
    selection.mux.set_topic_mark.assert_not_awaited()


async def test_remaining_binding_of_same_owner_under_another_route_keeps_mark(
    selection,
):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    selection.router.bind_thread(2, 43, "@new", chat_id=-200)
    assert lifecycle_state.record_topic_mark_owner("@new", "owner:@1")
    assert await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    selection.mux.set_topic_mark.assert_not_awaited()
    assert selection.router.get_window_for_thread(2, 43, -200) == "@new"


async def test_clear_never_falls_back_to_recycled_window_id(selection):
    lifecycle_state.set_topic_selection(
        "@1", detached=True, pending=False, owner="old-owner"
    )
    selection.mux.find_window_by_id.return_value = SimpleNamespace(
        window_id="@1", topic_markable=True, topic_mark_owner="replacement-owner"
    )
    await topic_selection.recover_topic_selections(mux=selection.mux)
    selection.mux.find_window_by_id.assert_not_awaited()
    selection.mux.set_topic_mark.assert_awaited_once_with(
        "@1", False, owner="old-owner"
    )


async def test_cross_route_last_detach_cancels_older_owner_mark_on_retry(
    selection, monkeypatch
):
    selection.router.bind_thread(1, 42, "@old", chat_id=-100)
    lifecycle_state.record_topic_mark_owner("@old", "same-owner")
    started = asyncio.Event()
    release = asyncio.Event()
    persisted = asyncio.Event()
    from ccgram.session import session_manager

    original_flush = session_manager.flush_state

    def flush():
        original_flush()
        if lifecycle_state.is_topic_detach_pending("@old", owner="same-owner"):
            persisted.set()

    monkeypatch.setattr(session_manager, "flush_state", flush)
    selection.mux.find_window_by_id.side_effect = lambda wid: SimpleNamespace(
        window_id=wid, topic_markable=True, topic_mark_owner="same-owner"
    )

    async def native(_wid, desired, *, owner):
        assert owner == "same-owner"
        if desired:
            started.set()
            await release.wait()
        return True

    selection.mux.set_topic_mark.side_effect = native
    marked = asyncio.create_task(
        topic_selection.select_topic_target("@new", mux=selection.mux)
    )
    await started.wait()
    detached = asyncio.create_task(
        topic_selection.detach_topic_target(
            1, 42, "@old", chat_id=-100, router=selection.router, mux=selection.mux
        )
    )
    try:
        async with asyncio.timeout(1):
            await persisted.wait()
    finally:
        release.set()
        results = await asyncio.gather(marked, detached)
    assert results == [False, True]
    assert lifecycle_state.get_pending_topic_mark("@new") is None
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert selection.mux.set_topic_mark.await_count == 2


async def test_owner_mark_success_does_not_authorize_a_vanished_routing_target(
    selection,
):
    selection.mux.find_window_by_id.side_effect = [
        SimpleNamespace(
            window_id="@1", topic_markable=True, topic_mark_owner="owner:@1"
        ),
        None,
    ]
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    selection.mux.set_topic_mark.assert_awaited_once_with("@1", True, owner="owner:@1")


@pytest.mark.parametrize("desired", [False, True])
async def test_ownerless_retry_never_claims_fresh_owner_of_reused_slot(
    selection, desired
):
    lifecycle_state.set_topic_selection(
        "@1", detached=not desired, pending=desired, owner=None
    )
    selection.mux.find_window_by_id.return_value = SimpleNamespace(
        window_id="@1",
        topic_markable=True,
        topic_mark_owner="replacement-owner",
        topic_marked=True,
    )
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert lifecycle_state.get_pending_topic_mark("@1") is desired
    assert lifecycle_state.get_topic_mark_owner("@1") is None
    selection.mux.set_topic_mark.assert_not_awaited()


@pytest.mark.parametrize("native_result", [False, True])
async def test_selection_preparation_preserves_detached_ttl_until_actual_binding(
    selection, native_result
):
    lifecycle_state.set_topic_selection(
        "@1", detached=True, pending=None, owner="owner:@1"
    )
    selection.mux.set_topic_mark.return_value = native_result
    assert (
        await topic_selection.select_topic_target("@1", mux=selection.mux)
        is native_result
    )
    assert lifecycle_state.is_topic_detached("@1")


async def test_same_owner_provisioning_claim_is_a_remaining_consumer(selection):
    selection.router.bind_thread(1, 42, "@1", chat_id=-100)
    lifecycle_state.record_topic_mark_owner("@new", "owner:@1")
    selection.router.begin_topic_provisioning(
        2, -200, thread_id=43, target_id="@new", kind="target_for_topic"
    )
    assert await topic_selection.detach_topic_target(
        1, 42, "@1", chat_id=-100, router=selection.router, mux=selection.mux
    )
    selection.mux.set_topic_mark.assert_not_awaited()
    assert lifecycle_state.is_topic_detached("@1")


async def test_old_owner_clear_completion_never_changes_reused_route_state(selection):
    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner="O")
    assert lifecycle_state.record_topic_mark_owner("@1", "P")
    selection.router.bind_thread(2, 43, "@1", chat_id=-200)
    assert lifecycle_state.complete_topic_admission("@1", "P")
    before = selection.store.window_states["@1"].to_dict()
    await topic_selection.recover_topic_selections(mux=selection.mux)
    selection.mux.find_window_by_id.assert_not_awaited()
    selection.mux.set_topic_mark.assert_awaited_once_with("@1", False, owner="O")
    assert selection.store.window_states["@1"].to_dict() == before
    assert selection.router.get_window_for_thread(2, 43, -200) == "@1"
    assert not lifecycle_state.pending_topic_selections()


async def test_old_owner_settlement_flush_failure_preserves_new_owner_and_both_ops(
    selection, monkeypatch
):
    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner="O")
    lifecycle_state.record_topic_mark_owner("@1", "P")
    lifecycle_state.queue_topic_mark("@1", "P", True)
    before = selection.store.window_states["@1"].to_dict()

    def fail_flush():
        raise OSError("disk full")

    monkeypatch.setattr(
        "ccgram.session.session_manager", SimpleNamespace(flush_state=fail_flush)
    )
    with pytest.raises(OSError, match="disk full"):
        await topic_selection.recover_topic_selections(
            mux=selection.mux,
            operation=next(
                op
                for op in lifecycle_state.pending_topic_selections()
                if op.owner == "O"
            ),
        )
    assert selection.store.window_states["@1"].to_dict() == before
    assert {
        op.owner: op.desired for op in lifecycle_state.pending_topic_selections()
    } == {"O": False, "P": True}


@pytest.mark.parametrize("consumer", ["binding", "claim"])
async def test_clear_rechecks_consumers_after_waiting_for_owner_lock(
    selection, monkeypatch, consumer
):
    from ccgram.session import session_manager

    selection.router.bind_thread(1, 42, "@old", chat_id=-100)
    lifecycle_state.record_topic_mark_owner("@old", "shared")
    started, release, persisted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_flush = session_manager.flush_state

    def flush():
        original_flush()
        if lifecycle_state.is_topic_detach_pending("@old", owner="shared"):
            persisted.set()

    async def native(_wid, desired, *, owner):
        assert owner == "shared"
        if desired:
            started.set()
            await release.wait()
        return True

    monkeypatch.setattr(session_manager, "flush_state", flush)
    selection.mux.find_window_by_id.side_effect = lambda wid: SimpleNamespace(
        window_id=wid, topic_markable=True, topic_mark_owner="shared", topic_marked=True
    )
    selection.mux.set_topic_mark.side_effect = native
    marked = asyncio.create_task(
        topic_selection.select_topic_target("@preparing", mux=selection.mux)
    )
    await started.wait()
    detached = asyncio.create_task(
        topic_selection.detach_topic_target(
            1, 42, "@old", chat_id=-100, router=selection.router, mux=selection.mux
        )
    )
    try:
        async with asyncio.timeout(1):
            await persisted.wait()
        lifecycle_state.set_topic_selection(
            "@consumer", detached=False, pending=None, owner="shared"
        )
        if consumer == "binding":
            selection.router.bind_thread(2, 43, "@consumer", chat_id=-200)
        else:
            selection.router.begin_topic_provisioning(
                2, -200, thread_id=43, target_id="@consumer", kind="target_for_topic"
            )
    finally:
        release.set()
        results = await asyncio.gather(marked, detached)
    assert results == [False, True]
    assert selection.mux.set_topic_mark.await_count == 1
    assert not lifecycle_state.is_topic_detach_pending("@old", owner="shared")
    assert lifecycle_state.is_topic_detached("@old")


async def test_marker_owner_is_never_used_as_a_routing_alias(selection):
    selection.mux.find_window_by_id.side_effect = [
        WindowRef("@1", "original", "/tmp", topic_mark_owner="O"),
        WindowRef(
            "@different",
            "same native owner",
            "/tmp",
            topic_marked=True,
            topic_mark_owner="O",
        ),
    ]
    assert not await topic_selection.select_topic_target("@1", mux=selection.mux)
    selection.mux.set_topic_mark.assert_awaited_once_with("@1", True, owner="O")


async def test_retry_requires_durable_intent_before_native_mutation(
    selection, monkeypatch
):
    lifecycle_state.set_topic_selection("@1", detached=True, pending=False, owner="O")

    def fail_flush():
        raise OSError("cannot persist intent")

    monkeypatch.setattr(
        "ccgram.session.session_manager", SimpleNamespace(flush_state=fail_flush)
    )
    with pytest.raises(OSError, match="cannot persist intent"):
        await topic_selection.recover_topic_selections(mux=selection.mux)
    selection.mux.set_topic_mark.assert_not_awaited()
    assert lifecycle_state.is_topic_detach_pending("@1", owner="O")


async def test_unknown_operation_cannot_settle_through_a_split_routing_alias(selection):
    lifecycle_state.set_topic_selection(
        "old-route", detached=True, pending=False, owner=None
    )
    selection.mux.find_window_by_id.return_value = WindowRef(
        "different-split", "split", "/tmp", topic_markable=False
    )
    await topic_selection.recover_topic_selections(mux=selection.mux)
    assert lifecycle_state.is_topic_detach_pending("old-route", owner=None)
    selection.mux.set_topic_mark.assert_not_awaited()


async def test_owned_mark_on_retry_waits_for_conflicting_route_quarantine(selection):
    lifecycle_state.queue_topic_mark("@1", "O", True)
    lifecycle_state.queue_topic_mark("@1", None, False)
    await topic_selection.recover_topic_selections(mux=selection.mux)
    selection.mux.set_topic_mark.assert_not_awaited()
    assert len(lifecycle_state.pending_topic_selections()) == 2


@pytest.mark.parametrize(
    "expected_owner,current_owner,marked,require_marked,allowed",
    [
        pytest.param("O", None, None, False, False, id="known-witness-lost-auto"),
        pytest.param("O", "Q", False, False, False, id="known-witness-changed-auto"),
        pytest.param("O", "O", False, False, True, id="stable-witness-unmarked-auto"),
        pytest.param(None, None, None, False, True, id="initially-ownerless-auto"),
        pytest.param(
            None, "P", False, False, True, id="initially-ownerless-current-known"
        ),
        pytest.param("O", "O", False, True, False, id="marked-eligibility-independent"),
        pytest.param("O", "O", True, True, True, id="stable-selected-marked"),
    ],
)
async def test_current_target_requires_retention_of_a_captured_witness(
    selection, expected_owner, current_owner, marked, require_marked, allowed
):
    snapshot = WindowRef(
        "@1",
        "same routing target",
        "/tmp",
        topic_mark_owner=current_owner,
        topic_marked=marked,
    )
    selection.mux.find_window_by_id.return_value = snapshot
    current = await topic_selection.verify_topic_target(
        "@1", owner=expected_owner, require_marked=require_marked, mux=selection.mux
    )
    assert (current is snapshot) is allowed
    selection.mux.set_topic_mark.assert_not_awaited()


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
    selection.mux.set_topic_mark.assert_awaited_once_with(
        "@1", False, owner="owner:" + "@1"
    )
