from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.handlers import topic_pause
from ccgram.multiplexer.base import WindowRef
from ccgram.telegram_client import FakeTelegramClient
from ccgram.thread_router import ThreadRouter
from ccgram.session_protocol.telegram import TelegramSessionAdapter
from ccgram.window_state_ports import lifecycle_state
from ccgram.window_state_store import WindowStateStore


@pytest.fixture
def env(monkeypatch):
    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _: None
    )
    monkeypatch.setattr(lifecycle_state, "window_store", store)
    monkeypatch.setattr(
        topic_pause, "resolve_session_for_window", AsyncMock(return_value=None)
    )
    router = ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: True)
    router.bind_thread(1, 10, "session", chat_id=-100)
    monkeypatch.setattr(topic_pause, "thread_router", router)
    window = WindowRef(
        "session", "work", "/tmp", topic_mark_owner="owner", topic_marked=False
    )
    backend = SimpleNamespace(find_window_by_id=AsyncMock(return_value=window))
    monkeypatch.setattr(topic_pause, "enabled", lambda: True, raising=False)
    return SimpleNamespace(
        router=router,
        window=window,
        backend=backend,
        client=FakeTelegramClient(),
        flush=lambda: None,
    )


async def reconcile(env, window=None):
    window = window or env.window
    env.backend.find_window_by_id.return_value = window
    await topic_pause.reconcile_topic_pauses(
        env.client, [window], router=env.router, backend=env.backend, flush=env.flush
    )


async def test_unflag_closes_without_losing_binding_and_reflag_reopens(env):
    await reconcile(env)
    assert env.router.get_window_for_thread(1, 10, -100) == "session"
    assert env.router.get_topic_pause(1, 10, -100).paused
    assert env.client.last_call("close_forum_topic")
    await reconcile(env, replace(env.window, topic_marked=True))
    assert not env.router.get_topic_pause(1, 10, -100).paused
    assert env.client.last_call("reopen_forum_topic")


@pytest.mark.parametrize(
    "marked,markable,owner",
    [(None, True, "owner"), (False, False, "owner"), (False, True, None)],
)
async def test_unknown_metadata_and_splits_are_untouched(env, marked, markable, owner):
    await reconcile(
        env,
        replace(
            env.window,
            topic_marked=marked,
            topic_markable=markable,
            topic_mark_owner=owner,
        ),
    )
    assert env.router.get_topic_pause(1, 10, -100) is None
    assert not env.client.calls


async def test_paused_binding_survives_restart(env):
    await reconcile(env)
    restored = ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: True)
    restored.from_dict(env.router.to_dict())
    assert (pause := restored.get_topic_pause(1, 10, -100)) is not None
    assert pause.paused
    assert restored.get_window_for_thread(1, 10, -100) == "session"


async def test_resume_frontier_excludes_paused_records_even_in_mixed_batches(
    env, tmp_path, monkeypatch
):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("paused output\n")
    monkeypatch.setattr(
        topic_pause,
        "resolve_session_for_window",
        AsyncMock(
            return_value=SimpleNamespace(
                session_id="conversation", file_path=transcript
            )
        ),
    )
    await reconcile(env)
    await reconcile(env, replace(env.window, topic_marked=True))
    frontier = transcript.stat().st_size
    assert not topic_pause.accepts_message(1, 10, -100, "conversation", frontier - 1)
    assert topic_pause.accepts_message(1, 10, -100, "conversation", frontier)
    assert topic_pause.accepts_message(1, 10, -100, "new-conversation", 0)
    restored = ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: True)
    restored.from_dict(env.router.to_dict())
    monkeypatch.setattr(topic_pause, "thread_router", restored)
    assert not topic_pause.accepts_message(1, 10, -100, "conversation", 0)


async def test_resume_failure_keeps_delivery_fenced(env):
    from telegram.error import Forbidden

    await reconcile(env)
    env.client.set_side_effect("reopen_forum_topic", [Forbidden("no permission")])
    await reconcile(env, replace(env.window, topic_marked=True))
    assert env.router.get_topic_pause(1, 10, -100).paused
    assert not topic_pause.accepts_message(1, 10, -100, "conversation", 99)
    assert not env.router.iter_topic_provisionings()


async def test_pause_is_durable_before_close_request(env):
    snapshots = []
    env.flush = lambda: snapshots.append(env.router.to_dict())

    def close(**kwargs):
        assert snapshots[-1]["topic_pauses"][0]["paused"]
        return True

    env.client.returns["close_forum_topic"] = close
    await reconcile(env)


async def test_close_is_idempotent(env):
    await reconcile(env)
    await reconcile(env)
    assert env.client.call_count("close_forum_topic") == 1


async def test_missing_topic_prepares_replacement_without_closing_terminal(env):
    from telegram.error import BadRequest

    await reconcile(env)
    env.client.set_side_effect("reopen_forum_topic", [BadRequest("TOPIC_ID_INVALID")])
    await reconcile(env, replace(env.window, topic_marked=True))
    (claim,) = env.router.iter_topic_provisionings()
    assert claim.kind == "topic_for_target"
    assert claim.topic_mark_owner == "owner"
    assert claim.retry_thread_id == 10
    assert env.router.get_topic_pause(1, 10, -100).missing
    assert not env.client.last_call("delete_forum_topic")
    restored = ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: True)
    restored.from_dict(env.router.to_dict())
    assert (pause := restored.get_topic_pause(1, 10, -100)) is not None
    assert pause.missing
    restored.attach_provisioning_topic(claim.claim_id, 11)
    assert restored.commit_topic_provisioning(claim.claim_id)
    assert (pause := restored.get_topic_pause(1, 11, -100)) is not None
    assert pause.paused
    assert restored.get_topic_pause(1, 10, -100) is None


async def test_deleted_paused_topic_stays_missing_until_reflag(env):
    await reconcile(env)
    candidates = topic_pause.cleanup_candidates(1, -100, env.router)
    result = await topic_pause.delete_paused_topics(
        env.client, candidates, router=env.router, backend=env.backend, flush=env.flush
    )
    assert result == {"deleted": 1, "protected": 0, "failed": 0}
    assert env.router.get_topic_pause(1, 10, -100).missing
    assert env.router.get_window_for_thread(1, 10, -100) == "session"
    await reconcile(env)
    assert not env.router.iter_topic_provisionings()
    await reconcile(env, replace(env.window, topic_marked=True))
    assert env.router.iter_topic_provisionings()


@pytest.mark.parametrize(
    "change", ["reflag", "unknown", "owner", "rebind", "resume", "provisioning"]
)
async def test_stale_cleanup_protects_changed_candidate(env, change):
    await reconcile(env)
    candidates = topic_pause.cleanup_candidates(1, -100, env.router)
    if change == "reflag":
        env.backend.find_window_by_id.return_value = replace(
            env.window, topic_marked=True
        )
    elif change == "unknown":
        env.backend.find_window_by_id.return_value = None
    elif change == "owner":
        env.backend.find_window_by_id.return_value = replace(
            env.window, topic_mark_owner="replacement"
        )
    elif change == "rebind":
        env.router.bind_thread(1, 10, "replacement", chat_id=-100)
    elif change == "resume":
        await reconcile(env, replace(env.window, topic_marked=True))
        await reconcile(env)
    else:
        env.router.begin_topic_provisioning(
            1, -100, target_id="session", kind="topic_for_target"
        )
    result = await topic_pause.delete_paused_topics(
        env.client, candidates, router=env.router, backend=env.backend, flush=env.flush
    )
    assert result["protected"] == 1
    assert env.client.last_call("delete_forum_topic") is None


async def test_cleanup_claim_blocks_rebinding_during_delete(env):
    await reconcile(env)
    candidates = topic_pause.cleanup_candidates(1, -100, env.router)

    def deleting(**kwargs):
        with pytest.raises(ValueError, match="deletion is in progress"):
            env.router.bind_thread(1, 10, "another", chat_id=-100)
        return True

    env.client.returns["delete_forum_topic"] = deleting
    result = await topic_pause.delete_paused_topics(
        env.client, candidates, router=env.router, backend=env.backend, flush=env.flush
    )
    assert result["deleted"] == 1
    env.router.bind_thread(1, 10, "another", chat_id=-100)


async def test_failed_cleanup_is_not_retired_for_unguarded_retries(env):
    from telegram.error import Forbidden

    await reconcile(env)
    candidates = topic_pause.cleanup_candidates(1, -100, env.router)
    env.client.set_side_effect(
        "delete_forum_topic", [Forbidden("no delete permission")]
    )
    result = await topic_pause.delete_paused_topics(
        env.client, candidates, router=env.router, backend=env.backend, flush=env.flush
    )
    assert result["failed"] == 1
    assert not env.router.get_topic_pause(1, 10, -100).missing
    assert not list(env.router.iter_retired_topics())


async def test_auto_mode_does_not_pause_unflagged_binding(env, monkeypatch):
    monkeypatch.setattr(topic_pause, "enabled", lambda: False)
    await reconcile(env)
    assert not env.router.topic_pauses
    assert not env.client.calls


async def test_late_own_close_never_clears_native_flag_or_unbinds(env, monkeypatch):
    monkeypatch.setattr(topic_pause, "_flush", env.flush)
    await reconcile(env)
    await reconcile(env, replace(env.window, topic_marked=True))
    assert topic_pause.note_closed(1, -100, 10)
    await reconcile(env, replace(env.window, topic_marked=True))
    assert env.router.get_window_for_thread(1, 10, -100) == "session"
    assert not env.router.get_topic_pause(1, 10, -100).paused


async def test_queue_generation_drops_old_work_and_settles_receipts(env):
    import asyncio
    from ccgram.delivery_contract import DeliveryOutcome, new_delivery_receipt
    from ccgram.handlers.messaging_pipeline import message_queue
    from ccgram.handlers.messaging_pipeline.message_task import ContentTask

    task = ContentTask(window_id="session", parts=("old",), thread_id=10, chat_id=-100)
    await reconcile(env)
    await reconcile(env, replace(env.window, topic_marked=True))
    result = await message_queue._dispatch_with_retry(
        env.client,
        1,
        message_queue.RetryDispatchState(task),
        asyncio.Queue(),
        asyncio.Lock(),
    )
    assert result is DeliveryOutcome.INTENTIONALLY_DROPPED
    receipt = new_delivery_receipt(checkpoint=100)
    receipt.track()
    receipt.settle(result)
    receipt.close()
    assert receipt.commit_ready
    assert env.client.last_call("send_message") is None


async def test_network_dispatch_fence_rechecks_after_await(env):
    from ccgram.telegram_client import DeliveryWithdrawnError, delivery_guard

    import asyncio

    lifecycle = asyncio.create_task(reconcile(env))
    current_revision = topic_pause.revision(1, 10, -100)
    with delivery_guard(
        lambda: (
            not topic_pause.paused(1, 10, -100)
            and topic_pause.revision(1, 10, -100) == current_revision
        )
    ):
        await lifecycle
        with pytest.raises(DeliveryWithdrawnError):
            await env.client.send_message(-100, "must not send", message_thread_id=10)
    assert env.client.last_call("send_message") is None


async def test_deleted_paused_topic_recovers_through_real_provisioning_after_restart(
    env, monkeypatch
):
    from telegram import ForumTopic
    from ccgram.config import config
    from ccgram.handlers.topics import topic_orchestration, topic_provisioning_recovery
    from ccgram.providers import _ensure_registered

    target = "81be230a-77a0-4247-b248-b5ef76d13c6d"
    env.window = replace(env.window, window_id=target)
    env.router.bind_thread(1, 10, target, chat_id=-100)
    _ensure_registered()
    monkeypatch.setattr(config, "topic_discovery_mode", "marked")
    await reconcile(env)
    state = env.router.get_topic_pause(1, 10, -100)
    assert state is not None
    await topic_pause.delete_paused_topics(
        env.client, [state], router=env.router, backend=env.backend, flush=env.flush
    )
    marked = replace(env.window, topic_marked=True)
    await reconcile(env, marked)
    restored = ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: True)
    restored.from_dict(env.router.to_dict())
    env.router = restored
    monkeypatch.setattr(topic_pause, "thread_router", restored)
    env.backend.capabilities = SimpleNamespace(name="agterm")
    env.backend.list_windows_for_reconciliation = AsyncMock(return_value=[marked])
    for module in (topic_orchestration, topic_provisioning_recovery):
        monkeypatch.setattr(module, "thread_router", restored)
        monkeypatch.setattr(
            module, "session_manager", SimpleNamespace(flush_state=env.flush)
        )
    monkeypatch.setattr(topic_orchestration, "tmux_manager", env.backend)
    monkeypatch.setattr(topic_provisioning_recovery, "multiplexer", env.backend)
    monkeypatch.setattr(
        topic_provisioning_recovery.window_query,
        "get_window_provider",
        lambda _: "shell",
    )
    monkeypatch.setattr(
        topic_provisioning_recovery.window_query, "view_window", lambda _: None
    )
    monkeypatch.setattr(
        topic_provisioning_recovery.window_query,
        "resolve_window_alias",
        lambda wid: wid,
    )
    env.client.returns["create_forum_topic"] = ForumTopic(11, "work", 0)
    result = await topic_provisioning_recovery.recover_topic_provisioning(
        env.client,
        router=restored,
        backend=env.backend,
        adapter=cast(TelegramSessionAdapter, MagicMock(spec=TelegramSessionAdapter)),
    )
    assert result == {"recreated": 1}
    assert restored.get_window_for_thread(1, 11, -100) == target
    assert (pause := restored.get_topic_pause(1, 11, -100)) is not None
    assert pause.paused
    await reconcile(env, marked)
    assert (pause := restored.get_topic_pause(1, 11, -100)) is not None
    assert not pause.paused
    assert env.client.call_count("create_forum_topic") == 1


@pytest.mark.parametrize("count", [1, 25])
async def test_preview_never_deletes_and_confirmation_is_single_use(
    env, monkeypatch, count
):
    import importlib
    from telegram import Update

    sync = importlib.import_module("ccgram.handlers.sync_command")
    await reconcile(env)
    from ccgram.thread_router import TopicPause

    for index in range(1, count):
        target = f"session-{index}"
        env.router.bind_thread(1, 10 + index, target, chat_id=-100)
        env.router.set_topic_pause(
            TopicPause(1, -100, 10 + index, target, f"owner-{index}", closed=True)
        )
    env.backend.find_window_by_id.side_effect = lambda target: replace(
        env.window,
        window_id=target,
        topic_mark_owner=next(
            state.owner
            for state in env.router.topic_pauses.values()
            if state.window_id == target
        ),
    )
    candidates = topic_pause.cleanup_candidates(1, -100, env.router)

    async def reconcile_service(client, windows):
        await topic_pause.reconcile_topic_pauses(
            client, windows, router=env.router, backend=env.backend, flush=env.flush
        )

    async def delete_service(client, candidates):
        return await topic_pause.delete_paused_topics(
            client, candidates, router=env.router, backend=env.backend, flush=env.flush
        )

    services = SimpleNamespace(
        enabled=lambda: True,
        reconcile_topic_pauses=reconcile_service,
        cleanup_candidates=lambda uid, chat: topic_pause.cleanup_candidates(
            uid, chat, env.router
        ),
        delete_paused_topics=delete_service,
    )
    monkeypatch.setattr(sync, "topic_pause", services)
    monkeypatch.setattr(sync, "thread_router", env.router)
    monkeypatch.setattr(
        sync, "list_windows_for_reconciliation", AsyncMock(return_value=[env.window])
    )
    monkeypatch.setattr(sync, "PTBTelegramClient", lambda _: env.client)
    reply = AsyncMock()
    monkeypatch.setattr(sync, "safe_reply", reply)
    monkeypatch.setattr(sync, "safe_edit", AsyncMock())
    update = MagicMock(spec=Update)
    update.effective_user.id = 1
    update.message.chat.id = -100
    context = SimpleNamespace(user_data={}, bot=None)
    await sync._preview_paused_cleanup(update, context)
    assert not env.client.last_call("delete_forum_topic")
    preview = context.user_data["paused_topic_cleanup"]
    assert preview[2] == candidates[:20]
    assert "permanently" in reply.call_args.args[1]
    update.callback_query.data = "pauseclean:" + preview[0]
    update.callback_query.message.chat.id = -100
    update.callback_query.answer = AsyncMock()
    await sync._confirm_paused_cleanup(update, context)
    assert env.client.call_count("delete_forum_topic") == min(count, 20)
    await sync._confirm_paused_cleanup(update, context)
    assert env.client.call_count("delete_forum_topic") == min(count, 20)


async def test_rebinding_same_target_cannot_drop_pause_or_frontier(env):
    await reconcile(env)
    before = env.router.get_topic_pause(1, 10, -100)
    env.router.bind_thread(1, 10, "session", chat_id=-100)
    assert env.router.get_topic_pause(1, 10, -100) == before
    env.router.bind_thread(1, 11, "session", chat_id=-100)
    assert (after := env.router.get_topic_pause(1, 11, -100)) is not None
    assert after.paused
    assert before is not None and before.owner == after.owner


async def test_switching_to_auto_restores_normal_close_semantics(env, monkeypatch):
    await reconcile(env)
    monkeypatch.setattr(topic_pause, "enabled", lambda: False)
    await reconcile(env)
    assert not topic_pause.note_closed(1, -100, 10)
    assert not topic_pause.note_missing(1, -100, 10)


async def test_flood_control_backoff_survives_restart_and_does_not_repeat_close(env):
    from telegram.error import RetryAfter

    env.client.set_side_effect("close_forum_topic", [RetryAfter(60)])
    await reconcile(env)
    await reconcile(env)
    assert env.client.call_count("close_forum_topic") == 1
    restored = ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: True)
    restored.from_dict(env.router.to_dict())
    env.router = restored
    await reconcile(env)
    assert env.client.call_count("close_forum_topic") == 1


@pytest.mark.parametrize("native", ["mismatching_id", "unavailable"])
async def test_cleanup_protects_bad_native_response(env, native):
    await reconcile(env)
    candidates = topic_pause.cleanup_candidates(1, -100, env.router)
    if native == "mismatching_id":
        env.backend.find_window_by_id.return_value = replace(
            env.window, window_id="another"
        )
    else:
        env.backend.find_window_by_id.side_effect = OSError("backend down")
    result = await topic_pause.delete_paused_topics(
        env.client, candidates, router=env.router, backend=env.backend, flush=env.flush
    )
    assert result["protected"] == 1
    assert not env.client.last_call("delete_forum_topic")


async def test_remote_close_handler_does_not_detach_a_flag_controlled_topic(
    env, monkeypatch
):
    from ccgram.config import config
    from ccgram.handlers.topics import topic_lifecycle

    await reconcile(env)
    await reconcile(env, replace(env.window, topic_marked=True))
    monkeypatch.setattr(topic_lifecycle, "thread_router", env.router)
    monkeypatch.setattr(topic_pause, "_flush", env.flush)
    monkeypatch.setattr(config, "allowed_users", [1])
    update = MagicMock()
    update.effective_user.id = 1
    update.effective_chat.id = -100
    update.message.chat.type = "supergroup"
    update.message.message_thread_id = 10
    await topic_lifecycle.topic_closed_handler(update, MagicMock())
    assert env.router.get_window_for_thread(1, 10, -100) == "session"
    assert (state := env.router.get_topic_pause(1, 10, -100)) is not None
    assert state.paused
    await reconcile(env, replace(env.window, topic_marked=True))
    assert not topic_pause.paused(1, 10, -100)


async def test_missing_paused_topic_probe_never_closes_running_terminal(
    env, monkeypatch
):
    from ccgram.handlers.topics import topic_lifecycle

    await reconcile(env)
    monkeypatch.setattr(topic_pause, "_flush", env.flush)
    adapter = MagicMock(spec=TelegramSessionAdapter)
    await topic_lifecycle._unbind_deleted_topic(
        env.client, 1, -100, 10, "session", adapter
    )
    adapter.close.assert_not_called()
    assert env.router.get_window_for_thread(1, 10, -100) == "session"
    assert (state := env.router.get_topic_pause(1, 10, -100)) is not None
    assert state.missing


async def test_paused_interactive_ui_skips_terminal_capture(env, monkeypatch):
    from ccgram.handlers.interactive import interactive_ui

    await reconcile(env)
    capture = AsyncMock()
    monkeypatch.setattr(interactive_ui, "_capture_interactive_content", capture)
    assert not await interactive_ui.handle_interactive_ui(
        env.client, 1, "session", 10, chat_id=-100
    )
    capture.assert_not_called()
