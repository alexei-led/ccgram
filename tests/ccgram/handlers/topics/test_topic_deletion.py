import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import BadRequest, RetryAfter, TelegramError

from ccgram.handlers.topics.topic_deletion import (
    cleanup_retired_topic,
    cleanup_retired_topics,
    retire_topic_binding,
)
from ccgram.state_persistence import StatePersistence
from ccgram.thread_router import RetiredTopic, ThreadRouter


@pytest.fixture
def router():
    with patch("ccgram.handlers.topics.topic_deletion.session_manager"):
        yield ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _: False)


def _retire(
    router: ThreadRouter,
    thread_id: int = 42,
    *,
    eligible: bool = True,
    reason: str = "session_closed",
) -> RetiredTopic:
    router.bind_thread(100, thread_id, f"@{thread_id}", chat_id=-999)
    router.unbind_thread(
        100,
        thread_id,
        chat_id=-999,
        retirement_reason=reason,
        cleanup_eligible=eligible,
    )
    return next(t for t in router.iter_retired_topics() if t.thread_id == thread_id)


async def test_delete_removes_record_without_close(router):
    topic = _retire(router)
    client = AsyncMock()

    assert await cleanup_retired_topic(client, topic, router=router) == "deleted"
    assert list(router.iter_retired_topics()) == []
    client.close_forum_topic.assert_not_awaited()


@pytest.mark.parametrize(
    "error", [BadRequest("Topic_id_invalid"), BadRequest("Message thread not found")]
)
async def test_already_deleted_is_terminal(router, error):
    topic = _retire(router)
    client = AsyncMock()
    client.delete_forum_topic.side_effect = error

    assert await cleanup_retired_topic(client, topic, router=router) == "already_gone"
    assert list(router.iter_retired_topics()) == []
    client.close_forum_topic.assert_not_awaited()


async def test_failed_delete_and_successful_close_survive_file_restart(tmp_path):
    router = ThreadRouter(
        schedule_save=lambda: persistence.schedule_save(),
        has_window_state=lambda _: False,
    )
    persistence = StatePersistence(tmp_path / "state.json", router.to_dict)
    topic = _retire(router)
    client = AsyncMock()
    client.delete_forum_topic.side_effect = TelegramError("not enough rights")

    with (
        patch("ccgram.handlers.topics.topic_deletion.time.time", return_value=1000),
        patch(
            "ccgram.handlers.topics.topic_deletion.session_manager.flush_state",
            side_effect=persistence.flush,
        ),
    ):
        assert await cleanup_retired_topic(client, topic, router=router) == "closed"

    restored = ThreadRouter(
        schedule_save=lambda: None, has_window_state=lambda _: False
    )
    restored.from_dict(persistence.load())
    pending = next(restored.iter_retired_topics())
    assert pending.closed is True
    assert pending.cleanup_eligible is True
    assert pending.retry_at == 1060

    retry = AsyncMock()
    with (
        patch("ccgram.handlers.topics.topic_deletion.time.time", return_value=1061),
        patch("ccgram.handlers.topics.topic_deletion.session_manager"),
    ):
        assert await cleanup_retired_topics(retry, router=restored) == {"deleted": 1}
    assert list(restored.iter_retired_topics()) == []
    retry.delete_forum_topic.assert_awaited_once()
    retry.close_forum_topic.assert_not_awaited()


async def test_false_results_remain_pending(router):
    topic = _retire(router)
    client = AsyncMock()
    client.delete_forum_topic.return_value = False
    client.close_forum_topic.return_value = False

    assert await cleanup_retired_topic(client, topic, router=router) == "failed"
    pending = next(router.iter_retired_topics())
    assert pending.closed is False
    assert pending.retry_at > 0


async def test_backoff_does_not_repeat_close_or_delete(router):
    topic = _retire(router)
    client = AsyncMock()
    client.delete_forum_topic.side_effect = TelegramError("denied")
    with patch("ccgram.handlers.topics.topic_deletion.time.time", return_value=1000):
        assert await cleanup_retired_topic(client, topic, router=router) == "closed"
        assert await cleanup_retired_topics(client, router=router) == {"deferred": 1}
    client.delete_forum_topic.assert_awaited_once()
    client.close_forum_topic.assert_awaited_once()


async def test_closed_pending_topic_retries_delete_without_reclosing(router):
    topic = _retire(router)
    topic = router.update_retired_topic(topic, retry_at=0, closed=True)
    assert topic is not None
    client = AsyncMock()
    client.delete_forum_topic.side_effect = TelegramError("denied")

    assert await cleanup_retired_topic(client, topic, router=router) == "closed"
    client.close_forum_topic.assert_not_awaited()
    assert next(router.iter_retired_topics()).closed is True


async def test_rate_limit_defers_without_close_and_stops_batch(router):
    _retire(router)
    _retire(router, 43)
    client = AsyncMock()
    client.delete_forum_topic.side_effect = RetryAfter(120)
    with patch("ccgram.handlers.topics.topic_deletion.time.time", return_value=1000):
        assert await cleanup_retired_topics(client, router=router) == {
            "rate_limited": 1
        }
    assert next(router.iter_retired_topics()).retry_at == 1120
    assert len(list(router.iter_retired_topics())) == 2
    client.delete_forum_topic.assert_awaited_once()
    client.close_forum_topic.assert_not_awaited()


async def test_rebind_is_rejected_while_delete_is_in_flight(router):
    topic = _retire(router)
    client = AsyncMock()

    async def rebind(*args, **kwargs):
        with pytest.raises(ValueError, match="deletion is in progress"):
            router.bind_thread(100, 42, "@new", chat_id=-999)
        return True

    client.delete_forum_topic.side_effect = rebind
    assert await cleanup_retired_topic(client, topic, router=router) == "deleted"
    client.close_forum_topic.assert_not_awaited()
    assert router.get_window_for_chat_thread(-999, 42) is None
    assert list(router.iter_retired_topics()) == []


async def test_multiple_active_users_protect_retired_topic(router):
    router.bind_thread(100, 42, "@one", chat_id=-999)
    router.bind_thread(200, 42, "@two", chat_id=-999)
    router.bind_thread(300, 42, "@three", chat_id=-999)
    router.unbind_thread(100, 42, chat_id=-999, cleanup_eligible=True)
    topic = next(router.iter_retired_topics())
    assert router.get_window_for_chat_thread(-999, 42) is None
    client = AsyncMock()

    assert (
        await cleanup_retired_topic(client, topic, router=router) == "protected_active"
    )
    client.delete_forum_topic.assert_not_awaited()
    client.close_forum_topic.assert_not_awaited()
    assert len(list(router.iter_thread_bindings())) == 2


async def test_retirement_clears_state_under_claim_after_unbinding(router):
    router.bind_thread(100, 42, "@one", chat_id=-999)
    client = AsyncMock()

    async def clear_state():
        assert router.get_window_for_chat_thread(-999, 42) is None
        with pytest.raises(ValueError, match="deletion is in progress"):
            router.bind_thread(100, 42, "@new", chat_id=-999)

    assert (
        await retire_topic_binding(
            client, 100, 42, "@one", router=router, before_delete=clear_state
        )
        == "deleted"
    )
    assert list(router.iter_retired_topics()) == []


async def test_missing_chat_identity_keeps_binding(router):
    router.bind_thread(100, 42, "@one")
    client = AsyncMock()
    assert (
        await retire_topic_binding(client, 100, 42, "@one", router=router)
        == "protected_active"
    )
    assert router.get_window_for_thread(100, 42) == "@one"
    client.delete_forum_topic.assert_not_awaited()


async def test_legacy_binding_with_recorded_chat_can_be_retired(router):
    router.bind_thread(100, 42, "@one")
    router.group_chat_ids["100:42"] = -999
    client = AsyncMock()

    assert (
        await retire_topic_binding(client, 100, 42, "@one", router=router) == "deleted"
    )
    assert list(router.iter_thread_bindings()) == []
    assert list(router.iter_retired_topics()) == []


async def test_retry_failures_do_not_starve_later_topics(router):
    for tid in range(10, 31):
        _retire(router, tid)
    client = AsyncMock()

    async def delete(chat_id, thread_id, **kwargs):
        if thread_id < 30:
            raise TelegramError("denied")
        return True

    client.delete_forum_topic.side_effect = delete
    with patch("ccgram.handlers.topics.topic_deletion.time.time", return_value=1000):
        assert await cleanup_retired_topics(client, router=router) == {"closed": 20}
    with patch("ccgram.handlers.topics.topic_deletion.time.time", return_value=1061):
        outcomes = await cleanup_retired_topics(client, router=router)
    assert outcomes["deleted"] == 1
    assert all(t.thread_id != 30 for t in router.iter_retired_topics())


async def test_old_record_cannot_delete_new_retirement_of_same_topic(router):
    old = _retire(router)
    new = _retire(router)
    client = AsyncMock()

    assert await cleanup_retired_topic(client, old, router=router) == "protected_active"
    assert list(router.iter_retired_topics()) == [new]
    client.delete_forum_topic.assert_not_awaited()


async def test_general_topic_is_protected(router):
    topic = _retire(router, 1)
    client = AsyncMock()

    assert (
        await cleanup_retired_topic(client, topic, router=router) == "protected_general"
    )
    client.delete_forum_topic.assert_not_awaited()
    client.close_forum_topic.assert_not_awaited()


async def test_sweep_limits_api_work_without_forgetting_remaining_topics(router):
    for tid in range(10, 40):
        _retire(router, tid)
    client = AsyncMock()

    assert await cleanup_retired_topics(client, router=router, limit=2) == {
        "deleted": 2
    }
    assert client.delete_forum_topic.await_count == 2
    assert len(list(router.iter_retired_topics())) == 28


async def test_exclude_reasons_skips_dead_session_retirements(router):
    dead = _retire(router, reason="dead_session")
    other = _retire(router, 43, reason="session_closed")
    client = AsyncMock()

    outcomes = await cleanup_retired_topics(
        client, router=router, exclude_reasons=frozenset({"dead_session"})
    )

    assert outcomes == {"deleted": 1}
    assert list(router.iter_retired_topics()) == [dead]
    assert other not in router.iter_retired_topics()


async def test_old_closed_topics_require_explicit_sync_cleanup(router):
    _retire(router, eligible=False, reason="remote_closed")
    keep = _retire(router, 43, eligible=False, reason="keep_remote")
    client = AsyncMock()

    assert await cleanup_retired_topics(client, router=router) == {}
    client.delete_forum_topic.assert_not_awaited()
    assert await cleanup_retired_topics(client, router=router, include_closed=True) == {
        "deleted": 1
    }
    assert list(router.iter_retired_topics()) == [keep]


async def test_reviewed_old_topic_failure_becomes_pending(router):
    _retire(router, eligible=False, reason="remote_removed")
    client = AsyncMock()
    client.delete_forum_topic.side_effect = TelegramError("denied")

    assert await cleanup_retired_topics(client, router=router, include_closed=True) == {
        "closed": 1
    }
    assert next(router.iter_retired_topics()).cleanup_eligible is True


async def test_concurrent_cleanup_and_cancellation_release_claim(router):
    topic = _retire(router)
    client = AsyncMock()
    entered = asyncio.Event()

    async def wait_for_cancellation(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    client.delete_forum_topic.side_effect = wait_for_cancellation
    first = asyncio.create_task(cleanup_retired_topic(client, topic, router=router))
    await entered.wait()
    try:
        assert await cleanup_retired_topic(client, topic, router=router) == "deferred"
    finally:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    client.delete_forum_topic.side_effect = None
    assert await cleanup_retired_topic(client, topic, router=router) == "deleted"


async def test_retire_topic_binding_protects_active_provisioning_claim(router):
    router.bind_thread(100, 42, "@old", chat_id=-999)
    router.begin_topic_provisioning(
        100,
        -999,
        thread_id=42,
        target_id="@new",
        kind="replacement",
    )
    client = AsyncMock()

    assert (
        await retire_topic_binding(client, 100, 42, "@old", router=router)
        == "protected_provisioning"
    )
    assert router.get_window_for_chat_thread(-999, 42) == "@old"
    client.delete_forum_topic.assert_not_awaited()


async def test_target_provisioning_protects_matching_window_before_unbind(router):
    router.bind_thread(100, 42, "durable", chat_id=-999)
    router.begin_topic_provisioning(
        100,
        -999,
        target_id="durable",
        kind="topic_for_target",
    )
    client = AsyncMock()

    assert (
        await retire_topic_binding(client, 100, 42, "durable", router=router)
        == "protected_provisioning"
    )
    assert router.get_window_for_chat_thread(-999, 42) == "durable"
    client.delete_forum_topic.assert_not_awaited()


async def test_cleanup_and_deletion_claim_exclude_provisioned_topic(router):
    topic = _retire(router)
    router.begin_topic_provisioning(
        100,
        -999,
        thread_id=42,
        target_id="new-target",
        kind="target_for_topic",
    )
    client = AsyncMock()

    assert await cleanup_retired_topic(client, topic, router=router) == (
        "protected_provisioning"
    )
    assert router.begin_topic_deletion(topic) is False
    client.delete_forum_topic.assert_not_awaited()


async def test_flush_failure_releases_topic_and_target_deletion_claims(router):
    topic = _retire(router)
    client = AsyncMock()

    with (
        patch(
            "ccgram.handlers.topics.topic_deletion.session_manager.flush_state",
            side_effect=OSError("state write failed"),
        ),
        pytest.raises(OSError, match="state write failed"),
    ):
        await cleanup_retired_topic(client, topic, router=router)

    assert list(router.iter_retired_topics()) == [topic]
    assert router.begin_topic_deletion(topic) is True
    router.end_topic_deletion(topic)
    claim = router.begin_topic_provisioning(
        100,
        -999,
        target_id=topic.target_id,
        kind="topic_for_target",
    )
    router.abort_topic_provisioning(
        claim.claim_id,
        target_confirmed_absent=False,
        topic_confirmed_absent=True,
    )
    client.delete_forum_topic.assert_not_awaited()
    client.close_forum_topic.assert_not_awaited()
