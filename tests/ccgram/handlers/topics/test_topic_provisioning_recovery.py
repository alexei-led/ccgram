from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, RetryAfter

from ccgram.handlers.topics.topic_provisioning_recovery import (
    recover_topic_provisioning,
)
from ccgram.telegram_rate_limiter import NO_RETRY_RATE_LIMIT_ARGS
from ccgram.thread_router import ThreadRouter


def _router() -> ThreadRouter:
    return ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _wid: False)


def _restored_claim(*, target_id="@2", thread_id=42, previous_target_id=None):
    original = _router()
    if previous_target_id:
        original.bind_thread(1, 42, previous_target_id, chat_id=-100)
    claim = original.begin_topic_provisioning(
        1,
        -100,
        thread_id=thread_id,
        target_id=target_id,
        previous_target_id=previous_target_id,
        kind="topic_for_target" if thread_id is None else "target_for_topic",
    )
    restored = _router()
    restored.from_dict(original.to_dict())
    return restored, claim


@pytest.fixture(autouse=True)
def _isolate_persistence():
    with (
        patch("ccgram.handlers.topics.topic_provisioning_recovery.session_manager"),
        patch("ccgram.handlers.topics.topic_deletion.session_manager"),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.clear_topic_state",
            new_callable=AsyncMock,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_query.resolve_window_alias",
            side_effect=lambda wid: wid,
        ),
    ):
        yield


@pytest.mark.parametrize("presence", [True, False, None])
async def test_restored_known_topic_follows_authoritative_presence(presence):
    router, claim = _restored_claim()
    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=presence,
    ) as probe:
        outcome = await recover_topic_provisioning(client, router=router)

    probe.assert_awaited_once_with("@2", None)
    if presence is True:
        assert outcome == {"bound": 1}
        assert router.get_window_for_chat_thread(-100, 42) == "@2"
        assert not router.iter_topic_provisionings()
        client.delete_forum_topic.assert_not_awaited()
    elif presence is False:
        assert outcome == {"deleted": 1}
        assert not router.iter_topic_provisionings()
        assert not list(router.iter_retired_topics())
        client.delete_forum_topic.assert_awaited_once_with(
            -100, 42, rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS
        )
    else:
        assert outcome == {"unresolved": 1}
        assert router.iter_topic_provisionings() == [claim]
        client.delete_forum_topic.assert_not_awaited()


async def test_probe_cleanup_rate_limit_commits_topic_then_stops_batch():
    router, _claim = _restored_claim()
    client = AsyncMock()
    client.send_message.return_value = MagicMock(message_id=99)
    client.delete_message.side_effect = RetryAfter(60)
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=True,
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "rate_limited": 1
        }

    assert router.get_window_for_chat_thread(-100, 42) == "@2"
    assert not router.iter_topic_provisionings()
    client.delete_message.assert_awaited_once_with(-100, 99)


async def test_active_creation_is_never_reconciled_as_abandoned():
    router = _router()
    claim = router.begin_topic_provisioning(
        1, -100, thread_id=42, target_id="@2", kind="target_for_topic"
    )
    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=False,
    ) as probe:
        assert await recover_topic_provisioning(client, router=router) == {}
    probe.assert_not_awaited()
    client.delete_forum_topic.assert_not_awaited()
    assert router.iter_topic_provisionings() == [claim]


@pytest.mark.parametrize(("target_id", "thread_id"), [(None, 42), ("@2", None)])
async def test_missing_remote_identity_remains_protected(target_id, thread_id):
    router, claim = _restored_claim(target_id=target_id, thread_id=thread_id)
    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
    ) as probe:
        assert await recover_topic_provisioning(client, router=router) == {
            "unresolved": 1
        }
    probe.assert_not_awaited()
    client.delete_forum_topic.assert_not_awaited()
    assert router.iter_topic_provisionings() == [claim]


async def test_creation_changed_during_probe_is_not_deleted():
    router, claim = _restored_claim()

    async def probe(*_args):
        router.attach_provisioning_target(claim.claim_id, "@3")
        return False

    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        side_effect=probe,
    ):
        assert await recover_topic_provisioning(client, router=router) == {"changed": 1}
    assert router.iter_topic_provisionings()[0].target_id == "@3"
    client.delete_forum_topic.assert_not_awaited()


async def test_deleted_topic_is_unbound_and_recreated_for_live_target():
    router, claim = _restored_claim(previous_target_id="@dead")
    client = AsyncMock()

    async def recreate(
        _client,
        chat_id,
        target_id,
        topic_name,
        *,
        user_id,
        propagate_retry_after,
        claim_id,
    ):
        router.attach_provisioning_topic(claim_id, 77)
        return router.commit_topic_provisioning(claim_id, window_name=topic_name)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            side_effect=recreate,
        ) as create,
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "recreated": 1
        }

    assert router.get_window_for_chat_thread(-100, 42) is None
    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    assert not router.iter_topic_provisionings()
    assert not list(router.iter_retired_topics())
    create.assert_awaited_once_with(
        client,
        -100,
        "@2",
        "@2",
        user_id=1,
        propagate_retry_after=True,
        claim_id=claim.claim_id,
    )


async def test_deleted_topic_recreation_uses_cached_window_name():
    router, claim = _restored_claim(previous_target_id="@dead")
    router.window_display_names["@2"] = "cached-project"
    client = AsyncMock()

    async def recreate(
        _client,
        _chat_id,
        _target_id,
        _topic_name,
        *,
        user_id,
        propagate_retry_after,
        claim_id,
    ):
        return True

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            side_effect=recreate,
        ) as create,
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "recreated": 1
        }

    create.assert_awaited_once_with(
        client,
        -100,
        "@2",
        "cached-project",
        user_id=1,
        propagate_retry_after=True,
        claim_id=claim.claim_id,
    )


async def test_deleted_topic_preserves_existing_target_binding():
    router, _claim = _restored_claim()
    router.bind_thread(1, 77, "@2", chat_id=-100)
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "released": 1
        }

    assert router.get_window_for_chat_thread(-100, 42) is None
    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    create.assert_not_awaited()


async def test_existing_target_binding_blocks_commit_of_present_topic():
    router, claim = _restored_claim()
    router.bind_thread(1, 77, "@2", chat_id=-100)
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "unresolved": 1
        }

    assert router.iter_topic_provisionings() == [claim]
    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    assert router.get_window_for_chat_thread(-100, 42) is None


async def test_unknown_topic_probe_retains_claim():
    router, claim = _restored_claim()
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "unresolved": 1
        }

    assert router.iter_topic_provisionings() == [claim]
    create.assert_not_awaited()


async def test_claim_changed_during_topic_probe_is_protected():
    router, claim = _restored_claim()
    client = AsyncMock()

    async def probe(*_args, **_kwargs):
        router.attach_provisioning_target(claim.claim_id, "@3")
        return False

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            side_effect=probe,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(client, router=router) == {"changed": 1}

    assert router.iter_topic_provisionings()[0].target_id == "@3"
    create.assert_not_awaited()


async def test_retry_after_stops_recovery_batch_without_settling_claims():
    router, first = _restored_claim()
    second = router.begin_topic_provisioning(
        1, -100, thread_id=43, target_id="@3", kind="target_for_topic"
    )
    router.mark_provisioning_uncertain(second.claim_id)
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ) as presence,
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            side_effect=RetryAfter(60),
        ),
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "rate_limited": 1
        }

    presence.assert_awaited_once_with("@2", None)
    assert [item.claim_id for item in router.iter_topic_provisionings()] == [
        first.claim_id,
        second.claim_id,
    ]


async def test_retry_after_during_real_recreation_stops_recovery_batch():
    router, first = _restored_claim()
    second = router.begin_topic_provisioning(
        1, -100, thread_id=43, target_id="@3", kind="target_for_topic"
    )
    router.mark_provisioning_uncertain(second.claim_id)
    client = AsyncMock()
    client.create_forum_topic.side_effect = RetryAfter(60)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ) as probe,
        patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router",
            router,
        ),
        patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
        patch(
            "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
            {},
        ) as retry_until,
        patch(
            "ccgram.handlers.topics.topic_orchestration.time.monotonic",
            return_value=100.0,
        ),
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "rate_limited": 1
        }

    probe.assert_awaited_once()
    client.create_forum_topic.assert_awaited_once_with(
        chat_id=-100,
        name="@2",
    )
    claims = router.iter_topic_provisionings()
    assert [item.claim_id for item in claims] == [
        first.claim_id,
        second.claim_id,
    ]
    assert claims[0].thread_id is None
    assert claims[0].retry_thread_id == 42
    assert claims[0].retry_at > 0
    assert claims[0].uncertain is False
    assert router.owns_topic_provisioning(claims[0].claim_id) is False
    assert retry_until[-100] > 100.0


async def test_deferred_recreation_survives_reload_and_recovers_without_event():
    router, first = _restored_claim()
    client = AsyncMock()
    client.create_forum_topic.side_effect = BadRequest("denied")

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router",
            router,
        ),
        patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
        patch(
            "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
            {},
        ),
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "deferred": 1
        }

    deferred = router.get_topic_provisioning(first.claim_id)
    assert deferred is not None
    assert deferred.thread_id is None
    assert deferred.retry_thread_id == 42
    assert deferred.retry_at > 0
    assert deferred.uncertain is False

    restored = _router()
    restored.from_dict(router.to_dict())
    retry_client = AsyncMock()
    retry_client.create_forum_topic.return_value = MagicMock(message_thread_id=77)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.time.time",
            return_value=deferred.retry_at + 1,
        ),
        patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router",
            restored,
        ),
        patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
        patch(
            "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
            {},
        ),
    ):
        assert await recover_topic_provisioning(retry_client, router=restored) == {
            "recreated": 1
        }

    assert restored.get_window_for_chat_thread(-100, 42) is None
    assert restored.get_window_for_chat_thread(-100, 77) == "@2"
    assert restored.iter_topic_provisionings() == []


async def test_uncertain_recreation_stays_quarantined_when_target_is_closed():
    router, claim = _restored_claim()
    prepared = router.prepare_topic_recreation(claim.claim_id)
    router.defer_topic_recreation(
        prepared.claim_id,
        retry_at=0.0,
        uncertain=True,
    )
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=False,
        ) as presence,
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
        ) as probe,
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "unresolved": 1
        }

    presence.assert_not_awaited()
    probe.assert_not_awaited()
    create.assert_not_awaited()
    current = router.get_topic_provisioning(claim.claim_id)
    assert current is not None
    assert current.uncertain is True
    assert current.retry_thread_id == 42


async def test_recreation_flush_failure_releases_ownership_for_later_retry():
    router, claim = _restored_claim()
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.session_manager.flush_state",
            side_effect=OSError("persist failed"),
        ) as flush,
    ):
        assert await recover_topic_provisioning(client, router=router) == {"changed": 1}

    deferred = router.get_topic_provisioning(claim.claim_id)
    assert deferred is not None
    assert deferred.thread_id is None
    assert deferred.retry_thread_id == 42
    assert deferred.uncertain is False
    assert router.owns_topic_provisioning(claim.claim_id) is False

    flush.side_effect = None

    async def recreate(
        _client,
        _chat_id,
        _target_id,
        _topic_name,
        *,
        user_id,
        propagate_retry_after,
        claim_id,
    ):
        router.attach_provisioning_topic(claim_id, 77)
        return router.commit_topic_provisioning(claim_id)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            side_effect=recreate,
        ),
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "recreated": 1
        }

    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    assert router.iter_topic_provisionings() == []


async def test_failed_replacement_preserves_previous_binding():
    router, _claim = _restored_claim(previous_target_id="@1")
    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=False,
    ):
        assert await recover_topic_provisioning(client, router=router) == {
            "released": 1
        }
    assert router.get_window_for_chat_thread(-100, 42) == "@1"
    assert not router.iter_topic_provisionings()
    client.delete_forum_topic.assert_not_awaited()


async def test_rate_limit_preserves_remaining_recovery_claims():
    router, _claim = _restored_claim()
    second = router.begin_topic_provisioning(
        1, -100, thread_id=43, target_id="@3", kind="target_for_topic"
    )
    router.mark_provisioning_uncertain(second.claim_id)
    client = AsyncMock()
    client.delete_forum_topic.side_effect = RetryAfter(60)
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=False,
    ) as probe:
        assert await recover_topic_provisioning(client, router=router) == {
            "rate_limited": 1
        }
    assert probe.await_count == 1
    assert [item.claim_id for item in router.iter_topic_provisionings()] == [
        second.claim_id
    ]
    retired = list(router.iter_retired_topics())
    assert len(retired) == 1
    assert retired[0].retry_at > 0
