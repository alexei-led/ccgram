from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import RetryAfter

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
