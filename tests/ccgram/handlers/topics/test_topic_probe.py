from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest, RetryAfter, TelegramError

from ccgram.handlers.topics.topic_probe import probe_topic_exists
from ccgram.telegram_rate_limiter import NO_RETRY_RATE_LIMIT_ARGS


async def test_probe_restores_intended_title_without_posting_messages():
    client = AsyncMock()

    assert await probe_topic_exists(client, -100, 42, topic_name="project") is True

    client.edit_forum_topic.assert_awaited_once_with(
        -100,
        42,
        name="project",
        rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS,
    )
    client.send_message.assert_not_awaited()
    client.delete_message.assert_not_awaited()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(
            BadRequest("Message thread not found"), False, id="missing-thread"
        ),
        pytest.param(BadRequest("TOPIC_ID_INVALID"), False, id="deleted-topic"),
        pytest.param(BadRequest("TOPIC_NOT_MODIFIED"), True, id="unchanged-title"),
        pytest.param(BadRequest("Not enough rights"), None, id="permission-denied"),
        pytest.param(TelegramError("network"), None, id="network-error"),
        pytest.param(RetryAfter(60), None, id="flood-control"),
    ],
)
async def test_probe_distinguishes_present_deleted_and_unknown(error, expected):
    client = AsyncMock()
    client.edit_forum_topic.side_effect = error

    assert await probe_topic_exists(client, -100, 42, topic_name="project") is expected
    client.send_message.assert_not_awaited()
    client.delete_message.assert_not_awaited()


async def test_probe_propagates_retry_after_when_requested():
    client = AsyncMock()
    client.edit_forum_topic.side_effect = RetryAfter(60)

    with pytest.raises(RetryAfter):
        await probe_topic_exists(
            client,
            -100,
            42,
            topic_name="project",
            propagate_retry_after=True,
        )


async def test_empty_title_cannot_confirm_topic_existence():
    client = AsyncMock()

    assert await probe_topic_exists(client, -100, 42, topic_name="") is None
    client.edit_forum_topic.assert_not_awaited()
    client.send_message.assert_not_awaited()


async def test_false_edit_result_cannot_confirm_topic_existence():
    client = AsyncMock()
    client.edit_forum_topic.return_value = False

    assert await probe_topic_exists(client, -100, 42, topic_name="project") is None


async def test_recovery_title_respects_telegram_limit():
    client = AsyncMock()

    assert await probe_topic_exists(client, -100, 42, topic_name="a" * 150) is True
    assert client.edit_forum_topic.call_args.kwargs["name"] == "a" * 128
