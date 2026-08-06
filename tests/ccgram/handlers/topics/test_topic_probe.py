from unittest.mock import AsyncMock, MagicMock, patch

from telegram.error import BadRequest, TelegramError

from ccgram.handlers.topics.topic_probe import probe_topic_exists


async def test_probe_uses_valid_text_and_deletes_message() -> None:
    client = AsyncMock()
    client.send_message.return_value = MagicMock(message_id=99)

    assert await probe_topic_exists(client, -100, 42) is True

    client.send_message.assert_awaited_once_with(
        -100,
        ".",
        message_thread_id=42,
        disable_notification=True,
    )
    client.delete_message.assert_awaited_once_with(-100, 99)


async def test_probe_reports_deleted_topic() -> None:
    client = AsyncMock()
    client.send_message.side_effect = BadRequest("Message thread not found")

    assert await probe_topic_exists(client, -100, 42) is False


async def test_probe_reports_unknown_transport_error() -> None:
    client = AsyncMock()
    client.send_message.side_effect = TelegramError("network")

    assert await probe_topic_exists(client, -100, 42) is None


async def test_probe_logs_cleanup_failure_but_keeps_exists_result() -> None:
    client = AsyncMock()
    client.send_message.return_value = MagicMock(message_id=99)
    client.delete_message.side_effect = TelegramError("delete failed")

    with patch("ccgram.handlers.topics.topic_probe.logger") as mock_logger:
        assert await probe_topic_exists(client, -100, 42) is True

    mock_logger.warning.assert_called_once()
