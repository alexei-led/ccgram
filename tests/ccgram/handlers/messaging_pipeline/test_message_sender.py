import asyncio
from unittest.mock import AsyncMock, patch

from telegram import Message
from telegram.error import BadRequest, RetryAfter, TelegramError

from ccgram.expandable_quote import EXPANDABLE_QUOTE_END as EXP_END
from ccgram.expandable_quote import EXPANDABLE_QUOTE_START as EXP_START
from ccgram.handlers.messaging_pipeline.message_sender import (
    MESSAGE_SEND_INTERVAL,
    _last_send_time,
    _send_with_fallback,
    edit_with_fallback,
    rate_limit_send,
)
from ccgram.telegram_client import FakeTelegramClient

import pytest


@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    _last_send_time.clear()
    yield
    _last_send_time.clear()


@pytest.fixture
def client() -> FakeTelegramClient:
    return FakeTelegramClient()


def _fake_message() -> Message:
    return AsyncMock(spec=Message)


class TestRateLimitSend:
    @pytest.fixture(autouse=True)
    def _real_send_interval(self, monkeypatch):
        # Conftest zeroes MESSAGE_SEND_INTERVAL for speed; restore the real
        # value here so wait-time assertions hold.
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.message_sender.MESSAGE_SEND_INTERVAL",
            0.5,
        )

    async def test_first_call_no_wait(self) -> None:
        with patch(
            "ccgram.handlers.messaging_pipeline.message_sender.asyncio.sleep",
            new_callable=AsyncMock,
            spec=asyncio.sleep,
        ) as mock_sleep:
            await rate_limit_send(123)
            mock_sleep.assert_not_called()

    async def test_second_call_within_interval_waits(self) -> None:
        await rate_limit_send(123)

        with patch(
            "ccgram.handlers.messaging_pipeline.message_sender.asyncio.sleep",
            new_callable=AsyncMock,
            spec=asyncio.sleep,
        ) as mock_sleep:
            await rate_limit_send(123)
            mock_sleep.assert_called_once()
            wait_time = mock_sleep.call_args[0][0]
            assert 0 < wait_time <= MESSAGE_SEND_INTERVAL

    async def test_different_chat_ids_independent(self) -> None:
        await rate_limit_send(1)

        with patch(
            "ccgram.handlers.messaging_pipeline.message_sender.asyncio.sleep",
            new_callable=AsyncMock,
            spec=asyncio.sleep,
        ) as mock_sleep:
            await rate_limit_send(2)
            mock_sleep.assert_not_called()

    async def test_updates_last_send_time(self) -> None:
        assert 123 not in _last_send_time
        await rate_limit_send(123)
        assert 123 in _last_send_time
        first_time = _last_send_time[123]

        await asyncio.sleep(0.01)
        await rate_limit_send(123)
        assert _last_send_time[123] > first_time


class TestSendWithFallback:
    @pytest.fixture(autouse=True)
    def _instant_retry_sleep(self, monkeypatch):
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.message_sender.asyncio.sleep",
            AsyncMock(),
        )

    async def test_entity_success(self, client: FakeTelegramClient) -> None:
        sent = _fake_message()
        client.returns["send_message"] = sent

        result = await _send_with_fallback(client, 123, "hello")
        assert result is sent
        assert client.call_count("send_message") == 1
        last = client.last_call("send_message")
        assert last is not None
        assert "entities" in last.kwargs
        assert "parse_mode" not in last.kwargs

    async def test_fallback_to_plain(self, client: FakeTelegramClient) -> None:
        sent = _fake_message()
        client.set_side_effect("send_message", [TelegramError("entity error"), sent])

        result = await _send_with_fallback(client, 123, "hello")
        assert result is sent
        assert client.call_count("send_message") == 2
        fallback_kwargs = client.calls[1].kwargs
        assert "parse_mode" not in fallback_kwargs
        assert "entities" not in fallback_kwargs

    async def test_both_fail_returns_none(self, client: FakeTelegramClient) -> None:
        client.set_side_effect(
            "send_message",
            [TelegramError("entity fail"), TelegramError("plain fail")],
        )

        result = await _send_with_fallback(client, 123, "hello")
        assert result is None

    async def test_retry_after_sleeps_and_retries(
        self, client: FakeTelegramClient
    ) -> None:
        sent = _fake_message()
        client.set_side_effect("send_message", [RetryAfter(1), sent])

        result = await _send_with_fallback(client, 123, "hello")
        assert result is sent
        assert client.call_count("send_message") == 2

    async def test_retry_after_then_permanent_fail_falls_through_to_plain(
        self, client: FakeTelegramClient
    ) -> None:
        sent = _fake_message()
        client.set_side_effect(
            "send_message",
            [RetryAfter(1), TelegramError("permanent fail"), sent],
        )
        result = await _send_with_fallback(client, 123, "hello")
        assert result is sent
        assert client.call_count("send_message") == 3

    async def test_plain_text_retry_after_then_success(
        self, client: FakeTelegramClient
    ) -> None:
        sent = _fake_message()
        client.set_side_effect(
            "send_message",
            [TelegramError("entity fail"), RetryAfter(1), sent],
        )
        result = await _send_with_fallback(client, 123, "hello")
        assert result is sent
        assert client.call_count("send_message") == 3

    async def test_plain_text_retry_after_then_permanent_fail(
        self, client: FakeTelegramClient
    ) -> None:
        client.set_side_effect(
            "send_message",
            [
                TelegramError("entity fail"),
                RetryAfter(1),
                TelegramError("plain also dead"),
            ],
        )
        result = await _send_with_fallback(client, 123, "hello")
        assert result is None
        assert client.call_count("send_message") == 3

    async def test_bold_formatting_sends_entities(
        self, client: FakeTelegramClient
    ) -> None:
        sent = _fake_message()
        client.returns["send_message"] = sent

        await _send_with_fallback(client, 123, "**bold text**")

        last = client.last_call("send_message")
        assert last is not None
        assert last.kwargs["text"] == "bold text"
        entities = last.kwargs["entities"]
        assert len(entities) >= 1
        assert any(e.type == "bold" for e in entities)


class TestEditWithFallback:
    async def test_entity_success(self, client: FakeTelegramClient) -> None:
        result = await edit_with_fallback(client, 123, 1, "hello")
        assert result is True
        last = client.last_call("edit_message_text")
        assert last is not None
        assert "entities" in last.kwargs

    async def test_entity_fail_plain_success(self, client: FakeTelegramClient) -> None:
        client.set_side_effect(
            "edit_message_text", [TelegramError("entity fail"), True]
        )
        result = await edit_with_fallback(client, 123, 1, "hello")
        assert result is True
        assert client.call_count("edit_message_text") == 2
        fallback_kwargs = client.calls[1].kwargs
        assert "entities" not in fallback_kwargs

    async def test_both_fail_returns_false(self, client: FakeTelegramClient) -> None:
        client.set_side_effect(
            "edit_message_text",
            [TelegramError("entity fail"), TelegramError("plain fail")],
        )
        result = await edit_with_fallback(client, 123, 1, "hello")
        assert result is False

    async def test_retry_after_reraised(self, client: FakeTelegramClient) -> None:
        client.set_side_effect("edit_message_text", [RetryAfter(5)])
        with pytest.raises(RetryAfter):
            await edit_with_fallback(client, 123, 1, "hello")

    async def test_retry_after_in_fallback_reraised(
        self, client: FakeTelegramClient
    ) -> None:
        client.set_side_effect(
            "edit_message_text", [TelegramError("entity fail"), RetryAfter(5)]
        )
        with pytest.raises(RetryAfter):
            await edit_with_fallback(client, 123, 1, "hello")

    async def test_not_modified_returns_true_no_plain_fallback(
        self, client: FakeTelegramClient
    ) -> None:
        """Telegram's 'not modified' error must not strip entities via fallback.

        Re-editing with identical text triggers BadRequest("Message is not
        modified"); falling back to plain text would clear the entities
        Telegram already has, leaving the message visibly unformatted.
        """
        client.set_side_effect(
            "edit_message_text", [BadRequest("Message is not modified")]
        )
        result = await edit_with_fallback(client, 123, 1, "hello")
        assert result is True
        assert client.call_count("edit_message_text") == 1


class TestEmptyAndOverlongGuards:
    async def test_empty_text_skips_send(self, client: FakeTelegramClient) -> None:
        result = await _send_with_fallback(client, 123, "")
        assert result is None
        assert client.call_count("send_message") == 0

    async def test_whitespace_only_text_skips_send(
        self, client: FakeTelegramClient
    ) -> None:
        result = await _send_with_fallback(client, 123, "   \n\t  ")
        assert result is None
        assert client.call_count("send_message") == 0

    async def test_overlong_text_truncates_under_limit(
        self, client: FakeTelegramClient
    ) -> None:
        from ccgram.telegram_sender import TELEGRAM_MAX_MESSAGE_LENGTH

        client.returns["send_message"] = _fake_message()
        long_text = "x" * 8000
        await _send_with_fallback(client, 123, long_text)

        last = client.last_call("send_message")
        assert last is not None
        sent_text = last.kwargs["text"]
        assert len(sent_text) <= TELEGRAM_MAX_MESSAGE_LENGTH
        assert sent_text.endswith("…")


class TestFallbackNoSentinelLeak:
    async def test_no_sentinel_bytes_in_fallback(
        self, client: FakeTelegramClient
    ) -> None:
        sent = _fake_message()
        client.set_side_effect("send_message", [TelegramError("entity error"), sent])

        text_with_sentinels = f"before {EXP_START}quoted{EXP_END} after"
        await _send_with_fallback(client, 123, text_with_sentinels)

        fallback_text = client.calls[1].kwargs["text"]
        assert "\x02" not in fallback_text

    async def test_edit_fallback_no_sentinel_bytes(
        self, client: FakeTelegramClient
    ) -> None:
        client.set_side_effect(
            "edit_message_text", [TelegramError("entity fail"), True]
        )
        text_with_sentinels = f"before {EXP_START}quoted{EXP_END} after"
        await edit_with_fallback(client, 123, 1, text_with_sentinels)

        fallback_kwargs = client.calls[1].kwargs
        assert "\x02" not in fallback_kwargs["text"]
