"""Tests for message_sender rate limiting and send-with-fallback."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from telegram import Message
from telegram.error import RetryAfter, TelegramError

from ccgram.handlers.message_sender import (
    MESSAGE_SEND_INTERVAL,
    _last_send_time,
    _send_with_fallback,
    rate_limit_send,
    strip_mdv2,
)


@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    _last_send_time.clear()
    yield
    _last_send_time.clear()


class TestRateLimitSend:
    async def test_first_call_no_wait(self) -> None:
        with patch(
            "ccgram.handlers.message_sender.asyncio.sleep",
            new_callable=AsyncMock,
            spec=asyncio.sleep,
        ) as mock_sleep:
            await rate_limit_send(123)
            mock_sleep.assert_not_called()

    async def test_second_call_within_interval_waits(self) -> None:
        await rate_limit_send(123)

        with patch(
            "ccgram.handlers.message_sender.asyncio.sleep",
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
            "ccgram.handlers.message_sender.asyncio.sleep",
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
    async def test_markdown_success(self) -> None:
        bot = AsyncMock()
        sent = AsyncMock(spec=Message)
        bot.send_message.return_value = sent

        result = await _send_with_fallback(bot, 123, "hello")
        assert result is sent
        bot.send_message.assert_called_once()
        assert bot.send_message.call_args.kwargs["parse_mode"] == "MarkdownV2"

    async def test_fallback_to_plain(self) -> None:
        bot = AsyncMock()
        sent = AsyncMock(spec=Message)
        bot.send_message.side_effect = [TelegramError("parse error"), sent]

        result = await _send_with_fallback(bot, 123, "hello")
        assert result is sent
        assert bot.send_message.call_count == 2
        assert "parse_mode" not in bot.send_message.call_args_list[1].kwargs

    async def test_both_fail_returns_none(self) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = [
            TelegramError("md fail"),
            TelegramError("plain fail"),
        ]

        result = await _send_with_fallback(bot, 123, "hello")
        assert result is None

    async def test_retry_after_sleeps_and_retries(self) -> None:
        bot = AsyncMock()
        sent = AsyncMock(spec=Message)
        bot.send_message.side_effect = [RetryAfter(1), sent]

        result = await _send_with_fallback(bot, 123, "hello")
        assert result is sent
        assert bot.send_message.call_count == 2

    async def test_retry_after_then_permanent_fail_returns_none(self) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = [
            RetryAfter(1),
            TelegramError("permanent fail"),
        ]
        result = await _send_with_fallback(bot, 123, "hello")
        assert result is None
        assert bot.send_message.call_count == 2

    async def test_plain_text_retry_after_then_success(self) -> None:
        bot = AsyncMock()
        sent = AsyncMock(spec=Message)
        bot.send_message.side_effect = [
            TelegramError("md fail"),
            RetryAfter(1),
            sent,
        ]
        result = await _send_with_fallback(bot, 123, "hello")
        assert result is sent
        assert bot.send_message.call_count == 3

    async def test_plain_text_retry_after_then_permanent_fail(self) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = [
            TelegramError("md fail"),
            RetryAfter(1),
            TelegramError("plain also dead"),
        ]
        result = await _send_with_fallback(bot, 123, "hello")
        assert result is None
        assert bot.send_message.call_count == 3

    async def test_fallback_strips_mdv2_escapes(self) -> None:
        bot = AsyncMock()
        sent = AsyncMock(spec=Message)
        bot.send_message.side_effect = [TelegramError("parse error"), sent]

        await _send_with_fallback(bot, 123, r"Hello \*world\* from bot\.py")

        assert bot.send_message.call_count == 2
        fallback_text = bot.send_message.call_args_list[1].kwargs["text"]
        assert "\\" not in fallback_text
        assert fallback_text == "Hello *world* from bot.py"


class TestStripMdv2:
    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            pytest.param(r"\*bold\*", "*bold*", id="escaped-asterisks"),
            pytest.param(r"\_italic\_", "_italic_", id="escaped-underscores"),
            pytest.param(r"\~strike\~", "~strike~", id="escaped-tildes"),
            pytest.param(r"\[link\]", "[link]", id="escaped-brackets"),
            pytest.param(r"\(url\)", "(url)", id="escaped-parens"),
            pytest.param(r"\#heading", "#heading", id="escaped-hash"),
            pytest.param(r"\+plus", "+plus", id="escaped-plus"),
            pytest.param(r"\-minus", "-minus", id="escaped-minus"),
            pytest.param(r"\=equals", "=equals", id="escaped-equals"),
            pytest.param(r"\|pipe", "|pipe", id="escaped-pipe"),
            pytest.param(r"\{brace\}", "{brace}", id="escaped-braces"),
            pytest.param(r"\!bang", "!bang", id="escaped-bang"),
            pytest.param("\\\\backslash", "\\backslash", id="escaped-backslash"),
            pytest.param(
                r"src/ccgram/bot\.py", "src/ccgram/bot.py", id="file-path-dot"
            ),
            pytest.param(
                r"src/ccgram/message\_sender\.py",
                "src/ccgram/message_sender.py",
                id="file-path-underscore-dot",
            ),
            pytest.param(">line1\n>line2", "line1\nline2", id="blockquote-prefix"),
            pytest.param(">content||", "content", id="expandable-quote-close"),
            pytest.param(
                ">first||\n>second||",
                "first\nsecond",
                id="expandable-quote-multi",
            ),
            pytest.param("plain text here", "plain text here", id="plain-passthrough"),
            pytest.param(
                "no special chars 123", "no special chars 123", id="no-changes"
            ),
            pytest.param("", "", id="empty-string"),
        ],
    )
    def test_strip_mdv2(self, input_text: str, expected: str) -> None:
        assert strip_mdv2(input_text) == expected
