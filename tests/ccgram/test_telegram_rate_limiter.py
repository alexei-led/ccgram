from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import RetryAfter

from ccgram.telegram_rate_limiter import CCGramAIORateLimiter


async def test_zero_retry_override_propagates_first_retry_after() -> None:
    limiter = CCGramAIORateLimiter(max_retries=5)
    callback = AsyncMock(side_effect=RetryAfter(3))

    with pytest.raises(RetryAfter):
        await limiter.process_request(
            callback=callback,
            args=(),
            kwargs={},
            endpoint="unpinAllForumTopicMessages",
            data={"chat_id": 42},
            rate_limit_args=0,
        )

    callback.assert_awaited_once()


async def test_default_retry_policy_is_preserved() -> None:
    limiter = CCGramAIORateLimiter(max_retries=1)
    callback = AsyncMock(side_effect=[RetryAfter(timedelta(milliseconds=1)), True])

    with patch("telegram.ext._aioratelimiter.asyncio.sleep", new=AsyncMock()):
        result = await limiter.process_request(
            callback=callback,
            args=(),
            kwargs={},
            endpoint="sendMessage",
            data={"chat_id": 42},
            rate_limit_args=None,
        )

    assert result is True
    assert callback.await_count == 2
