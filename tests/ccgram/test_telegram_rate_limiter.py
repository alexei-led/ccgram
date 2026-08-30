from datetime import timedelta
from unittest.mock import AsyncMock, call, patch

import pytest
from telegram.error import RetryAfter
from telegram.ext import ExtBot

from ccgram.telegram_rate_limiter import (
    CCGramAIORateLimiter,
    NO_RETRY_RATE_LIMIT_ARGS,
)


@pytest.mark.parametrize("rate_limit_args", [0, -2, NO_RETRY_RATE_LIMIT_ARGS])
async def test_no_retry_override_propagates_first_retry_after(
    rate_limit_args: int,
) -> None:
    limiter = CCGramAIORateLimiter(max_retries=5)
    callback = AsyncMock(side_effect=RetryAfter(timedelta(seconds=3)))

    with pytest.raises(RetryAfter):
        await limiter.process_request(
            callback=callback,
            args=(),
            kwargs={},
            endpoint="unpinAllForumTopicMessages",
            data={"chat_id": 42},
            rate_limit_args=rate_limit_args,
        )

    callback.assert_awaited_once()


def test_no_retry_sentinel_survives_extbot_transport() -> None:
    api_kwargs = ExtBot._merge_api_rl_kwargs(  # pyright: ignore[reportPrivateUsage]
        None, NO_RETRY_RATE_LIMIT_ARGS
    )

    assert api_kwargs is not None
    assert NO_RETRY_RATE_LIMIT_ARGS in api_kwargs.values()


async def test_default_retry_policy_uses_exponential_backoff_with_jitter() -> None:
    limiter = CCGramAIORateLimiter(max_retries=3)
    callback = AsyncMock(
        side_effect=[
            RetryAfter(timedelta(seconds=3)),
            RetryAfter(timedelta(seconds=3)),
            RetryAfter(timedelta(seconds=3)),
            True,
        ]
    )

    with (
        patch("telegram.ext._aioratelimiter.asyncio.sleep", new=AsyncMock()) as sleep,
        patch("ccgram.telegram_rate_limiter.logger", create=True) as logger,
        patch("random.uniform", return_value=0.25),
    ):
        result = await limiter.process_request(
            callback=callback,
            args=(),
            kwargs={},
            endpoint="sendMessage",
            data={"chat_id": 42},
            rate_limit_args=None,
        )

    assert result is True
    assert callback.await_count == 4
    assert sleep.await_args_list == [call(4.25), call(5.25), call(7.25)]
    assert logger.warning.call_count == 3
    assert all("exc_info" not in item.kwargs for item in logger.warning.call_args_list)


async def test_retry_exhaustion_warns_without_ptb_exception_traceback() -> None:
    limiter = CCGramAIORateLimiter(max_retries=1)
    callback = AsyncMock(side_effect=RetryAfter(timedelta(seconds=3)))

    with (
        patch("telegram.ext._aioratelimiter.asyncio.sleep", new=AsyncMock()),
        patch("telegram.ext._aioratelimiter._LOGGER.exception") as ptb_exception,
        patch("ccgram.telegram_rate_limiter.logger", create=True) as logger,
        pytest.raises(RetryAfter),
    ):
        await limiter.process_request(
            callback=callback,
            args=(),
            kwargs={},
            endpoint="sendMessage",
            data={"chat_id": 42},
            rate_limit_args=None,
        )

    ptb_exception.assert_not_called()
    assert logger.warning.call_count == 2
    assert all("exc_info" not in item.kwargs for item in logger.warning.call_args_list)
