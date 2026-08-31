import asyncio
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


async def test_retry_after_does_not_pause_unrelated_chat() -> None:
    limiter = CCGramAIORateLimiter(max_retries=1)
    retry_sleep_started = asyncio.Event()
    release_retry = asyncio.Event()
    retrying_callback = AsyncMock(
        side_effect=[RetryAfter(timedelta(seconds=3)), "retried"]
    )
    unrelated_callback = AsyncMock(return_value="unrelated")

    async def controlled_sleep(_delay: float) -> None:
        retry_sleep_started.set()
        await release_retry.wait()

    with (
        patch("ccgram.telegram_rate_limiter.asyncio.sleep", controlled_sleep),
        patch("random.uniform", return_value=0),
    ):
        retrying_task = asyncio.create_task(
            limiter.process_request(
                callback=retrying_callback,
                args=(),
                kwargs={},
                endpoint="editForumTopic",
                data={"chat_id": 1001},
                rate_limit_args=None,
            )
        )
        await asyncio.wait_for(retry_sleep_started.wait(), timeout=0.5)
        unrelated_task = asyncio.create_task(
            limiter.process_request(
                callback=unrelated_callback,
                args=(),
                kwargs={},
                endpoint="sendMessage",
                data={"chat_id": 1002},
                rate_limit_args=None,
            )
        )

        try:
            result = await asyncio.wait_for(asyncio.shield(unrelated_task), timeout=0.1)
        finally:
            release_retry.set()
            await asyncio.gather(retrying_task, unrelated_task)

    assert result == "unrelated"
    unrelated_callback.assert_awaited_once()
    assert retrying_callback.await_count == 2


async def test_concurrent_retry_after_waits_start_independently() -> None:
    limiter = CCGramAIORateLimiter(max_retries=1)
    started = {4.0: asyncio.Event(), 6.0: asyncio.Event()}
    release = {4.0: asyncio.Event(), 6.0: asyncio.Event()}
    callback_a = AsyncMock(side_effect=[RetryAfter(timedelta(seconds=3)), "a"])
    callback_b = AsyncMock(side_effect=[RetryAfter(timedelta(seconds=5)), "b"])

    async def controlled_sleep(delay: float) -> None:
        started[delay].set()
        await release[delay].wait()

    with (
        patch("ccgram.telegram_rate_limiter.asyncio.sleep", controlled_sleep),
        patch("random.uniform", return_value=0),
    ):
        task_a = asyncio.create_task(
            limiter.process_request(
                callback=callback_a,
                args=(),
                kwargs={},
                endpoint="editForumTopic",
                data={"chat_id": 1001},
                rate_limit_args=None,
            )
        )
        await asyncio.wait_for(started[4.0].wait(), timeout=0.5)
        task_b = asyncio.create_task(
            limiter.process_request(
                callback=callback_b,
                args=(),
                kwargs={},
                endpoint="sendMessage",
                data={"chat_id": 1002},
                rate_limit_args=None,
            )
        )

        try:
            await asyncio.wait_for(started[6.0].wait(), timeout=0.1)
            release[6.0].set()
            assert await asyncio.wait_for(asyncio.shield(task_b), timeout=0.1) == "b"
            assert not task_a.done()
        finally:
            release[4.0].set()
            release[6.0].set()
            await asyncio.gather(task_a, task_b)

    assert callback_a.await_count == 2
    assert callback_b.await_count == 2


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
    assert callback.await_count == 2
    assert logger.warning.call_count == 2
    assert all("exc_info" not in item.kwargs for item in logger.warning.call_args_list)
