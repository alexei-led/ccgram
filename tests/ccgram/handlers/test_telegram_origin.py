import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.handlers.telegram_origin import (
    _PENDING_INJECTION_TTL_S,
    clear_pending_telegram_injections,
    consume_telegram_injection,
    forget_telegram_injection,
    remember_telegram_injection,
    send_telegram_to_window,
)


def setup_function() -> None:
    clear_pending_telegram_injections()


def teardown_function() -> None:
    clear_pending_telegram_injections()


def test_matching_injections_are_consumed_once_in_fifo_order() -> None:
    with patch("ccgram.handlers.telegram_origin.time.monotonic", return_value=100.0):
        remember_telegram_injection(1, "@1", 42, "same")
        remember_telegram_injection(1, "@1", 42, "same")

    with patch("ccgram.handlers.telegram_origin.time.monotonic", return_value=101.0):
        assert consume_telegram_injection(1, "@1", 42, "same") is True
        assert consume_telegram_injection(1, "@1", 42, "same") is True
        assert consume_telegram_injection(1, "@1", 42, "same") is False


def test_matching_skips_unrelated_pending_entries() -> None:
    remember_telegram_injection(1, "@1", 42, "/local-command")
    remember_telegram_injection(1, "@1", 42, "second")

    assert consume_telegram_injection(1, "@1", 42, "second") is True
    assert consume_telegram_injection(1, "@1", 42, "/local-command") is True


def test_failed_send_forgets_exact_reservation() -> None:
    first = remember_telegram_injection(1, "@1", 42, "same")
    second = remember_telegram_injection(1, "@1", 42, "same")

    forget_telegram_injection(1, "@1", 42, second)

    assert consume_telegram_injection(1, "@1", 42, "same") is True
    assert consume_telegram_injection(1, "@1", 42, "same") is False
    forget_telegram_injection(1, "@1", 42, first)


def test_expired_injection_does_not_suppress_terminal_input() -> None:
    with patch("ccgram.handlers.telegram_origin.time.monotonic", return_value=100.0):
        remember_telegram_injection(1, "@1", 42, "hello")

    with patch(
        "ccgram.handlers.telegram_origin.time.monotonic",
        return_value=100.0 + _PENDING_INJECTION_TTL_S,
    ):
        assert consume_telegram_injection(1, "@1", 42, "hello") is False


def test_injection_is_scoped_to_its_bound_topic() -> None:
    remember_telegram_injection(1, "@1", 42, "hello", -100)

    assert consume_telegram_injection(1, "@1", 42, "hello", -200) is False
    assert consume_telegram_injection(1, "@1", 42, "hello", -100) is True


def test_injection_is_scoped_to_chat_when_thread_ids_collide() -> None:
    remember_telegram_injection(1, "@1", 42, "hello", -100)

    assert consume_telegram_injection(1, "@1", 42, "hello", -200) is False
    assert consume_telegram_injection(1, "@1", 42, "hello", -100) is True


def test_matches_provider_trimmed_user_text() -> None:
    remember_telegram_injection(1, "@1", 42, "  hello  \n")

    assert consume_telegram_injection(1, "@1", 42, "hello") is True


@pytest.mark.asyncio
async def test_origin_aware_send_reserves_before_terminal_injection() -> None:
    async def send_after_reservation(
        window_id: str, text: str, *, raw: bool = False
    ) -> tuple[bool, str]:
        assert consume_telegram_injection(1, window_id, 42, text) is True
        return True, "ok"

    with patch(
        "ccgram.handlers.telegram_origin.send_to_window",
        AsyncMock(side_effect=send_after_reservation),
    ):
        assert await send_telegram_to_window(1, "@1", 42, "hello") == (True, "ok")


@pytest.mark.asyncio
async def test_origin_aware_send_rolls_back_on_false_result() -> None:
    with patch(
        "ccgram.handlers.telegram_origin.send_to_window",
        AsyncMock(return_value=(False, "window gone")),
    ):
        assert await send_telegram_to_window(1, "@1", 42, "hello") == (
            False,
            "window gone",
        )

    assert consume_telegram_injection(1, "@1", 42, "hello") is False


@pytest.mark.asyncio
async def test_origin_aware_send_rolls_back_on_cancellation() -> None:
    with (
        patch(
            "ccgram.handlers.telegram_origin.send_to_window",
            AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await send_telegram_to_window(1, "@1", 42, "hello")

    assert consume_telegram_injection(1, "@1", 42, "hello") is False


@pytest.mark.asyncio
async def test_origin_aware_send_rolls_back_on_exception() -> None:
    with (
        patch(
            "ccgram.handlers.telegram_origin.send_to_window",
            AsyncMock(side_effect=RuntimeError("send failed")),
        ),
        pytest.raises(RuntimeError, match="send failed"),
    ):
        await send_telegram_to_window(1, "@1", 42, "hello")

    assert consume_telegram_injection(1, "@1", 42, "hello") is False


def test_pending_correlations_are_not_evicted_before_ttl() -> None:
    for index in range(300):
        remember_telegram_injection(index, f"@{index}", 42, f"message-{index}")
    for index in range(20):
        remember_telegram_injection(999, "@burst", 42, f"burst-{index}")

    assert consume_telegram_injection(0, "@0", 42, "message-0") is True
    assert consume_telegram_injection(999, "@burst", 42, "burst-0") is True
