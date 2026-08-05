from unittest.mock import patch

from ccgram.handlers.telegram_origin import (
    _PENDING_INJECTION_TTL_S,
    clear_pending_telegram_injections,
    consume_telegram_injection,
    forget_telegram_injection,
    remember_telegram_injection,
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


def test_out_of_order_text_does_not_consume_later_injection() -> None:
    remember_telegram_injection(1, "@1", 42, "first")
    remember_telegram_injection(1, "@1", 42, "second")

    assert consume_telegram_injection(1, "@1", 42, "second") is False
    assert consume_telegram_injection(1, "@1", 42, "first") is True
    assert consume_telegram_injection(1, "@1", 42, "second") is True


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
    remember_telegram_injection(1, "@1", 42, "hello")

    assert consume_telegram_injection(1, "@1", 43, "hello") is False
    assert consume_telegram_injection(1, "@1", 42, "hello") is True
