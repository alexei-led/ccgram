"""Tests for bot-level error handler, shutdown notification, and signal diagnostics."""

import contextlib
import io
import signal
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, Conflict, NetworkError, RetryAfter, TelegramError
from telegram.request import HTTPXRequest

from ccgram.bot import (
    _error_handler,
    _record_successful_poll,
    _reset_polling_conflict_state,
    _send_shutdown_notification,
    polling_conflict_requires_restart,
)
from ccgram.telegram_request import ResilientPollingHTTPXRequest


@pytest.fixture(autouse=True)
def _reset_conflict_state() -> None:
    _reset_polling_conflict_state()


def _make_context(error: BaseException) -> MagicMock:
    ctx = MagicMock()
    ctx.error = error
    return ctx


class TestErrorHandlerNetworkError:
    async def test_network_error_logged_as_info(self) -> None:
        ctx = _make_context(NetworkError("httpx.ConnectError:"))

        with patch("ccgram.bot.logger") as mock_logger:
            await _error_handler(None, ctx)

        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    async def test_retry_after_logged_as_warning_without_traceback(self) -> None:
        ctx = _make_context(RetryAfter(timedelta(seconds=3)))

        with patch("ccgram.bot.logger") as mock_logger:
            await _error_handler(None, ctx)

        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args.kwargs["retry_after_seconds"] == 3.0
        assert "exc_info" not in mock_logger.warning.call_args.kwargs
        mock_logger.error.assert_not_called()


class TestErrorHandlerStaleCallback:
    async def test_bad_request_query_too_old_is_debug_not_error(self) -> None:
        ctx = _make_context(BadRequest("Query is too old and response timeout expired"))

        with patch("ccgram.bot.logger") as mock_logger:
            await _error_handler(None, ctx)

        mock_logger.debug.assert_called_once()
        assert "expired" in mock_logger.debug.call_args[0][0]
        mock_logger.error.assert_not_called()

    async def test_bad_request_query_id_invalid_is_debug(self) -> None:
        ctx = _make_context(BadRequest("query id is invalid and too old"))

        with patch("ccgram.bot.logger") as mock_logger:
            await _error_handler(None, ctx)

        mock_logger.debug.assert_called_once()
        mock_logger.error.assert_not_called()

    async def test_other_bad_request_still_logged_as_error(self) -> None:
        ctx = _make_context(BadRequest("Chat not found"))

        with patch("ccgram.bot.logger") as mock_logger:
            await _error_handler(None, ctx)

        mock_logger.error.assert_called_once()
        mock_logger.debug.assert_not_called()

    async def test_other_telegram_error_logged_as_error(self) -> None:
        ctx = _make_context(TelegramError("Network timeout"))

        with patch("ccgram.bot.logger") as mock_logger:
            await _error_handler(None, ctx)

        mock_logger.error.assert_called_once()

    async def test_conflict_retries_during_grace_period(self) -> None:
        ctx = _make_context(Conflict("409 Conflict"))

        with (
            patch("ccgram.bot.logger") as mock_logger,
            patch("ccgram.bot.time.monotonic", return_value=100.0),
        ):
            await _error_handler(None, ctx)

        ctx.application.stop_running.assert_not_called()
        assert polling_conflict_requires_restart() is False
        mock_logger.warning.assert_called_once()

    async def test_sustained_conflict_stops_for_supervisor_restart(self) -> None:
        ctx = _make_context(Conflict("409 Conflict"))

        with (
            patch("ccgram.bot.logger") as mock_logger,
            patch("ccgram.bot.arm_shutdown_watchdog") as arm_watchdog,
            patch("ccgram.bot.time.monotonic", side_effect=[100.0, 190.0]),
        ):
            await _error_handler(None, ctx)
            await _error_handler(None, ctx)

        ctx.application.stop_running.assert_called_once()
        arm_watchdog.assert_called_once_with()
        assert polling_conflict_requires_restart() is True
        mock_logger.critical.assert_called_once()

    async def test_successful_poll_resets_conflict_grace_period(self) -> None:
        ctx = _make_context(Conflict("409 Conflict"))

        with patch("ccgram.bot.time.monotonic", side_effect=[100.0, 1_000.0]):
            await _error_handler(None, ctx)
            _record_successful_poll()
            await _error_handler(None, ctx)

        ctx.application.stop_running.assert_not_called()
        assert polling_conflict_requires_restart() is False

    async def test_raw_conflicts_accumulate_until_shutdown(self) -> None:
        request = ResilientPollingHTTPXRequest(on_success=_record_successful_poll)
        response = (
            409,
            b'{"ok": false, "error_code": 409, "description": "Conflict"}',
        )
        ctx = _make_context(Conflict("placeholder"))

        with (
            patch.object(HTTPXRequest, "do_request", AsyncMock(return_value=response)),
            patch("ccgram.bot.time.monotonic", side_effect=[100.0, 190.0]),
        ):
            for _ in range(2):
                with pytest.raises(Conflict) as raised:
                    await request.post("https://example.com")
                ctx.error = raised.value
                await _error_handler(None, ctx)

        ctx.application.stop_running.assert_called_once()
        assert polling_conflict_requires_restart() is True


class TestShutdownExitWatchdog:
    async def test_post_stop_arms_and_post_shutdown_cancels(self) -> None:
        # The v4.13 watchdog is armed via arm_shutdown_watchdog() from
        # post_stop (the common funnel) and disarmed in post_shutdown.
        from ccgram import bot
        from ccgram.bot import post_shutdown, post_stop

        bot._shutdown_watchdog = None
        timer = MagicMock()
        app = MagicMock()
        with (
            patch("ccgram.bot.threading.Timer", return_value=timer) as timer_cls,
            patch("ccgram.bot.bootstrap.stop_delivery_runtime", new=AsyncMock()),
            patch("ccgram.bot._send_shutdown_notification", new=AsyncMock()) as goodbye,
            patch("ccgram.bot.bootstrap.shutdown_runtime", new=AsyncMock()),
        ):
            await post_stop(app)
            timer_cls.assert_called_once_with(
                bot._SHUTDOWN_WATCHDOG_SECONDS,
                bot._force_exit_after_wedged_shutdown,
            )
            assert timer.daemon is True
            timer.start.assert_called_once()
            goodbye.assert_awaited_once()
            assert bot._shutdown_watchdog is timer

            await post_shutdown(app)
        timer.cancel.assert_called_once()
        assert bot._shutdown_watchdog is None

    async def test_forced_exit_runs_bounded_flush_and_exits(self) -> None:
        from ccgram.bot import _force_exit_after_wedged_shutdown

        with (
            patch("ccgram.bot.os._exit") as exit_mock,
            patch("ccgram.bot.threading.Thread") as thread_cls,
        ):
            _force_exit_after_wedged_shutdown()
        exit_mock.assert_called_once_with(1)
        from ccgram.bot import _WATCHDOG_FLUSH_SECONDS

        thread_cls.assert_called_once_with(
            target=thread_cls.call_args.kwargs["target"], daemon=True
        )
        flusher = thread_cls.return_value
        flusher.start.assert_called_once()
        flusher.join.assert_called_once_with(timeout=_WATCHDOG_FLUSH_SECONDS)

    async def test_conflict_stop_arms_the_watchdog_too(self) -> None:
        # The conflict stop path never reaches the signal handler, so
        # the watchdog is armed right there (b0f01ed0): a wedge in
        # stop() must not leave the supervisor waiting.
        ctx = _make_context(Conflict("409 Conflict"))
        with (
            patch("ccgram.bot.threading.Timer") as timer_cls,
            patch("ccgram.bot.time.monotonic", side_effect=[100.0, 190.0]),
        ):
            await _error_handler(None, ctx)
            await _error_handler(None, ctx)
        ctx.application.stop_running.assert_called_once()
        timer_cls.assert_called_once()


class TestShutdownNotification:
    async def test_sends_to_general_topic(self) -> None:
        app = MagicMock()
        app.bot.send_message = AsyncMock()

        with (
            patch("ccgram.bot.config") as mock_config,
            patch("ccgram.main._shutdown_signal", signal.SIGINT),
        ):
            mock_config.group_id = -100123
            await _send_shutdown_notification(app)

        app.bot.send_message.assert_called_once()
        call_kwargs = app.bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == -100123
        assert "message_thread_id" not in call_kwargs
        assert "SIGINT" in call_kwargs["text"]

    async def test_skipped_without_group_id(self) -> None:
        app = MagicMock()
        app.bot.send_message = AsyncMock()

        with patch("ccgram.bot.config") as mock_config:
            mock_config.group_id = None
            await _send_shutdown_notification(app)

        app.bot.send_message.assert_not_called()

    async def test_clean_exit_reason(self) -> None:
        app = MagicMock()
        app.bot.send_message = AsyncMock()

        with (
            patch("ccgram.bot.config") as mock_config,
            patch("ccgram.main._shutdown_signal", 0),
        ):
            mock_config.group_id = -100123
            await _send_shutdown_notification(app)

        text = app.bot.send_message.call_args.kwargs["text"]
        assert "Clean exit" in text

    async def test_send_failure_does_not_crash(self) -> None:
        app = MagicMock()
        app.bot.send_message = AsyncMock(side_effect=TelegramError("forbidden"))

        with (
            patch("ccgram.bot.config") as mock_config,
            patch("ccgram.main._shutdown_signal", 0),
        ):
            mock_config.group_id = -100123
            await _send_shutdown_notification(app)


class TestSignalDiagnostics:
    def test_signal_handler_records_signum_and_raises(self) -> None:
        from ccgram import main

        stderr_capture = io.StringIO()
        with (
            patch.object(main, "_shutdown_signal", 0),
            patch("ccgram.bot.arm_shutdown_watchdog") as arm_watchdog,
            patch("sys.stderr", stderr_capture),
        ):
            with contextlib.suppress(SystemExit):
                main._on_signal(signal.SIGINT)
            assert main._shutdown_signal == signal.SIGINT

        arm_watchdog.assert_called_once_with(exit_code=128 + signal.SIGINT)

        output = stderr_capture.getvalue()
        assert "SIGINT" in output
        assert "call stack" not in output

    def test_install_uses_loop_add_signal_handler(self) -> None:
        from ccgram.main import _install_signal_handlers, _on_signal

        loop = MagicMock()
        _install_signal_handlers(loop)

        registered = {call.args[0] for call in loop.add_signal_handler.call_args_list}
        assert registered == {signal.SIGINT, signal.SIGTERM, signal.SIGQUIT}
        for call in loop.add_signal_handler.call_args_list:
            assert call.args[1] is _on_signal
            assert call.args[2] == call.args[0]
