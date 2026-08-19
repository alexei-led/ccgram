"""Tests for resilient Telegram polling requests."""

import asyncio
from pathlib import Path
import tomllib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import Conflict, NetworkError, TimedOut
from telegram.request import HTTPXRequest

from ccgram.bot import create_bot
from ccgram.telegram_request import ResilientPollingHTTPXRequest


class TestResilientPollingHTTPXRequest:
    async def test_rebuilds_client_after_timeout(self) -> None:
        request = ResilientPollingHTTPXRequest()
        old_client = request._client

        with (
            patch.object(
                HTTPXRequest,
                "do_request",
                AsyncMock(side_effect=TimedOut("pool timeout")),
            ),
            pytest.raises(TimedOut),
        ):
            await request.do_request("https://example.com", "POST")

        assert request._client is not old_client
        assert old_client.is_closed
        assert not request._client.is_closed

    async def test_rebuilds_client_after_network_error(self) -> None:
        request = ResilientPollingHTTPXRequest()
        old_client = request._client

        with (
            patch.object(
                HTTPXRequest,
                "do_request",
                AsyncMock(side_effect=NetworkError("proxy broken")),
            ),
            pytest.raises(NetworkError),
        ):
            await request.do_request("https://example.com", "POST")

        assert request._client is not old_client
        assert old_client.is_closed
        assert not request._client.is_closed

    async def test_concurrent_failures_reset_shared_client_once(self) -> None:
        request = ResilientPollingHTTPXRequest()
        old_client = request._client
        both_entered = asyncio.Event()
        release = asyncio.Event()
        entered = 0

        async def fail_together(*_args, **_kwargs) -> None:
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            raise TimedOut("shared client failed")

        with (
            patch.object(
                HTTPXRequest,
                "do_request",
                AsyncMock(side_effect=fail_together),
            ),
            patch.object(
                request, "_build_client", wraps=request._build_client
            ) as mock_build,
        ):
            calls = [
                asyncio.create_task(request.do_request("https://example.com", "POST"))
                for _ in range(2)
            ]
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            release.set()
            results = await asyncio.gather(*calls, return_exceptions=True)

        assert all(isinstance(result, TimedOut) for result in results)
        assert mock_build.call_count == 1
        assert request._client is not old_client
        assert old_client.is_closed
        assert not request._client.is_closed

    async def test_cancelled_reset_finishes_closing_stale_client(self) -> None:
        request = ResilientPollingHTTPXRequest()
        old_client = request._client
        close_started = asyncio.Event()
        release_close = asyncio.Event()
        close_finished = asyncio.Event()

        async def slow_close() -> None:
            close_started.set()
            await release_close.wait()
            close_finished.set()

        with patch.object(old_client, "aclose", AsyncMock(side_effect=slow_close)):
            reset = asyncio.create_task(
                request._reset_client(failed_client=old_client, reason="TimedOut")
            )
            await asyncio.wait_for(close_started.wait(), timeout=1)
            reset.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reset

            assert request._client_close_tasks
            release_close.set()
            await asyncio.wait_for(close_finished.wait(), timeout=1)
            await asyncio.sleep(0)

        assert not request._client_close_tasks
        assert request._client is not old_client

    async def test_calls_success_callback_after_successful_api_response(self) -> None:
        on_success = MagicMock()
        request = ResilientPollingHTTPXRequest(on_success=on_success)
        response = (200, b'{"ok": true, "result": []}')

        with patch.object(HTTPXRequest, "do_request", AsyncMock(return_value=response)):
            assert await request.post("https://example.com") == []

        on_success.assert_called_once()

    async def test_conflict_does_not_call_success_callback(self) -> None:
        on_success = MagicMock()
        request = ResilientPollingHTTPXRequest(on_success=on_success)
        response = (
            409,
            b'{"ok": false, "error_code": 409, "description": "Conflict"}',
        )

        with (
            patch.object(HTTPXRequest, "do_request", AsyncMock(return_value=response)),
            pytest.raises(Conflict),
        ):
            await request.post("https://example.com")

        on_success.assert_not_called()


def _reset_log_calls(mock_logger, level: str) -> list:
    return [
        c
        for c in getattr(mock_logger, level).call_args_list
        if c.args and "Reset Telegram HTTP client" in c.args[0]
    ]


class TestResetWarningRateLimit:
    async def _fail(self, request) -> None:
        with pytest.raises(TimedOut):
            await request.do_request("https://example.com", "POST")

    async def test_isolated_reset_logs_info_not_warning(self) -> None:
        """A single dropped connection is routine; PTB recovers on the next try."""
        request = ResilientPollingHTTPXRequest()
        with (
            patch.object(
                HTTPXRequest,
                "do_request",
                AsyncMock(side_effect=TimedOut("t")),
            ),
            patch("ccgram.telegram_request.logger") as mock_logger,
        ):
            await self._fail(request)

        assert _reset_log_calls(mock_logger, "warning") == []
        assert len(_reset_log_calls(mock_logger, "info")) == 1

    async def test_sustained_outage_warns_once_per_interval(self) -> None:
        request = ResilientPollingHTTPXRequest()
        with (
            patch.object(
                HTTPXRequest,
                "do_request",
                AsyncMock(side_effect=TimedOut("t")),
            ),
            patch("ccgram.telegram_request.logger") as mock_logger,
        ):
            for _ in range(5):
                await self._fail(request)

        # Resets 1-2 are info; reset 3 crosses the threshold and warns; the
        # rest fall back to info until the warn interval elapses.
        assert len(_reset_log_calls(mock_logger, "warning")) == 1
        assert len(_reset_log_calls(mock_logger, "info")) == 4

    async def test_success_resets_consecutive_counter(self) -> None:
        request = ResilientPollingHTTPXRequest()
        response = (200, b'{"ok": true, "result": []}')
        mock = AsyncMock(
            side_effect=[TimedOut("t"), TimedOut("t"), response, TimedOut("t")]
        )

        with (
            patch.object(HTTPXRequest, "do_request", mock),
            patch("ccgram.telegram_request.logger") as mock_logger,
        ):
            for _ in range(2):
                with pytest.raises(TimedOut):
                    await request.post("u")
            await request.post("u")
            with pytest.raises(TimedOut):
                await request.post("u")

        # The success cleared the streak, so the third failure is isolated.
        assert _reset_log_calls(mock_logger, "warning") == []

    async def test_sustained_outage_warns_before_monotonic_interval(self) -> None:
        """The first warning must not wait out the interval from process start."""
        request = ResilientPollingHTTPXRequest()
        with (
            patch.object(
                HTTPXRequest,
                "do_request",
                AsyncMock(side_effect=TimedOut("t")),
            ),
            patch("ccgram.telegram_request.time.monotonic", return_value=1.0),
            patch("ccgram.telegram_request.logger") as mock_logger,
        ):
            for _ in range(3):
                await self._fail(request)

        assert len(_reset_log_calls(mock_logger, "warning")) == 1


class TestCreateBotPollingRequest:
    @patch("ccgram.bot.config")
    def test_uses_resilient_request_for_telegram_traffic(
        self, mock_config: MagicMock
    ) -> None:
        mock_config.telegram_bot_token = "fake:token"

        app = create_bot()

        assert isinstance(app.bot._request[0], ResilientPollingHTTPXRequest)
        assert isinstance(app.bot._request[1], ResilientPollingHTTPXRequest)
        assert app.bot._request[0]._client._transport._pool._max_connections == 1
        assert app.bot._request[1]._client._transport._pool._max_connections == 256
        assert app.bot._request[0].read_timeout == 10
        assert app.bot._request[1].read_timeout == 10
        assert app.bot._request[0].request_name == "getUpdates"
        assert app.bot._request[1].request_name == "Bot API"


class TestProjectDependencies:
    def test_declares_ptb_socks_support(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())
        dependencies = pyproject["project"]["dependencies"]

        assert any(
            dependency.startswith("python-telegram-bot[")
            and "socks" in dependency.partition("[")[2].partition("]")[0].split(",")
            for dependency in dependencies
        )
