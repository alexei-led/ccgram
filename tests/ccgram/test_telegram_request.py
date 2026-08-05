"""Tests for resilient Telegram polling requests."""

from pathlib import Path
import tomllib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import NetworkError, TimedOut
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

    async def test_calls_success_callback_after_successful_request(self) -> None:
        on_success = MagicMock()
        request = ResilientPollingHTTPXRequest(on_success=on_success)

        with patch.object(HTTPXRequest, "do_request", AsyncMock(return_value=b"ok")):
            assert await request.do_request("https://example.com", "POST") == b"ok"

        on_success.assert_called_once()


def _reset_log_calls(mock_logger, level: str) -> list:
    return [
        c
        for c in getattr(mock_logger, level).call_args_list
        if c.args and "Reset Telegram polling" in c.args[0]
    ]


class TestResetWarningRateLimit:
    async def test_first_reset_warns(self) -> None:
        request = ResilientPollingHTTPXRequest()
        with (
            patch.object(
                HTTPXRequest,
                "do_request",
                AsyncMock(side_effect=TimedOut("t")),
            ),
            patch("ccgram.telegram_request.logger") as mock_logger,
            pytest.raises(TimedOut),
        ):
            await request.do_request("https://example.com", "POST")
        assert len(_reset_log_calls(mock_logger, "warning")) == 1
        assert _reset_log_calls(mock_logger, "debug") == []

    async def test_repeated_resets_within_interval_demoted_to_debug(self) -> None:
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
                with pytest.raises(TimedOut):
                    await request.do_request("https://example.com", "POST")

        assert len(_reset_log_calls(mock_logger, "warning")) == 1
        assert len(_reset_log_calls(mock_logger, "debug")) == 4

    async def test_success_resets_warn_eligibility(self) -> None:
        request = ResilientPollingHTTPXRequest()
        sentinel = object()
        mock = AsyncMock(side_effect=[TimedOut("t"), sentinel, TimedOut("t")])

        with (
            patch.object(HTTPXRequest, "do_request", mock),
            patch("ccgram.telegram_request.logger") as mock_logger,
        ):
            with pytest.raises(TimedOut):
                await request.do_request("u", "POST")
            await request.do_request("u", "POST")
            with pytest.raises(TimedOut):
                await request.do_request("u", "POST")

        assert len(_reset_log_calls(mock_logger, "warning")) == 2


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
