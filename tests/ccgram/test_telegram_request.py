"""Tests for resilient Telegram polling requests."""

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


class TestCreateBotPollingRequest:
    @patch("ccgram.bot.config")
    def test_uses_resilient_request_for_get_updates(
        self, mock_config: MagicMock
    ) -> None:
        mock_config.telegram_bot_token = "fake:token"

        app = create_bot()

        assert isinstance(app.bot._request[0], ResilientPollingHTTPXRequest)
        assert isinstance(app.bot._request[1], HTTPXRequest)
        assert not isinstance(app.bot._request[1], ResilientPollingHTTPXRequest)
        assert app.bot._request[0]._client._transport._pool._max_connections == 1
