"""Herdr event-stream acknowledgement regressions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.multiplexer.herdr_events import SUBSCRIBED, open_socket_stream


async def test_subscribed_sentinel_requires_successful_ack() -> None:
    reader = MagicMock()
    reader.readline = AsyncMock(return_value=b'{"error":{"message":"no"}}\n')
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.wait_closed = AsyncMock()

    with patch(
        "ccgram.multiplexer.herdr_events.asyncio.open_unix_connection",
        new=AsyncMock(return_value=(reader, writer)),
    ):
        stream = open_socket_stream("/tmp/herdr.sock", [])
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    writer.close.assert_called_once()


async def test_subscribed_sentinel_follows_successful_ack() -> None:
    reader = MagicMock()
    reader.readline = AsyncMock(return_value=b'{"result":{}}\n')
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.wait_closed = AsyncMock()

    with patch(
        "ccgram.multiplexer.herdr_events.asyncio.open_unix_connection",
        new=AsyncMock(return_value=(reader, writer)),
    ):
        stream = open_socket_stream("/tmp/herdr.sock", [])
        assert await anext(stream) == SUBSCRIBED
        await stream.aclose()
