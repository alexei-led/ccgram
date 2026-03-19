"""Telegram request helpers for resilient long polling."""

import asyncio

import httpx
import structlog
from telegram.error import NetworkError, TimedOut
from telegram.request import HTTPXRequest

logger = structlog.get_logger()


class ResilientPollingHTTPXRequest(HTTPXRequest):
    """Reset the polling HTTP client after transient transport failures.

    PTB uses a dedicated request object for ``getUpdates`` with a single
    connection. If that connection gets stuck in a bad proxy/tunnel state,
    subsequent polls can queue behind it forever. Rebuilding the client after a
    timeout/network failure gives the polling loop a fresh pool on the next
    retry.
    """

    async def _reset_client(self, *, reason: str) -> None:
        old_client = self._client
        self._client = self._build_client()

        try:
            async with asyncio.timeout(1.0):
                await old_client.aclose()
        except (TimeoutError, RuntimeError, OSError, httpx.HTTPError) as exc:
            logger.debug(
                "Ignoring error while closing stale polling client after %s: %s",
                reason,
                exc,
            )

    async def do_request(self, *args, **kwargs):  # type: ignore[override]
        try:
            return await super().do_request(*args, **kwargs)
        except (TimedOut, NetworkError) as exc:
            await self._reset_client(reason=exc.__class__.__name__)
            logger.warning(
                "Reset Telegram polling HTTP client after %s: %s",
                exc.__class__.__name__,
                exc,
            )
            raise
