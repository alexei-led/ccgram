"""Telegram request helpers for resilient long polling."""

import asyncio
import time
from collections.abc import Callable

import httpx
import structlog
from telegram.error import NetworkError, TimedOut
from telegram.request import HTTPXRequest

logger = structlog.get_logger()

# Minimum interval between reset warnings during a sustained outage.
# Without this, every failed poll (~5s apart) emits a warning, flooding logs.
_RESET_WARN_INTERVAL_S: float = 30.0


class ResilientPollingHTTPXRequest(HTTPXRequest):
    """Reset the polling HTTP client after transient transport failures.

    PTB uses a dedicated request object for ``getUpdates`` with a single
    connection. If that connection gets stuck in a bad proxy/tunnel state,
    subsequent polls can queue behind it forever. Rebuilding the client after a
    timeout/network failure gives the polling loop a fresh pool on the next
    retry.

    The first reset after a successful request logs at warning; subsequent
    resets within `_RESET_WARN_INTERVAL_S` log at debug to avoid floods during
    sustained outages.
    """

    def __init__(
        self, *args, on_success: Callable[[], None] | None = None, **kwargs
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._on_success = on_success
        # 0.0 means "no warn yet" — first reset will warn.
        self._last_reset_warn_ts: float = 0.0

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

    def _should_warn_for_reset(self, now: float) -> bool:
        """Throttle: warn once per interval, then debug. Reset by success."""
        if now - self._last_reset_warn_ts >= _RESET_WARN_INTERVAL_S:
            self._last_reset_warn_ts = now
            return True
        return False

    async def do_request(self, *args, **kwargs):  # type: ignore[override]
        try:
            result = await super().do_request(*args, **kwargs)
        except (TimedOut, NetworkError) as exc:
            await self._reset_client(reason=exc.__class__.__name__)
            log = (
                logger.warning
                if self._should_warn_for_reset(time.monotonic())
                else logger.debug
            )
            log(
                "Reset Telegram polling HTTP client after %s: %s",
                exc.__class__.__name__,
                exc,
            )
            raise
        else:
            # Successful request — next reset gets to warn again.
            self._last_reset_warn_ts = 0.0
            if self._on_success is not None:
                self._on_success()
            return result
