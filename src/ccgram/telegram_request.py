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
    """Reset a Telegram HTTP client after transient transport failures.

    PTB uses one instance for ``getUpdates`` and another shared instance for
    normal Bot API traffic. Rebuilding a stuck client gives the next request a
    fresh pool. Concurrent failures from the same stale client must perform one
    reset only; otherwise a late failure can close the replacement client.

    The first reset after a successful request logs at warning; subsequent
    resets within `_RESET_WARN_INTERVAL_S` log at debug to avoid floods during
    sustained outages.
    """

    def __init__(
        self,
        *args,
        on_success: Callable[[], None] | None = None,
        request_name: str = "Telegram",
        **kwargs,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._on_success = on_success
        self.request_name = request_name
        self._last_reset_warn_ts: float | None = None
        self._reset_lock = asyncio.Lock()
        self._client_close_tasks: set[asyncio.Task[None]] = set()

    async def _reset_client(
        self, *, failed_client: httpx.AsyncClient, reason: str
    ) -> bool:
        async with self._reset_lock:
            if self._client is not failed_client:
                return False
            self._client = self._build_client()

        close_task = asyncio.create_task(failed_client.aclose())
        self._client_close_tasks.add(close_task)

        def discard_close_task(task: asyncio.Task[None]) -> None:
            self._client_close_tasks.discard(task)
            if not task.cancelled():
                task.exception()

        close_task.add_done_callback(discard_close_task)
        try:
            async with asyncio.timeout(1.0):
                await asyncio.shield(close_task)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, RuntimeError, OSError, httpx.HTTPError) as exc:
            logger.debug(
                "Ignoring error while closing stale Telegram client after %s: %s",
                reason,
                exc,
            )
        return True

    def _should_warn_for_reset(self, now: float) -> bool:
        """Throttle: warn once per interval, then debug. Reset by success."""
        if (
            self._last_reset_warn_ts is None
            or now - self._last_reset_warn_ts >= _RESET_WARN_INTERVAL_S
        ):
            self._last_reset_warn_ts = now
            return True
        return False

    async def post(self, *args, **kwargs):  # type: ignore[override]
        result = await super().post(*args, **kwargs)
        # BaseRequest.post validates the Bot API response before returning.
        self._last_reset_warn_ts = None
        if self._on_success is not None:
            self._on_success()
        return result

    async def do_request(self, *args, **kwargs):  # type: ignore[override]
        failed_client = self._client
        try:
            return await super().do_request(*args, **kwargs)
        except (TimedOut, NetworkError) as exc:
            if await self._reset_client(
                failed_client=failed_client, reason=exc.__class__.__name__
            ):
                log = (
                    logger.warning
                    if self._should_warn_for_reset(time.monotonic())
                    else logger.debug
                )
                log(
                    "Reset Telegram HTTP client (%s) after %s: %s",
                    self.request_name,
                    exc.__class__.__name__,
                    exc,
                )
            raise
