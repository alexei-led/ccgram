"""Telegram rate limiting with endpoint-controlled RetryAfter handling."""

import contextlib
from collections.abc import Callable, Coroutine
from typing import Any

from telegram.ext import AIORateLimiter


class CCGramAIORateLimiter(AIORateLimiter):
    """Allow selected requests to propagate the first RetryAfter response.

    PTB 22.x treats ``rate_limit_args=0`` as falsy and falls back to the global
    retry count. Calling ``_run_request`` directly preserves PTB's proactive
    rate limits while letting ccgram's endpoint-specific backoff handle the
    first flood-control response.
    """

    async def process_request(
        self,
        callback: Callable[..., Coroutine[Any, Any, Any]],
        args: Any,
        kwargs: dict[str, Any],
        endpoint: str,
        data: dict[str, Any],
        rate_limit_args: int | None,
    ) -> Any:
        if rate_limit_args != 0:
            return await super().process_request(
                callback=callback,
                args=args,
                kwargs=kwargs,
                endpoint=endpoint,
                data=data,
                rate_limit_args=rate_limit_args,
            )

        chat_id = data.get("chat_id")
        if chat_id is not None:
            with contextlib.suppress(TypeError, ValueError):
                chat_id = int(chat_id)
        group: int | str | bool = False
        if (isinstance(chat_id, int) and chat_id < 0) or isinstance(chat_id, str):
            group = chat_id

        return await self._run_request(
            chat=chat_id is not None,
            group=group,
            allow_paid_broadcast=data.get("allow_paid_broadcast", False),
            callback=callback,
            args=args,
            kwargs=kwargs,
        )
