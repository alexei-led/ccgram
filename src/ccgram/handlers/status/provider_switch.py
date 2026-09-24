"""One quiet notice for a stable provider change, not every poll or startup."""

from dataclasses import dataclass
import time

import structlog
from telegram.error import RetryAfter, TelegramError

from ...telegram_client import TelegramClient
from ...telegram_rate_limiter import retry_after_seconds
from ...topic_state_registry import topic_state
from ...window_state_ports import identity_state
from ..messaging_pipeline.message_sender import rate_limit_send
from ..provider_display import provider_label
from .topic_emoji import sync_topic_name

logger = structlog.get_logger()
_SWITCH_SETTLE_SECONDS = 3.0
_RETRY_SECONDS = 30.0


@dataclass
class _Observation:
    window_id: str
    provider: str
    pending: str = ""
    since: float = 0.0
    retry_at: float = 0.0


_observed: dict[tuple[int, int], _Observation] = {}


def remember_provider_selection(
    chat_id: int, thread_id: int, window_id: str, provider: str
) -> None:
    """The /agent reply already acknowledges this selection; do not echo it."""
    _observed[(chat_id, thread_id)] = _Observation(window_id, provider)


@topic_state.register("chat")
def clear_provider_switch_state(chat_id: int, thread_id: int) -> None:
    _observed.pop((chat_id, thread_id), None)


async def observe_provider_switch(
    client: TelegramClient,
    chat_id: int,
    thread_id: int,
    window_id: str,
    display_name: str,
) -> None:
    provider = identity_state.get_provider_name(window_id)
    if not provider:
        return
    key = (chat_id, thread_id)
    observed = _observed.get(key)
    if observed is None or observed.window_id != window_id:
        remember_provider_selection(chat_id, thread_id, window_id, provider)
        return
    if provider == observed.provider:
        observed.pending = ""
        return
    now = time.monotonic()
    if observed.pending != provider:
        observed.pending, observed.since, observed.retry_at = provider, now, 0.0
        return
    if now - observed.since < _SWITCH_SETTLE_SECONDS or now < observed.retry_at:
        return
    text = f"{provider_label(observed.provider)} → {provider_label(provider)}."
    if provider == "shell":
        text += " Terminal mode: text routes to shell commands."
    await rate_limit_send(chat_id)
    # A manual selection or another hook can supersede us while rate-limited.
    if (
        _observed.get(key) is not observed
        or identity_state.get_provider_name(window_id) != provider
    ):
        return
    try:
        await client.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=text,
            disable_notification=True,
        )
    except RetryAfter as exc:
        observed.retry_at = time.monotonic() + max(
            _RETRY_SECONDS, retry_after_seconds(exc)
        )
        return
    except TelegramError:
        observed.retry_at = time.monotonic() + _RETRY_SECONDS
        logger.debug("Provider notice failed for %s", window_id, exc_info=True)
        return
    observed.provider, observed.pending = provider, ""
    await sync_topic_name(client, chat_id, thread_id, display_name)
