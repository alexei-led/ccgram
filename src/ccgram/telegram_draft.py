"""Best-effort streaming delivery through Telegram Bot API 9.5 drafts.

``sendMessageDraft`` creates a temporary preview. It returns ``True`` and has
no message id. A normal ``sendMessage`` is required to persist the final text.
For unsupported peers or older Bot API servers, ``DraftStream`` falls back to
``send_message`` plus ``edit_message_text``.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from datetime import timedelta
import time
from typing import Any, Final, Literal

import structlog
from telegram import Bot, InlineKeyboardMarkup
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError, TimedOut

from .utils import log_throttled

_KEEP_MARKUP: Final[Any] = object()
_MAX_LEN: Final[int] = 4096
_DEGRADE_AFTER_FAILURES: Final[int] = 2
_MIN_DRAFT_INTERVAL: Final[float] = 0.35
_UNSUPPORTED_MARKERS: Final[tuple[str, ...]] = (
    "method not found",
    "method is not implemented",
    "unknown method",
    "endpoint not found",
)
_PEER_INVALID_MARKERS: Final[tuple[str, ...]] = (
    "draft_peer_invalid",
    "peer_invalid",
    "chat not found",
)

logger = structlog.get_logger()

__all__ = [
    "DRAFT_LEGACY",
    "DRAFT_STREAMING",
    "DRAFT_UNSET",
    "DraftStream",
    "is_draft_unavailable",
    "is_peer_draft_unsupported",
    "mark_draft_unavailable",
    "mark_peer_draft_unsupported",
    "reset_draft_state",
]

DRAFT_STREAMING: Final[str] = "streaming"
DRAFT_LEGACY: Final[str] = "legacy"
DRAFT_UNSET: Final[str] = "unset"

_DRAFT_UNAVAILABLE = False
_DRAFT_REASON = ""
_UNSUPPORTED_PEERS: set[tuple[int, int | None]] = set()


def is_draft_unavailable() -> bool:
    """Return whether the Bot API draft method is globally unavailable."""
    return _DRAFT_UNAVAILABLE


def mark_draft_unavailable(reason: str = "") -> None:
    """Disable draft probing after a server reports an unknown method."""
    global _DRAFT_UNAVAILABLE, _DRAFT_REASON
    if not _DRAFT_UNAVAILABLE:
        _DRAFT_UNAVAILABLE = True
        _DRAFT_REASON = reason
        logger.info("Draft streaming disabled: %s", reason or "no reason given")


def draft_unavailable_reason() -> str:
    """Return the first reason recorded for global draft disablement."""
    return _DRAFT_REASON


def reset_draft_state() -> None:
    """Reset availability caches. Intended for tests and process restart."""
    global _DRAFT_UNAVAILABLE, _DRAFT_REASON
    _DRAFT_UNAVAILABLE = False
    _DRAFT_REASON = ""
    _UNSUPPORTED_PEERS.clear()


def is_peer_draft_unsupported(chat_id: int, thread_id: int | None) -> bool:
    """Return whether this chat/topic previously rejected drafts."""
    return (chat_id, thread_id) in _UNSUPPORTED_PEERS


def mark_peer_draft_unsupported(chat_id: int, thread_id: int | None) -> None:
    """Cache a peer-specific draft rejection."""
    _UNSUPPORTED_PEERS.add((chat_id, thread_id))


def _is_unsupported_error(exc: BadRequest) -> bool:
    message = (exc.message or "").lower()
    return any(marker in message for marker in _UNSUPPORTED_MARKERS)


def _is_peer_invalid_error(exc: BadRequest) -> bool:
    message = (exc.message or "").lower()
    return any(marker in message for marker in _PEER_INVALID_MARKERS)


def _retry_after_seconds(exc: RetryAfter) -> float:
    retry_after = exc.retry_after
    if isinstance(retry_after, timedelta):
        return retry_after.total_seconds()
    return float(retry_after)


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_LEN else text[:_MAX_LEN]


class DraftStream:
    """Accumulate text and deliver it as a Telegram draft or editable message.

    The streaming lifecycle is ``start`` → ``append``/``replace``* →
    ``finalize``. In streaming mode ``start`` returns ``None`` because a draft
    has no message id. ``finalize`` sends the persisted message and stores its
    id. In legacy mode ``start`` returns the initial message id.
    """

    def __init__(
        self,
        bot: Bot,
        chat_id: int,
        *,
        message_thread_id: int | None = None,
        reply_to_message_id: int | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._thread_id = message_thread_id
        self._reply_to = reply_to_message_id
        self._reply_markup = reply_markup
        self._draft_id = secrets.randbelow(2_147_483_646) + 1
        self._message_id: int | None = None
        self._buffer = ""
        self._mode: Literal["streaming", "legacy", "unset"] = DRAFT_UNSET  # type: ignore[assignment]
        self._closed = False
        self._stream_failures = 0
        self._last_draft_at = 0.0
        self._retry_not_before = 0.0
        self._pending_flush: asyncio.Task[None] | None = None
        self._update_lock = asyncio.Lock()

    @property
    def message_id(self) -> int | None:
        """Return the persisted message id, if one exists."""
        return self._message_id

    @property
    def mode(self) -> str:
        """Return ``streaming``, ``legacy``, or ``unset``."""
        return self._mode

    @property
    def closed(self) -> bool:
        """Return whether the stream has been finalized or aborted."""
        return self._closed

    @property
    def text(self) -> str:
        """Return the text that fits in one Telegram message."""
        return _truncate(self._buffer)

    async def start(self, initial_text: str) -> int | None:
        """Open a stream and send its first snapshot."""
        if self._mode != DRAFT_UNSET:
            raise RuntimeError("DraftStream.start called twice")
        if not initial_text:
            return None
        self._buffer = initial_text

        try:
            if _DRAFT_UNAVAILABLE or is_peer_draft_unsupported(
                self._chat_id, self._thread_id
            ):
                await self._start_legacy()
            else:
                await self._start_streaming()
        except (TimedOut, NetworkError, RetryAfter, TelegramError) as exc:
            logger.warning("DraftStream.start failed: %s", exc)
            return None
        return self._message_id

    async def append(self, delta: str) -> None:
        """Append a delta and publish the cumulative text."""
        self._ensure_open()
        self._buffer += delta
        await self._push_update()

    async def replace(
        self,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None | Any = _KEEP_MARKUP,
    ) -> None:
        """Replace the cumulative text and publish the new snapshot."""
        self._ensure_open()
        self._buffer = text
        if reply_markup is not _KEEP_MARKUP:
            self._reply_markup = reply_markup
        await self._push_update()

    async def replace_confirmed(
        self,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None | Any = _KEEP_MARKUP,
    ) -> bool:
        """Replace text only after Telegram confirms the snapshot.

        Normal streaming updates may be deferred for rate limiting. Delivery
        receipts cannot settle on that weaker contract, so queued transcript
        work uses this method and waits for the rate-limit slot itself.
        """
        self._ensure_open()
        self._buffer = text
        if reply_markup is not _KEEP_MARKUP:
            self._reply_markup = reply_markup
        await self._cancel_pending_flush()

        if self._mode == DRAFT_LEGACY:
            await self._push_legacy(raise_on_error=True)
            return True
        if _DRAFT_UNAVAILABLE or is_peer_draft_unsupported(
            self._chat_id, self._thread_id
        ):
            return False

        while True:
            async with self._update_lock:
                now = time.monotonic()
                remaining = max(
                    _MIN_DRAFT_INTERVAL - (now - self._last_draft_at),
                    self._retry_not_before - now,
                )
                if remaining <= 0:
                    await self._send_draft()
                    self._last_draft_at = time.monotonic()
                    self._retry_not_before = 0.0
                    self._stream_failures = 0
                    return True
            await asyncio.sleep(remaining)

    async def finalize(
        self,
        final_text: str | None = None,
        *,
        reply_markup: InlineKeyboardMarkup | None | Any = _KEEP_MARKUP,
    ) -> None:
        """Persist the final text and close the stream."""
        self._ensure_open()
        if final_text is not None:
            self._buffer = final_text
        if reply_markup is not _KEEP_MARKUP:
            self._reply_markup = reply_markup

        await self._cancel_pending_flush()
        if self._mode == DRAFT_STREAMING:
            await self._send_final_message()
        else:
            await self._push_legacy(raise_on_error=True)
        self._closed = True

    async def abort(self) -> None:
        """Close a draft, or delete the fallback message."""
        if self._closed:
            return
        await self._cancel_pending_flush()
        if self._mode == DRAFT_LEGACY and self._message_id is not None:
            try:
                await self._bot.delete_message(
                    chat_id=self._chat_id, message_id=self._message_id
                )
            except TelegramError as exc:
                logger.warning("DraftStream.abort delete failed: %s", exc)
        self._closed = True

    def _ensure_open(self) -> None:
        if self._mode == DRAFT_UNSET:
            raise RuntimeError("DraftStream not started")
        if self._closed:
            raise RuntimeError("DraftStream already closed")

    def _send_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._thread_id is not None:
            kwargs["message_thread_id"] = self._thread_id
        if self._reply_to is not None:
            kwargs["reply_to_message_id"] = self._reply_to
        if self._reply_markup is not None:
            kwargs["reply_markup"] = self._reply_markup
        return kwargs

    def _draft_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "chat_id": self._chat_id,
            "draft_id": self._draft_id,
            "text": self.text,
        }
        if self._thread_id is not None:
            kwargs["message_thread_id"] = self._thread_id
        return kwargs

    async def _start_streaming(self) -> None:
        try:
            await self._send_draft()
        except BadRequest as exc:
            if _is_unsupported_error(exc):
                mark_draft_unavailable(f"sendMessageDraft: {exc.message}")
            elif _is_peer_invalid_error(exc):
                mark_peer_draft_unsupported(self._chat_id, self._thread_id)
                logger.info(
                    "sendMessageDraft peer-invalid for chat=%s thread=%s",
                    self._chat_id,
                    self._thread_id,
                )
            else:
                logger.warning("sendMessageDraft rejected: %s", exc)
            await self._start_legacy()
            return
        except RetryAfter as exc:
            await asyncio.sleep(_retry_after_seconds(exc) + 1)
            await self._start_legacy()
            return
        except TelegramError as exc:
            logger.warning("sendMessageDraft failed: %s", exc)
            await self._start_legacy()
            return
        self._last_draft_at = time.monotonic()
        self._mode = DRAFT_STREAMING

    async def _start_legacy(self) -> None:
        message = await self._bot.send_message(
            chat_id=self._chat_id,
            text=self.text,
            **self._send_kwargs(),
        )
        self._message_id = message.message_id
        self._mode = DRAFT_LEGACY

    async def _push_update(self) -> None:
        if self._mode == DRAFT_STREAMING:
            await self._push_streaming()
        else:
            await self._push_legacy()

    async def _push_streaming(self) -> None:
        if _DRAFT_UNAVAILABLE or is_peer_draft_unsupported(
            self._chat_id, self._thread_id
        ):
            return
        async with self._update_lock:
            now = time.monotonic()
            remaining = max(
                _MIN_DRAFT_INTERVAL - (now - self._last_draft_at),
                self._retry_not_before - now,
            )
            if remaining > 0:
                self._schedule_pending_flush(remaining)
                return
            retry_delay = await self._send_stream_snapshot()
        if retry_delay is not None:
            self._schedule_pending_flush(retry_delay)

    def _schedule_pending_flush(self, delay: float) -> None:
        if self._pending_flush is not None and not self._pending_flush.done():
            return
        self._pending_flush = asyncio.create_task(
            self._flush_pending_after(delay),
            name=f"telegram-draft-flush:{self._chat_id}:{self._draft_id}",
        )

    async def _flush_pending_after(self, delay: float) -> None:
        retry_delay: float | None = None
        try:
            await asyncio.sleep(delay)
            async with self._update_lock:
                if self._closed or self._mode != DRAFT_STREAMING:
                    return
                retry_delay = await self._send_stream_snapshot()
        except asyncio.CancelledError:
            raise
        except TelegramError as exc:
            logger.warning("DraftStream trailing update failed: %s", exc)
        finally:
            self._pending_flush = None
        if retry_delay is not None and not self._closed:
            self._schedule_pending_flush(retry_delay)

    async def _cancel_pending_flush(self) -> None:
        task = self._pending_flush
        self._pending_flush = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _send_stream_snapshot(self) -> float | None:
        if _DRAFT_UNAVAILABLE or is_peer_draft_unsupported(
            self._chat_id, self._thread_id
        ):
            return None
        try:
            await self._send_draft()
            self._last_draft_at = time.monotonic()
            self._retry_not_before = 0.0
            self._stream_failures = 0
        except BadRequest as exc:
            if _is_unsupported_error(exc):
                mark_draft_unavailable(f"sendMessageDraft: {exc.message}")
            elif _is_peer_invalid_error(exc):
                mark_peer_draft_unsupported(self._chat_id, self._thread_id)
            await self._handle_stream_failure(exc)
        except RetryAfter as exc:
            retry_delay = _retry_after_seconds(exc)
            self._retry_not_before = time.monotonic() + retry_delay
            await self._handle_stream_failure(exc)
            return retry_delay
        except TelegramError as exc:
            await self._handle_stream_failure(exc)
        return None

    async def _send_draft(self) -> None:
        """Call PTB's typed method, with an escape hatch for older PTB."""
        payload = self._draft_kwargs()
        send_draft = getattr(self._bot, "send_message_draft", None)
        if send_draft is not None:
            await send_draft(**payload)
            return
        await self._bot.do_api_request("sendMessageDraft", api_kwargs=payload)

    async def _send_final_message(self) -> None:
        try:
            message = await self._bot.send_message(
                chat_id=self._chat_id,
                text=self.text,
                **self._send_kwargs(),
            )
        except TelegramError as exc:
            logger.warning("DraftStream final send failed: %s", exc)
            raise
        self._message_id = message.message_id

    async def _push_legacy(  # noqa: C901
        self, *, raise_on_error: bool = False
    ) -> None:
        if self._message_id is None:
            return
        markup = self._reply_markup
        if markup is None:
            markup = InlineKeyboardMarkup([])
        try:
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._message_id,
                text=self.text,
                reply_markup=markup,
            )
        except BadRequest as exc:
            if "not modified" in (exc.message or "").lower():
                return
            self._warn_legacy_edit_failed(exc)
            if raise_on_error:
                raise
        except RetryAfter as exc:
            await asyncio.sleep(_retry_after_seconds(exc) + 1)
            try:
                await self._bot.edit_message_text(
                    chat_id=self._chat_id,
                    message_id=self._message_id,
                    text=self.text,
                    reply_markup=markup,
                )
            except TelegramError as retry_exc:
                self._warn_legacy_edit_failed(retry_exc)
                if raise_on_error:
                    raise
        except TelegramError as exc:
            self._warn_legacy_edit_failed(exc)
            if raise_on_error:
                raise

    async def _handle_stream_failure(self, exc: TelegramError) -> None:
        self._stream_failures += 1
        logger.warning(
            "DraftStream streaming update failed (%d/%d): %s",
            self._stream_failures,
            _DEGRADE_AFTER_FAILURES,
            exc,
        )
        if self._stream_failures >= _DEGRADE_AFTER_FAILURES:
            logger.warning(
                "DraftStream keeping stale preview after repeated update failures"
            )

    def _warn_legacy_edit_failed(self, exc: TelegramError) -> None:
        log_throttled(
            logger,
            f"draft-legacy-edit:{self._chat_id}:{self._message_id}",
            "DraftStream legacy edit failed: %s",
            exc,
        )
