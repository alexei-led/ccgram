"""Correlate Telegram-to-terminal input with transcript user messages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

from ..multiplexer.window_ops import send_followup_to_window, send_to_window

_PENDING_INJECTION_TTL_S = 30.0


@dataclass(frozen=True, slots=True, eq=False)
class _PendingInjection:
    text: str
    created_at: float


_pending_injections: dict[
    tuple[int, int | None, str, int], deque[_PendingInjection]
] = {}


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _key(user_id: int, window_id: str, thread_id: int, chat_id: int | None):
    return user_id, chat_id, window_id, thread_id


def remember_telegram_injection(
    user_id: int,
    window_id: str,
    thread_id: int,
    text: str,
    chat_id: int | None = None,
) -> _PendingInjection:
    now = time.monotonic()
    _discard_all_expired(now)
    pending = _pending_injections.setdefault(
        _key(user_id, window_id, thread_id, chat_id), deque()
    )
    injection = _PendingInjection(_normalize(text), now)
    pending.append(injection)
    return injection


def forget_telegram_injection(
    user_id: int,
    window_id: str,
    thread_id: int,
    injection: _PendingInjection,
    chat_id: int | None = None,
) -> None:
    pending = _pending_injections.get(_key(user_id, window_id, thread_id, chat_id))
    if pending is None:
        return
    try:
        pending.remove(injection)
    except ValueError:
        return
    if not pending:
        _pending_injections.pop(_key(user_id, window_id, thread_id, chat_id), None)


async def send_telegram_to_window(
    user_id: int,
    window_id: str,
    thread_id: int | None,
    text: str,
    chat_id: int | None = None,
    *,
    raw: bool = False,
) -> tuple[bool, str]:
    if thread_id is None:
        return await send_to_window(window_id, text, raw=raw)
    injection = remember_telegram_injection(
        user_id, window_id, thread_id, text, chat_id
    )
    success = False
    try:
        success, message = await send_to_window(window_id, text, raw=raw)
        return success, message
    finally:
        if not success:
            forget_telegram_injection(user_id, window_id, thread_id, injection, chat_id)


async def send_telegram_followup_to_window(
    user_id: int,
    window_id: str,
    thread_id: int | None,
    text: str,
    chat_id: int | None = None,
) -> tuple[bool, str]:
    if thread_id is None:
        return await send_followup_to_window(window_id, text)
    injection = remember_telegram_injection(
        user_id, window_id, thread_id, text, chat_id
    )
    success = False
    try:
        success, message = await send_followup_to_window(window_id, text)
        return success, message
    finally:
        if not success:
            forget_telegram_injection(user_id, window_id, thread_id, injection, chat_id)


def consume_telegram_injection(
    user_id: int,
    window_id: str,
    thread_id: int,
    text: str,
    chat_id: int | None = None,
) -> bool:
    key = _key(user_id, window_id, thread_id, chat_id)
    pending = _pending_injections.get(key)
    if pending is None:
        return False
    _discard_expired(pending, time.monotonic())
    if not pending:
        _pending_injections.pop(key, None)
        return False
    normalized = _normalize(text)
    for index, injection in enumerate(pending):
        if injection.text == normalized:
            del pending[index]
            if not pending:
                _pending_injections.pop(key, None)
            return True
    return False


def clear_pending_telegram_injections() -> None:
    _pending_injections.clear()


def _discard_expired(pending: deque[_PendingInjection], now: float) -> None:
    while pending and now - pending[0].created_at >= _PENDING_INJECTION_TTL_S:
        pending.popleft()


def _discard_all_expired(now: float) -> None:
    for key, pending in list(_pending_injections.items()):
        _discard_expired(pending, now)
        if not pending:
            _pending_injections.pop(key, None)
