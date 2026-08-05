"""Correlate Telegram-to-terminal input with transcript user messages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

_PENDING_INJECTION_TTL_S = 30.0
_MAX_PENDING_INJECTIONS = 16


@dataclass(frozen=True, slots=True, eq=False)
class _PendingInjection:
    text: str
    created_at: float


_pending_injections: dict[tuple[int, str, int], deque[_PendingInjection]] = {}


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip("\n")


def remember_telegram_injection(
    user_id: int, window_id: str, thread_id: int, text: str
) -> _PendingInjection:
    """Reserve a Telegram prompt before terminal injection."""
    key = (user_id, window_id, thread_id)
    pending = _pending_injections.setdefault(key, deque())
    now = time.monotonic()
    _discard_expired(pending, now)
    injection = _PendingInjection(_normalize(text), now)
    pending.append(injection)
    while len(pending) > _MAX_PENDING_INJECTIONS:
        pending.popleft()
    return injection


def forget_telegram_injection(
    user_id: int,
    window_id: str,
    thread_id: int,
    injection: _PendingInjection,
) -> None:
    """Remove a reservation when terminal injection fails."""
    key = (user_id, window_id, thread_id)
    pending = _pending_injections.get(key)
    if pending is None:
        return
    try:
        pending.remove(injection)
    except ValueError:
        return
    if not pending:
        _pending_injections.pop(key, None)


def consume_telegram_injection(
    user_id: int, window_id: str, thread_id: int, text: str
) -> bool:
    """Consume one matching pending prompt, returning whether it was Telegram-originated."""
    key = (user_id, window_id, thread_id)
    pending = _pending_injections.get(key)
    if pending is None:
        return False

    _discard_expired(pending, time.monotonic())
    if not pending:
        _pending_injections.pop(key, None)
        return False
    if pending[0].text != _normalize(text):
        return False

    pending.popleft()
    if not pending:
        _pending_injections.pop(key, None)
    return True


def clear_pending_telegram_injections() -> None:
    """Clear pending input correlations. Used by tests and process shutdown."""
    _pending_injections.clear()


def _discard_expired(pending: deque[_PendingInjection], now: float) -> None:
    while pending and now - pending[0].created_at >= _PENDING_INJECTION_TTL_S:
        pending.popleft()
