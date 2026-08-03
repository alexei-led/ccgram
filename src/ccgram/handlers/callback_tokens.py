"""Short-lived callback indirection for opaque window identifiers.

Telegram limits callback_data to 64 UTF-8 bytes.  Opaque Herdr session targets
are deliberately 81 ASCII bytes, so a callback cannot carry one verbatim.
This in-memory mapping preserves the complete payload and verifies the clicker
still owns its target when it is resolved.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

_CALLBACK_LIMIT = 64
_TOKEN_TTL_SECONDS = 3600.0
_TOKEN_MARKER = "~"


@dataclass(frozen=True, slots=True)
class _CallbackToken:
    payload: str
    window_id: str
    expires_at: float


_tokens: dict[str, _CallbackToken] = {}


def compact_callback_data(prefix: str, payload: str, window_id: str) -> str:
    """Return *payload* or a short callback token retaining it server-side."""
    if len(payload.encode("utf-8")) <= _CALLBACK_LIMIT:
        return payload
    _prune_expired()
    while True:
        token = secrets.token_urlsafe(9)
        callback_data = f"{prefix}{_TOKEN_MARKER}{token}"
        if token not in _tokens and len(callback_data.encode("utf-8")) <= _CALLBACK_LIMIT:
            _tokens[token] = _CallbackToken(
                payload=payload,
                window_id=window_id,
                expires_at=time.monotonic() + _TOKEN_TTL_SECONDS,
            )
            return callback_data


def resolve_callback_data(
    data: str,
    user_id: int,
    owns_window: Callable[[int, str], bool],
) -> str | None:
    """Resolve a compact callback after expiry and target-ownership checks.

    Ordinary callbacks pass through unchanged. ``None`` means the token is
    expired, unknown, or belongs to a target the clicking user no longer owns.
    """
    marker_index = data.find(_TOKEN_MARKER)
    if marker_index < 0:
        return data
    token = data[marker_index + len(_TOKEN_MARKER) :]
    entry = _tokens.get(token)
    if entry is None:
        return None
    if entry.expires_at <= time.monotonic():
        del _tokens[token]
        return None
    if not owns_window(user_id, entry.window_id):
        return None
    return entry.payload


def _prune_expired() -> None:
    now = time.monotonic()
    for token, entry in list(_tokens.items()):
        if entry.expires_at <= now:
            del _tokens[token]
