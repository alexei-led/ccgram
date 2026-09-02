"""Short-lived callback indirection for opaque window identifiers.

Telegram limits callback_data to 64 UTF-8 bytes.  Opaque Herdr session targets
are deliberately 81 ASCII bytes, so a callback cannot carry one verbatim.
A bounded on-disk mapping preserves the complete payload across service restarts
and verifies the clicker still owns its target when it is resolved.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..utils import atomic_write_json, ccgram_dir

_CALLBACK_LIMIT = 64
_TOKEN_TTL_SECONDS = 3600.0
_MAX_CALLBACK_TOKENS = 512
_TOKEN_MARKER = "~"
_TOKEN_STORE_PATH = ccgram_dir() / "callback_tokens.json"
# token_urlsafe(9) produces exactly 12 URL-safe base64 characters. Require the
# complete envelope so normal callback payloads containing ``~`` pass through.
_TOKEN_ENVELOPE_RE = re.compile(r"^.+~([A-Za-z0-9_-]{12})$")


@dataclass(frozen=True, slots=True)
class _CallbackToken:
    payload: str
    window_id: str
    expires_at: float
    expires_at_wall: float


_tokens: dict[str, _CallbackToken] = {}
_loaded_store_path: Path | None = None


def _ensure_loaded() -> None:
    """Load callback indirections once so Telegram keyboards survive restarts."""
    global _loaded_store_path
    if _loaded_store_path == _TOKEN_STORE_PATH:
        return
    _loaded_store_path = _TOKEN_STORE_PATH
    try:
        raw = json.loads(_TOKEN_STORE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(raw, dict):
        return
    now_wall = time.time()
    now_mono = time.monotonic()
    for token, value in raw.items():
        if not isinstance(token, str) or not isinstance(value, dict):
            continue
        payload = value.get("payload")
        window_id = value.get("window_id")
        expires_at_wall = value.get("expires_at")
        if (
            not isinstance(payload, str)
            or not isinstance(window_id, str)
            or not isinstance(expires_at_wall, (int, float))
            or expires_at_wall <= now_wall
        ):
            continue
        _tokens[token] = _CallbackToken(
            payload=payload,
            window_id=window_id,
            expires_at=now_mono + expires_at_wall - now_wall,
            expires_at_wall=float(expires_at_wall),
        )
        if len(_tokens) >= _MAX_CALLBACK_TOKENS:
            break


def _persist_tokens() -> None:
    """Persist the bounded token map without making delivery fail on I/O errors."""
    data = {
        token: {
            "payload": entry.payload,
            "window_id": entry.window_id,
            "expires_at": entry.expires_at_wall,
        }
        for token, entry in _tokens.items()
    }
    try:
        if data:
            atomic_write_json(_TOKEN_STORE_PATH, data)
        else:
            _TOKEN_STORE_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def compact_callback_data(prefix: str, payload: str, window_id: str) -> str:
    """Return *payload* or a short callback token retaining it server-side."""
    if len(payload.encode("utf-8")) <= _CALLBACK_LIMIT:
        return payload
    _ensure_loaded()
    _prune_expired()
    while len(_tokens) >= _MAX_CALLBACK_TOKENS:
        _tokens.pop(next(iter(_tokens)))
    while True:
        token = secrets.token_urlsafe(9)
        callback_data = f"{prefix}{_TOKEN_MARKER}{token}"
        if (
            token not in _tokens
            and len(callback_data.encode("utf-8")) <= _CALLBACK_LIMIT
        ):
            expires_at_wall = time.time() + _TOKEN_TTL_SECONDS
            _tokens[token] = _CallbackToken(
                payload=payload,
                window_id=window_id,
                expires_at=time.monotonic() + _TOKEN_TTL_SECONDS,
                expires_at_wall=expires_at_wall,
            )
            _persist_tokens()
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
    match = _TOKEN_ENVELOPE_RE.fullmatch(data)
    if match is None:
        return data
    _ensure_loaded()
    _prune_expired()
    token = match.group(1)
    entry = _tokens.get(token)
    if entry is None:
        return None
    if entry.expires_at <= time.monotonic():
        del _tokens[token]
        _persist_tokens()
        return None
    if not owns_window(user_id, entry.window_id):
        return None
    return entry.payload


def revoke_window_tokens(window_id: str) -> None:
    """Invalidate callback tokens targeting a window during topic cleanup."""
    _ensure_loaded()
    removed = False
    for token, entry in list(_tokens.items()):
        if entry.window_id == window_id:
            del _tokens[token]
            removed = True
    if removed:
        _persist_tokens()


def _prune_expired() -> None:
    now = time.monotonic()
    removed = False
    for token, entry in list(_tokens.items()):
        if entry.expires_at <= now:
            del _tokens[token]
            removed = True
    if removed:
        _persist_tokens()
