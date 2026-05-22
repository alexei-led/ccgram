"""Transcript surface — paginated history + substring search per window.

Two routes registered by :func:`register_transcript_routes`:

- ``GET /api/transcript/{token}`` — cursor-paginated message list. Optional
  query params: ``cursor`` (integer offset, default ``0``) and ``limit``
  (default ``50``, capped at ``MAX_LIMIT``). Returns ``{messages, total,
  next_cursor}``; ``next_cursor`` is ``None`` when the page contained the
  tail of the transcript.

- ``GET /api/transcript/{token}/search`` — case-insensitive substring search
  over the message ``text`` field. Required query param ``q``. Returns
  ``{matches}`` where each match carries ``index``, the matching entry, and
  one preceding/following entry (``before``/``after``) so the SPA can render
  light context without a second round-trip.

Auth follows the same model as the terminal websocket: the token in the
path is the only authority for ``window_id`` — there is no path-level
``window_id`` to spoof.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aiohttp import web

from ..auth import InvalidTokenError, authorize_api_request

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    Reader = Callable[[str], Awaitable[list[dict[str, Any]]]]

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 50
MAX_LIMIT = 200
MAX_SEARCH_RESULTS = 50
MIN_QUERY_LEN = 1
MAX_QUERY_LEN = 200

_BOT_TOKEN_KEY = web.AppKey("transcript_bot_token", str)
_READER_KEY: web.AppKey[Reader] = web.AppKey("transcript_reader")


async def _default_reader(window_id: str) -> list[dict[str, Any]]:
    """Read all messages for ``window_id`` via the global session resolver."""
    # Lazy: tests substitute alternate readers before any request
    # arrives; the singleton lookup stays inside the default factory.
    # Lazy: miniapp routes resolve singletons per-request so tests can stub them
    from ...session_query import get_recent_messages

    messages, _ = await get_recent_messages(window_id)
    return messages


_INIT_DATA_HEADER = "X-Telegram-Init-Data"


def _init_data_from_header(request: web.Request) -> str | None:
    """Read initData from the header only — keep secrets out of URLs/logs."""
    raw = request.headers.get(_INIT_DATA_HEADER)
    return raw or None


def _verify(request: web.Request) -> str | web.Response:
    """Return ``window_id`` from the path token + initData, or a 403 response."""
    token = request.match_info["token"]
    bot_token = request.app[_BOT_TOKEN_KEY]
    init_data = _init_data_from_header(request)
    try:
        payload = authorize_api_request(
            bot_token=bot_token, token=token, init_data=init_data
        )
    except InvalidTokenError as exc:
        logger.debug("rejected transcript token: %s", exc)
        return web.Response(status=403, text="invalid or expired token")
    return payload.window_id


def _parse_int(raw: str | None, *, default: int, lo: int, hi: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


async def _handle_list(request: web.Request) -> web.Response:
    auth = _verify(request)
    if isinstance(auth, web.Response):
        return auth
    window_id = auth

    cursor = _parse_int(request.query.get("cursor"), default=0, lo=0, hi=10_000_000)
    limit = _parse_int(
        request.query.get("limit"), default=DEFAULT_LIMIT, lo=1, hi=MAX_LIMIT
    )

    reader = request.app[_READER_KEY]
    messages = await reader(window_id)
    if not messages:
        return web.json_response(
            {"reason": "no_session", "messages": [], "total": 0, "next_cursor": None},
            status=404,
        )

    total = len(messages)
    if cursor >= total:
        return web.json_response({"messages": [], "total": total, "next_cursor": None})

    end = cursor + limit
    page = messages[cursor:end]
    next_cursor = end if end < total else None
    return web.json_response(
        {"messages": page, "total": total, "next_cursor": next_cursor}
    )


async def _handle_search(request: web.Request) -> web.Response:
    auth = _verify(request)
    if isinstance(auth, web.Response):
        return auth
    window_id = auth

    query = (request.query.get("q") or "").strip()
    if len(query) < MIN_QUERY_LEN:
        return web.json_response({"error": "query too short"}, status=400)
    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN]

    reader = request.app[_READER_KEY]
    messages = await reader(window_id)
    if not messages:
        return web.json_response({"reason": "no_session", "matches": []}, status=404)

    needle = query.casefold()
    matches: list[dict[str, Any]] = []
    truncated = False
    for idx, msg in enumerate(messages):
        text = msg.get("text") or ""
        if needle not in text.casefold():
            continue
        if len(matches) >= MAX_SEARCH_RESULTS:
            # Result cap reached — surface a truncated flag so the SPA can
            # show a "more results — refine your query" hint instead of
            # implying the cap is the whole match count.
            truncated = True
            break
        matches.append(
            {
                "index": idx,
                "entry": msg,
                "before": messages[idx - 1] if idx > 0 else None,
                "after": messages[idx + 1] if idx + 1 < len(messages) else None,
            }
        )

    return web.json_response(
        {
            "matches": matches,
            "query": query,
            "total": len(matches),
            "truncated": truncated,
        }
    )


def register_transcript_routes(
    app: web.Application,
    *,
    bot_token: str,
    reader: Reader | None = None,
) -> None:
    """Attach transcript list/search routes to ``app``.

    ``reader`` is injected by tests; production passes ``None`` to use the
    global session resolver.
    """
    app[_BOT_TOKEN_KEY] = bot_token
    app[_READER_KEY] = reader or _default_reader
    app.router.add_get("/api/transcript/{token}", _handle_list)
    app.router.add_get("/api/transcript/{token}/search", _handle_search)


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MAX_SEARCH_RESULTS",
    "register_transcript_routes",
]
