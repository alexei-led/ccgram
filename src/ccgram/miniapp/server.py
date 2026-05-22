"""Mini App HTTP server — aiohttp app with token-scoped routes.

Run via :func:`start_server` (returns an ``AppRunner``) and torn down via
:func:`stop_server`. The server binds to a local host/port; an external
reverse-proxy is expected to terminate TLS and forward to it.

Routes:
- ``GET /healthz`` — liveness probe (no auth).
- ``GET /app/{token}`` — serves ``static/index.html``. Token is verified by
  :func:`verify_token`; the validated payload is exposed to the page through
  a meta tag injected at render time.
- ``GET /static/{path}`` — static file serving rooted at ``static/``.

The bot token is captured at server-build time and used to verify all tokens.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from .api import register_terminal_routes, register_transcript_routes
from .auth import InvalidTokenError, verify_token

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    PaneCapture = Callable[[str], Awaitable[str | None]]
    PaneByIdCapture = Callable[[str, str], Awaitable[str | None]]
    PaneList = Callable[[str], Awaitable[list[dict[str, Any]]]]
    TranscriptReader = Callable[[str], Awaitable[list[dict[str, Any]]]]

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"
# Cached at import time — the template never changes at runtime, so re-reading
# it on every authenticated page hit is wasted I/O. Tolerate a missing asset
# at import: a partial install must not crash bot startup.
try:
    _INDEX_TEMPLATE: str | None = _INDEX_FILE.read_text(encoding="utf-8")
except OSError as _exc:
    logger.warning("miniapp index.html not loaded: %s", _exc)
    _INDEX_TEMPLATE = None

# Key used to stash the bot token on the aiohttp Application.
_BOT_TOKEN_KEY = web.AppKey("bot_token", str)


async def _handle_health(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


def _render_index(payload_meta: str) -> str:
    """Inject the validated token payload into the index template.

    The placeholder ``<!-- CCGRAM_PAYLOAD -->`` is replaced with a
    ``<meta>`` tag carrying the JSON payload; the SPA reads it via
    ``document.querySelector('meta[name=ccgram-payload]').content``.
    """
    assert _INDEX_TEMPLATE is not None  # _handle_app guards this
    return _INDEX_TEMPLATE.replace("<!-- CCGRAM_PAYLOAD -->", payload_meta)


async def _handle_app(request: web.Request) -> web.Response:
    token = request.match_info["token"]
    bot_token = request.app[_BOT_TOKEN_KEY]
    if _INDEX_TEMPLATE is None:
        return web.Response(status=503, text="dashboard assets unavailable")
    try:
        payload = verify_token(token, bot_token=bot_token)
    except InvalidTokenError as exc:
        logger.debug("rejected miniapp token: %s", exc)
        return web.Response(status=403, text="invalid or expired token")

    # Escape every interpolated value: today the payload fields are
    # constrained shapes (``@\d+``, ints), but the meta tag is the only
    # path through which trusted server data reaches HTML, so treating
    # all values as untrusted keeps future field additions safe.
    meta = (
        f'<meta name="ccgram-payload" '
        f'data-window-id="{html.escape(str(payload.window_id), quote=True)}" '
        f'data-user-id="{html.escape(str(payload.user_id), quote=True)}" '
        f'data-exp="{html.escape(str(payload.exp), quote=True)}">'
    )
    body = _render_index(meta)
    return web.Response(text=body, content_type="text/html")


def build_app(
    *,
    bot_token: str,
    terminal_capture: PaneCapture | None = None,
    pane_capture: PaneByIdCapture | None = None,
    pane_list: PaneList | None = None,
    transcript_reader: TranscriptReader | None = None,
) -> web.Application:
    """Build the aiohttp application without starting it.

    ``terminal_capture`` overrides the tmux pane reader used by the live
    terminal websocket; tests inject a stub, production leaves it ``None``
    to fall through to the global ``TmuxManager`` singleton.

    ``pane_capture`` overrides the per-pane capture used by the multi-pane
    grid; ``pane_list`` overrides the pane enumerator. Both fall through to
    ``tmux_manager`` defaults when ``None``.

    ``transcript_reader`` overrides the per-window message loader used by
    the transcript HTTP routes; tests inject a stub, production leaves it
    ``None`` to fall through to ``session_resolver``.
    """
    app = web.Application()
    app[_BOT_TOKEN_KEY] = bot_token
    app.router.add_get("/healthz", _handle_health)
    app.router.add_get("/app/{token}", _handle_app)
    app.router.add_static("/static/", path=_STATIC_DIR, show_index=False)
    register_terminal_routes(
        app,
        bot_token=bot_token,
        capture=terminal_capture,
        pane_capture=pane_capture,
        pane_list=pane_list,
    )
    register_transcript_routes(app, bot_token=bot_token, reader=transcript_reader)
    return app


async def start_server(
    *,
    bot_token: str,
    host: str,
    port: int,
    app_factory: Callable[..., web.Application] | None = None,
) -> web.AppRunner:
    """Bind the aiohttp app on ``host:port`` and return a runner.

    ``app_factory`` allows tests to inject a customised app; production calls
    leave it ``None`` to use :func:`build_app`.
    """
    factory = app_factory or build_app
    app = factory(bot_token=bot_token)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    logger.info("miniapp server listening on %s:%d", host, port)
    return runner


async def stop_server(runner: web.AppRunner) -> None:
    """Tear down the runner returned by :func:`start_server`."""
    await runner.cleanup()
    logger.info("miniapp server stopped")
