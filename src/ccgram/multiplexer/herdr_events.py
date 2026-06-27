"""herdr push-event stream — the only long-lived unix-socket reader in ccgram.

``HerdrManager`` is otherwise strictly request/response (one ``herdr`` subprocess
per call). The push event stream (``events.subscribe``) needs a persistent
connection, so the socket I/O lives here, separate from the manager, and is
injected into ``HerdrManager`` for unit tests (canned event lines, no socket).

Wire protocol (verified live against herdr 0.7.1): newline-delimited JSON over
the unix socket. ``events.subscribe`` returns one ack line (``{"result": …}``)
then keeps the connection open, pushing one event per line as
``{"data": {…}, "event": "<name>"}``. herdr is inconsistent about the ``event``
form — ``pane.agent_status_changed`` (dot) vs ``tab_closed`` (underscore) — so
event names are matched in both forms. Pane agent-status subscriptions require a
``pane_id``; ``tab.closed`` is global.

After the ack, ``open_socket_stream`` yields a one-shot ``SUBSCRIBED`` sentinel so
the caller can reprime *after* the subscription is live (events that arrive
during the reprime are buffered by the socket and read next), closing the
reprime-vs-subscribe race.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence

import structlog

from .base import AgentStatus, MuxEvent

logger = structlog.get_logger()

# Sentinel yielded once, right after a successful subscribe, before any event.
_SUBSCRIBED_KEY = "__subscribed__"
SUBSCRIBED: dict = {_SUBSCRIBED_KEY: True}

# herdr event names that map onto neutral MuxEvents (matched in both the dot and
# underscore forms herdr uses). Window death is keyed on ``tab.closed`` only:
# ``pane.exited`` fires for any single pane and would falsely kill a multi-pane
# (agent-team) tab whose other panes are alive — the poll loop backstops the
# rare case where a tab vanishes without a ``tab.closed`` push.
_EVT_AGENT_STATUS = frozenset(
    {"pane.agent_status_changed", "pane_agent_status_changed"}
)
_EVT_TAB_CLOSED = frozenset({"tab.closed", "tab_closed"})


def is_subscribed_sentinel(obj: Mapping[str, object]) -> bool:
    """True when *obj* is the post-subscribe sentinel from ``open_socket_stream``."""
    return bool(obj.get(_SUBSCRIBED_KEY))


async def open_socket_stream(
    socket_path: str, subscriptions: Sequence[Mapping[str, object]]
) -> AsyncIterator[dict]:
    """Open the herdr socket, subscribe, and yield the sentinel then pushed events.

    Yields ``SUBSCRIBED`` once after the ack, then each pushed event (lines with
    an ``"event"`` key); the ack and malformed lines are skipped. Returns on EOF.
    Raises ``OSError`` on connect/read failure — the caller reconnects. The socket
    is always closed on exit (including cancellation).
    """
    reader, writer = await asyncio.open_unix_connection(socket_path)
    try:
        request = json.dumps(
            {
                "id": "ccgram-events",
                "method": "events.subscribe",
                "params": {"subscriptions": list(subscriptions)},
            }
        )
        writer.write((request + "\n").encode())
        await writer.drain()

        # First line is the subscription ack ({"result": …}) or an error payload.
        ack = await reader.readline()
        if ack:
            with contextlib.suppress(ValueError):
                payload = json.loads(ack)
                if isinstance(payload, dict) and "error" in payload:
                    logger.warning("herdr events.subscribe error: %s", payload["error"])
        yield SUBSCRIBED

        while True:
            line = await reader.readline()
            if not line:  # EOF — server closed the stream
                return
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except ValueError:
                logger.debug("herdr event stream: non-JSON line")
                continue
            if isinstance(obj, dict) and "event" in obj:
                yield obj
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


def translate_event(
    obj: Mapping[str, object], pane_to_window: Mapping[str, str]
) -> MuxEvent | None:
    """Map a herdr push-event dict to a neutral ``MuxEvent`` (None to ignore).

    Filters the firehose: pane/tab events for windows outside *pane_to_window*
    are dropped (herdr pushes lifecycle events for every pane on the server).
    ``tab.closed`` → ``window_died``; ``pane.agent_status_changed`` →
    ``agent_status``.
    """
    event = obj.get("event")
    data = obj.get("data")
    if not isinstance(data, Mapping):
        data = {}

    if event in _EVT_AGENT_STATUS:
        window_id = pane_to_window.get(_str(data.get("pane_id")))
        if not window_id:
            return None
        return MuxEvent(
            kind="agent_status",
            window_id=window_id,
            pane_id=_str(data.get("pane_id")),
            status=AgentStatus(
                state=_str(data.get("agent_status")) or "unknown",
                agent=_str(data.get("agent")),
                custom_status=_str(data.get("custom_status")),
            ),
        )
    if event in _EVT_TAB_CLOSED:
        tab_id = _str(data.get("tab_id"))
        if tab_id and tab_id in set(pane_to_window.values()):
            return MuxEvent(kind="window_died", window_id=tab_id)
        return None
    return None


def _str(value: object) -> str:
    """Coerce an optional JSON scalar to a string ('' for None/missing)."""
    return value if isinstance(value, str) else ""
