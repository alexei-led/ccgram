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
event names are matched in both forms. Pane agent-status and pane-exit
subscriptions require a ``pane_id``; ``tab.closed`` is global.

After the ack, ``open_socket_stream`` yields a one-shot ``SUBSCRIBED`` sentinel so
the caller can reprime *after* the subscription is live (events that arrive
during the reprime are buffered by the socket and read next), closing the
reprime-vs-subscribe race.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, Mapping, Sequence

import structlog

from .base import AgentStatus, MuxEvent

logger = structlog.get_logger()

# Sentinel yielded once, right after a successful subscribe, before any event.
_SUBSCRIBED_KEY = "__subscribed__"
SUBSCRIBED: dict = {_SUBSCRIBED_KEY: True}

# herdr event names that map onto neutral MuxEvents (matched in both the dot and
# underscore forms herdr uses). Pane exits are target-specific; tab closures
# fan out to every guarded target currently sharing that tab.
_EVT_AGENT_STATUS = frozenset(
    {"pane.agent_status_changed", "pane_agent_status_changed"}
)
_EVT_PANE_EXITED = frozenset(
    {"pane.exited", "pane_exited", "pane.closed", "pane_closed"}
)
_EVT_TAB_CLOSED = frozenset({"tab.closed", "tab_closed"})


def is_subscribed_sentinel(obj: Mapping[str, object]) -> bool:
    """True when *obj* is the post-subscribe sentinel from ``open_socket_stream``."""
    return bool(obj.get(_SUBSCRIBED_KEY))


async def open_socket_stream(
    socket_path: str, subscriptions: Sequence[Mapping[str, object]]
) -> AsyncGenerator[dict, None]:
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

        # Yield SUBSCRIBED only after a successful JSON-RPC acknowledgement.
        # Empty, malformed, or error acks never establish a live subscription.
        ack = await reader.readline()
        try:
            payload = json.loads(ack) if ack else None
        except ValueError:
            logger.warning("herdr events.subscribe returned malformed acknowledgement")
            return
        if (
            not isinstance(payload, dict)
            or "error" in payload
            or not isinstance(payload.get("result"), Mapping)
        ):
            logger.warning("herdr events.subscribe rejected subscription: %r", payload)
            return
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
    obj: Mapping[str, object],
    pane_to_window: Mapping[str, str],
    tab_to_windows: Mapping[str, Sequence[str]],
) -> tuple[MuxEvent, ...]:
    """Map a herdr push-event dict to neutral events for guarded targets.

    The stream is a server-wide firehose. Pane events are filtered through the
    current pane-to-target map; a ``tab.closed`` can affect multiple independent
    session targets sharing a tab, so it produces one durable-target event per
    mapped target. Raw Herdr pane and tab locators never escape this boundary.
    """
    event = obj.get("event")
    data = obj.get("data")
    if not isinstance(data, Mapping):
        data = {}

    if event in _EVT_AGENT_STATUS:
        pane_id = _event_locator(data, "pane_id", "pane")
        window_id = pane_to_window.get(pane_id)
        if not window_id:
            return ()
        return (
            MuxEvent(
                kind="agent_status",
                window_id=window_id,
                pane_id=pane_id,
                status=AgentStatus(
                    state=_str(data.get("agent_status")) or "unknown",
                    agent=_str(data.get("agent")),
                    custom_status=_str(data.get("custom_status")),
                ),
            ),
        )
    if event in _EVT_PANE_EXITED:
        pane_id = _event_locator(data, "pane_id", "pane")
        target_id = pane_to_window.get(pane_id)
        return (MuxEvent(kind="window_died", window_id=target_id),) if target_id else ()
    if event in _EVT_TAB_CLOSED:
        tab_id = _event_locator(data, "tab_id", "tab")
        return tuple(
            MuxEvent(kind="window_died", window_id=target_id)
            for target_id in tab_to_windows.get(tab_id, ())
        )
    return ()


def _event_locator(data: Mapping[str, object], key: str, nested_key: str) -> str:
    """Read a locator from the live flat protocol or its nested variant."""
    direct = _str(data.get(key))
    if direct:
        return direct
    nested = data.get(nested_key)
    return _str(nested.get(key)) if isinstance(nested, Mapping) else ""


def _str(value: object) -> str:
    """Coerce an optional JSON scalar to a string ('' for None/missing)."""
    return value if isinstance(value, str) else ""
