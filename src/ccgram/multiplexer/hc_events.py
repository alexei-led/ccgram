"""herdr-core push events: the ``events.subscribe`` reader and its translation.

``HcManager`` is otherwise request/response (one connection per call). The
event stream needs one long-lived connection, so its socket I/O lives here and
is injected into the manager, which lets unit tests drive ``watch_events`` with
canned notifications and no socket.

Wire protocol (verified live against herdr-core 0.1.18, protocol generation 1):

* ``events.subscribe`` answers once with ``{"result": {"seq", "terminals",
  "channel"}}``: a snapshot of every matching terminal, after which the
  connection carries only notifications.
* Fact changes arrive as ``{"method": "event", "params": {"type", "terminal",
  "seq", "data"}}``. Observed types: ``created``, ``foreground``, ``status``
  (``data.status`` is ``{app, state, msg}`` or null once the program stops
  reporting), ``finished`` (``data.exit_code`` / ``reason``) and ``removed``.
  The schema also names ``lost``, ``restarted``, ``keep``, ``turn``,
  ``title``, ``labels``, ``size`` and ``doc``.
* ``{"method": "heartbeat", "params": {"seq"}}`` every 15 s.
* A subscriber that fell behind gets ``missed`` plus a fresh snapshot. Its
  exact shape has not been observed, so the reader surfaces it as a resync
  request and the manager re-subscribes (which yields a fresh snapshot).

Every herdr-core shape stays inside this module and ``hc.py``; callers only see
the neutral ``MuxEvent`` / ``AgentStatus`` values.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, Mapping, Sequence

import structlog

from .base import AgentStatus, MuxEvent
from .hc_socket import MAX_FRAME_BYTES, encode_request

__all__ = [
    "RESYNC",
    "agent_name_for",
    "is_resync",
    "is_subscribed",
    "open_event_stream",
    "parse_status",
    "snapshot_events",
    "translate_event",
]

logger = structlog.get_logger()

_SUBSCRIBED_KEY = "__hc_subscribed__"
_RESYNC_KEY = "__hc_resync__"

# Yielded when the server says this subscriber missed events; the consumer
# re-subscribes to get a consistent snapshot instead of guessing what changed.
RESYNC: dict = {_RESYNC_KEY: True}

# Event types that end a terminal for ccgram's purposes: the program exited
# (``finished``), the core lost it (``lost``), or it was removed outright.
_DEATH_TYPES = frozenset({"finished", "lost", "removed"})
_STATUS_TYPE = "status"
_MISSED = "missed"

# herdr-core reports the program's own ``app`` word (OSC 7501). Map the ones
# ccgram has a provider for onto that provider's name; anything else passes
# through unchanged so the label still says what is running.
_APP_TO_AGENT: Mapping[str, str] = {
    "claude-code": "claude",
    "claude": "claude",
    "codex": "codex",
    "codex-cli": "codex",
    "gemini": "gemini",
    "gemini-cli": "gemini",
    "pi": "pi",
    "antigravity": "antigravity",
}


def agent_name_for(app: str) -> str:
    """Map herdr-core's ``status.app`` word to ccgram's provider name."""
    word = app.strip()
    return _APP_TO_AGENT.get(word.casefold(), word)


def parse_status(raw: object) -> AgentStatus | None:
    """Parse herdr-core's ``status`` object into ``AgentStatus``; None if absent.

    ``state`` is herdr-core's word (``idle`` / ``working`` / ``done`` /
    ``blocked`` / ``error``), ``app`` becomes the agent name and ``msg`` the
    short activity label.
    """
    if not isinstance(raw, Mapping):
        return None
    state = raw.get("state")
    if not isinstance(state, str) or not state.strip():
        return None
    app = raw.get("app")
    msg = raw.get("msg")
    return AgentStatus(
        state=state.strip(),
        agent=agent_name_for(app) if isinstance(app, str) else "",
        custom_status=msg.strip() if isinstance(msg, str) else "",
    )


def is_subscribed(obj: Mapping[str, object]) -> bool:
    """True for the sentinel yielded once the subscription is live."""
    return bool(obj.get(_SUBSCRIBED_KEY))


def is_resync(obj: Mapping[str, object]) -> bool:
    """True when the stream asks the consumer to re-subscribe."""
    return bool(obj.get(_RESYNC_KEY))


def subscribed_sentinel(terminals: Sequence[object]) -> dict:
    """Build the post-subscribe sentinel carrying the snapshot terminals."""
    return {_SUBSCRIBED_KEY: True, "terminals": list(terminals)}


async def open_event_stream(
    socket_path: str, *, idle_timeout: float = 45.0
) -> AsyncGenerator[dict, None]:
    """Subscribe to every terminal and yield the sentinel, then notifications.

    Yields one sentinel (see ``is_subscribed``) carrying the snapshot's
    ``terminals`` right after a successful acknowledgement, then each
    notification line as a dict. A ``missed`` notice is turned into ``RESYNC``.
    Returns on EOF or a rejected subscription; raises ``OSError`` /
    ``TimeoutError`` on transport failure or when nothing (not even a
    heartbeat) arrives for *idle_timeout* seconds. The socket is always closed
    on exit, including cancellation.
    """
    reader, writer = await asyncio.open_unix_connection(
        socket_path, limit=MAX_FRAME_BYTES
    )
    try:
        writer.write(encode_request(1, "events.subscribe", {}))
        await writer.drain()
        ack = await asyncio.wait_for(reader.readline(), idle_timeout)
        try:
            payload = json.loads(ack) if ack else None
        except ValueError:
            logger.warning("hc events.subscribe returned a malformed acknowledgement")
            return
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, Mapping):
            logger.warning("hc events.subscribe was rejected: %r", payload)
            return
        terminals = result.get("terminals")
        yield subscribed_sentinel(terminals if isinstance(terminals, list) else [])

        while True:
            line = await asyncio.wait_for(reader.readline(), idle_timeout)
            if not line:
                return
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except ValueError:
                logger.debug("hc event stream: non-JSON line")
                continue
            if not isinstance(obj, dict):
                continue
            if _is_missed(obj):
                yield RESYNC
                continue
            yield obj
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


def _is_missed(obj: Mapping[str, object]) -> bool:
    if obj.get("method") == _MISSED:
        return True
    params = obj.get("params")
    return isinstance(params, Mapping) and params.get("type") == _MISSED


def snapshot_events(
    terminals: Sequence[object], watched: Mapping[str, str]
) -> tuple[MuxEvent, ...]:
    """Agent-status events for watched terminals in a subscribe snapshot.

    This is the reprime: the snapshot is consistent with the stream position,
    so no extra request is needed. Terminals without a status are skipped;
    terminals missing from the snapshot are left to the poll loop's liveness
    checks rather than declared dead here.
    """
    events: list[MuxEvent] = []
    for terminal in terminals:
        if not isinstance(terminal, Mapping):
            continue
        address = terminal.get("address")
        window_id = watched.get(address) if isinstance(address, str) else None
        if not window_id or terminal.get("state") != "live":
            continue
        status = parse_status(terminal.get("status"))
        if status is not None:
            events.append(
                MuxEvent(kind="agent_status", window_id=window_id, status=status)
            )
    return tuple(events)


def translate_event(
    obj: Mapping[str, object], watched: Mapping[str, str]
) -> tuple[MuxEvent, ...]:
    """Map one herdr-core notification onto neutral events for watched terminals.

    *watched* maps a terminal address to the window id the consumer asked
    about. ``heartbeat`` and every unknown or irrelevant type map to nothing.
    A ``status`` whose status is null (the program stopped reporting, as
    happens when the agent exits) becomes state ``unknown`` so a cached
    ``working`` does not outlive the agent.
    """
    if obj.get("method") != "event":
        return ()
    params = obj.get("params")
    if not isinstance(params, Mapping):
        return ()
    address = params.get("terminal")
    window_id = watched.get(address) if isinstance(address, str) else None
    if not window_id:
        return ()
    event_type = params.get("type")
    if event_type in _DEATH_TYPES:
        return (MuxEvent(kind="window_died", window_id=window_id),)
    if event_type == _STATUS_TYPE:
        data = params.get("data")
        raw = data.get("status") if isinstance(data, Mapping) else None
        status = parse_status(raw) or AgentStatus(state="unknown")
        return (MuxEvent(kind="agent_status", window_id=window_id, status=status),)
    return ()
