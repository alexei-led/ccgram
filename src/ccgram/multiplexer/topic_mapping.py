"""Backend-neutral projection of multiplexer windows onto Telegram topics.

Consumes the multiplexer seam; it is **not** part of the ``Multiplexer``
contract (which stops at opaque ``window_id``). It defines how a backend's
windows/panes project onto ccgram's flat ``group → topic`` structure.

The design ("Telegram topic mapping (herdr)") maps one herdr agent pane to one
Telegram topic — "topic = pane = agent". Because herdr uses thin identity
(``window_id`` *is* the ``wN:pN`` pane id), per-pane topics, per-pane inbound
routing, and session-id-anchored restart re-resolution (Task 8) already fall out
of ccgram's window-id-centric machinery. The one behavior this module adds is
the discovery filter: on a backend that exposes agent status natively, only
panes herdr reports as running an agent become topics — a bare shell pane does
not.

Lives in ``multiplexer/`` (not ``handlers/``) so both the core session monitor
and the topic handlers can import it without crossing the F1 boundary, and
because it is pure logic over the neutral value types.
"""

from __future__ import annotations

from .base import MultiplexerCapabilities, WindowRef


def is_agent_topic_window(window: WindowRef, caps: MultiplexerCapabilities) -> bool:
    """Return True when a discovered window should surface as its own topic.

    Gated on ``caps.native_agent_status`` — a capability flag, never a backend
    name (architecture rule: gate on capabilities, not ``name == "herdr"``):

    * Backends without native agent status (tmux): every window is eligible,
      so the historical auto-topic behavior is unchanged.
    * Backends with native agent status (herdr): only agent panes qualify.
      herdr carries the agent label in ``WindowRef.pane_current_command``
      (empty for a bare shell pane), so a non-empty label marks an agent. Each
      agent pane — including the extra panes a tab split (agent team) spawns —
      has a distinct ``window_id`` and therefore becomes a distinct topic.
    """
    if not caps.native_agent_status:
        return True
    return bool(window.pane_current_command.strip())
