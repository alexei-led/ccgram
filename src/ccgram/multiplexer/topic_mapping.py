"""Backend-neutral projection of multiplexer windows onto Telegram topics.

Consumes the multiplexer seam; it is **not** part of the ``Multiplexer``
contract (which stops at opaque ``window_id``). It defines how a backend's
windows/panes project onto ccgram's flat ``group → topic`` structure.

The design ("Telegram topic mapping (herdr)") maps one herdr agent pane to one
Telegram topic — "topic = pane = agent". Because herdr uses thin identity
(``window_id`` *is* the ``wN:pN`` pane id), per-pane topics, per-pane inbound
routing, and session-id-anchored restart re-resolution (Task 8) already fall out
of ccgram's window-id-centric machinery. The behaviors this module adds are the
discovery filter — on a backend that exposes agent status natively, only panes
herdr reports as running an agent become topics, a bare shell pane does not —
and the adaptive topic-title rendering (``format_agent_topic_prefix``) the herdr
adapter stamps into ``WindowRef.window_name``.

Lives in ``multiplexer/`` (not ``handlers/``) so both the core session monitor
and the topic handlers can import it without crossing the F1 boundary, and
because it is pure logic over the neutral value types.
"""

from __future__ import annotations

from .base import MultiplexerCapabilities, WindowRef

# Separates the workspace prefix from the agent name in a herdr topic title
# (design "Adaptive topic title": ``"<workspace> ▸ <agent>"``).
TOPIC_PREFIX_SEPARATOR = " ▸ "


def format_agent_topic_prefix(
    workspace: str, agent: str, tab: str = "", *, split: bool = False
) -> str:
    """Render a herdr agent pane's adaptive topic label (no status emoji).

    Produces ``"<workspace> ▸ <agent>"`` and appends ``"/<tab>"`` only when the
    pane's tab holds more than one pane (``split`` — an agent team), so a lone
    agent stays terse while team members stay distinguishable (design "Adaptive
    topic title"). The status emoji is prepended later by the topic-emoji
    machinery; this is the clean name it composes onto.

    Backend-neutral and pure: the herdr adapter sources the labels (workspace
    from ``workspace list``, agent from ``display_agent``/``agent``, tab from
    ``tab list``) and the split flag (pane count per tab) and calls this. Empty
    parts degrade gracefully so a half-populated pane never renders a stray
    separator: missing workspace falls back to the agent alone, missing agent to
    the workspace alone.
    """
    workspace = workspace.strip()
    agent = agent.strip()
    tab = tab.strip()
    if workspace and agent:
        label = f"{workspace}{TOPIC_PREFIX_SEPARATOR}{agent}"
    else:
        label = workspace or agent
    if split and tab:
        label = f"{label}/{tab}" if label else tab
    return label


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
