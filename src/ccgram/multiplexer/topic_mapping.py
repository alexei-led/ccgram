"""Backend-neutral projection of multiplexer windows onto Telegram topics.

Consumes the multiplexer seam; it is **not** part of the ``Multiplexer``
contract (which stops at opaque ``window_id``). It defines how a backend's
windows/tabs project onto ccgram's flat ``group → topic`` structure.

The design ("Telegram topic mapping (herdr)") maps one herdr agent tab to one
Telegram topic — "topic = tab = agent". Because herdr uses tab identity
(``window_id`` *is* the ``wN:tM`` tab id), per-tab topics, per-tab inbound
routing, and session-id-anchored restart re-resolution (Task 8) already fall out
of ccgram's window-id-centric machinery. The behaviors this module adds are the
discovery filter — on a backend that exposes agent status natively, only tabs
herdr reports as running an agent become topics, a bare shell tab does not —
and the adaptive topic-title rendering (``format_agent_topic_prefix``) the herdr
adapter stamps into ``WindowRef.window_name``.

Lives in ``multiplexer/`` (not ``handlers/``) so both the core session monitor
and the topic handlers can import it without crossing the F1 boundary, and
because it is pure logic over the neutral value types.
"""

from __future__ import annotations

from .base import MultiplexerCapabilities, WindowRef

# Separates the workspace prefix from the tab name in a herdr topic title
# (design "Adaptive topic title": ``"<workspace> ▸ <tab>"``).
TOPIC_PREFIX_SEPARATOR = " ▸ "


def format_agent_topic_prefix(workspace: str, tab: str) -> str:
    """Render a herdr tab's adaptive topic label (no status emoji).

    Produces ``"<workspace> ▸ <tab>"`` — the tab name is primary so two tabs
    running the same agent in one workspace get distinct titles
    (``"ccgram ▸ herdr-support"`` vs ``"ccgram ▸ ralphex"``). The status emoji
    is prepended later by the topic-emoji machinery; this is the clean name it
    composes onto.

    Backend-neutral and pure: the herdr adapter sources the labels (workspace
    from ``workspace list``, tab label from ``tab list``) and calls this. Empty
    parts degrade gracefully so a half-populated tab never renders a stray
    separator: missing workspace falls back to the tab label alone, missing tab
    to the workspace alone.
    """
    workspace = workspace.strip()
    tab = tab.strip()
    if workspace and tab:
        return f"{workspace}{TOPIC_PREFIX_SEPARATOR}{tab}"
    return workspace or tab


def is_agent_topic_window(window: WindowRef, caps: MultiplexerCapabilities) -> bool:
    """Return True when a discovered window should surface as its own topic.

    Gated on ``caps.native_agent_status`` — a capability flag, never a backend
    name (architecture rule: gate on capabilities, not ``name == "herdr"``):

    * Backends without native agent status (tmux): every window is eligible,
      so the historical auto-topic behavior is unchanged.
    * Backends with native agent status (herdr): only agent tabs qualify.
      herdr carries the agent label in ``WindowRef.pane_current_command``
      (empty for a bare shell tab), so a non-empty label marks an agent. Each
      agent tab has a distinct ``window_id`` and therefore becomes a distinct
      topic. A split tab (agent team) is one topic with multiple panes — not
      multiple topics.
    """
    if not caps.native_agent_status:
        return True
    return bool(window.pane_current_command.strip())
