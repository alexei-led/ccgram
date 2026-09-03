"""Backend-neutral projection of multiplexer windows onto Telegram topics.

Consumes the multiplexer seam; it is **not** part of the ``Multiplexer``
contract (which stops at opaque ``window_id``). It defines how a backend's
windows/tabs project onto ccgram's flat ``group → topic`` structure.

The design maps each reported Herdr agent to one Telegram topic. Herdr
``WindowRef.window_id`` is an opaque target, never a raw tab, pane, or other
locator. A published agent session has a durable target; a sessionless detected
agent does not become a topic. A bare shell record does not become a topic, and
raw locator aliases are never auto-migrated. tmux behavior remains unchanged.

Lives in ``multiplexer/`` (not ``handlers/``) so both the core session monitor
and the topic handlers can import it without crossing the F1 boundary, and
because it is pure logic over the neutral value types.
"""

from __future__ import annotations

from ..herdr_targets import is_herdr_session_target
from .base import MultiplexerCapabilities, WindowRef

# Separates workspace, tab, and optional pane parts in a Herdr topic title.
TOPIC_PREFIX_SEPARATOR = " ▸ "


def format_agent_topic_prefix(
    workspace: str, tab: str, pane: str = "", *, provider: str = ""
) -> str:
    """Render a Herdr topic label with an easy-to-search provider prefix.

    Produces ``"<Provider> ▸ <workspace> ▸ <tab> ▸ <pane>"`` when *provider*
    is present. Without a provider, the legacy workspace/tab label is kept.
    The status emoji is prepended later by the topic-emoji machinery.

    Empty parts degrade gracefully so a half-populated tab never renders a
    stray separator.
    """
    provider_label = provider.strip().capitalize()
    parts = [
        part.strip() for part in (provider_label, workspace, tab, pane) if part.strip()
    ]
    return TOPIC_PREFIX_SEPARATOR.join(parts)


def is_agent_topic_window(window: WindowRef, caps: MultiplexerCapabilities) -> bool:
    """Return True when a discovered window should surface as its own topic.

    Gated on ``caps.native_agent_status`` — a capability flag, never a backend
    name (architecture rule: gate on capabilities, not ``name == "herdr"``):

    * Backends without native agent status (tmux): every window is eligible,
      so the historical auto-topic behavior is unchanged.
    * Backends with native agent status (herdr): every detected agent record
      qualifies. The adapter exposes each with a versioned opaque target and an
      agent label; a raw tab/pane locator or bare shell record is rejected.
    """
    if not caps.native_agent_status:
        return True
    return is_herdr_session_target(window.window_id) and bool(
        window.pane_current_command.strip()
    )
