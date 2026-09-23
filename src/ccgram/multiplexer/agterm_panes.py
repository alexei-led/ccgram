"""Private agterm split projections shared by discovery and hook resolution.

The primary keeps its existing session UUID. A split target pins the reported
foreground argv; closing, promoting or replacing that peer invalidates it.
These are guarded locators, not agterm's unexposed stable surface tokens.
"""

from __future__ import annotations

import hashlib
import json

_SPLIT_MARKER = ":split-"


def split_session_id(target: str) -> str | None:
    """Return the owning session for a guarded split target."""
    session_id, separator, _guard = target.partition(_SPLIT_MARKER)
    return session_id if separator and session_id else None


def pane_sessions(session: dict) -> list[dict]:
    """Project primary and live (including hidden) split panes as topic targets."""
    primary = dict(session)
    if session.get("statusPane", "left") != "left":
        primary.pop("status", None)
    panes = [primary]
    if not session.get("hasSplit", session.get("split", False)):
        return panes
    foreground = session.get("splitForeground") or []
    signature = hashlib.sha256(json.dumps(foreground).encode()).hexdigest()[:20]
    peer = dict(session)
    peer.update(
        id=f"{session['id']}{_SPLIT_MARKER}{signature}",
        name=f"{session.get('name') or 'agterm'} · split",
        cwd=session.get("splitCwd") or "",
        foreground=foreground,
        title="",
        _agterm_session_id=session["id"],
    )
    if session.get("statusPane") != "right":
        peer.pop("status", None)
    panes.append(peer)
    return panes


def hook_pane(session: dict, provider: str, session_id: str) -> dict | None:
    """Resolve a hook against live argv, not its possibly stale baked pane role.

    Exact CLI session IDs distinguish two agents of the same kind. Otherwise
    require a unique provider match; ambiguity must not overwrite either binding.
    """
    # Lazy: providers reference the multiplexer seam during registry setup.
    from ..providers.process_detection import classify_provider_from_argv

    matches: list[dict] = []
    for pane in pane_sessions(session):
        argv = pane.get("foreground")
        if not isinstance(argv, list) or classify_provider_from_argv(argv) != provider:
            continue
        if "--session-id" in argv:
            index = argv.index("--session-id") + 1
            if index >= len(argv) or argv[index] != session_id:
                continue
        matches.append(pane)
    return matches[0] if len(matches) == 1 else None
