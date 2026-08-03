"""Backend-neutral hook identity resolver.

The Claude Code hook runs as a separate process spawned inside a multiplexer
pane; it cannot import bot config or wire the ``multiplexer`` proxy. It only
needs to answer "which window am I?" from the environment. Each backend exposes
that differently — tmux via ``$TMUX_PANE`` + ``tmux display-message``, herdr via
``$HERDR_PANE_ID`` — so this module picks the backend by which
``self_identify_env`` variable is present (never a ``name == "<backend>"``
conditional) and returns a neutral ``SelfIdentity``.

The tmux probe (a ``display-message`` subprocess) is injected as ``tmux_query``
so this module stays I/O-free and table-testable; the hook supplies its own
``_resolve_window_id`` as the default probe. The herdr branch resolves the
``(workspace_id, pane_id)`` locator through an injected ``herdr_query``. It
never falls back to a pane or tab identifier: absent, missing, or ambiguous
live records cause the hook to skip its write.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

# tmux_query returns ``(session_window_key, window_id, window_name, pane_tty)``
# or None on failure — the exact shape of ``hook._resolve_window_id``.
TmuxQuery = Callable[[str], "tuple[str, str, str, str] | None"]

# herdr_query returns one opaque session target for a ``(workspace_id, pane_id)``
# locator, or None when no unique live record exists.  The resolver must fail
# closed rather than bind a pane or tab key.
HerdrQuery = Callable[[str, str], "str | None"]


@dataclass(frozen=True)
class SelfIdentity:
    """Neutral identity of the window that fired the hook.

    ``session_window_key`` is the ``session_map.json`` key (``<session>:<id>``
    for tmux, ``herdr:<opaque-target-id>`` for herdr). ``pane_tty`` is tmux-only
    (herdr does not expose a tty).
    """

    mux: str
    session_window_key: str
    window_id: str
    window_name: str
    pane_tty: str = ""


def resolve_self_identity(
    env: Mapping[str, str],
    *,
    tmux_query: TmuxQuery,
    herdr_query: HerdrQuery | None = None,
) -> SelfIdentity | None:
    """Resolve the firing window's identity from ``env``.

    Dispatches on which backend's ``self_identify_env`` var is present:
    ``$TMUX_PANE`` → tmux (via ``tmux_query``), ``$HERDR_PANE_ID`` → herdr.
    Returns None when neither is set or the tmux probe fails (today's
    "cannot determine window" path). tmux wins when both are present (a herdr
    pane running inside a tmux pane still reports the outer tmux identity).

    For herdr: ``herdr_query(workspace_id, pane_id)`` resolves the exact live
    locator to a guarded session target, so ``session_window_key`` becomes
    ``herdr:<opaque-target-id>``. Returns None when either locator component is
    unavailable or the probe has zero or multiple matches; the hook then skips
    the session_map write rather than recording a location as identity.
    """
    tmux_pane = env.get("TMUX_PANE", "")
    if tmux_pane:
        resolved = tmux_query(tmux_pane)
        if resolved is None:
            return None
        session_window_key, window_id, window_name, pane_tty = resolved
        return SelfIdentity(
            mux="tmux",
            session_window_key=session_window_key,
            window_id=window_id,
            window_name=window_name,
            pane_tty=pane_tty,
        )

    herdr_pane = env.get("HERDR_PANE_ID", "")
    if herdr_pane:
        workspace_id = env.get("HERDR_WORKSPACE_ID", "")
        target_id = (
            herdr_query(workspace_id, herdr_pane)
            if herdr_query is not None and workspace_id
            else None
        )
        if target_id is None:
            return None
        return SelfIdentity(
            mux="herdr",
            session_window_key=f"herdr:{target_id}",
            window_id=target_id,
            window_name="",
        )

    return None
