"""Backward-compat shim — the tmux backend now lives in ``multiplexer.tmux``.

Task 2 moved the implementation into the ``multiplexer`` package (the first
backend behind the ``Multiplexer`` contract).  This module re-exports the
public surface so existing ``from ccgram.tmux_manager import ...`` call sites
keep working unchanged.  Call sites migrate to the ``multiplexer`` proxy in
Task 4, at which point this shim is removed.
"""

from __future__ import annotations

from .multiplexer.tmux import (
    PaneInfo,
    TmuxManager,
    TmuxWindow,
    _vim_locks,
    _vim_state,
    clear_vim_state,
    has_insert_indicator,
    notify_vim_insert_seen,
    reset_vim_state,
    send_followup_to_window,
    send_to_window,
    tmux_manager,
)

__all__ = [
    "PaneInfo",
    "TmuxManager",
    "TmuxWindow",
    "_vim_locks",
    "_vim_state",
    "clear_vim_state",
    "has_insert_indicator",
    "notify_vim_insert_seen",
    "reset_vim_state",
    "send_followup_to_window",
    "send_to_window",
    "tmux_manager",
]
