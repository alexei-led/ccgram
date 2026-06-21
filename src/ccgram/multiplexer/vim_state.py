"""Vim-insert detection state — backend-neutral, no I/O.

The tmux backend's ``send`` path probes for vim's ``-- INSERT --`` indicator
to avoid swallowing keystrokes; status polling notifies this cache when it
observes insert mode.  The state lives here (not in a concrete backend) so the
polling layer can import the detection helpers without importing the tmux
backend (F1 boundary).

``_vim_state`` / ``_vim_locks`` are shared mutable caches: the tmux backend
reads them in ``send``; ``notify_vim_insert_seen`` (called from polling) writes
them.  Both reference the dicts defined here, so mutations are visible across
importers.
"""

from __future__ import annotations

import asyncio
import re

from ..topic_state_registry import topic_state

# window_id → True (vim mode on) / False (off).  Missing key = unknown.
_vim_state: dict[str, bool] = {}

# Per-window locks serializing vim probe + send sequences to prevent
# interleaved keystrokes from concurrent send calls.
_vim_locks: dict[str, asyncio.Lock] = {}

_VIM_INSERT_RE = re.compile(r"^--\s*INSERT\s*--\s*$")


def has_insert_indicator(pane_text: str) -> bool:
    """Check if vim's ``-- INSERT --`` appears in the last 3 lines of pane text.

    Only matches lines where ``-- INSERT --`` is the sole content (with optional
    whitespace), avoiding false positives from Claude Code's own status bar which
    renders ``-- INSERT -- ⏸ plan mode on ...`` with trailing text.
    """
    return any(
        _VIM_INSERT_RE.search(line.strip()) for line in pane_text.splitlines()[-3:]
    )


def notify_vim_insert_seen(window_id: str) -> None:
    """Record that vim INSERT mode was observed (called from status polling)."""
    _vim_state[window_id] = True


@topic_state.register("window")
def clear_vim_state(window_id: str) -> None:
    """Remove vim state cache entry and lock for a window (called on cleanup)."""
    _vim_state.pop(window_id, None)
    _vim_locks.pop(window_id, None)


def reset_vim_state() -> None:
    """Reset all vim state (for testing)."""
    _vim_state.clear()
    _vim_locks.clear()
