"""Reliable window listings for destructive state reconciliation."""

from __future__ import annotations

import inspect
from typing import Protocol, cast

from .base import WindowRef


class _ReconciliationWindowLister(Protocol):
    async def list_windows_for_reconciliation(self) -> list[WindowRef] | None:
        """Return None when a reliable window listing is unavailable."""


class _TargetedPresenceProbe(Protocol):
    async def window_exists(self, window_id: str) -> bool | None:
        """True, False, or None when the backend could not answer."""


async def list_windows_for_reconciliation(
    backend: object | None = None,
) -> list[WindowRef] | None:
    """Return a confirmed window listing, or None when it is unavailable.

    ``Multiplexer.list_windows()`` remains best-effort for user-facing reads.
    State cleanup must call this stronger backend contract so a failed listing
    cannot be treated as proof that every tracked window is gone.
    """
    backend = _resolve_backend(backend)

    method = getattr(backend, "list_windows_for_reconciliation", None)
    if not callable(method):
        name = getattr(getattr(backend, "capabilities", None), "name", "unknown")
        raise RuntimeError(
            f"Multiplexer backend {name!r} does not support reconciliation listings"
        )
    lister = cast("_ReconciliationWindowLister", backend)
    return await lister.list_windows_for_reconciliation()


def _resolve_backend(backend: object | None) -> object:
    """The concrete backend behind whatever the caller passed.

    Callers hold the module facade, whose type defines only ``__getattr__``, so
    a static lookup on it finds no methods at all. Resolving through it first
    is what lets the targeted-probe check below see the real backend instead of
    silently deciding it has none.
    """
    # Lazy: importing multiplexer package state at module load forms a cycle.
    from . import _MultiplexerProxy, get_active_multiplexer

    if backend is None or isinstance(backend, _MultiplexerProxy):
        return get_active_multiplexer()
    return backend


async def window_snapshot(
    window_id: str, backend: object | None = None
) -> tuple[bool, WindowRef | None]:
    """One confirmed read: ``(confirmed, window)``.

    ``confirmed`` is False when the backend could not be asked, and ``window``
    is then meaningless. When confirmed, ``window`` is the ref or ``None`` if
    it is genuinely gone.

    Callers that need to inspect the window as well as its existence take this
    instead of a presence check followed by ``find_window_by_id``: that second
    lookup is a second chance to fail, and its ``None`` is ambiguous again.
    """
    windows = await list_windows_for_reconciliation(backend)
    if windows is None:
        return False, None
    return True, next((w for w in windows if w.matches(window_id)), None)


async def window_presence(window_id: str, backend: object | None = None) -> bool | None:
    """Is this window there? ``True`` yes, ``False`` gone, ``None`` cannot say.

    The tri-state form of the existence question, on the seam because every
    layer asks it. ``Multiplexer.find_window_by_id`` cannot answer it: it
    returns ``None`` both for a window that is gone and for a backend that
    could not be reached, so a caller that closes a topic, unbinds a thread or
    switches a provider on that answer does so during an outage too.

    Read it immediately before the mutation, per candidate — these loops span
    Telegram round-trips, so one verdict for a batch is stale by the second
    item.
    """
    backend = _resolve_backend(backend)

    # A backend that can answer about one window directly is authoritative, and
    # once it declares that, its answer is the only one taken. agterm is such a
    # backend: its listing is assembled from per-window RPCs and cannot be a
    # snapshot, while a targeted call distinguishes "no such session" from a
    # failure to ask. Falling back to the aggregate on a malformed answer would
    # reach for the listing this probe exists to avoid, so anything that is not
    # a bool becomes unknown.
    #
    # getattr_static, because it reads the type without running __getattr__: a
    # plain getattr is satisfied by any test double that generates attributes
    # on demand, which would route real backends through a probe that answers
    # nothing.
    if inspect.iscoroutinefunction(
        inspect.getattr_static(type(backend), "window_exists", None)
    ):
        prober = cast("_TargetedPresenceProbe", backend)
        answer = await prober.window_exists(window_id)
        return answer if isinstance(answer, bool) else None

    confirmed, window = await window_snapshot(window_id, backend)
    return window is not None if confirmed else None
