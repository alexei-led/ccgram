"""Multiplexer package — backend-neutral terminal-multiplexer seam.

Exposes:

- ``get_multiplexer(name)`` — registry resolver (re-exported from ``registry``).
- ``multiplexer`` — module-level proxy that forwards to the wired backend.
- ``install_multiplexer`` / ``get_active_multiplexer`` — wiring used by
  ``bootstrap.py`` to select the backend from ``config.multiplexer_name``.

Callers import the ``multiplexer`` proxy and type against
``multiplexer.base.Multiplexer``. They must not import a concrete backend
(``multiplexer.tmux`` / ``multiplexer.herdr``) directly — enforced by the F1
boundary audit (Task 4). The proxy mirrors the ``window_store`` /
``thread_router`` proxy pattern: it resolves to the instance installed by
bootstrap and raises a clear error before wiring.
"""

from __future__ import annotations

from typing import Any, cast

from ccgram.multiplexer.base import Multiplexer
from ccgram.multiplexer.registry import (
    UnknownMultiplexerError,
    get_multiplexer,
    multiplexer_names,
)

__all__ = [
    "Multiplexer",
    "UnknownMultiplexerError",
    "get_active_multiplexer",
    "get_multiplexer",
    "install_multiplexer",
    "multiplexer",
    "multiplexer_names",
]


_active_multiplexer: Multiplexer | None = None


def get_active_multiplexer() -> Multiplexer:
    """Return the wired multiplexer backend.

    Raises:
        RuntimeError: when called before bootstrap has installed a backend.
    """
    if _active_multiplexer is None:
        raise RuntimeError(
            "Multiplexer not yet wired. "
            "bootstrap_application() must install a backend before use."
        )
    return _active_multiplexer


def install_multiplexer(backend: Multiplexer) -> None:
    """Install *backend* as the module-level multiplexer (called by bootstrap)."""
    global _active_multiplexer
    _active_multiplexer = backend


def _reset_multiplexer_for_testing() -> None:
    """Clear the wired backend so the proxy is unwired again (test isolation)."""
    global _active_multiplexer
    _active_multiplexer = None


class _MultiplexerProxy:
    """Module-level facade resolving to the wired backend.

    Mirrors the ``window_store`` / ``thread_router`` proxy pattern. All
    attribute access delegates to the bootstrap-installed ``Multiplexer``;
    raises ``RuntimeError`` if accessed before wiring.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(get_active_multiplexer(), name)

    def __repr__(self) -> str:
        if _active_multiplexer is None:
            return "<MultiplexerProxy unwired>"
        return f"<MultiplexerProxy → {_active_multiplexer!r}>"


multiplexer: Multiplexer = cast("Multiplexer", _MultiplexerProxy())
