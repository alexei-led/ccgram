"""Multiplexer registry — resolves backend names to cached instances.

Mirrors ``providers/registry.py``: one instance per backend name, created
lazily on first ``get_multiplexer(name)``. Backends are imported lazily inside
their factory so importing this module pulls in no backend I/O library
(``libtmux``, herdr socket client) — the core stays I/O-free.

The "tmux" backend reuses the existing ``multiplexer.tmux.tmux_manager``
singleton so there is exactly one tmux server connection. "herdr" is added in
Task 7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import structlog

if TYPE_CHECKING:
    from ccgram.multiplexer.base import Multiplexer

logger = structlog.get_logger()


class UnknownMultiplexerError(LookupError):
    """Raised when requesting a multiplexer name that is not registered."""


def _make_tmux() -> Multiplexer:
    # Lazy: importing the tmux backend pulls in libtmux; keep it out of the
    # registry's import-time dependencies so importing the seam stays I/O-free.
    from ccgram.multiplexer.tmux import tmux_manager

    return tmux_manager


_FACTORIES: dict[str, Callable[[], Multiplexer]] = {
    "tmux": _make_tmux,
}

_instances: dict[str, Multiplexer] = {}


def multiplexer_names() -> list[str]:
    """Return all registered backend names."""
    return list(_FACTORIES)


def get_multiplexer(name: str) -> Multiplexer:
    """Return a cached multiplexer backend instance for *name*.

    One instance per name — repeated calls return the same object.
    Raises ``UnknownMultiplexerError`` if *name* is not registered.
    """
    cached = _instances.get(name)
    if cached is not None:
        return cached
    factory = _FACTORIES.get(name)
    if factory is None:
        available = ", ".join(sorted(_FACTORIES)) or "(none)"
        raise UnknownMultiplexerError(
            f"Unknown multiplexer {name!r}. Available: {available}"
        )
    instance = factory()
    _instances[name] = instance
    logger.debug("Resolved multiplexer backend %r", name)
    return instance


def _reset_multiplexer_cache_for_testing() -> None:
    """Drop cached backend instances so tests build fresh backends."""
    _instances.clear()
