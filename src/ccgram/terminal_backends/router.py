"""Terminal backend registry — maps ``TerminalUnitRef`` to a backend.

The router is the single lookup seam handlers use after Task 2: code
that needs to send/capture/close a terminal unit asks the router for
the backend instance keyed by ``ref.backend`` and dispatches through
the protocol. Backend-specific imports stay inside backend modules.

Default registrations are lazy: the first call to ``get_router()``
constructs the singleton and registers ``TmuxBackend`` wrapping the
existing ``tmux_manager`` so legacy behaviour keeps working with no
explicit bootstrap step. Tests call ``reset_router_for_testing()`` to
get a fresh registry.
"""

from __future__ import annotations

from .base import (
    KNOWN_BACKENDS,
    TerminalBackend,
    TerminalBackendUnavailableError,
    TerminalUnitRef,
)


class TerminalBackendRouter:
    """In-memory map of ``backend name -> TerminalBackend`` adapter."""

    def __init__(self) -> None:
        self._backends: dict[str, TerminalBackend] = {}

    def register(self, backend: TerminalBackend) -> None:
        """Register or replace a backend implementation.

        Raises ``ValueError`` for unknown backend names so a typo never
        silently becomes a dead route.
        """
        name = backend.name
        if name not in KNOWN_BACKENDS:
            raise ValueError(
                f"unknown terminal backend {name!r}; "
                f"expected one of {sorted(KNOWN_BACKENDS)!r}"
            )
        self._backends[name] = backend

    def unregister(self, name: str) -> None:
        """Remove a registered backend (no-op if absent)."""
        self._backends.pop(name, None)

    def known(self) -> frozenset[str]:
        """Names of currently registered backends."""
        return frozenset(self._backends)

    def get(self, name: str) -> TerminalBackend:
        """Look up a backend by name.

        Raises ``TerminalBackendUnavailableError`` when the backend has
        not been registered (e.g., cmux disabled by config).
        """
        backend = self._backends.get(name)
        if backend is None:
            raise TerminalBackendUnavailableError(
                f"terminal backend {name!r} is not registered"
            )
        return backend

    def for_ref(self, ref: TerminalUnitRef) -> TerminalBackend:
        """Look up the backend that owns ``ref``."""
        return self.get(ref.backend)


_router_instance: TerminalBackendRouter | None = None


def get_router() -> TerminalBackendRouter:
    """Return the process-wide router, registering defaults on first use."""
    global _router_instance
    if _router_instance is None:
        _router_instance = TerminalBackendRouter()
        # Lazy: tmux adapter pulls in libtmux/tmux_manager — keep the
        # router module import-light so config/test code can import it
        # without dragging the whole tmux stack.
        from .tmux import TmuxBackend

        # Lazy: tmux_manager module owns libtmux startup; keep router
        # registration the only entry point that pays that import cost.
        from ..tmux_manager import tmux_manager

        _router_instance.register(TmuxBackend(tmux_manager))
    return _router_instance


def reset_router_for_testing() -> None:
    """Drop the cached router so the next ``get_router()`` rebuilds it."""
    global _router_instance
    _router_instance = None


__all__ = [
    "TerminalBackendRouter",
    "get_router",
    "reset_router_for_testing",
]
