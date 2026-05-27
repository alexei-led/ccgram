"""Backend lifecycle wiring — bridge typed config to the router."""

from __future__ import annotations

from .base import BACKEND_CMUX, TerminalBackend
from .cmux import CmuxBackend
from .cmux_native_client import CmuxNativeClient
from .config import TerminalBackendConfig
from .router import get_router


def register_cmux_backend_if_enabled(
    config: TerminalBackendConfig,
    *,
    backend: TerminalBackend | None = None,
) -> TerminalBackend | None:
    """Register native cmux backend with the router when config permits."""
    if not config.cmux_active:
        return None
    if backend is None:
        backend = CmuxBackend(CmuxNativeClient())
    if backend.name != BACKEND_CMUX:
        raise ValueError(f"expected a cmux backend, got {backend.name!r}")
    get_router().register(backend)
    return backend


def unregister_cmux_backend() -> None:
    """Drop cmux from the router (used by test teardown)."""
    get_router().unregister(BACKEND_CMUX)


__all__ = [
    "register_cmux_backend_if_enabled",
    "unregister_cmux_backend",
]
