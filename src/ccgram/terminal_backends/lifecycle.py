"""Backend lifecycle wiring — bridge typed config to the router.

Task 3 introduced :class:`TerminalBackendConfig` and the cmux backend
adapter but deliberately left registration unwired so the router stayed
tmux-only. Task 4 needs a single, testable entry point that bootstrap
calls once at startup to register cmux when the config says so.

The helper is intentionally tiny — it owns the config → backend wiring
decision and nothing else. The router stays import-light (no config
dependency), and tests can call this with a synthetic config (or a
pre-built :class:`CmuxBackend`) without touching environment state.
"""

from __future__ import annotations

from .base import BACKEND_CMUX, TerminalBackend
from .cmux import CmuxBackend
from .cmux_client import CmuxSidecarClient
from .config import TerminalBackendConfig
from .router import get_router


def register_cmux_backend_if_enabled(
    config: TerminalBackendConfig,
    *,
    backend: TerminalBackend | None = None,
) -> TerminalBackend | None:
    """Register cmux backend with the router when config permits.

    ``backend`` is for tests that want to inject a fake-backed
    :class:`CmuxBackend`; production wires a real
    :class:`CmuxSidecarClient` over the configured socket path.

    Returns the registered backend (or ``None`` when cmux is disabled).
    """
    if not config.cmux_active:
        return None
    if backend is None:
        assert config.cmux_sidecar_socket is not None  # cmux_active guarantees this
        client = CmuxSidecarClient.with_unix_socket(
            str(config.cmux_sidecar_socket),
            timeout_ms=config.cmux_sidecar_timeout_ms,
            required_protocol_version=config.cmux_protocol_version,
        )
        backend = CmuxBackend(client)
    if backend.name != BACKEND_CMUX:
        raise ValueError(
            f"expected a cmux backend, got {backend.name!r}",
        )
    get_router().register(backend)
    return backend


def unregister_cmux_backend() -> None:
    """Drop cmux from the router (used by test teardown)."""
    get_router().unregister(BACKEND_CMUX)


__all__ = [
    "register_cmux_backend_if_enabled",
    "unregister_cmux_backend",
]
