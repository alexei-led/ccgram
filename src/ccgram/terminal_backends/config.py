"""Terminal backend config — typed projection over ``CCGRAM_*`` env vars.

This module is the only place inside ``src/ccgram`` that reads
``CCGRAM_TERMINAL_BACKEND*`` and ``CCGRAM_CMUX_*`` environment
variables. Handlers and adapters consume the typed
``TerminalBackendConfig`` instead, so cmux env knobs never leak into
handler code. The boundary is enforced by
``tests/ccgram/test_backend_config_access_audit.py``.

Defaults keep cmux disabled. The router never auto-registers cmux —
bootstrap consults this config and calls a dedicated registration step.

Env vars:

* ``CCGRAM_TERMINAL_BACKEND_DEFAULT`` — ``tmux`` (default) or ``cmux``.
* ``CCGRAM_CMUX_ENABLED`` — ``false`` by default; truthy enables cmux.
* ``CCGRAM_CMUX_SIDECAR_SOCKET`` — Unix socket path; defaults to
  ``$CCGRAM_DIR/cmux-sidecar.sock``.
* ``CCGRAM_CMUX_SIDECAR_TIMEOUT_MS`` — per-request timeout (default
  ``5000``; clamped to a non-negative integer).
* ``CCGRAM_CMUX_PROTOCOL_VERSION`` — optional minimum protocol version
  the client requires from the sidecar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .base import BACKEND_CMUX, BACKEND_TMUX, KNOWN_BACKENDS

ENV_TERMINAL_BACKEND_DEFAULT = "CCGRAM_TERMINAL_BACKEND_DEFAULT"
ENV_CMUX_ENABLED = "CCGRAM_CMUX_ENABLED"
ENV_CMUX_SIDECAR_SOCKET = "CCGRAM_CMUX_SIDECAR_SOCKET"
ENV_CMUX_SIDECAR_TIMEOUT_MS = "CCGRAM_CMUX_SIDECAR_TIMEOUT_MS"
ENV_CMUX_PROTOCOL_VERSION = "CCGRAM_CMUX_PROTOCOL_VERSION"

DEFAULT_BACKEND = BACKEND_TMUX
DEFAULT_CMUX_TIMEOUT_MS = 5000
DEFAULT_SIDECAR_SOCKET_BASENAME = "cmux-sidecar.sock"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"", "0", "false", "no", "off"})


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSEY:
        return False
    raise ValueError(
        f"expected boolean (one of {sorted(_TRUTHY | _FALSEY)!r}), got {raw!r}"
    )


def _parse_positive_int(raw: str | None, *, default: int, name: str) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class TerminalBackendConfig:
    """Typed projection of terminal-backend related env vars.

    Constructed by :func:`load_terminal_backend_config` once at startup
    and passed to backend registration. Equality on this dataclass is
    by value, so tests can assemble synthetic configs without touching
    the environment.
    """

    default_backend: str = DEFAULT_BACKEND
    cmux_enabled: bool = False
    cmux_sidecar_socket: Path | None = None
    cmux_sidecar_timeout_ms: int = DEFAULT_CMUX_TIMEOUT_MS
    cmux_protocol_version: str | None = None
    raw_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default_backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"default_backend must be one of {sorted(KNOWN_BACKENDS)!r}, "
                f"got {self.default_backend!r}"
            )
        if self.cmux_sidecar_timeout_ms < 0:
            raise ValueError(
                f"cmux_sidecar_timeout_ms must be non-negative, "
                f"got {self.cmux_sidecar_timeout_ms}"
            )
        if self.cmux_enabled and self.cmux_sidecar_socket is None:
            raise ValueError("cmux_enabled=True requires cmux_sidecar_socket to be set")

    @property
    def cmux_active(self) -> bool:
        """True when cmux is fully usable (enabled and configured)."""
        return self.cmux_enabled and self.cmux_sidecar_socket is not None

    def with_default(self, backend: str) -> TerminalBackendConfig:
        """Return a copy with a new default backend (test helper)."""
        return TerminalBackendConfig(
            default_backend=backend,
            cmux_enabled=self.cmux_enabled,
            cmux_sidecar_socket=self.cmux_sidecar_socket,
            cmux_sidecar_timeout_ms=self.cmux_sidecar_timeout_ms,
            cmux_protocol_version=self.cmux_protocol_version,
            raw_env=dict(self.raw_env),
        )


def _resolve_socket_path(raw: str | None, *, config_dir: Path | None) -> Path | None:
    if raw is not None and raw.strip():
        return Path(raw).expanduser()
    if config_dir is None:
        return None
    return config_dir / DEFAULT_SIDECAR_SOCKET_BASENAME


def load_terminal_backend_config(
    env: "os._Environ[str] | dict[str, str] | None" = None,
    *,
    config_dir: Path | None = None,
) -> TerminalBackendConfig:
    """Build a :class:`TerminalBackendConfig` from environment variables.

    ``env`` defaults to :data:`os.environ`. ``config_dir`` is the
    ``~/.ccgram`` directory (or override) used to derive the default
    sidecar socket path when no override is set; pass ``None`` to skip
    that fallback.

    Raises :class:`ValueError` for malformed entries. Unknown backend
    names in ``CCGRAM_TERMINAL_BACKEND_DEFAULT`` are rejected so a typo
    never silently degrades to tmux.
    """
    source = os.environ if env is None else env

    raw_env: dict[str, str] = {}
    for name in (
        ENV_TERMINAL_BACKEND_DEFAULT,
        ENV_CMUX_ENABLED,
        ENV_CMUX_SIDECAR_SOCKET,
        ENV_CMUX_SIDECAR_TIMEOUT_MS,
        ENV_CMUX_PROTOCOL_VERSION,
    ):
        if name in source:
            raw_env[name] = source[name]

    default_backend = (
        source.get(ENV_TERMINAL_BACKEND_DEFAULT, DEFAULT_BACKEND) or DEFAULT_BACKEND
    ).strip()
    if default_backend not in KNOWN_BACKENDS:
        raise ValueError(
            f"{ENV_TERMINAL_BACKEND_DEFAULT} must be one of "
            f"{sorted(KNOWN_BACKENDS)!r}, got {default_backend!r}"
        )

    cmux_enabled = _parse_bool(source.get(ENV_CMUX_ENABLED), default=False)
    cmux_socket = _resolve_socket_path(
        source.get(ENV_CMUX_SIDECAR_SOCKET), config_dir=config_dir
    )
    timeout_ms = _parse_positive_int(
        source.get(ENV_CMUX_SIDECAR_TIMEOUT_MS),
        default=DEFAULT_CMUX_TIMEOUT_MS,
        name=ENV_CMUX_SIDECAR_TIMEOUT_MS,
    )
    protocol_version_raw = source.get(ENV_CMUX_PROTOCOL_VERSION)
    protocol_version = (
        protocol_version_raw.strip()
        if protocol_version_raw is not None and protocol_version_raw.strip()
        else None
    )

    if cmux_enabled and cmux_socket is None:
        raise ValueError(
            f"{ENV_CMUX_ENABLED} is truthy but no sidecar socket is configured; "
            f"set {ENV_CMUX_SIDECAR_SOCKET} or provide a config_dir"
        )

    return TerminalBackendConfig(
        default_backend=default_backend,
        cmux_enabled=cmux_enabled,
        cmux_sidecar_socket=cmux_socket,
        cmux_sidecar_timeout_ms=timeout_ms,
        cmux_protocol_version=protocol_version,
        raw_env=raw_env,
    )


__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_CMUX_TIMEOUT_MS",
    "DEFAULT_SIDECAR_SOCKET_BASENAME",
    "ENV_CMUX_ENABLED",
    "ENV_CMUX_PROTOCOL_VERSION",
    "ENV_CMUX_SIDECAR_SOCKET",
    "ENV_CMUX_SIDECAR_TIMEOUT_MS",
    "ENV_TERMINAL_BACKEND_DEFAULT",
    "TerminalBackendConfig",
    "load_terminal_backend_config",
]


# Re-export only stable backend names so unrelated imports don't widen
# the public surface of this module.
assert BACKEND_TMUX in KNOWN_BACKENDS
assert BACKEND_CMUX in KNOWN_BACKENDS
