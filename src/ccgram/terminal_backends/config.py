"""Terminal backend config — typed projection over ``CCGRAM_*`` env vars.

This module is the only place inside ``src/ccgram`` that reads
``CCGRAM_TERMINAL_BACKEND*`` and ``CCGRAM_CMUX_*`` environment variables.
Handlers and adapters consume the typed ``TerminalBackendConfig`` instead.

User-facing knobs are intentionally small:

* ``CCGRAM_TERMINAL_BACKEND_DEFAULT`` — ``tmux`` (default) or ``cmux``.
* ``CCGRAM_TERMINAL_BACKEND`` — alias for ``CCGRAM_TERMINAL_BACKEND_DEFAULT``.
* ``CCGRAM_CMUX_ENABLED`` — ``false`` by default; truthy enables native cmux.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .base import BACKEND_CMUX, BACKEND_TMUX, KNOWN_BACKENDS

ENV_TERMINAL_BACKEND = "CCGRAM_TERMINAL_BACKEND"
ENV_TERMINAL_BACKEND_DEFAULT = "CCGRAM_TERMINAL_BACKEND_DEFAULT"
ENV_CMUX_ENABLED = "CCGRAM_CMUX_ENABLED"

DEFAULT_BACKEND = BACKEND_TMUX

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


@dataclass(frozen=True, slots=True)
class TerminalBackendConfig:
    """Typed projection of terminal-backend related env vars."""

    default_backend: str = DEFAULT_BACKEND
    cmux_enabled: bool = False
    raw_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default_backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"default_backend must be one of {sorted(KNOWN_BACKENDS)!r}, "
                f"got {self.default_backend!r}"
            )

    @property
    def cmux_active(self) -> bool:
        """True when native cmux integration should be registered."""
        return self.cmux_enabled

    def with_default(self, backend: str) -> TerminalBackendConfig:
        """Return a copy with a new default backend (test helper)."""
        return TerminalBackendConfig(
            default_backend=backend,
            cmux_enabled=self.cmux_enabled,
            raw_env=dict(self.raw_env),
        )


def load_terminal_backend_config(
    env: "os._Environ[str] | dict[str, str] | None" = None,
    *,
    config_dir: object | None = None,
) -> TerminalBackendConfig:
    """Build a :class:`TerminalBackendConfig` from environment variables.

    ``config_dir`` is accepted for call-site compatibility; native cmux does not
    need a ccgram-managed socket path.
    """
    del config_dir
    source = os.environ if env is None else env

    raw_env: dict[str, str] = {}
    for name in (
        ENV_TERMINAL_BACKEND,
        ENV_TERMINAL_BACKEND_DEFAULT,
        ENV_CMUX_ENABLED,
    ):
        if name in source:
            raw_env[name] = source[name]

    raw_backend = source.get(
        ENV_TERMINAL_BACKEND,
        source.get(ENV_TERMINAL_BACKEND_DEFAULT, DEFAULT_BACKEND),
    )
    default_backend = (raw_backend or DEFAULT_BACKEND).strip()
    if default_backend not in KNOWN_BACKENDS:
        raise ValueError(
            f"{ENV_TERMINAL_BACKEND} must be one of "
            f"{sorted(KNOWN_BACKENDS)!r}, got {default_backend!r}"
        )

    cmux_enabled = _parse_bool(source.get(ENV_CMUX_ENABLED), default=False)

    return TerminalBackendConfig(
        default_backend=default_backend,
        cmux_enabled=cmux_enabled,
        raw_env=raw_env,
    )


__all__ = [
    "DEFAULT_BACKEND",
    "ENV_CMUX_ENABLED",
    "ENV_TERMINAL_BACKEND",
    "ENV_TERMINAL_BACKEND_DEFAULT",
    "TerminalBackendConfig",
    "load_terminal_backend_config",
]


assert BACKEND_TMUX in KNOWN_BACKENDS
assert BACKEND_CMUX in KNOWN_BACKENDS
