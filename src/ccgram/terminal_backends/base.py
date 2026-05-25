"""Terminal backend contract — DTOs, capability flags, error taxonomy, protocol.

This module is the published vocabulary that handlers and the backend
router share. Backend implementations (``tmux.py``, ``cmux.py``) consume
the protocol; backend-specific identifiers (``@N`` for tmux, workspace
UUIDs for cmux) never escape the adapter that owns them.

Key types:

* ``TerminalUnitRef`` — opaque ``(backend, unit_id)`` pair used by every
  terminal-bound operation. Handlers do not parse ``unit_id``.
* ``TerminalUnit`` — read projection used by dashboards and pickers.
* ``TerminalBackendCapabilities`` — capability flags gating UI buttons.
* ``TerminalBackend`` — the operation protocol (Task 2 wires this up).
* ``TerminalBackendError`` and subclasses — normalised error taxonomy
  matching the sidecar's stable error codes
  (``unavailable``/``not_found``/``unsupported``/``timeout``/
  ``rejected``/``internal_error``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, runtime_checkable

BACKEND_TMUX: Final[str] = "tmux"
BACKEND_CMUX: Final[str] = "cmux"
KNOWN_BACKENDS: Final[frozenset[str]] = frozenset({BACKEND_TMUX, BACKEND_CMUX})

TerminalUnitState = Literal[
    "starting", "working", "waiting", "ready", "dead", "unknown"
]

TerminalBackendErrorCode = Literal[
    "unavailable",
    "not_found",
    "unsupported",
    "no_terminal_surface",
    "timeout",
    "rejected",
    "internal_error",
]


@dataclass(frozen=True, slots=True)
class TerminalUnitRef:
    """Opaque ``(backend, unit_id)`` pair identifying a terminal unit.

    ``backend`` must be one of ``KNOWN_BACKENDS``; ``unit_id`` is
    backend-local. Handlers never parse the unit id — they call backend
    operations through the router. ``display_id`` is a stable human/debug
    string (``tmux:@3``, ``cmux:workspace:<uuid>``) safe to log.
    """

    backend: str
    unit_id: str

    def __post_init__(self) -> None:
        if not self.backend or not self.backend.strip():
            raise ValueError("TerminalUnitRef.backend must be a non-empty string")
        if self.backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"TerminalUnitRef.backend must be one of {sorted(KNOWN_BACKENDS)!r}, "
                f"got {self.backend!r}"
            )
        if not self.unit_id or not self.unit_id.strip():
            raise ValueError("TerminalUnitRef.unit_id must be a non-empty string")

    @property
    def display_id(self) -> str:
        return f"{self.backend}:{self.unit_id}"


@dataclass(frozen=True, slots=True)
class TerminalUnit:
    """Read-only projection of a terminal unit for dashboards/pickers.

    ``backend_metadata`` is for diagnostics only — user-visible code
    should rely on the first-class fields so backend specifics never
    become a covert API.
    """

    ref: TerminalUnitRef
    title: str = ""
    cwd: str | None = None
    provider_name: str | None = None
    state: TerminalUnitState = "unknown"
    supports_capture: bool = False
    supports_send_text: bool = False
    supports_send_key: bool = False
    supports_close: bool = False
    supports_resume: bool = False
    backend_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalBackendCapabilities:
    """Capability flags exposed by a backend implementation.

    Used by the router to gate UI buttons and reject unsupported flows
    before they reach the backend.
    """

    backend: str
    supports_create: bool = False
    supports_list: bool = False
    supports_capture: bool = False
    supports_send_text: bool = False
    supports_send_key: bool = False
    supports_close: bool = False
    supports_resume: bool = False
    supports_event_stream: bool = False
    protocol_version: str = ""

    def __post_init__(self) -> None:
        if self.backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"TerminalBackendCapabilities.backend must be one of "
                f"{sorted(KNOWN_BACKENDS)!r}, got {self.backend!r}"
            )


class TerminalBackendError(Exception):
    """Base error normalised across all terminal backend implementations.

    Carries a stable ``code`` matching the sidecar error taxonomy and an
    optional ``ref`` identifying the target unit. Subclasses set the
    appropriate code by default — raising the base class with an
    explicit ``code`` is allowed for ``internal_error`` cases that do not
    fit a subclass.
    """

    code: TerminalBackendErrorCode = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: TerminalBackendErrorCode | None = None,
        ref: TerminalUnitRef | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.ref = ref

    def __str__(self) -> str:
        base = super().__str__()
        if self.ref is None:
            return f"[{self.code}] {base}"
        return f"[{self.code}] {base} (unit={self.ref.display_id})"


class TerminalBackendUnavailableError(TerminalBackendError):
    """Backend (or its sidecar) is not reachable."""

    code: TerminalBackendErrorCode = "unavailable"


class TerminalNotFoundError(TerminalBackendError):
    """Target unit (workspace/window) no longer exists."""

    code: TerminalBackendErrorCode = "not_found"


class TerminalUnsupportedOperationError(TerminalBackendError):
    """Operation not supported by this backend or sidecar version."""

    code: TerminalBackendErrorCode = "unsupported"


class TerminalOperationTimeoutError(TerminalBackendError):
    """Operation exceeded its configured deadline."""

    code: TerminalBackendErrorCode = "timeout"


class TerminalOperationRejectedError(TerminalBackendError):
    """Backend refused the request (invalid input or unsafe target)."""

    code: TerminalBackendErrorCode = "rejected"


@runtime_checkable
class TerminalBackend(Protocol):
    """Backend-neutral terminal operations contract.

    Task 2 wires this up: ``TmuxBackend`` wraps the existing
    ``tmux_manager`` behaviour and the backend router dispatches to the
    matching adapter by ``TerminalUnitRef.backend``. Implementations
    expose their feature surface through ``capabilities()`` so the
    router can gate UI before issuing calls.

    Operations raise ``TerminalBackendError`` subclasses for normalised
    failure modes (``TerminalNotFoundError`` for missing units,
    ``TerminalUnsupportedOperationError`` for backend mismatches,
    ``TerminalBackendUnavailableError`` when the backend cannot be
    reached). A ``False`` / ``None`` return models a transient failure
    the caller may retry or surface as a generic error.
    """

    @property
    def name(self) -> str:
        """Backend name (``tmux`` / ``cmux``)."""
        ...

    def capabilities(self) -> TerminalBackendCapabilities:
        """Return the backend's capability projection.

        Implementations should refresh capability negotiation lazily —
        callers must not assume this is free.
        """
        ...

    async def list_units(self) -> list[TerminalUnit]:
        """Enumerate terminal units known to this backend."""
        ...

    async def capture(
        self, ref: TerminalUnitRef, *, with_ansi: bool = False
    ) -> str | None:
        """Capture the visible text of the target unit.

        Returns the captured text or ``None`` on transient capture
        failure. Raises ``TerminalNotFoundError`` when the unit is gone.
        """
        ...

    async def send_text(
        self, ref: TerminalUnitRef, text: str, *, raw: bool = False
    ) -> bool:
        """Send literal text (followed by Enter) to the target unit.

        ``raw`` requests a TUI-bypass send path for plain shells.
        Returns ``True`` on success, ``False`` on transient failure.
        Raises ``TerminalNotFoundError`` when the unit is gone.
        """
        ...

    async def send_key(self, ref: TerminalUnitRef, key: str) -> bool:
        """Send a single non-literal key (``Up``, ``Escape``, ``C-c``).

        Returns ``True`` on success, ``False`` on transient failure.
        Raises ``TerminalNotFoundError`` when the unit is gone.
        """
        ...

    async def close(self, ref: TerminalUnitRef) -> bool:
        """Close/kill the target unit. Returns ``True`` on success."""
        ...


__all__ = [
    "BACKEND_CMUX",
    "BACKEND_TMUX",
    "KNOWN_BACKENDS",
    "TerminalBackend",
    "TerminalBackendCapabilities",
    "TerminalBackendError",
    "TerminalBackendErrorCode",
    "TerminalBackendUnavailableError",
    "TerminalNotFoundError",
    "TerminalOperationRejectedError",
    "TerminalOperationTimeoutError",
    "TerminalUnit",
    "TerminalUnitRef",
    "TerminalUnitState",
    "TerminalUnsupportedOperationError",
]
