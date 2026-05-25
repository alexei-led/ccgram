"""Terminal backend contract package — backend-neutral terminal identity and ops.

Public surface is intentionally narrow: handlers and the future backend
router import only DTOs and the ``TerminalBackend`` protocol from here.
Backend implementations live in sibling modules (``tmux.py``, ``cmux.py``)
and stay private to this package; the router introduced in Task 2 is the
only consumer.

This module exists in Task 1 as a safety net: the contract types ship
before any behaviour-bearing code is wrapped behind them. Adding new
exports here means widening the published vocabulary — keep it small.
"""

from __future__ import annotations

from .base import (
    BACKEND_CMUX,
    BACKEND_TMUX,
    KNOWN_BACKENDS,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalBackendError,
    TerminalBackendErrorCode,
    TerminalBackendUnavailableError,
    TerminalNotFoundError,
    TerminalOperationRejectedError,
    TerminalOperationTimeoutError,
    TerminalUnit,
    TerminalUnitRef,
    TerminalUnitState,
    TerminalUnsupportedOperationError,
)

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
