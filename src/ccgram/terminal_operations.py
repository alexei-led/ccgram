"""Backend-neutral terminal operation service.

Thin façade over the backend router that exposes the operations
handlers actually call (``send_text``, ``capture``, ``send_key``,
``close``, ``list_units``) keyed by the legacy ``window_id``. The
service resolves the window's terminal identity via the feature port,
dispatches to the matching backend, and translates the normalised
error taxonomy back into the user-facing strings the existing tmux
flow already returns.

Existence checks live inside the backend adapter (``TmuxBackend``
raises ``TerminalNotFoundError`` when the underlying window is gone);
this module converts those into the same ``(False, "Window not
found …")`` tuples ``tmux_manager.send_to_window`` used to return.
"""

from __future__ import annotations

import structlog

from .terminal_backends.base import (
    BACKEND_TMUX,
    TerminalBackend,
    TerminalBackendError,
    TerminalNotFoundError,
    TerminalUnit,
    TerminalUnitRef,
)
from .thread_router import thread_router
from .window_state_ports.terminal_identity import get_terminal_identity

logger = structlog.get_logger()


def _resolve_ref(window_id: str) -> TerminalUnitRef:
    """Return the ``TerminalUnitRef`` for ``window_id``.

    Legacy rows with no terminal identity stored resolve to a tmux ref
    with ``unit_id == window_id`` — this is the back-compat contract
    Task 1 established.
    """
    identity = get_terminal_identity(window_id)
    if identity is None:
        return TerminalUnitRef(backend=BACKEND_TMUX, unit_id=window_id)
    unit_id = identity.unit_id or window_id
    return TerminalUnitRef(backend=identity.backend, unit_id=unit_id)


def _backend_for(ref: TerminalUnitRef) -> TerminalBackend:
    # Lazy: keep terminal_operations import-light so config/test code
    # can import it without dragging the router (and libtmux) in.
    from .terminal_backends.router import get_router

    return get_router().for_ref(ref)


async def send_text(
    window_id: str, text: str, *, raw: bool = False
) -> tuple[bool, str]:
    """Send literal text to the window's terminal unit.

    Returns ``(success, message)`` matching the legacy
    ``send_to_window`` contract: success carries ``"Sent to {display}"``
    so existing handler code keeps its UX. Backend errors degrade to a
    user-facing string scoped to the failure code.
    """
    display = thread_router.get_display_name(window_id)
    logger.debug(
        "terminal_operations.send_text: window_id=%s (%s), text_len=%d",
        window_id,
        display,
        len(text),
    )
    ref = _resolve_ref(window_id)
    try:
        backend = _backend_for(ref)
    except TerminalBackendError as exc:
        logger.debug(
            "send_text backend unavailable", window_id=window_id, code=exc.code
        )
        return False, f"Terminal backend unavailable ({exc.code})"
    try:
        success = await backend.send_text(ref, text, raw=raw)
    except TerminalNotFoundError:
        return False, "Window not found (may have been closed)"
    except TerminalBackendError as exc:
        logger.debug("send_text rejected", window_id=window_id, code=exc.code)
        return False, f"Terminal backend error ({exc.code})"
    if success:
        return True, f"Sent to {display}"
    return False, "Failed to send keys"


async def capture(window_id: str, *, with_ansi: bool = False) -> str | None:
    """Capture pane text for the window.

    Returns the captured text or ``None`` on transient failure. Raises
    ``TerminalNotFoundError`` when the unit is gone so callers can
    surface the existing "Window no longer exists" message distinctly
    from a capture failure.
    """
    ref = _resolve_ref(window_id)
    backend = _backend_for(ref)
    return await backend.capture(ref, with_ansi=with_ansi)


async def send_key(window_id: str, key: str) -> bool:
    """Send a non-literal key (``Up``, ``Escape``, ``C-c``)."""
    ref = _resolve_ref(window_id)
    backend = _backend_for(ref)
    try:
        return await backend.send_key(ref, key)
    except TerminalNotFoundError:
        return False


async def close(window_id: str) -> bool:
    """Close/kill the window's terminal unit."""
    ref = _resolve_ref(window_id)
    backend = _backend_for(ref)
    return await backend.close(ref)


async def list_units(backend_name: str = BACKEND_TMUX) -> list[TerminalUnit]:
    """Enumerate terminal units known to a specific backend."""
    # Lazy: keep terminal_operations import-light so config/test code
    # can import it without dragging the router (and libtmux) in.
    from .terminal_backends.router import get_router

    backend = get_router().get(backend_name)
    return await backend.list_units()


__all__ = [
    "capture",
    "close",
    "list_units",
    "send_key",
    "send_text",
]
