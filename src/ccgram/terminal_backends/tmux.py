"""TmuxBackend — ``TerminalBackend`` adapter over the existing tmux manager.

Wraps ``TmuxManager`` so callers can route through the backend
protocol without changing user-visible tmux behaviour. The adapter
preserves the explicit window pre-check that ``send_to_window`` and
the screenshot path rely on today: when the underlying tmux window is
gone, methods raise ``TerminalNotFoundError`` so the operation service
can map it back to the existing "Window not found" UX.

The adapter does not own state — all bookkeeping (vim cache, external
discovery, foreign session routing) stays in ``tmux_manager.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BACKEND_TMUX,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalNotFoundError,
    TerminalUnit,
    TerminalUnitRef,
    TerminalUnsupportedOperationError,
)

if TYPE_CHECKING:
    from ..tmux_manager import TmuxManager


class TmuxBackend(TerminalBackend):
    """``TerminalBackend`` adapter delegating to ``TmuxManager``."""

    def __init__(self, manager: TmuxManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return BACKEND_TMUX

    def capabilities(self) -> TerminalBackendCapabilities:
        return TerminalBackendCapabilities(
            backend=BACKEND_TMUX,
            supports_create=True,
            supports_list=True,
            supports_capture=True,
            supports_send_text=True,
            supports_send_key=True,
            supports_close=True,
            supports_resume=True,
            supports_event_stream=False,
            protocol_version="tmux-local",
        )

    def _check_backend(self, ref: TerminalUnitRef) -> None:
        if ref.backend != self.name:
            raise TerminalUnsupportedOperationError(
                f"TmuxBackend cannot service {ref.backend!r} units",
                ref=ref,
            )

    async def list_units(self) -> list[TerminalUnit]:
        windows = await self._manager.list_windows()
        units: list[TerminalUnit] = []
        for window in windows:
            ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id=window.window_id)
            units.append(
                TerminalUnit(
                    ref=ref,
                    title=window.window_name,
                    cwd=window.cwd or None,
                    state="unknown",
                    supports_capture=True,
                    supports_send_text=True,
                    supports_send_key=True,
                    supports_close=True,
                    supports_resume=True,
                    backend_metadata={
                        "pane_current_command": window.pane_current_command,
                        "pane_tty": window.pane_tty,
                    },
                )
            )
        return units

    async def capture(
        self, ref: TerminalUnitRef, *, with_ansi: bool = False
    ) -> str | None:
        self._check_backend(ref)
        window = await self._manager.find_window_by_id(ref.unit_id)
        if window is None:
            raise TerminalNotFoundError("tmux window not found", ref=ref)
        return await self._manager.capture_pane(window.window_id, with_ansi=with_ansi)

    async def send_text(
        self, ref: TerminalUnitRef, text: str, *, raw: bool = False
    ) -> bool:
        self._check_backend(ref)
        window = await self._manager.find_window_by_id(ref.unit_id)
        if window is None:
            raise TerminalNotFoundError("tmux window not found", ref=ref)
        return await self._manager.send_keys(window.window_id, text, raw=raw)

    async def send_key(self, ref: TerminalUnitRef, key: str) -> bool:
        self._check_backend(ref)
        window = await self._manager.find_window_by_id(ref.unit_id)
        if window is None:
            raise TerminalNotFoundError("tmux window not found", ref=ref)
        return await self._manager.send_keys(
            window.window_id, key, enter=False, literal=False
        )

    async def close(self, ref: TerminalUnitRef) -> bool:
        self._check_backend(ref)
        return await self._manager.kill_window(ref.unit_id)


__all__ = ["TmuxBackend"]
