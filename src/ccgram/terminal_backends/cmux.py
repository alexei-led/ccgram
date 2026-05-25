"""CmuxBackend — ``TerminalBackend`` adapter over the sidecar JSON-RPC client.

The adapter is intentionally thin: it maps :class:`TerminalUnitRef`
values (``backend="cmux"``, ``unit_id=<workspace_id>``) onto sidecar
calls and translates :class:`CmuxWorkspace` results into the neutral
:class:`TerminalUnit` projection. Capability negotiation is delegated
to the sidecar's ``hello`` handshake.

The backend is never registered automatically — bootstrap consults the
typed :class:`TerminalBackendConfig` and registers a CmuxBackend only
when ``cmux_active`` is true. Tests construct backends with an injected
:class:`CmuxSidecarClient` (and a fake transport behind it).
"""

from __future__ import annotations

from typing import cast

from .base import (
    BACKEND_CMUX,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalBackendError,
    TerminalUnit,
    TerminalUnitRef,
    TerminalUnitState,
    TerminalUnsupportedOperationError,
)
from .cmux_client import CmuxSidecarClient
from .cmux_protocol import (
    PROTOCOL_VERSION,
    CmuxHelloResult,
    CmuxProtocolError,
    CmuxWorkspace,
)


class CmuxBackend(TerminalBackend):
    """``TerminalBackend`` adapter delegating to :class:`CmuxSidecarClient`.

    Capability flags are derived from the sidecar handshake the first
    time :meth:`capabilities` is called within an async context. Until
    a handshake is observed a conservative default is returned so the
    router can still gate UI without forcing a network round-trip.
    """

    def __init__(self, client: CmuxSidecarClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return BACKEND_CMUX

    def capabilities(self) -> TerminalBackendCapabilities:
        return self._capabilities_from(self._client.cached_hello())

    def _capabilities_from(
        self, hello: CmuxHelloResult | None
    ) -> TerminalBackendCapabilities:
        if hello is None:
            return TerminalBackendCapabilities(
                backend=BACKEND_CMUX,
                supports_create=False,
                supports_list=True,
                supports_capture=True,
                supports_send_text=True,
                supports_send_key=True,
                supports_close=False,
                supports_resume=False,
                supports_event_stream=False,
                protocol_version=PROTOCOL_VERSION,
            )
        return TerminalBackendCapabilities(
            backend=BACKEND_CMUX,
            supports_create=hello.supports_create,
            supports_list=True,
            supports_capture=hello.supports_capture,
            supports_send_text=hello.supports_send_text,
            supports_send_key=hello.supports_send_key,
            supports_close=hello.supports_close,
            supports_resume=False,
            supports_event_stream=hello.supports_event_stream,
            protocol_version=hello.protocol_version,
        )

    def _check_backend(self, ref: TerminalUnitRef) -> None:
        if ref.backend != self.name:
            raise TerminalUnsupportedOperationError(
                f"CmuxBackend cannot service {ref.backend!r} units",
                ref=ref,
            )

    async def negotiate(self) -> TerminalBackendCapabilities:
        """Force a handshake and return the resolved capability projection.

        Converts sidecar protocol violations into the neutral
        backend-error taxonomy so callers see a single exception family.
        """
        try:
            hello = await self._client.hello()
        except CmuxProtocolError as exc:
            raise TerminalBackendError(str(exc), code="internal_error") from exc
        return self._capabilities_from(hello)

    async def list_units(self) -> list[TerminalUnit]:
        try:
            workspaces = await self._client.list_workspaces()
        except CmuxProtocolError as exc:
            raise TerminalBackendError(str(exc), code="internal_error") from exc
        return [self._workspace_to_unit(ws) for ws in workspaces]

    def _workspace_to_unit(self, workspace: CmuxWorkspace) -> TerminalUnit:
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id=workspace.workspace_id)
        return TerminalUnit(
            ref=ref,
            title=workspace.title,
            cwd=workspace.cwd,
            provider_name=workspace.provider_name,
            state=_normalise_state(workspace.state),
            supports_capture=workspace.has_terminal_surface,
            supports_send_text=workspace.has_terminal_surface,
            supports_send_key=workspace.has_terminal_surface,
            supports_close=False,
            supports_resume=False,
            backend_metadata={"has_terminal_surface": workspace.has_terminal_surface},
        )

    async def capture(
        self, ref: TerminalUnitRef, *, with_ansi: bool = False
    ) -> str | None:
        self._check_backend(ref)
        try:
            return await self._client.capture_screen(ref.unit_id, with_ansi=with_ansi)
        except CmuxProtocolError as exc:
            raise TerminalBackendError(
                str(exc), code="internal_error", ref=ref
            ) from exc

    async def send_text(
        self, ref: TerminalUnitRef, text: str, *, raw: bool = False
    ) -> bool:
        self._check_backend(ref)
        try:
            return await self._client.send_text(ref.unit_id, text, raw=raw)
        except CmuxProtocolError as exc:
            raise TerminalBackendError(
                str(exc), code="internal_error", ref=ref
            ) from exc

    async def send_key(self, ref: TerminalUnitRef, key: str) -> bool:
        self._check_backend(ref)
        try:
            return await self._client.send_key(ref.unit_id, key)
        except CmuxProtocolError as exc:
            raise TerminalBackendError(
                str(exc), code="internal_error", ref=ref
            ) from exc

    async def close(self, ref: TerminalUnitRef) -> bool:
        self._check_backend(ref)
        try:
            return await self._client.close_workspace(ref.unit_id)
        except CmuxProtocolError as exc:
            raise TerminalBackendError(
                str(exc), code="internal_error", ref=ref
            ) from exc


_VALID_STATES: frozenset[str] = frozenset(
    {"starting", "working", "waiting", "ready", "dead", "unknown"}
)


def _normalise_state(raw: str) -> TerminalUnitState:
    """Coerce a sidecar state string to the literal taxonomy or ``unknown``."""
    if raw in _VALID_STATES:
        return cast(TerminalUnitState, raw)
    return "unknown"


__all__ = ["CmuxBackend"]
