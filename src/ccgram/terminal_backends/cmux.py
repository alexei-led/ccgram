"""CmuxBackend — ``TerminalBackend`` adapter for cmux terminal surfaces.

The adapter is intentionally thin: it maps :class:`TerminalUnitRef`
values (``backend="cmux"``, ``unit_id=<surface_id>``) onto a cmux client
and translates :class:`CmuxTerminalSession` results into the neutral
:class:`TerminalUnit` projection. cmux workspaces can scope a ccgram instance;
panes stay metadata because terminal surfaces move between panes.
"""

from __future__ import annotations

from typing import Protocol, cast

from .base import (
    BACKEND_CMUX,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalBackendError,
    TerminalNotFoundError,
    TerminalUnit,
    TerminalUnitRef,
    TerminalUnitState,
    TerminalUnsupportedOperationError,
)
from .cmux_protocol import (
    PROTOCOL_VERSION,
    CmuxHelloResult,
    CmuxProtocolError,
    CmuxTerminalSession,
)


class CmuxClient(Protocol):
    """Client shape consumed by :class:`CmuxBackend`."""

    def cached_hello(self) -> CmuxHelloResult | None: ...

    async def hello(self) -> CmuxHelloResult: ...

    async def list_terminal_sessions(self) -> list[CmuxTerminalSession]: ...

    async def capture_screen(
        self, terminal_id: str, *, with_ansi: bool = False
    ) -> str: ...

    async def send_text(
        self, terminal_id: str, text: str, *, raw: bool = False
    ) -> bool: ...

    async def send_key(self, terminal_id: str, key: str) -> bool: ...

    async def close_terminal_session(self, terminal_id: str) -> bool: ...


class CmuxBackend(TerminalBackend):
    """``TerminalBackend`` adapter delegating to a cmux client."""

    def __init__(self, client: CmuxClient, *, workspace_id: str | None = None) -> None:
        self._client = client
        if workspace_id is None:
            self._workspace_id = None
            return
        cleaned = workspace_id.strip()
        self._workspace_id = cleaned or None

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
                supports_capture=False,
                supports_send_text=False,
                supports_send_key=False,
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
            sessions = await self._client.list_terminal_sessions()
        except CmuxProtocolError as exc:
            raise TerminalBackendError(str(exc), code="internal_error") from exc
        return [
            self._terminal_session_to_unit(session)
            for session in sessions
            if self._session_in_workspace(session)
        ]

    def _session_in_workspace(self, session: CmuxTerminalSession) -> bool:
        if self._workspace_id is None:
            return True
        return session.workspace_id == self._workspace_id

    async def _require_workspace_membership(self, ref: TerminalUnitRef) -> None:
        if self._workspace_id is None:
            return
        sessions = await self._client.list_terminal_sessions()
        for session in sessions:
            if session.terminal_id == ref.unit_id and self._session_in_workspace(
                session
            ):
                return
        raise TerminalNotFoundError(
            f"cmux terminal surface not found in workspace {self._workspace_id}",
            ref=ref,
        )

    def _terminal_session_to_unit(self, session: CmuxTerminalSession) -> TerminalUnit:
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id=session.terminal_id)
        caps = self._capabilities_from(self._client.cached_hello())
        return TerminalUnit(
            ref=ref,
            title=session.title,
            cwd=session.cwd,
            provider_name=session.provider_name,
            state=_normalise_state(session.state),
            supports_capture=caps.supports_capture,
            supports_send_text=caps.supports_send_text,
            supports_send_key=caps.supports_send_key,
            supports_close=caps.supports_close,
            supports_resume=False,
            backend_metadata={
                key: value
                for key, value in {
                    "workspace_id": session.workspace_id,
                    "workspace_title": session.workspace_title,
                    "pane_id": session.pane_id,
                    "surface_id": session.surface_id,
                    "panel_id": session.panel_id,
                    "window_id": session.window_id,
                    "window_ref": session.window_ref,
                    "workspace_ref": session.workspace_ref,
                    "pane_ref": session.pane_ref,
                    "surface_ref": session.surface_ref,
                    "focused": session.focused,
                    "selected_in_pane": session.selected_in_pane,
                }.items()
                if value is not None and value != ""
            },
        )

    async def capture(
        self, ref: TerminalUnitRef, *, with_ansi: bool = False
    ) -> str | None:
        self._check_backend(ref)
        try:
            await self._require_workspace_membership(ref)
            return await self._client.capture_screen(ref.unit_id, with_ansi=with_ansi)
        except CmuxProtocolError as exc:
            raise TerminalBackendError(
                str(exc), code="internal_error", ref=ref
            ) from exc
        except TerminalBackendError as exc:
            raise _error_with_ref(exc, ref) from exc

    async def send_text(
        self, ref: TerminalUnitRef, text: str, *, raw: bool = False
    ) -> bool:
        self._check_backend(ref)
        try:
            await self._require_workspace_membership(ref)
            return await self._client.send_text(ref.unit_id, text, raw=raw)
        except CmuxProtocolError as exc:
            raise TerminalBackendError(
                str(exc), code="internal_error", ref=ref
            ) from exc
        except TerminalBackendError as exc:
            raise _error_with_ref(exc, ref) from exc

    async def send_key(self, ref: TerminalUnitRef, key: str) -> bool:
        self._check_backend(ref)
        try:
            await self._require_workspace_membership(ref)
            return await self._client.send_key(ref.unit_id, key)
        except CmuxProtocolError as exc:
            raise TerminalBackendError(
                str(exc), code="internal_error", ref=ref
            ) from exc
        except TerminalBackendError as exc:
            raise _error_with_ref(exc, ref) from exc

    async def close(self, ref: TerminalUnitRef) -> bool:
        self._check_backend(ref)
        try:
            await self._require_workspace_membership(ref)
            return await self._client.close_terminal_session(ref.unit_id)
        except CmuxProtocolError as exc:
            raise TerminalBackendError(
                str(exc), code="internal_error", ref=ref
            ) from exc
        except TerminalBackendError as exc:
            raise _error_with_ref(exc, ref) from exc


def _error_with_ref(
    exc: TerminalBackendError, ref: TerminalUnitRef
) -> TerminalBackendError:
    if exc.ref is not None:
        return exc
    message = str(exc.args[0]) if exc.args else str(exc)
    if type(exc) is TerminalBackendError:
        return TerminalBackendError(message, code=exc.code, ref=ref)
    return type(exc)(message, ref=ref)


_VALID_STATES: frozenset[str] = frozenset(
    {"starting", "working", "waiting", "ready", "dead", "unknown"}
)


def _normalise_state(raw: str) -> TerminalUnitState:
    """Coerce a sidecar state string to the literal taxonomy or ``unknown``."""
    if raw in _VALID_STATES:
        return cast(TerminalUnitState, raw)
    return "unknown"


__all__ = ["CmuxBackend"]
