"""Native cmux CLI/RPC client for terminal-surface operations.

This module talks to the installed ``cmux`` CLI directly. ccgram treats cmux
terminal surfaces as terminal units; workspaces and panes are metadata only.
The client owns the external command contract and normalises current cmux RPC
responses into the existing cmux backend DTOs.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .base import (
    TerminalBackendError,
    TerminalBackendUnavailableError,
    TerminalNotFoundError,
    TerminalOperationRejectedError,
    TerminalOperationTimeoutError,
)
from .cmux_protocol import (
    PROTOCOL_VERSION,
    CmuxHelloResult,
    CmuxProtocolError,
    CmuxTerminalSession,
)

_DEFAULT_TIMEOUT = 5.0


class CmuxNativeClient:
    """Small async wrapper around ``cmux rpc``.

    The public methods mirror the old sidecar client closely enough that
    ``CmuxBackend`` can stay a thin adapter. Each call spawns ``cmux rpc``;
    that is fine for picker/send/capture frequency and keeps the first native
    integration boring. Boring survives contact with terminals.
    """

    def __init__(
        self, *, timeout: float = _DEFAULT_TIMEOUT, cmux_bin: str = "cmux"
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        self._timeout = timeout
        self._cmux_bin = cmux_bin
        self._hello = CmuxHelloResult(
            protocol_version=PROTOCOL_VERSION,
            sidecar_version="native",
            supports_create=False,
            supports_capture=True,
            supports_send_text=True,
            supports_send_key=True,
            supports_close=False,
            supports_event_stream=False,
        )

    async def hello(self) -> CmuxHelloResult:
        return self._hello

    def cached_hello(self) -> CmuxHelloResult | None:
        return self._hello

    async def list_terminal_sessions(self) -> list[CmuxTerminalSession]:
        payload = await self._rpc("system.tree", {"all": True})
        return _sessions_from_tree(payload)

    async def capture_screen(self, terminal_id: str, *, with_ansi: bool = False) -> str:
        del with_ansi  # cmux read_text currently returns text, not ANSI frames.
        payload = await self._rpc(
            "surface.read_text", {"surface_id": terminal_id, "lines": 200}
        )
        return _read_text_payload(payload)

    async def send_text(
        self, terminal_id: str, text: str, *, raw: bool = False
    ) -> bool:
        del raw
        await self._rpc("surface.send_text", {"surface_id": terminal_id, "text": text})
        return True

    async def send_key(self, terminal_id: str, key: str) -> bool:
        await self._rpc("surface.send_key", {"surface_id": terminal_id, "key": key})
        return True

    async def close_terminal_session(self, terminal_id: str) -> bool:
        await self._rpc("surface.close", {"surface_id": terminal_id})
        return True

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                self._cmux_bin,
                "rpc",
                method,
                json.dumps(params),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise TerminalBackendUnavailableError("cmux CLI not found on PATH") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise TerminalOperationTimeoutError(
                f"cmux rpc {method} timed out after {self._timeout:.1f}s"
            ) from exc

        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise _error_from_cmux_failure(method, stderr_text)
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise CmuxProtocolError(f"cmux rpc {method} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise CmuxProtocolError(f"cmux rpc {method} returned non-object JSON")
        return payload


class FakeCmuxNativeClient(CmuxNativeClient):
    """Test helper avoiding subprocesses while exercising native parsing."""

    def __init__(self, responses: Mapping[str, dict[str, Any]]) -> None:
        super().__init__()
        self._responses = dict(responses)
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, dict(params)))
        if method not in self._responses:
            raise AssertionError(f"unexpected cmux rpc method {method!r}")
        return self._responses[method]


def _error_from_cmux_failure(method: str, stderr: str) -> TerminalBackendError:
    message = stderr or f"cmux rpc {method} failed"
    lowered = message.lower()
    if "not found" in lowered or "unknown surface" in lowered:
        return TerminalNotFoundError(message)
    if "unknown command" in lowered or "unknown method" in lowered:
        return TerminalOperationRejectedError(message)
    if "permission" in lowered or "socket" in lowered or "connection" in lowered:
        return TerminalBackendUnavailableError(message)
    return TerminalBackendError(message, code="internal_error")


def _sessions_from_tree(payload: Mapping[str, Any]) -> list[CmuxTerminalSession]:
    sessions: list[CmuxTerminalSession] = []
    for window in _list_field(payload, "windows", context="system.tree"):
        if isinstance(window, Mapping):
            sessions.extend(_sessions_from_window(window))
    return sessions


def _sessions_from_window(window: Mapping[str, Any]) -> list[CmuxTerminalSession]:
    window_id = _str_or_none(window.get("id"))
    window_ref = _str_or_none(window.get("ref"))
    sessions: list[CmuxTerminalSession] = []
    for workspace in _list_field(window, "workspaces", context="window"):
        if isinstance(workspace, Mapping):
            sessions.extend(
                _sessions_from_workspace(
                    workspace,
                    window_id=window_id,
                    window_ref=window_ref,
                )
            )
    return sessions


def _sessions_from_workspace(
    workspace: Mapping[str, Any],
    *,
    window_id: str | None,
    window_ref: str | None,
) -> list[CmuxTerminalSession]:
    workspace_id = _str_or_none(workspace.get("id"))
    workspace_ref = _str_or_none(workspace.get("ref"))
    workspace_title = _str_or_none(workspace.get("title")) or ""
    sessions: list[CmuxTerminalSession] = []
    for pane in _list_field(workspace, "panes", context="workspace"):
        if isinstance(pane, Mapping):
            sessions.extend(
                _sessions_from_pane(
                    pane,
                    window_id=window_id,
                    window_ref=window_ref,
                    workspace_id=workspace_id,
                    workspace_ref=workspace_ref,
                    workspace_title=workspace_title,
                )
            )
    return sessions


def _sessions_from_pane(
    pane: Mapping[str, Any],
    *,
    window_id: str | None,
    window_ref: str | None,
    workspace_id: str | None,
    workspace_ref: str | None,
    workspace_title: str,
) -> list[CmuxTerminalSession]:
    pane_id = _str_or_none(pane.get("id"))
    pane_ref = _str_or_none(pane.get("ref"))
    sessions: list[CmuxTerminalSession] = []
    for surface in _list_field(pane, "surfaces", context="pane"):
        if not isinstance(surface, Mapping):
            continue
        session = _session_from_surface(
            surface,
            window_id=window_id,
            window_ref=window_ref,
            workspace_id=workspace_id,
            workspace_ref=workspace_ref,
            workspace_title=workspace_title,
            pane_id=pane_id,
            pane_ref=pane_ref,
        )
        if session is not None:
            sessions.append(session)
    return sessions


def _session_from_surface(
    surface: Mapping[str, Any],
    *,
    window_id: str | None,
    window_ref: str | None,
    workspace_id: str | None,
    workspace_ref: str | None,
    workspace_title: str,
    pane_id: str | None,
    pane_ref: str | None,
) -> CmuxTerminalSession | None:
    if surface.get("type") != "terminal":
        return None
    surface_id = _str_or_none(surface.get("id"))
    if surface_id is None:
        return None
    title = (
        _str_or_none(surface.get("title"))
        or _str_or_none(surface.get("ref"))
        or surface_id
    )
    return CmuxTerminalSession(
        terminal_id=surface_id,
        title=title,
        state="ready",
        workspace_id=workspace_id,
        workspace_title=workspace_title,
        pane_id=pane_id,
        surface_id=surface_id,
        panel_id=None,
        cwd=None,
        provider_name=None,
        window_id=window_id,
        window_ref=window_ref,
        workspace_ref=workspace_ref,
        pane_ref=pane_ref,
        surface_ref=_str_or_none(surface.get("ref")),
        focused=_bool_or_none(surface.get("focused")),
        selected_in_pane=_bool_or_none(surface.get("selected_in_pane")),
    )


def _list_field(payload: Mapping[str, Any], key: str, *, context: str) -> list[Any]:
    value = payload.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CmuxProtocolError(f"{context}: {key} must be a list")
    return list(value)


def _read_text_payload(payload: Mapping[str, Any]) -> str:
    text = payload.get("text")
    if isinstance(text, str):
        return text
    encoded = payload.get("base64")
    if isinstance(encoded, str):
        try:
            return base64.b64decode(encoded).decode("utf-8", errors="replace")
        except ValueError as exc:
            raise CmuxProtocolError(
                "surface.read_text returned invalid base64"
            ) from exc
    raise CmuxProtocolError("surface.read_text returned neither text nor base64")


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


__all__ = ["CmuxNativeClient", "FakeCmuxNativeClient"]
