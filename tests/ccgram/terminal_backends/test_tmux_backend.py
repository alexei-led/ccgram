from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.terminal_backends import (
    BACKEND_CMUX,
    BACKEND_TMUX,
    TerminalNotFoundError,
    TerminalUnitRef,
    TerminalUnsupportedOperationError,
)
from ccgram.terminal_backends.tmux import TmuxBackend


def _make_window(window_id: str = "@0") -> MagicMock:
    w = MagicMock()
    w.window_id = window_id
    w.window_name = "demo"
    w.cwd = "/tmp/demo"
    w.pane_current_command = "claude"
    w.pane_tty = "/dev/ttys001"
    return w


def _make_manager(
    *,
    find_result=None,
    capture_result=None,
    send_result: bool = True,
    list_result=None,
    kill_result: bool = True,
) -> MagicMock:
    mgr = MagicMock()
    mgr.find_window_by_id = AsyncMock(return_value=find_result)
    mgr.capture_pane = AsyncMock(return_value=capture_result)
    mgr.send_keys = AsyncMock(return_value=send_result)
    mgr.list_windows = AsyncMock(return_value=list_result or [])
    mgr.kill_window = AsyncMock(return_value=kill_result)
    return mgr


class TestTmuxBackendCapabilities:
    def test_name_is_tmux(self) -> None:
        backend = TmuxBackend(_make_manager())
        assert backend.name == BACKEND_TMUX

    def test_capabilities_reflect_full_surface(self) -> None:
        caps = TmuxBackend(_make_manager()).capabilities()
        assert caps.backend == BACKEND_TMUX
        assert caps.supports_capture
        assert caps.supports_send_text
        assert caps.supports_send_key
        assert caps.supports_close
        assert caps.supports_create
        assert caps.supports_list
        assert caps.supports_resume
        assert caps.supports_event_stream is False
        assert caps.protocol_version == "tmux-local"


class TestTmuxBackendCapture:
    async def test_capture_delegates_to_manager(self) -> None:
        mgr = _make_manager(find_result=_make_window("@5"), capture_result="hello")
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@5")
        text = await backend.capture(ref, with_ansi=True)
        assert text == "hello"
        mgr.find_window_by_id.assert_awaited_once_with("@5")
        mgr.capture_pane.assert_awaited_once_with("@5", with_ansi=True)

    async def test_capture_missing_window_raises_not_found(self) -> None:
        mgr = _make_manager(find_result=None)
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@99")
        with pytest.raises(TerminalNotFoundError) as excinfo:
            await backend.capture(ref)
        assert excinfo.value.code == "not_found"
        assert excinfo.value.ref == ref

    async def test_capture_failure_returns_none(self) -> None:
        mgr = _make_manager(find_result=_make_window("@5"), capture_result=None)
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@5")
        assert await backend.capture(ref) is None

    async def test_capture_rejects_foreign_backend_ref(self) -> None:
        backend = TmuxBackend(_make_manager())
        ref = TerminalUnitRef(backend=BACKEND_CMUX, unit_id="ws-1")
        with pytest.raises(TerminalUnsupportedOperationError):
            await backend.capture(ref)


class TestTmuxBackendSendText:
    async def test_send_text_delegates_to_manager(self) -> None:
        mgr = _make_manager(find_result=_make_window("@5"))
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@5")
        assert await backend.send_text(ref, "hi") is True
        mgr.send_keys.assert_awaited_once_with("@5", "hi", raw=False)

    async def test_send_text_raw_propagated(self) -> None:
        mgr = _make_manager(find_result=_make_window("@5"))
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@5")
        await backend.send_text(ref, "ls", raw=True)
        mgr.send_keys.assert_awaited_once_with("@5", "ls", raw=True)

    async def test_send_text_missing_window_raises(self) -> None:
        mgr = _make_manager(find_result=None)
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@99")
        with pytest.raises(TerminalNotFoundError):
            await backend.send_text(ref, "hi")

    async def test_send_text_failure_propagates_false(self) -> None:
        mgr = _make_manager(find_result=_make_window("@5"), send_result=False)
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@5")
        assert await backend.send_text(ref, "hi") is False


class TestTmuxBackendSendKey:
    async def test_send_key_uses_non_literal_no_enter(self) -> None:
        mgr = _make_manager(find_result=_make_window("@5"))
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@5")
        await backend.send_key(ref, "Escape")
        mgr.send_keys.assert_awaited_once_with(
            "@5", "Escape", enter=False, literal=False
        )

    async def test_send_key_missing_window_raises(self) -> None:
        mgr = _make_manager(find_result=None)
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@99")
        with pytest.raises(TerminalNotFoundError):
            await backend.send_key(ref, "Escape")


class TestTmuxBackendClose:
    async def test_close_delegates_to_manager(self) -> None:
        mgr = _make_manager()
        backend = TmuxBackend(mgr)
        ref = TerminalUnitRef(backend=BACKEND_TMUX, unit_id="@5")
        assert await backend.close(ref) is True
        mgr.kill_window.assert_awaited_once_with("@5")


class TestTmuxBackendListUnits:
    async def test_list_units_projects_windows(self) -> None:
        mgr = _make_manager(list_result=[_make_window("@0"), _make_window("@1")])
        backend = TmuxBackend(mgr)
        units = await backend.list_units()
        assert [u.ref.unit_id for u in units] == ["@0", "@1"]
        assert all(u.ref.backend == BACKEND_TMUX for u in units)
        assert all(u.supports_send_text for u in units)
        assert units[0].backend_metadata["pane_current_command"] == "claude"
