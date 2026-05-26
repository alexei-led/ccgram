"""Tests for /sessions dashboard command."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import (
    CB_SESSIONS_NEW,
    CB_SESSIONS_REFRESH,
    CB_STATUS_ESC,
    CB_STATUS_SCREENSHOT,
)
from ccgram.handlers.sessions_dashboard import (
    _build_dashboard,
    handle_sessions_kill,
    handle_sessions_kill_confirm,
    handle_sessions_refresh,
    sessions_command,
)
from ccgram.session import WindowState
from ccgram.terminal_backends.base import (
    BACKEND_CMUX,
    TerminalBackendUnavailableError,
    TerminalUnit,
    TerminalUnitRef,
)
from ccgram.terminal_backends.router import reset_router_for_testing


@pytest.fixture(autouse=True)
def _isolated_router():
    reset_router_for_testing()
    yield
    reset_router_for_testing()


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch("ccgram.handlers.sessions_dashboard.view_window") as mock_view,
        patch("ccgram.handlers.sessions_dashboard.thread_router") as mock_tr,
        patch("ccgram.handlers.sessions_dashboard.tmux_manager") as mock_tm,
        patch("ccgram.handlers.sessions_dashboard.config") as mock_cfg,
        patch(
            "ccgram.handlers.sessions_dashboard.get_backend",
            return_value="tmux",
        ),
        patch(
            "ccgram.handlers.sessions_dashboard.get_unit_id",
            side_effect=lambda wid: wid,
        ),
    ):
        mock_tr.get_all_thread_windows.return_value = {}
        mock_tr.get_display_name.side_effect = lambda wid: wid
        mock_view.side_effect = lambda wid: WindowState()
        mock_tm.list_windows = AsyncMock(return_value=[])
        mock_tm.discover_external_sessions = AsyncMock(return_value=[])
        mock_cfg.is_user_allowed.return_value = True
        yield mock_view, mock_tr, mock_tm, mock_cfg


class TestBuildDashboard:
    async def test_empty(self, _patch_deps) -> None:
        text, keyboard = await _build_dashboard(100)
        assert "No active sessions" in text
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert CB_SESSIONS_REFRESH in data
        assert CB_SESSIONS_NEW in data

    async def test_alive_session(self, _patch_deps) -> None:
        mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "myproject"
        mock_sm.side_effect = lambda wid: WindowState(cwd="/home/user/myproject")
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        text, _kb = await _build_dashboard(100)
        assert "\U0001f7e2 myproject" in text

    async def test_alive_session_shows_cwd(self, _patch_deps) -> None:
        mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "myproject"
        mock_sm.side_effect = lambda wid: WindowState(cwd="/home/user/myproject")
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        text, _kb = await _build_dashboard(100)
        assert "/home/user/myproject" in text

    async def test_no_cwd_shows_no_path(self, _patch_deps) -> None:
        mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "myproject"
        mock_sm.side_effect = lambda wid: WindowState(cwd="")
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        text, _kb = await _build_dashboard(100)
        assert "    " not in text

    async def test_dead_session(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "oldproject"
        mock_tm.list_windows = AsyncMock(return_value=[])

        text, _kb = await _build_dashboard(100)
        assert "\u26ab oldproject" in text

    async def test_multiple_sessions(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {10: "@0", 20: "@5"}
        mock_tr.get_display_name.side_effect = lambda wid: {
            "@0": "alive",
            "@5": "dead",
        }[wid]
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        text, _kb = await _build_dashboard(100)
        assert "\U0001f7e2 alive" in text
        assert "\u26ab dead" in text

    async def test_refresh_and_new_buttons(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        _text, keyboard = await _build_dashboard(100)
        labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert any("Refresh" in label for label in labels)
        assert any("New" in label for label in labels)
        assert CB_SESSIONS_REFRESH in data
        assert CB_SESSIONS_NEW in data

    async def test_alive_session_has_esc_button(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        _text, keyboard = await _build_dashboard(100)
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert any(d.startswith(CB_STATUS_ESC) for d in data)

    async def test_alive_session_has_screenshot_button(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        _text, keyboard = await _build_dashboard(100)
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert any(d.startswith(CB_STATUS_SCREENSHOT) for d in data)

    async def test_alive_session_shows_provider(self, _patch_deps) -> None:
        mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "myproject"
        mock_sm.side_effect = lambda wid: WindowState(
            cwd="/home/user/myproject", provider_name="codex"
        )
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        text, _kb = await _build_dashboard(100)
        assert "[codex]" in text

    async def test_default_provider_shows_no_tag(self, _patch_deps) -> None:
        mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "myproject"
        mock_sm.side_effect = lambda wid: WindowState(
            cwd="/home/user/myproject", provider_name=""
        )
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        text, _kb = await _build_dashboard(100)
        assert "[" not in text

    async def test_yolo_mode_shows_tag(self, _patch_deps) -> None:
        mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "myproject"
        mock_sm.side_effect = lambda wid: WindowState(
            cwd="/home/user/myproject",
            provider_name="codex",
            approval_mode="yolo",
        )
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        text, _kb = await _build_dashboard(100)
        assert "[YOLO]" in text

    async def test_dead_session_no_action_buttons(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "deadproject"
        mock_tm.list_windows = AsyncMock(return_value=[])

        _text, keyboard = await _build_dashboard(100)
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert not any(d.startswith(CB_STATUS_ESC) for d in data)
        assert not any(d.startswith(CB_STATUS_SCREENSHOT) for d in data)


class TestSessionsCommand:
    async def test_calls_reply(self, _patch_deps) -> None:
        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()

        with patch("ccgram.handlers.sessions_dashboard.safe_reply") as mock_reply:
            await sessions_command(update, MagicMock())
            mock_reply.assert_called_once()
            assert update.message == mock_reply.call_args[0][0]
            assert "No active sessions" in mock_reply.call_args[0][1]

    async def test_unauthorized(self, _patch_deps) -> None:
        _, _, _, mock_cfg = _patch_deps
        mock_cfg.is_user_allowed.return_value = False

        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()

        with patch("ccgram.handlers.sessions_dashboard.safe_reply") as mock_reply:
            await sessions_command(update, MagicMock())
            mock_reply.assert_called_once()
            assert "not authorized" in mock_reply.call_args[0][1]

    async def test_no_user(self) -> None:
        update = MagicMock()
        update.effective_user = None
        update.message = AsyncMock()

        with patch("ccgram.handlers.sessions_dashboard.safe_reply") as mock_reply:
            await sessions_command(update, MagicMock())
            mock_reply.assert_not_called()

    async def test_no_message(self) -> None:
        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = None

        with patch("ccgram.handlers.sessions_dashboard.safe_reply") as mock_reply:
            await sessions_command(update, MagicMock())
            mock_reply.assert_not_called()


class TestSessionsRefresh:
    async def test_refresh_edits(self, _patch_deps) -> None:
        query = AsyncMock()

        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit:
            await handle_sessions_refresh(query, 100)
            mock_edit.assert_called_once()
            assert query == mock_edit.call_args[0][0]
            assert "No active sessions" in mock_edit.call_args[0][1]


class TestKillButtons:
    async def test_alive_session_has_kill_button(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "myproject"
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        _text, keyboard = await _build_dashboard(100)
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert any(d.startswith("sess:kill:") for d in data)

    async def test_dead_session_no_kill_button(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "@0"}
        mock_tr.get_display_name.side_effect = lambda wid: "oldproject"
        mock_tm.list_windows = AsyncMock(return_value=[])

        _text, keyboard = await _build_dashboard(100)
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert not any(d.startswith("sess:kill:") for d in data)

    async def test_empty_dashboard_no_kill_button(self, _patch_deps) -> None:
        _text, keyboard = await _build_dashboard(100)
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert not any(d.startswith("sess:kill:") for d in data)

    async def test_cmux_kill_prompt_is_rejected(self, _patch_deps) -> None:
        query = AsyncMock()

        with (
            patch(
                "ccgram.handlers.sessions_dashboard.get_backend", return_value="cmux"
            ),
            patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit,
        ):
            await handle_sessions_kill(query, 100, "cmux:ws-a")

        mock_edit.assert_awaited_once()
        assert "tmux" in mock_edit.call_args[0][1]

    async def test_cmux_kill_confirm_does_not_unbind(self, _patch_deps) -> None:
        _mock_view, mock_tr, mock_tm, _mock_cfg = _patch_deps
        query = AsyncMock()
        client = AsyncMock()

        with (
            patch(
                "ccgram.handlers.sessions_dashboard.get_backend", return_value="cmux"
            ),
            patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit,
        ):
            killed = await handle_sessions_kill_confirm(query, 100, "cmux:ws-a", client)

        assert killed is False
        mock_edit.assert_awaited_once()
        mock_tm.find_window_by_id.assert_not_called()
        mock_tr.unbind_thread.assert_not_called()


def _cmux_unit(workspace_id: str, title: str = "alpha") -> TerminalUnit:
    return TerminalUnit(
        ref=TerminalUnitRef(backend=BACKEND_CMUX, unit_id=workspace_id),
        title=title,
        cwd="/repo/a",
        provider_name="claude",
        supports_capture=True,
        supports_send_text=True,
        supports_send_key=True,
    )


class TestCmuxRendering:
    async def test_cmux_row_shown_alive_when_workspace_listed(
        self, _patch_deps
    ) -> None:
        mock_view, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "cmux:ws-a"}
        mock_tr.get_display_name.side_effect = lambda wid: "alpha"
        mock_view.side_effect = lambda wid: WindowState(cwd="/repo/a")

        fake_backend = MagicMock()
        fake_backend.name = BACKEND_CMUX
        fake_backend.list_units = AsyncMock(return_value=[_cmux_unit("ws-a")])
        from ccgram.terminal_backends.router import get_router

        get_router().register(fake_backend)

        with (
            patch(
                "ccgram.handlers.sessions_dashboard.get_backend",
                side_effect=lambda wid: "cmux" if wid == "cmux:ws-a" else "tmux",
            ),
            patch(
                "ccgram.handlers.sessions_dashboard.get_unit_id",
                side_effect=lambda wid: "ws-a" if wid == "cmux:ws-a" else wid,
            ),
        ):
            text, _kb = await _build_dashboard(100)
        assert "[cmux]" in text
        assert "\U0001f7e2 alpha" in text
        assert "[cmux unavailable]" not in text

    async def test_cmux_row_renders_unavailable_when_backend_not_registered(
        self, _patch_deps
    ) -> None:
        mock_view, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "cmux:ws-a"}
        mock_tr.get_display_name.side_effect = lambda wid: "alpha"
        mock_view.side_effect = lambda wid: WindowState()

        with (
            patch(
                "ccgram.handlers.sessions_dashboard.get_backend",
                side_effect=lambda wid: "cmux" if wid == "cmux:ws-a" else "tmux",
            ),
            patch(
                "ccgram.handlers.sessions_dashboard.get_unit_id",
                side_effect=lambda wid: "ws-a" if wid == "cmux:ws-a" else wid,
            ),
        ):
            text, _kb = await _build_dashboard(100)
        assert "[cmux unavailable]" in text

    async def test_cmux_row_renders_unavailable_when_sidecar_errors(
        self, _patch_deps
    ) -> None:
        mock_view, mock_tr, _mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "cmux:ws-a"}
        mock_tr.get_display_name.side_effect = lambda wid: "alpha"
        mock_view.side_effect = lambda wid: WindowState()

        fake_backend = MagicMock()
        fake_backend.name = BACKEND_CMUX
        fake_backend.list_units = AsyncMock(
            side_effect=TerminalBackendUnavailableError("sidecar down")
        )
        from ccgram.terminal_backends.router import get_router

        get_router().register(fake_backend)

        with (
            patch(
                "ccgram.handlers.sessions_dashboard.get_backend",
                side_effect=lambda wid: "cmux" if wid == "cmux:ws-a" else "tmux",
            ),
            patch(
                "ccgram.handlers.sessions_dashboard.get_unit_id",
                side_effect=lambda wid: "ws-a" if wid == "cmux:ws-a" else wid,
            ),
        ):
            text, _kb = await _build_dashboard(100)
        assert "[cmux unavailable]" in text

    async def test_cmux_row_no_kill_button_when_alive(self, _patch_deps) -> None:
        mock_view, mock_tr, _mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {42: "cmux:ws-a"}
        mock_tr.get_display_name.side_effect = lambda wid: "alpha"
        mock_view.side_effect = lambda wid: WindowState()

        fake_backend = MagicMock()
        fake_backend.name = BACKEND_CMUX
        fake_backend.list_units = AsyncMock(return_value=[_cmux_unit("ws-a")])
        from ccgram.terminal_backends.router import get_router

        get_router().register(fake_backend)

        with (
            patch(
                "ccgram.handlers.sessions_dashboard.get_backend",
                side_effect=lambda wid: "cmux" if wid == "cmux:ws-a" else "tmux",
            ),
            patch(
                "ccgram.handlers.sessions_dashboard.get_unit_id",
                side_effect=lambda wid: "ws-a" if wid == "cmux:ws-a" else wid,
            ),
        ):
            _text, keyboard = await _build_dashboard(100)
        data = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert not any(d.startswith("sess:kill:") for d in data)
        assert not any(d.startswith(CB_STATUS_ESC) for d in data)
        assert not any(d.startswith(CB_STATUS_SCREENSHOT) for d in data)

    async def test_tmux_alive_row_unaffected_by_cmux_backend(self, _patch_deps) -> None:
        mock_view, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_all_thread_windows.return_value = {10: "@0", 20: "cmux:ws-x"}
        mock_tr.get_display_name.side_effect = lambda wid: {
            "@0": "tmux-alive",
            "cmux:ws-x": "cmux-alive",
        }[wid]
        mock_view.side_effect = lambda wid: WindowState()
        mock_tm.list_windows = AsyncMock(return_value=[MagicMock(window_id="@0")])

        fake_backend = MagicMock()
        fake_backend.name = BACKEND_CMUX
        fake_backend.list_units = AsyncMock(return_value=[])
        from ccgram.terminal_backends.router import get_router

        get_router().register(fake_backend)

        with (
            patch(
                "ccgram.handlers.sessions_dashboard.get_backend",
                side_effect=lambda wid: "cmux" if wid.startswith("cmux:") else "tmux",
            ),
            patch(
                "ccgram.handlers.sessions_dashboard.get_unit_id",
                side_effect=lambda wid: (
                    wid.split(":", 1)[1] if wid.startswith("cmux:") else wid
                ),
            ),
        ):
            text, _kb = await _build_dashboard(100)
        assert "\U0001f7e2 tmux-alive" in text
        assert "⚫ cmux-alive" in text or "⚫ cmux-alive" in text
