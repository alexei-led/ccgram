"""Tests for session kill via sessions dashboard (two-step confirmation)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.sessions_dashboard import (
    handle_sessions_kill,
    handle_sessions_kill_confirm,
)
from ccgram.session import WindowState


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch("ccgram.handlers.sessions_dashboard.session_manager") as mock_sm,
        patch("ccgram.handlers.sessions_dashboard.tmux_manager") as mock_tm,
        patch(
            "ccgram.handlers.sessions_dashboard.clear_topic_state",
            new_callable=AsyncMock,
        ) as mock_clear,
    ):
        mock_sm.get_display_name.side_effect = lambda wid: wid
        mock_sm.get_window_state.side_effect = lambda wid: WindowState()
        mock_sm.get_all_thread_windows.return_value = {}
        mock_tm.list_windows = AsyncMock(return_value=[])
        yield mock_sm, mock_tm, mock_clear


class TestHandleSessionsKill:
    async def test_shows_confirmation(self, _patch_deps) -> None:
        mock_sm, _, _ = _patch_deps
        mock_sm.get_display_name.side_effect = lambda wid: "myproj"

        query = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit:
            await handle_sessions_kill(query, 100, "@5")
            mock_edit.assert_called_once()
            text = mock_edit.call_args[0][1]
            assert "Kill session" in text
            assert "myproj" in text
            keyboard = mock_edit.call_args.kwargs["reply_markup"]
            data = [
                btn.callback_data for row in keyboard.inline_keyboard for btn in row
            ]
            assert any("sess:killok:" in d for d in data)


class TestHandleSessionsKillConfirm:
    async def test_kills_and_unbinds(self, _patch_deps) -> None:
        mock_sm, mock_tm, mock_clear = _patch_deps
        mock_sm.get_display_name.side_effect = lambda wid: "myproj"
        mock_sm.iter_thread_bindings.return_value = [
            (100, 42, "@5"),
            (200, 99, "@5"),
            (300, 10, "@9"),
        ]
        mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@5"))
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        bot = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit"):
            await handle_sessions_kill_confirm(query, 100, "@5", bot)

        mock_tm.kill_window.assert_called_once_with("@5")
        assert mock_sm.unbind_thread.call_count == 2
        mock_sm.unbind_thread.assert_any_call(100, 42)
        mock_sm.unbind_thread.assert_any_call(200, 99)
        assert mock_clear.call_count == 2

    async def test_window_already_gone(self, _patch_deps) -> None:
        mock_sm, mock_tm, _ = _patch_deps
        mock_sm.get_display_name.side_effect = lambda wid: "myproj"
        mock_sm.iter_thread_bindings.return_value = [(100, 42, "@5")]
        mock_tm.find_window_by_id = AsyncMock(return_value=None)
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        bot = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit"):
            await handle_sessions_kill_confirm(query, 100, "@5", bot)

        mock_tm.kill_window.assert_not_called()
        mock_sm.unbind_thread.assert_called_once_with(100, 42)

    async def test_refreshes_dashboard_after_kill(self, _patch_deps) -> None:
        mock_sm, mock_tm, _ = _patch_deps
        mock_sm.get_display_name.side_effect = lambda wid: "proj"
        mock_sm.iter_thread_bindings.return_value = [(100, 42, "@5")]
        mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@5"))
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        bot = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit:
            await handle_sessions_kill_confirm(query, 100, "@5", bot)
            mock_edit.assert_called_once()
            text = mock_edit.call_args[0][1]
            assert "Killed" in text
