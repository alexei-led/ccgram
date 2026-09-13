"""Tests for session kill via sessions dashboard (two-step confirmation)."""

from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.multiplexer.base import WindowRef

import pytest
from telegram.error import TelegramError

from ccgram.handlers.sessions_dashboard import (
    handle_sessions_kill,
    handle_sessions_kill_confirm,
)
from ccgram.session import WindowState
from ccgram.telegram_rate_limiter import NO_RETRY_RATE_LIMIT_ARGS
from ccgram.thread_router import ThreadRouter


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch("ccgram.handlers.sessions_dashboard.view_window") as mock_view,
        patch("ccgram.handlers.sessions_dashboard.thread_router") as mock_tr,
        patch("ccgram.handlers.sessions_dashboard.tmux_manager") as mock_tm,
        patch(
            "ccgram.handlers.sessions_dashboard.clear_topic_state",
            new_callable=AsyncMock,
        ) as mock_clear,
    ):
        mock_tr.get_display_name.side_effect = lambda wid: wid
        mock_view.side_effect = lambda wid: WindowState()
        mock_tr.get_all_thread_windows.return_value = {}
        mock_tm.list_windows = AsyncMock(return_value=[])
        yield mock_view, mock_tr, mock_tm, mock_clear


class TestHandleSessionsKill:
    async def test_shows_confirmation(self, _patch_deps) -> None:
        _mock_sm, mock_tr, _, _ = _patch_deps
        mock_tr.get_display_name.side_effect = lambda wid: "myproj"

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
    @staticmethod
    def _router() -> ThreadRouter:
        return ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )

    async def test_kills_and_unbinds(self, _patch_deps) -> None:
        _, _, mock_tm, mock_clear = _patch_deps
        router = self._router()
        router.bind_thread(100, 42, "@5", chat_id=-100)
        router.bind_thread(200, 99, "@5", chat_id=-200)
        router.bind_thread(300, 10, "@9", chat_id=-300)
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        bot = AsyncMock()
        with (
            patch("ccgram.handlers.sessions_dashboard.safe_edit"),
            patch("ccgram.handlers.sessions_dashboard.thread_router", router),
            patch("ccgram.handlers.sessions_dashboard.session_manager"),
            patch(
                "ccgram.handlers.sessions_dashboard.window_presence",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("ccgram.handlers.topics.topic_deletion.session_manager"),
        ):
            await handle_sessions_kill_confirm(query, 100, "@5", bot)

        mock_tm.kill_window.assert_called_once_with("@5")
        assert {
            (call.args[0], call.args[1], call.kwargs["rate_limit_args"])
            for call in bot.delete_forum_topic.await_args_list
        } == {
            (-100, 42, NO_RETRY_RATE_LIMIT_ARGS),
            (-200, 99, NO_RETRY_RATE_LIMIT_ARGS),
        }
        assert mock_clear.call_count == 2
        bot.close_forum_topic.assert_not_awaited()
        assert router.get_window_for_chat_thread(-100, 42) is None
        assert router.get_window_for_chat_thread(-200, 99) is None
        assert router.get_window_for_chat_thread(-300, 10) == "@9"
        assert list(router.iter_retired_topics()) == []

    async def test_window_already_gone(self, _patch_deps) -> None:
        _, _, mock_tm, _ = _patch_deps
        router = self._router()
        router.bind_thread(100, 42, "@5", chat_id=-999)
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        bot = AsyncMock()
        with (
            patch("ccgram.handlers.sessions_dashboard.safe_edit") as edit,
            patch("ccgram.handlers.sessions_dashboard.thread_router", router),
            patch("ccgram.handlers.sessions_dashboard.session_manager"),
            patch(
                "ccgram.handlers.sessions_dashboard.window_presence",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("ccgram.handlers.topics.topic_deletion.session_manager"),
        ):
            await handle_sessions_kill_confirm(query, 100, "@5", bot)

        mock_tm.kill_window.assert_not_called()
        bot.delete_forum_topic.assert_awaited_once_with(
            -999, 42, rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS
        )
        assert "Was already gone" in edit.call_args.args[1]
        assert router.get_window_for_chat_thread(-999, 42) is None
        assert list(router.iter_retired_topics()) == []

    async def test_failed_topic_delete_keeps_pending_record(self, _patch_deps) -> None:
        _, _, mock_tm, mock_clear = _patch_deps
        router = self._router()
        router.bind_thread(100, 42, "@5", chat_id=-999)
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        bot = AsyncMock()
        bot.delete_forum_topic.side_effect = TelegramError("Forbidden")
        bot.close_forum_topic.side_effect = TelegramError("Forbidden")
        with (
            patch("ccgram.handlers.sessions_dashboard.safe_edit"),
            patch("ccgram.handlers.sessions_dashboard.thread_router", router),
            patch("ccgram.handlers.sessions_dashboard.session_manager"),
            patch(
                "ccgram.handlers.sessions_dashboard.window_presence",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("ccgram.handlers.topics.topic_deletion.session_manager"),
        ):
            await handle_sessions_kill_confirm(query, 100, "@5", bot)

        mock_tm.kill_window.assert_awaited_once_with("@5")
        bot.delete_forum_topic.assert_awaited_once_with(
            -999, 42, rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS
        )
        bot.close_forum_topic.assert_awaited_once_with(
            -999, 42, rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS
        )
        assert mock_clear.await_count == 1
        assert router.get_window_for_chat_thread(-999, 42) is None
        pending = next(router.iter_retired_topics())
        assert pending.reason == "session_closed"
        assert pending.cleanup_eligible is True
        assert pending.closed is False

    async def test_refreshes_dashboard_after_kill(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, _ = _patch_deps
        mock_tr.get_display_name.side_effect = lambda wid: "proj"
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@5")]
        mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@5"))
        mock_tm.list_windows_for_reconciliation = AsyncMock(
            return_value=[WindowRef(window_id="@5", window_name="proj", cwd="/p")]
        )
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        bot = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit:
            await handle_sessions_kill_confirm(query, 100, "@5", bot)
            mock_edit.assert_called_once()
            text = mock_edit.call_args[0][1]
            assert "Killed" in text

    async def test_kill_unbinds_case_variant_bindings(self, _patch_deps) -> None:
        _, _, mock_tm, mock_clear = _patch_deps
        router = self._router()
        router.bind_thread(100, 42, "9f1c2d3e-4a5b", chat_id=-100)
        router.bind_thread(200, 99, "9F1C2D3E-4A5B", chat_id=-200)
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        bot = AsyncMock()
        with (
            patch("ccgram.handlers.sessions_dashboard.safe_edit"),
            patch("ccgram.handlers.sessions_dashboard.thread_router", router),
            patch("ccgram.handlers.sessions_dashboard.session_manager"),
            patch(
                "ccgram.handlers.sessions_dashboard.window_presence",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("ccgram.handlers.topics.topic_deletion.session_manager"),
        ):
            await handle_sessions_kill_confirm(query, 100, "9f1c2d3e-4a5b", bot)

        mock_tm.kill_window.assert_awaited_once_with("9f1c2d3e-4a5b")
        assert {
            (call.args[0], call.args[1], call.kwargs["rate_limit_args"])
            for call in bot.delete_forum_topic.await_args_list
        } == {
            (-100, 42, NO_RETRY_RATE_LIMIT_ARGS),
            (-200, 99, NO_RETRY_RATE_LIMIT_ARGS),
        }
        assert mock_clear.call_count == 2
        bot.close_forum_topic.assert_not_awaited()
        assert router.get_window_for_chat_thread(-100, 42) is None
        assert router.get_window_for_chat_thread(-200, 99) is None
        assert list(router.iter_retired_topics()) == []


class TestKillNeedsConfirmedLiveness:
    """Kill deletes every route to a session, so an outage must change nothing.

    find_window_by_id answers None both for a window that is gone and for a
    backend that could not be reached. Acting on that during an outage leaves
    the session running with no topic left to reach it from.
    """

    async def test_unreachable_backend_kills_and_unbinds_nothing(
        self, _patch_deps
    ) -> None:
        _mock_sm, mock_tr, mock_tm, mock_clear = _patch_deps
        mock_tr.get_display_name.side_effect = lambda wid: "myproj"
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@5")]
        mock_tm.find_window_by_id = AsyncMock(return_value=None)
        mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=None)
        mock_tm.kill_window = AsyncMock()

        query = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as edit:
            await handle_sessions_kill_confirm(query, 100, "@5", AsyncMock())

        mock_tm.kill_window.assert_not_called()
        mock_tr.unbind_thread.assert_not_called()
        mock_clear.assert_not_called()
        assert "Could not reach" in edit.call_args[0][1]


class TestKillRequiresTheKillToSucceed:
    """A failed kill must not clear the routing to a session still running."""

    async def test_failed_kill_unbinds_nothing(self, _patch_deps) -> None:
        _mock_sm, mock_tr, mock_tm, mock_clear = _patch_deps
        mock_tr.get_display_name.side_effect = lambda wid: "myproj"
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@5")]
        mock_tm.list_windows_for_reconciliation = AsyncMock(
            return_value=[WindowRef(window_id="@5", window_name="proj", cwd="/p")]
        )
        mock_tm.kill_window = AsyncMock(return_value=False)

        query = AsyncMock()
        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as edit:
            await handle_sessions_kill_confirm(query, 100, "@5", AsyncMock())

        mock_tr.unbind_thread.assert_not_called()
        mock_clear.assert_not_called()
        assert "Could not kill" in edit.call_args[0][1]
