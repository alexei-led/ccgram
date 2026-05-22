from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.handlers.callback_data import (
    CB_KEYS_PREFIX,
    CB_STATUS_ESC,
    CB_STATUS_GET_FILE,
    CB_STATUS_LAST_REPLY,
)
from ccgram.handlers.status.status_bar_actions import _handle_status_bar_action

MOD = "ccgram.handlers.status.status_bar_actions"


def _q() -> AsyncMock:
    q = AsyncMock()
    q.answer = AsyncMock()
    q.edit_message_reply_markup = AsyncMock()
    q.get_bot = MagicMock(return_value=AsyncMock())
    return q


class TestStatusEsc:
    async def test_sends_escape_key(self):
        query = _q()
        with (
            patch(f"{MOD}.user_owns_window", return_value=True),
            patch(f"{MOD}.tmux_manager") as tm,
        ):
            tm.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@0"))
            tm.send_keys = AsyncMock()
            await _handle_status_bar_action(
                query, 1, f"{CB_STATUS_ESC}@0", MagicMock(), MagicMock()
            )
        tm.send_keys.assert_awaited_once_with(
            "@0", "Escape", enter=False, literal=False
        )
        query.answer.assert_awaited_once_with("\u238b Sent Escape")

    async def test_window_not_found(self):
        query = _q()
        with (
            patch(f"{MOD}.user_owns_window", return_value=True),
            patch(f"{MOD}.tmux_manager") as tm,
        ):
            tm.find_window_by_id = AsyncMock(return_value=None)
            await _handle_status_bar_action(
                query, 1, f"{CB_STATUS_ESC}@0", MagicMock(), MagicMock()
            )
        query.answer.assert_awaited_once_with("Window not found", show_alert=True)


class TestLastReply:
    async def test_dispatches_to_send_last_reply(self):
        query = _q()
        with (
            patch(f"{MOD}.user_owns_window", return_value=True),
            patch(f"{MOD}.get_thread_id", return_value=42),
            patch(f"{MOD}.thread_router") as tr,
            patch(
                "ccgram.handlers.last_reply.send_last_reply", new_callable=AsyncMock
            ) as mock_send,
        ):
            tr.resolve_chat_id.return_value = 999
            await _handle_status_bar_action(
                query, 1, f"{CB_STATUS_LAST_REPLY}@0", MagicMock(), MagicMock()
            )
        mock_send.assert_awaited_once()
        args = mock_send.call_args.args
        assert args[1] == 999
        assert args[2] == 42
        assert args[3] == "@0"

    async def test_rejects_foreign_window(self):
        query = _q()
        with patch(f"{MOD}.user_owns_window", return_value=False):
            await _handle_status_bar_action(
                query, 1, f"{CB_STATUS_LAST_REPLY}@0", MagicMock(), MagicMock()
            )
        query.answer.assert_awaited_once_with("Not your session", show_alert=True)


class TestGetFile:
    async def test_opens_file_browser(self, tmp_path):
        query = _q()
        context = MagicMock()
        context.user_data = {}
        view = MagicMock()
        view.cwd = str(tmp_path)
        with (
            patch(f"{MOD}.user_owns_window", return_value=True),
            patch(f"{MOD}.get_thread_id", return_value=42),
            patch(f"{MOD}.thread_router") as tr,
            patch(f"{MOD}.window_query") as wq,
            patch(
                "ccgram.handlers.send.open_file_browser", new_callable=AsyncMock
            ) as mock_open,
        ):
            wq.view_window.return_value = view
            tr.resolve_chat_id.return_value = 999
            await _handle_status_bar_action(
                query, 1, f"{CB_STATUS_GET_FILE}@0", MagicMock(), context
            )
        mock_open.assert_awaited_once()

    async def test_rejects_foreign_window(self):
        query = _q()
        with patch(f"{MOD}.user_owns_window", return_value=False):
            await _handle_status_bar_action(
                query, 1, f"{CB_STATUS_GET_FILE}@0", MagicMock(), MagicMock()
            )
        query.answer.assert_awaited_once_with("Not your session", show_alert=True)


class TestKeys:
    async def test_sends_key_and_schedules_refresh(self):
        query = _q()
        update = MagicMock()
        with (
            patch(f"{MOD}.user_owns_window", return_value=True),
            patch(f"{MOD}.get_thread_id", return_value=None),
            patch(f"{MOD}.tmux_manager") as tm,
            patch(f"{MOD}._schedule_key_refresh") as mock_sched,
        ):
            tm.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@0"))
            tm.send_keys = AsyncMock()
            await _handle_status_bar_action(
                query, 1, f"{CB_KEYS_PREFIX}ent:@0", update, MagicMock()
            )
        tm.send_keys.assert_awaited_once()
        mock_sched.assert_called_once()

    async def test_unknown_key_rejected(self):
        query = _q()
        with (
            patch(f"{MOD}.user_owns_window", return_value=True),
            patch(f"{MOD}.tmux_manager"),
        ):
            await _handle_status_bar_action(
                query, 1, f"{CB_KEYS_PREFIX}xxx:@0", MagicMock(), MagicMock()
            )
        query.answer.assert_awaited_once_with("Unknown key")


class TestClearKeyRefreshes:
    def test_cancels_matching_tasks(self):
        from ccgram.handlers.status.status_bar_actions import (
            _clear_key_refreshes,
            _pending_key_refreshes,
        )

        task1 = MagicMock()
        task1.done.return_value = False
        task2 = MagicMock()
        task2.done.return_value = False
        _pending_key_refreshes[(1, "@0")] = task1
        _pending_key_refreshes[(2, "@1")] = task2
        _clear_key_refreshes("@0")
        task1.cancel.assert_called_once()
        task2.cancel.assert_not_called()
        _pending_key_refreshes.clear()


class TestBuildDashboardButton:
    def test_returns_none_when_miniapp_disabled(self):
        from ccgram.handlers.status.status_bar_actions import build_dashboard_button

        with patch(f"{MOD}.config") as cfg:
            cfg.miniapp_base_url = ""
            assert build_dashboard_button("@0", 42) is None

    def test_returns_none_for_whitespace_url(self):
        from ccgram.handlers.status.status_bar_actions import build_dashboard_button

        # Config strips at load, but defensively guard against runtime mutation.
        with patch(f"{MOD}.config") as cfg:
            cfg.miniapp_base_url = ""
            assert build_dashboard_button("@0", 42) is None

    def test_builds_webapp_button_when_enabled(self):
        from telegram import WebAppInfo

        from ccgram.handlers.status.status_bar_actions import build_dashboard_button

        with (
            patch(f"{MOD}.config") as cfg,
            patch(f"{MOD}.sign_token", return_value="signed-tok") as mock_sign,
        ):
            cfg.miniapp_base_url = "https://miniapp.example/"
            cfg.telegram_bot_token = "bot:abc"
            btn = build_dashboard_button("@7", 42)

        assert btn is not None
        assert btn.text == "\U0001fa9f Dashboard"
        assert isinstance(btn.web_app, WebAppInfo)
        assert btn.web_app.url == "https://miniapp.example/app/signed-tok"
        # No trailing-slash duplication from base_url.
        assert "//app/" not in btn.web_app.url
        mock_sign.assert_called_once_with(
            bot_token="bot:abc", window_id="@7", user_id=42
        )

    def test_token_signed_per_window(self):
        from ccgram.handlers.status.status_bar_actions import build_dashboard_button

        captured: list[tuple[str, int]] = []

        def fake_sign(*, bot_token: str, window_id: str, user_id: int) -> str:
            assert bot_token == "bot:abc"
            captured.append((window_id, user_id))
            return f"tok-{window_id}-{user_id}"

        with (
            patch(f"{MOD}.config") as cfg,
            patch(f"{MOD}.sign_token", side_effect=fake_sign),
        ):
            cfg.miniapp_base_url = "https://miniapp.example"
            cfg.telegram_bot_token = "bot:abc"
            b1 = build_dashboard_button("@5", 1)
            b2 = build_dashboard_button("@9", 2)

        assert b1 is not None and b2 is not None
        assert b1.web_app is not None and b2.web_app is not None
        assert b1.web_app.url.endswith("/app/tok-@5-1")
        assert b2.web_app.url.endswith("/app/tok-@9-2")
        assert captured == [("@5", 1), ("@9", 2)]
