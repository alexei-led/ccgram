import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.text.text_handler import (
    _check_ui_guards,
    _forward_message,
    _handle_dead_window,
    _handle_unbound_topic,
)
from ccgram.handlers.topics.directory_browser import (
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    STATE_SELECTING_WINDOW,
)
from ccgram.handlers.user_state import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    RECOVERY_WINDOW_ID,
)

_TH = "ccgram.handlers.text.text_handler"


class TestCheckUiGuards:
    @pytest.mark.parametrize(
        ("state", "expected_text"),
        [
            (STATE_SELECTING_WINDOW, "window picker"),
            (STATE_BROWSING_DIRECTORY, "directory browser"),
        ],
    )
    async def test_same_thread_blocks(self, state, expected_text) -> None:
        message = AsyncMock()
        user_data = {STATE_KEY: state, PENDING_THREAD_ID: 42}

        with patch(f"{_TH}.safe_reply", new_callable=AsyncMock) as mock_reply:
            result = await _check_ui_guards(user_data, 42, message)

        assert result is True
        mock_reply.assert_called_once()
        assert expected_text in mock_reply.call_args.args[1]

    @pytest.mark.parametrize(
        "state", [STATE_SELECTING_WINDOW, STATE_BROWSING_DIRECTORY]
    )
    async def test_stale_thread_clears(self, state) -> None:
        message = AsyncMock()
        user_data = {
            STATE_KEY: state,
            PENDING_THREAD_ID: 99,
            PENDING_THREAD_TEXT: "old",
        }

        result = await _check_ui_guards(user_data, 42, message)

        assert result is False
        assert STATE_KEY not in user_data
        assert PENDING_THREAD_ID not in user_data
        assert PENDING_THREAD_TEXT not in user_data

    async def test_no_state_continues(self) -> None:
        message = AsyncMock()
        result = await _check_ui_guards({}, 42, message)
        assert result is False

    async def test_none_user_data_continues(self) -> None:
        message = AsyncMock()
        result = await _check_ui_guards(None, 42, message)
        assert result is False


class TestHandleUnboundTopic:
    @patch(f"{_TH}.thread_router")
    @patch(f"{_TH}.tmux_manager")
    async def test_bound_topic_returns_false(
        self, _mock_tm: MagicMock, mock_tr: MagicMock
    ) -> None:
        mock_tr.get_window_for_thread.return_value = "@0"
        message = AsyncMock()

        result = await _handle_unbound_topic(100, 42, "hello", {}, message)

        assert result is False

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_TH}.build_window_picker")
    @patch(f"{_TH}.tmux_manager")
    @patch(f"{_TH}.thread_router")
    async def test_shows_window_picker(
        self,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_picker: MagicMock,
        mock_reply: AsyncMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.iter_thread_bindings.return_value = []
        w = MagicMock(window_id="@5", window_name="proj", cwd="/tmp")
        mock_tm.list_windows = AsyncMock(return_value=[w])
        mock_picker.return_value = ("Pick:", MagicMock(), ["@5"])

        user_data: dict = {}
        message = MagicMock()

        result = await _handle_unbound_topic(100, 42, "hello", user_data, message)

        assert result is True
        mock_picker.assert_called_once()
        assert mock_reply.call_count == 2
        assert user_data[STATE_KEY] == STATE_SELECTING_WINDOW
        assert user_data[PENDING_THREAD_TEXT] == "hello"

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_TH}.build_directory_browser")
    @patch(f"{_TH}.tmux_manager")
    @patch(f"{_TH}.thread_router")
    async def test_shows_directory_browser(
        self,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_browser: MagicMock,
        mock_reply: AsyncMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.iter_thread_bindings.return_value = []
        mock_tm.list_windows = AsyncMock(return_value=[])
        mock_browser.return_value = ("Browse:", MagicMock(), [])

        user_data: dict = {}
        message = AsyncMock()

        result = await _handle_unbound_topic(100, 42, "hello", user_data, message)

        assert result is True
        mock_browser.assert_called_once()
        assert user_data[STATE_KEY] == STATE_BROWSING_DIRECTORY
        assert mock_reply.call_count == 2

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_TH}.build_window_picker")
    @patch(f"{_TH}.tmux_manager")
    @patch(f"{_TH}.thread_router")
    async def test_stores_pending_state(
        self,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_picker: MagicMock,
        _mock_reply: AsyncMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.iter_thread_bindings.return_value = []
        w = MagicMock(window_id="@5", window_name="proj", cwd="/tmp")
        mock_tm.list_windows = AsyncMock(return_value=[w])
        mock_picker.return_value = ("Pick:", MagicMock(), ["@5"])

        user_data: dict = {}
        message = AsyncMock()

        await _handle_unbound_topic(100, 42, "my text", user_data, message)

        assert user_data[PENDING_THREAD_ID] == 42
        assert user_data[PENDING_THREAD_TEXT] == "my text"

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_TH}.build_window_picker")
    @patch(f"{_TH}.tmux_manager")
    @patch(f"{_TH}.thread_router")
    async def test_window_picker_sends_pending_disclosure(
        self,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_picker: MagicMock,
        mock_reply: AsyncMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.iter_thread_bindings.return_value = []
        w = MagicMock(window_id="@5", window_name="proj", cwd="/tmp")
        mock_tm.list_windows = AsyncMock(return_value=[w])
        mock_picker.return_value = ("Pick:", MagicMock(), ["@5"])

        user_data: dict = {}
        message = AsyncMock()

        await _handle_unbound_topic(100, 42, "hello", user_data, message)

        from ccgram.handlers.text.text_handler import PENDING_DELIVERY_NOTICE

        assert mock_reply.call_count == 2
        assert mock_reply.call_args_list[1].args[1] == PENDING_DELIVERY_NOTICE

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_TH}.build_directory_browser")
    @patch(f"{_TH}.tmux_manager")
    @patch(f"{_TH}.thread_router")
    async def test_directory_browser_sends_pending_disclosure(
        self,
        mock_tr: MagicMock,
        mock_tm: MagicMock,
        mock_browser: MagicMock,
        mock_reply: AsyncMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.iter_thread_bindings.return_value = []
        mock_tm.list_windows = AsyncMock(return_value=[])
        mock_browser.return_value = ("Browse:", MagicMock(), [])

        user_data: dict = {}
        message = AsyncMock()

        await _handle_unbound_topic(100, 42, "hello", user_data, message)

        from ccgram.handlers.text.text_handler import PENDING_DELIVERY_NOTICE

        assert mock_reply.call_count == 2
        assert mock_reply.call_args_list[1].args[1] == PENDING_DELIVERY_NOTICE


class TestHandleDeadWindow:
    @patch(f"{_TH}.tmux_manager")
    async def test_alive_window_returns_false(self, mock_tm: MagicMock) -> None:
        mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock())
        message = AsyncMock()

        result = await _handle_dead_window("@0", 100, 42, "hello", {}, message)

        assert result is False

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_TH}.render_banner")
    @patch(f"{_TH}.tmux_manager")
    @patch(f"{_TH}.window_query")
    @patch(f"{_TH}.thread_router")
    async def test_shows_recovery_ui(
        self,
        mock_tr: MagicMock,
        mock_sm: MagicMock,
        mock_tm: MagicMock,
        mock_render: MagicMock,
        mock_reply: AsyncMock,
    ) -> None:
        mock_tm.find_window_by_id = AsyncMock(return_value=None)
        mock_tr.get_display_name.return_value = "project"
        ws = MagicMock()
        ws.cwd = "/tmp/project"
        mock_sm.view_window.return_value = ws
        mock_render.return_value = (
            "⚠ Session `project` ended.\n📂 `/tmp/project`",
            MagicMock(),
        )

        user_data: dict = {}
        message = AsyncMock()

        with patch(f"{_TH}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = True
            result = await _handle_dead_window(
                "@0", 100, 42, "hello", user_data, message
            )

        assert result is True
        mock_reply.assert_called_once()
        banner = mock_render.call_args.args[0]
        assert banner.window_id == "@0"
        assert banner.mode == "dead"
        assert banner.cwd == "/tmp/project"
        assert banner.display == "project"
        assert user_data[RECOVERY_WINDOW_ID] == "@0"

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_TH}.tmux_manager")
    @patch(f"{_TH}.window_query")
    @patch(f"{_TH}.thread_router")
    async def test_recovery_banner_includes_help_text(
        self,
        mock_tr: MagicMock,
        mock_sm: MagicMock,
        mock_tm: MagicMock,
        mock_reply: AsyncMock,
    ) -> None:
        mock_tm.find_window_by_id = AsyncMock(return_value=None)
        mock_tr.get_display_name.return_value = "project"
        ws = MagicMock()
        ws.cwd = "/tmp/project"
        mock_sm.view_window.return_value = ws

        user_data: dict = {}
        message = AsyncMock()

        with patch(
            "ccgram.handlers.recovery.recovery_banner.get_provider_for_window"
        ) as mock_gpw:
            caps = mock_gpw.return_value.capabilities
            caps.supports_continue = True
            caps.supports_resume = True
            with patch(f"{_TH}.Path") as mock_path:
                mock_path.return_value.is_dir.return_value = True
                await _handle_dead_window("@0", 100, 42, "hello", user_data, message)

        body = mock_reply.call_args.args[1]
        assert "Start fresh" in body
        assert "Continue last session" in body
        assert "Resume from list" in body

    @pytest.mark.parametrize("cwd", ["", "/nonexistent"])
    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(f"{_TH}.build_directory_browser")
    @patch(f"{_TH}.tmux_manager")
    @patch(f"{_TH}.window_query")
    @patch(f"{_TH}.thread_router")
    async def test_falls_back_to_browser(
        self,
        mock_tr: MagicMock,
        mock_sm: MagicMock,
        mock_tm: MagicMock,
        mock_browser: MagicMock,
        _mock_reply: AsyncMock,
        cwd: str,
    ) -> None:
        mock_tm.find_window_by_id = AsyncMock(return_value=None)
        mock_tr.get_display_name.return_value = "project"
        ws = MagicMock()
        ws.cwd = cwd
        mock_sm.view_window.return_value = ws
        mock_browser.return_value = ("Browse:", MagicMock(), [])

        user_data: dict = {}
        message = AsyncMock()

        with patch(f"{_TH}.Path") as mock_path:
            mock_path.return_value.is_dir.return_value = False
            mock_path.cwd.return_value = mock_path.return_value
            str_mock = MagicMock(return_value="/cwd")
            mock_path.cwd.return_value.__str__ = str_mock
            result = await _handle_dead_window(
                "@0", 100, 42, "hello", user_data, message
            )

        assert result is True
        mock_tr.unbind_thread.assert_called_once_with(100, 42)
        mock_browser.assert_called_once()


class TestShellProviderRouting:
    @patch(f"{_TH}.get_provider_for_window")
    @patch(f"{_TH}._handle_dead_window", new_callable=AsyncMock, return_value=False)
    @patch(f"{_TH}.thread_router")
    async def test_shell_provider_routes_to_handle_shell_message(
        self,
        mock_tr: MagicMock,
        _mock_dead: AsyncMock,
        mock_get_provider: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = "@0"

        provider = MagicMock()
        provider.capabilities.name = "shell"
        provider.capabilities.chat_first_command_path = True
        mock_get_provider.return_value = provider

        with patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_message",
            new_callable=AsyncMock,
        ) as mock_shell:
            from ccgram.handlers.text.text_handler import handle_text_message

            update = MagicMock()
            update.effective_user.id = 100
            context = MagicMock()
            context.bot = AsyncMock()
            context.user_data = {}
            message = AsyncMock()
            message.message_thread_id = 42
            message.text = "list files"
            message.chat_id = -100
            message.chat.type = "supergroup"
            update.message = message
            update.effective_user = MagicMock()
            update.effective_user.id = 100

            await handle_text_message(update, context)

            mock_shell.assert_called_once()
            call_args = mock_shell.call_args
            assert call_args[0][2] == 42
            assert call_args[0][3] == "@0"
            assert call_args[0][4] == "list files"

    @patch(f"{_TH}.get_provider_for_window")
    @patch(f"{_TH}._handle_dead_window", new_callable=AsyncMock, return_value=False)
    @patch(f"{_TH}.window_query")
    @patch(f"{_TH}.thread_router")
    async def test_non_shell_provider_does_not_route_to_shell(
        self,
        mock_tr: MagicMock,
        mock_sm: MagicMock,
        _mock_dead: AsyncMock,
        mock_get_provider: MagicMock,
    ) -> None:
        mock_tr.get_window_for_thread.return_value = "@0"
        mock_sm.send_to_window = AsyncMock(return_value=(True, ""))

        provider = MagicMock()
        provider.capabilities.name = "claude"
        provider.capabilities.chat_first_command_path = False
        mock_get_provider.return_value = provider

        with (
            patch(
                "ccgram.handlers.shell.shell_commands.handle_shell_message",
                new_callable=AsyncMock,
            ) as mock_shell,
            patch(f"{_TH}.get_interactive_window", return_value=None),
        ):
            from ccgram.handlers.text.text_handler import handle_text_message

            update = MagicMock()
            context = MagicMock()
            context.bot = AsyncMock()
            context.user_data = {}
            message = AsyncMock()
            message.message_thread_id = 42
            message.text = "hello"
            message.chat_id = -100
            message.chat.type = "supergroup"
            update.message = message
            update.effective_user = MagicMock()
            update.effective_user.id = 100

            await handle_text_message(update, context)

            mock_shell.assert_not_called()


class TestForwardMessage:
    @patch(f"{_TH}.send_to_window", new_callable=AsyncMock, return_value=(True, "ok"))
    @patch(f"{_TH}.window_query")
    async def test_sends_to_window(
        self, mock_sm: MagicMock, mock_send: AsyncMock
    ) -> None:
        bot = AsyncMock()
        message = AsyncMock()

        with patch(f"{_TH}.get_interactive_window", return_value=None):
            await _forward_message("@0", 100, 42, "hello", bot, message)

        mock_send.assert_called_once_with("@0", "hello")

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(
        f"{_TH}.send_to_window",
        new_callable=AsyncMock,
        return_value=(False, "Window not found"),
    )
    @patch(f"{_TH}.window_query")
    async def test_send_failure_replies_error(
        self, mock_sm: MagicMock, _mock_send: AsyncMock, mock_reply: AsyncMock
    ) -> None:
        bot = AsyncMock()
        message = AsyncMock()

        await _forward_message("@0", 100, 42, "hello", bot, message)

        mock_reply.assert_called_once()
        assert "Window not found" in mock_reply.call_args.args[1]

    @patch(f"{_TH}.get_interactive_window", return_value=None)
    @patch(f"{_TH}._capture_bash_output")
    @patch(f"{_TH}.send_to_window", new_callable=AsyncMock, return_value=(True, "ok"))
    @patch(f"{_TH}.window_query")
    async def test_bash_capture_for_bang_command(
        self,
        mock_sm: MagicMock,
        _mock_send: AsyncMock,
        mock_capture: MagicMock,
        _mock_interactive: MagicMock,
    ) -> None:
        bot = AsyncMock()
        message = AsyncMock()

        await _forward_message("@0", 100, 42, "!ls -la", bot, message)

        from ccgram.handlers.text.text_handler import _bash_capture_tasks

        key = (100, 42)
        assert key in _bash_capture_tasks
        task = _bash_capture_tasks.pop(key)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @patch(f"{_TH}.get_interactive_window", return_value=None)
    @patch(f"{_TH}.send_to_window", new_callable=AsyncMock, return_value=(True, "ok"))
    @patch(f"{_TH}.window_query")
    async def test_cancels_existing_bash_capture(
        self, mock_sm: MagicMock, _mock_send: AsyncMock, _mock_interactive: MagicMock
    ) -> None:
        bot = AsyncMock()
        message = AsyncMock()

        from ccgram.handlers.text.text_handler import _bash_capture_tasks

        dummy_task = AsyncMock(spec=asyncio.Task)
        dummy_task.done.return_value = False
        _bash_capture_tasks[(100, 42)] = dummy_task

        await _forward_message("@0", 100, 42, "hello", bot, message)

        dummy_task.cancel.assert_called_once()
        assert (100, 42) not in _bash_capture_tasks

    @patch(f"{_TH}.handle_interactive_ui", new_callable=AsyncMock)
    @patch(f"{_TH}.get_interactive_window", return_value="@0")
    @patch(f"{_TH}.send_to_window", new_callable=AsyncMock, return_value=(True, "ok"))
    @patch(f"{_TH}.window_query")
    async def test_refreshes_interactive_ui(
        self,
        mock_sm: MagicMock,
        _mock_send: AsyncMock,
        _mock_get_iw: MagicMock,
        mock_handle_ui: AsyncMock,
    ) -> None:
        bot = AsyncMock()
        message = AsyncMock()

        await _forward_message("@0", 100, 42, "hello", bot, message)

        mock_handle_ui.assert_called_once()
        assert mock_handle_ui.call_args.args[0] is bot
        assert mock_handle_ui.call_args.args[1:] == (100, "@0", 42)

    @patch(f"{_TH}.send_to_window", new_callable=AsyncMock, return_value=(True, "ok"))
    @patch(f"{_TH}.window_query")
    async def test_sends_typing_chat_action(
        self, _mock_sm: MagicMock, _mock_send: AsyncMock
    ) -> None:
        from telegram.constants import ChatAction

        bot = AsyncMock()
        message = AsyncMock()
        message.chat.send_action = AsyncMock()

        with patch(f"{_TH}.get_interactive_window", return_value=None):
            await _forward_message("@0", 100, 42, "hello", bot, message)

        message.chat.send_action.assert_awaited_once_with(ChatAction.TYPING)


class TestBashCaptureCleanup:
    @pytest.fixture(autouse=True)
    def _clear_bash_tasks(self):
        from ccgram.handlers.text.text_handler import _bash_capture_tasks

        _bash_capture_tasks.clear()
        yield
        _bash_capture_tasks.clear()

    async def test_cleanup_on_early_return(self, monkeypatch) -> None:
        from ccgram.handlers.text.text_handler import (
            _bash_capture_tasks,
            _capture_bash_output,
        )

        key = (999, 888)

        monkeypatch.setattr(f"{_TH}.asyncio.sleep", AsyncMock())
        with (
            patch(f"{_TH}.tmux_manager") as mock_tm,
            patch(f"{_TH}.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = 999
            mock_tm.capture_pane = AsyncMock(return_value=None)

            task = asyncio.create_task(
                _capture_bash_output(AsyncMock(), 999, 888, "@0", "ls")
            )
            _bash_capture_tasks[key] = task
            await task

        assert key not in _bash_capture_tasks

    async def test_cleanup_on_cancel(self) -> None:
        from ccgram.handlers.text.text_handler import (
            _bash_capture_tasks,
            _capture_bash_output,
        )

        key = (777, 666)

        with (
            patch(f"{_TH}.tmux_manager") as mock_tm,
            patch(f"{_TH}.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = 777
            mock_tm.capture_pane = AsyncMock(return_value=None)

            task = asyncio.create_task(
                _capture_bash_output(AsyncMock(), 777, 666, "@0", "ls")
            )
            _bash_capture_tasks[key] = task
            await asyncio.sleep(0)
            task.cancel()
            await task

        assert key not in _bash_capture_tasks

    async def test_identity_check_preserves_replacement_task(self) -> None:
        from ccgram.handlers.text.text_handler import (
            _bash_capture_tasks,
            _capture_bash_output,
        )

        key = (555, 444)
        sentinel = AsyncMock(spec=asyncio.Task)

        with (
            patch(f"{_TH}.tmux_manager") as mock_tm,
            patch(f"{_TH}.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = 555
            mock_tm.capture_pane = AsyncMock(return_value=None)

            task_a = asyncio.create_task(
                _capture_bash_output(AsyncMock(), 555, 444, "@0", "ls")
            )
            _bash_capture_tasks[key] = task_a
            await asyncio.sleep(0)

            task_a.cancel()
            _bash_capture_tasks[key] = sentinel  # Task B

            await task_a  # A's finally runs

        assert _bash_capture_tasks.get(key) is sentinel
