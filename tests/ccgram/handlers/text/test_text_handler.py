import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.text.text_handler import (
    PENDING_DELIVERY_NOTICE,
    _bash_capture_tasks,
    _capture_bash_output,
    _check_ui_guards,
    _forward_message,
    _handle_dead_window,
    _handle_unbound_topic,
    handle_text_message,
)
from ccgram.handlers.polling.polling_state import lifecycle_strategy
from ccgram.multiplexer.base import WindowRef
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


@pytest.fixture(autouse=True)
def _clean_lifecycle_state():
    lifecycle_strategy._states.clear()
    yield
    lifecycle_strategy._states.clear()


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
    @pytest.fixture
    def unbound_env(self) -> Iterator[SimpleNamespace]:
        with (
            patch(f"{_TH}.thread_router") as router,
            patch(f"{_TH}.tmux_manager") as mux,
            patch(f"{_TH}.build_window_picker") as picker,
            patch(f"{_TH}.build_directory_browser") as browser,
            patch(f"{_TH}.safe_reply", new_callable=AsyncMock) as reply,
        ):
            router.get_window_for_thread.return_value = None
            router.iter_thread_bindings.return_value = []
            mux.list_windows = AsyncMock(return_value=[])
            picker.return_value = ("Pick:", MagicMock(), ["@5"])
            browser.return_value = ("Browse:", MagicMock(), [])
            yield SimpleNamespace(
                router=router, mux=mux, picker=picker, browser=browser, reply=reply
            )

    async def test_bound_topic_returns_false(
        self, unbound_env: SimpleNamespace
    ) -> None:
        unbound_env.router.get_window_for_thread.return_value = "@0"

        result = await _handle_unbound_topic(100, 42, "hello", {}, AsyncMock())

        assert result is False
        unbound_env.picker.assert_not_called()
        unbound_env.browser.assert_not_called()

    async def test_adoptable_windows_show_the_picker(
        self, unbound_env: SimpleNamespace
    ) -> None:
        unbound_env.mux.list_windows = AsyncMock(
            return_value=[MagicMock(window_id="@5", window_name="proj", cwd="/tmp")]
        )
        user_data: dict = {}

        result = await _handle_unbound_topic(100, 42, "my text", user_data, AsyncMock())

        assert result is True
        unbound_env.picker.assert_called_once()
        assert user_data[STATE_KEY] == STATE_SELECTING_WINDOW
        assert user_data[PENDING_THREAD_ID] == 42
        assert user_data[PENDING_THREAD_TEXT] == "my text"
        assert unbound_env.reply.call_count == 2
        assert unbound_env.reply.call_args_list[1].args[1] == PENDING_DELIVERY_NOTICE

    async def test_no_adoptable_windows_shows_the_directory_browser(
        self, unbound_env: SimpleNamespace
    ) -> None:
        user_data: dict = {}

        result = await _handle_unbound_topic(100, 42, "my text", user_data, AsyncMock())

        assert result is True
        unbound_env.browser.assert_called_once()
        assert user_data[STATE_KEY] == STATE_BROWSING_DIRECTORY
        assert user_data[PENDING_THREAD_ID] == 42
        assert user_data[PENDING_THREAD_TEXT] == "my text"
        assert unbound_env.reply.call_count == 2
        assert unbound_env.reply.call_args_list[1].args[1] == PENDING_DELIVERY_NOTICE


class TestHandleDeadWindow:
    @patch(f"{_TH}.tmux_manager")
    async def test_alive_window_returns_false(self, mock_tm: MagicMock) -> None:
        mock_tm.list_windows_for_reconciliation = AsyncMock(
            return_value=[WindowRef(window_id="@0", window_name="p", cwd="/p")]
        )
        message = AsyncMock()

        result = await _handle_dead_window("@0", 100, 42, "hello", {}, message)

        assert result is False

    @patch(f"{_TH}.tmux_manager")
    async def test_alive_window_clears_stale_autoclose_timer(
        self, mock_tm: MagicMock
    ) -> None:
        lifecycle_strategy.start_autoclose_timer(100, 42, "dead", 100.0)
        mock_tm.list_windows_for_reconciliation = AsyncMock(
            return_value=[WindowRef(window_id="@0", window_name="p", cwd="/p")]
        )
        message = AsyncMock()

        result = await _handle_dead_window("@0", 100, 42, "hello", {}, message)

        assert result is False
        assert lifecycle_strategy.get_state(100, 42).autoclose is None

    async def test_live_shell_after_agent_exit_shows_recovery(self) -> None:
        # A real ref, not a MagicMock: WindowRef.matches compares ids, and a
        # mock would fail to match and route this through the window-is-gone
        # branch, passing the assertions below without exercising the case.
        window = WindowRef(
            window_id="@0",
            window_name="project",
            cwd="/tmp/project",
            pane_current_command="bash",
        )
        view = MagicMock(cwd="/tmp/project", provider_name="claude")
        message = AsyncMock()
        message.chat.id = -100
        with (
            patch(f"{_TH}.tmux_manager") as mock_tm,
            patch(f"{_TH}.window_query") as mock_query,
            patch(f"{_TH}.thread_router") as mock_router,
            patch(
                f"{_TH}.agent_origin_returned_to_shell",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(f"{_TH}.render_banner", return_value=("recovery", MagicMock())),
            patch(f"{_TH}.safe_reply", new_callable=AsyncMock) as mock_reply,
            patch(f"{_TH}.Path") as mock_path,
        ):
            mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=[window])
            mock_query.get_window_provider.return_value = "claude"
            mock_query.view_window.return_value = view
            mock_router.get_display_name.return_value = "project"
            mock_path.return_value.is_dir.return_value = True

            result = await _handle_dead_window("@0", 100, 42, "hello", {}, message)

        assert result is True
        mock_reply.assert_awaited_once()

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
        mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=[])
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
        mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=[])
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
        mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=[])
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
        mock_tr.unbind_thread.assert_called_once_with(
            100,
            42,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )
        mock_browser.assert_called_once()


def _text_update(text: str, *, user_id: int = 100, thread_id: int = 42) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    message = AsyncMock()
    message.message_thread_id = thread_id
    message.text = text
    message.chat_id = -100
    message.chat.type = "supergroup"
    update.message = message
    return update


def _text_context() -> MagicMock:
    ctx = MagicMock()
    ctx.bot = AsyncMock()
    ctx.user_data = {}
    return ctx


class TestShellProviderRouting:
    @pytest.fixture
    def routing_env(self) -> Iterator[SimpleNamespace]:
        with (
            patch(f"{_TH}.thread_router") as router,
            patch(f"{_TH}.window_query"),
            patch(f"{_TH}.get_provider_for_window") as get_provider,
            patch(
                f"{_TH}._handle_dead_window",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(f"{_TH}.get_interactive_window", return_value=None),
            patch(
                "ccgram.handlers.shell.shell_commands.handle_shell_message",
                new_callable=AsyncMock,
            ) as handle_shell,
        ):
            router.get_window_for_thread.return_value = "@0"
            yield SimpleNamespace(
                router=router, get_provider=get_provider, handle_shell=handle_shell
            )

    @staticmethod
    def _provider(name: str, *, chat_first: bool) -> MagicMock:
        provider = MagicMock()
        provider.capabilities.name = name
        provider.capabilities.chat_first_command_path = chat_first
        return provider

    async def test_shell_provider_routes_to_handle_shell_message(
        self, routing_env: SimpleNamespace
    ) -> None:
        routing_env.get_provider.return_value = self._provider("shell", chat_first=True)

        await handle_text_message(_text_update("list files"), _text_context())

        routing_env.handle_shell.assert_called_once()
        assert routing_env.handle_shell.call_args[0][2:5] == (42, "@0", "list files")

    async def test_non_shell_provider_does_not_route_to_shell(
        self, routing_env: SimpleNamespace
    ) -> None:
        routing_env.get_provider.return_value = self._provider(
            "claude", chat_first=False
        )

        await handle_text_message(_text_update("hello"), _text_context())

        routing_env.handle_shell.assert_not_called()


class TestForwardMessage:
    @patch(
        f"{_TH}.send_telegram_to_window",
        new_callable=AsyncMock,
        return_value=(True, "ok"),
    )
    @patch(f"{_TH}.window_query")
    async def test_sends_to_window(
        self, mock_sm: MagicMock, mock_send: AsyncMock
    ) -> None:
        bot = AsyncMock()
        message = AsyncMock()

        with patch(f"{_TH}.get_interactive_window", return_value=None):
            await _forward_message("@0", 100, 42, "hello", bot, message)

        mock_send.assert_called_once_with(100, "@0", 42, "hello", ANY)

    @patch(f"{_TH}.safe_reply", new_callable=AsyncMock)
    @patch(
        f"{_TH}.send_telegram_to_window",
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
    @patch(
        f"{_TH}.send_telegram_to_window",
        new_callable=AsyncMock,
        return_value=(True, "ok"),
    )
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

        key = (100, 42)
        assert key in _bash_capture_tasks
        task = _bash_capture_tasks.pop(key)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @patch(f"{_TH}.get_interactive_window", return_value=None)
    @patch(
        f"{_TH}.send_telegram_to_window",
        new_callable=AsyncMock,
        return_value=(True, "ok"),
    )
    @patch(f"{_TH}.window_query")
    async def test_cancels_existing_bash_capture(
        self, mock_sm: MagicMock, _mock_send: AsyncMock, _mock_interactive: MagicMock
    ) -> None:
        bot = AsyncMock()
        message = AsyncMock()

        dummy_task = AsyncMock(spec=asyncio.Task)
        dummy_task.done.return_value = False
        _bash_capture_tasks[(100, 42)] = dummy_task

        await _forward_message("@0", 100, 42, "hello", bot, message)

        dummy_task.cancel.assert_called_once()
        assert (100, 42) not in _bash_capture_tasks

    @patch(f"{_TH}.handle_interactive_ui", new_callable=AsyncMock)
    @patch(f"{_TH}.get_interactive_window", return_value="@0")
    @patch(
        f"{_TH}.send_telegram_to_window",
        new_callable=AsyncMock,
        return_value=(True, "ok"),
    )
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

    @patch(
        f"{_TH}.send_telegram_to_window",
        new_callable=AsyncMock,
        return_value=(True, "ok"),
    )
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
    def _clear_bash_tasks(self) -> Iterator[None]:
        _bash_capture_tasks.clear()
        yield
        _bash_capture_tasks.clear()

    async def test_cleanup_on_early_return(self, monkeypatch) -> None:
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


class TestDeadWindowNeedsAConfirmedRead:
    """An unreachable backend must not unbind a live topic.

    find_window_by_id answers None both for a window that is gone and for a
    backend that could not be reached. With a stale or invalid cached cwd, the
    dead branch unbinds the thread as system_replacement and drops the user
    into the directory browser, so the live session loses its topic.
    """

    async def test_unreachable_backend_preserves_the_binding(self) -> None:
        message = AsyncMock()
        with (
            patch(f"{_TH}.tmux_manager") as mock_tm,
            patch(f"{_TH}.window_query") as mock_query,
            patch(f"{_TH}.thread_router") as mock_router,
            patch(f"{_TH}.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=None)
            mock_query.is_legacy_herdr.return_value = False
            # An invalid cached cwd: the shape that reaches the unbind.
            mock_query.view_window.return_value = MagicMock(cwd="/nonexistent")

            result = await _handle_dead_window("@0", 100, 42, "hi", {}, message)

        assert result is True
        mock_router.unbind_thread.assert_not_called()
        assert "Could not reach" in mock_reply.await_args[0][1]


class TestStaleDeadMarkerDoesNotOutliveTheWindow:
    """The dead marker is sticky, so a live window must be able to clear it.

    tick_window returns early once the marker is set, so nothing else clears
    it. An agterm session comes back under the same UUID after an app restart,
    and with a stale cached cwd the dead branch unbinds the topic.
    """

    async def test_live_agent_clears_the_marker_instead_of_unbinding(self) -> None:
        lifecycle_strategy.mark_dead_notified(100, 42, "@0")
        window = WindowRef(
            window_id="@0",
            window_name="project",
            cwd="/tmp/project",
            pane_current_command="claude",
        )
        message = AsyncMock()
        with (
            patch(f"{_TH}.tmux_manager") as mock_tm,
            patch(f"{_TH}.window_query") as mock_query,
            patch(f"{_TH}.thread_router") as mock_router,
            patch(
                f"{_TH}.agent_origin_returned_to_shell",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=[window])
            mock_query.is_legacy_herdr.return_value = False
            mock_query.get_window_provider.return_value = "claude"
            # Stale cached cwd: the shape that otherwise reaches the unbind.
            mock_query.view_window.return_value = MagicMock(cwd="/nonexistent")

            result = await _handle_dead_window("@0", 100, 42, "hi", {}, message)

        assert result is False
        mock_router.unbind_thread.assert_not_called()
        assert not lifecycle_strategy.is_dead_notified(100, 42, "@0")

    async def test_shell_return_still_shows_recovery(self) -> None:
        """The marker clearing must not swallow the case recovery exists for."""
        window = WindowRef(
            window_id="@0",
            window_name="project",
            cwd="/tmp/project",
            pane_current_command="bash",
        )
        message = AsyncMock()
        message.chat.id = -100
        with (
            patch(f"{_TH}.tmux_manager") as mock_tm,
            patch(f"{_TH}.window_query") as mock_query,
            patch(f"{_TH}.thread_router") as mock_router,
            patch(
                f"{_TH}.agent_origin_returned_to_shell",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(f"{_TH}.render_banner", return_value=("recovery", MagicMock())),
            patch(f"{_TH}.safe_reply", new_callable=AsyncMock) as mock_reply,
            patch(f"{_TH}.Path") as mock_path,
        ):
            mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=[window])
            mock_query.is_legacy_herdr.return_value = False
            mock_query.get_window_provider.return_value = "claude"
            mock_query.view_window.return_value = MagicMock(
                cwd="/tmp/project", provider_name="claude"
            )
            mock_router.get_display_name.return_value = "project"
            mock_path.return_value.is_dir.return_value = True

            result = await _handle_dead_window("@0", 100, 42, "hi", {}, message)

        assert result is True
        mock_reply.assert_awaited_once()


class TestUnknownAgentStateDoesNotClearTheMarker:
    """agterm reports no foreground for a shell, an unreadable process group
    and a failed lookup alike, and detection maps all three to "". Treating
    that as "an agent is running" clears the dead marker, forwards the next
    message, and the shared send guard types it into the pane with Return, so
    a session restored under the same UUID as a plain shell runs it.
    """

    @staticmethod
    async def _run(pane_command: str) -> tuple[bool, AsyncMock, AsyncMock]:
        lifecycle_strategy.mark_dead_notified(100, 42, "@0")
        window = WindowRef(
            window_id="@0",
            window_name="project",
            cwd="/tmp/project",
            pane_current_command=pane_command,
        )
        message = AsyncMock()
        with (
            patch(f"{_TH}.tmux_manager") as mock_tm,
            patch(f"{_TH}.window_query") as mock_query,
            patch(f"{_TH}.thread_router") as mock_router,
            patch(
                f"{_TH}.agent_origin_returned_to_shell",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(f"{_TH}.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_tm.list_windows_for_reconciliation = AsyncMock(return_value=[window])
            mock_query.is_legacy_herdr.return_value = False
            mock_query.get_window_provider.return_value = "claude"
            # Stale cached cwd: the shape that reaches the unbind.
            mock_query.view_window.return_value = MagicMock(cwd="/nonexistent")
            result = await _handle_dead_window("@0", 100, 42, "hi", {}, message)
        return result, mock_router, mock_reply

    @pytest.mark.parametrize("pane_command", ["", "vim", "less", "top"])
    async def test_an_unknown_foreground_keeps_the_marker_and_forwards_nothing(
        self, pane_command: str
    ) -> None:
        handled, mock_router, mock_reply = await self._run(pane_command)

        assert handled is True, "the message must not be forwarded to the pane"
        assert lifecycle_strategy.is_dead_notified(100, 42, "@0")
        mock_router.unbind_thread.assert_not_called()
        assert "nothing confirms an agent" in mock_reply.await_args[0][1]

    async def test_a_named_agent_still_clears_the_marker(self) -> None:
        """The other side: a confirmed agent is why the clearing exists."""
        handled, mock_router, _ = await self._run("claude")

        assert handled is False, "a live agent's topic keeps forwarding"
        assert not lifecycle_strategy.is_dead_notified(100, 42, "@0")
        mock_router.unbind_thread.assert_not_called()
