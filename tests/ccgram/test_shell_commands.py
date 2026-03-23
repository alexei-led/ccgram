"""Tests for shell command generation and approval flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot, CallbackQuery, InlineKeyboardMarkup, Message

from ccgram.handlers.callback_data import (
    CB_SHELL_CANCEL,
    CB_SHELL_CONFIRM_DANGER,
    CB_SHELL_EDIT,
    CB_SHELL_RUN,
    CB_SHELL_SETUP,
    CB_SHELL_SETUP_SKIP,
)
from ccgram.handlers.shell_commands import (
    _build_approval_keyboard,
    _cancel_stuck_input,
    _generation_counter,
    _marker_setup_skipped,
    _shell_pending,
    clear_marker_skip,
    clear_shell_pending,
    handle_shell_callback,
    handle_shell_message,
    has_shell_pending,
    show_command_approval,
)
from ccgram.llm.base import CommandResult

_MOD = "ccgram.handlers.shell_commands"


@pytest.fixture(autouse=True)
def _clean_shell_state():
    _shell_pending.clear()
    _marker_setup_skipped.clear()
    _generation_counter.clear()
    yield
    _shell_pending.clear()
    _marker_setup_skipped.clear()
    _generation_counter.clear()


class TestPendingState:
    def test_clear_removes_entry(self) -> None:
        _shell_pending[(-100, 42)] = ("ls", 1)
        clear_shell_pending(-100, 42)
        assert _shell_pending.get((-100, 42)) is None

    def test_clear_nonexistent_no_error(self) -> None:
        clear_shell_pending(999, 999)


class TestBuildApprovalKeyboard:
    @pytest.mark.parametrize(
        ("is_dangerous", "expected_labels", "absent_labels"),
        [
            (False, ["Run", "Edit", "Cancel"], []),
            (True, ["Confirm", "Cancel"], ["Edit"]),
        ],
        ids=["non-dangerous", "dangerous"],
    )
    def test_button_labels(
        self,
        is_dangerous: bool,
        expected_labels: list[str],
        absent_labels: list[str],
    ) -> None:
        kb = _build_approval_keyboard("@0", is_dangerous=is_dangerous)
        assert isinstance(kb, InlineKeyboardMarkup)
        texts = [btn.text for row in kb.inline_keyboard for btn in row]
        for label in expected_labels:
            assert any(label in t for t in texts)
        for label in absent_labels:
            assert not any(label in t for t in texts)

    @pytest.mark.parametrize(
        ("is_dangerous", "btn_label", "expected_prefix"),
        [
            (False, "Run", CB_SHELL_RUN),
            (True, "Confirm", CB_SHELL_CONFIRM_DANGER),
        ],
        ids=["non-dangerous-run", "dangerous-confirm"],
    )
    def test_callback_data_includes_window_id(
        self, is_dangerous: bool, btn_label: str, expected_prefix: str
    ) -> None:
        kb = _build_approval_keyboard("@5", is_dangerous=is_dangerous)
        buttons = [btn for row in kb.inline_keyboard for btn in row]
        btn = next(b for b in buttons if btn_label in b.text)
        assert btn.callback_data == f"{expected_prefix}@5"


class TestHandleShellMessage:
    async def test_bang_prefix_sends_raw_command(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.start_shell_capture") as mock_capture,
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            await handle_shell_message(bot, 1, 42, "@0", "!ls -la", message)

            mock_sm.send_to_window.assert_called_once_with("@0", "ls -la", raw=True)
            mock_capture.assert_called_once_with(
                bot, 1, 42, "@0", baseline="", command="ls -la"
            )

    async def test_bang_with_space_strips_leading_space(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.start_shell_capture"),
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            await handle_shell_message(bot, 1, 42, "@0", "! ls", message)

            mock_sm.send_to_window.assert_called_once_with("@0", "ls", raw=True)

    async def test_bare_bang_is_ignored(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.session_manager") as mock_sm,
        ):
            mock_sm.send_to_window = AsyncMock()
            await handle_shell_message(bot, 1, 42, "@0", "!", message)

            mock_sm.send_to_window.assert_not_called()

    async def test_no_bang_no_llm_sends_raw(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=None),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.start_shell_capture"),
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            await handle_shell_message(bot, 1, 42, "@0", "find . -name foo", message)

            mock_sm.send_to_window.assert_called_once_with(
                "@0", "find . -name foo", raw=True
            )

    async def test_no_bang_with_llm_calls_completer(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="find . -name foo", explanation="Search", is_dangerous=False
            )
        )

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock),
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(
                bot, 1, 42, "@0", "find files named foo", message
            )

            mock_completer.generate_command.assert_called_once()
            assert (
                mock_completer.generate_command.call_args[0][0]
                == "find files named foo"
            )

    async def test_llm_error_notifies_user(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "do something", message)

            mock_send.assert_called_once()
            assert "LLM request failed" in mock_send.call_args[0][2]
            mock_sm.send_to_window.assert_not_called()

    async def test_llm_config_error_notifies_user(self) -> None:
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", side_effect=ValueError("bad provider")),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await handle_shell_message(bot, 1, 42, "@0", "do something")

            mock_send.assert_called_once()
            assert "LLM misconfigured" in mock_send.call_args[0][2]
            mock_sm.send_to_window.assert_not_called()

    async def test_send_failure_replies_error(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(False, "Window not found"))
            mock_sm.resolve_chat_id.return_value = -100

            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

            mock_send.assert_called_once()
            assert "Window not found" in mock_send.call_args[0][2]

    async def test_message_optional_uses_safe_send(self) -> None:
        bot = AsyncMock(spec=Bot)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(command="ls", explanation="", is_dangerous=False)
        )

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "list files")

            mock_send.assert_called_once()


class TestHandleShellCallback:
    async def test_run_with_pending_executes_and_clears(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(f"{_MOD}.start_shell_capture") as mock_capture,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_window_for_thread.return_value = "@0"
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            _shell_pending[(-100, 42)] = ("ls -la", 1)

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            query.answer.assert_called_once()
            mock_sm.send_to_window.assert_called_once_with("@0", "ls -la", raw=True)
            mock_capture.assert_called_once_with(
                bot, 1, 42, "@0", baseline="", command="ls -la"
            )
            assert _shell_pending.get((-100, 42)) is None

    async def test_run_wrong_user_rejects(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            _shell_pending[(-100, 42)] = ("ls -la", 999)

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            assert "Not your command" in mock_edit.call_args[0][1]

    async def test_confirm_danger_wrong_user_rejects(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            _shell_pending[(-100, 42)] = ("rm -rf /", 999)

            await handle_shell_callback(
                query, 1, f"{CB_SHELL_CONFIRM_DANGER}@0", bot, 42
            )

            assert "Not your command" in mock_edit.call_args[0][1]

    async def test_run_no_window_binding_rejects(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_window_for_thread.return_value = None
            _shell_pending[(-100, 42)] = ("ls -la", 1)

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            assert "No session bound" in mock_edit.call_args[0][1]
            assert _shell_pending.get((-100, 42)) is None

    async def test_run_without_pending_shows_expired(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_sm.resolve_chat_id.return_value = -100

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            mock_edit.assert_called_once()
            assert "expired" in mock_edit.call_args[0][1]

    async def test_cancel_clears_pending(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            _shell_pending[(-100, 42)] = ("rm -rf /", 1)

            await handle_shell_callback(query, 1, f"{CB_SHELL_CANCEL}@0", bot, 42)

            query.answer.assert_called_once_with("Cancelled")
            assert _shell_pending.get((-100, 42)) is None
            mock_edit.assert_called_once()
            assert "Cancelled" in mock_edit.call_args[0][1]

    async def test_edit_clears_pending_and_shows_command(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            _shell_pending[(-100, 42)] = ("grep -r pattern .", 1)

            await handle_shell_callback(query, 1, f"{CB_SHELL_EDIT}@0", bot, 42)

            mock_edit.assert_called_once()
            assert "grep -r pattern ." in mock_edit.call_args[0][1]
            assert _shell_pending.get((-100, 42)) is None

    async def test_edit_without_pending_shows_expired(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_sm.resolve_chat_id.return_value = -100

            await handle_shell_callback(query, 1, f"{CB_SHELL_EDIT}@0", bot, 42)

            mock_edit.assert_called_once()
            assert "expired" in mock_edit.call_args[0][1]

    async def test_thread_id_none_answers_no_context(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, None)

        query.answer.assert_called_once_with("No topic context")

    async def test_confirm_danger_with_pending_executes(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(f"{_MOD}.start_shell_capture"),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_window_for_thread.return_value = "@0"
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            _shell_pending[(-100, 42)] = ("rm -rf /tmp/test", 1)

            await handle_shell_callback(
                query, 1, f"{CB_SHELL_CONFIRM_DANGER}@0", bot, 42
            )

            mock_sm.send_to_window.assert_called_once_with(
                "@0", "rm -rf /tmp/test", raw=True
            )
            assert _shell_pending.get((-100, 42)) is None


class TestGatherLlmContext:
    async def test_assembles_cwd_shell_and_tools(self) -> None:
        from ccgram.handlers.shell_commands import gather_llm_context

        with (
            patch(
                "ccgram.providers.shell.detect_pane_shell",
                new_callable=AsyncMock,
                return_value="fish",
            ),
            patch(
                f"{_MOD}._detect_shell_tools",
                return_value="rg (grep replacement)",
            ),
            patch(f"{_MOD}.session_manager") as mock_sm,
        ):
            mock_sm.get_window_state.return_value = MagicMock(cwd="/home/user/project")
            ctx = await gather_llm_context("@0")

        assert ctx["cwd"] == "/home/user/project"
        assert ctx["shell"] == "fish"
        assert ctx["shell_tools"] == "rg (grep replacement)"

    async def test_empty_cwd_when_none(self) -> None:
        from ccgram.handlers.shell_commands import gather_llm_context

        with (
            patch(
                "ccgram.providers.shell.detect_pane_shell",
                new_callable=AsyncMock,
                return_value="bash",
            ),
            patch(
                f"{_MOD}._detect_shell_tools",
                return_value="",
            ),
            patch(f"{_MOD}.session_manager") as mock_sm,
        ):
            mock_sm.get_window_state.return_value = MagicMock(cwd=None)
            ctx = await gather_llm_context("@0")

        assert ctx["cwd"] == ""


class TestCancelStuckInput:
    def _mock_window(self, pane_cmd: str = "fish"):  # noqa: ANN202
        from ccgram.tmux_manager import TmuxWindow

        return TmuxWindow(
            window_id="@0",
            window_name="test",
            cwd="/tmp",
            pane_current_command=pane_cmd,
        )

    async def test_clean_prompt_does_nothing(self) -> None:
        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=self._mock_window())
            mock_tm.capture_pane = AsyncMock(return_value="output\nccgram:0❯ ")
            mock_tm.send_keys = AsyncMock()

            await _cancel_stuck_input("@0")

            mock_tm.send_keys.assert_not_called()

    async def test_stuck_continuation_sends_ctrl_c(self) -> None:
        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=self._mock_window())
            mock_tm.capture_pane = AsyncMock(
                return_value="ccgram:0❯ begin\n  for x in 1 2 3"
            )
            mock_tm.send_keys = AsyncMock()

            await _cancel_stuck_input("@0")

            mock_tm.send_keys.assert_called_once_with(
                "@0", "C-c", enter=False, literal=False
            )

    async def test_running_command_skips(self) -> None:

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(
                return_value=self._mock_window(pane_cmd="python3")
            )
            mock_tm.send_keys = AsyncMock()

            await _cancel_stuck_input("@0")

            mock_tm.send_keys.assert_not_called()

    async def test_no_window_skips(self) -> None:
        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.send_keys = AsyncMock()

            await _cancel_stuck_input("@0")

            mock_tm.send_keys.assert_not_called()

    async def test_partial_typed_text_sends_ctrl_c(self) -> None:
        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=self._mock_window())
            mock_tm.capture_pane = AsyncMock(return_value="ccgram:0❯ some partial inp")
            mock_tm.send_keys = AsyncMock()

            await _cancel_stuck_input("@0")

            mock_tm.send_keys.assert_called_once()

    async def test_tail_dash_f_running_skips(self) -> None:

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(
                return_value=self._mock_window(pane_cmd="tail")
            )
            mock_tm.send_keys = AsyncMock()

            await _cancel_stuck_input("@0")

            mock_tm.send_keys.assert_not_called()

    async def test_login_shell_detected(self) -> None:

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(
                return_value=self._mock_window(pane_cmd="-bash")
            )
            mock_tm.capture_pane = AsyncMock(return_value="ccgram:0❯ echo 'unclosed")
            mock_tm.send_keys = AsyncMock()

            await _cancel_stuck_input("@0")

            mock_tm.send_keys.assert_called_once()


class TestShowCommandApprovalPaths:
    async def test_message_present_uses_safe_reply(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)
        result = CommandResult(
            command="ls", explanation="List files", is_dangerous=False
        )

        with patch(f"{_MOD}.safe_reply", new_callable=AsyncMock) as mock_reply:
            await show_command_approval(bot, -100, 42, "@0", result, 1, message)

        mock_reply.assert_called_once()
        assert "`ls`" in mock_reply.call_args[0][1]
        assert _shell_pending[(-100, 42)] == ("ls", 1)

    async def test_message_none_uses_safe_send(self) -> None:
        bot = AsyncMock(spec=Bot)
        result = CommandResult(command="pwd", explanation="", is_dangerous=False)

        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send:
            await show_command_approval(bot, -100, 42, "@0", result, 1, None)

        mock_send.assert_called_once()
        assert "`pwd`" in mock_send.call_args[0][2]
        assert _shell_pending[(-100, 42)] == ("pwd", 1)


class TestOfferPromptSetup:
    async def test_skips_when_user_chose_skip(self) -> None:
        from ccgram.handlers.shell_commands import (
            _marker_setup_skipped,
            offer_prompt_setup,
        )

        _marker_setup_skipped.add("@0")
        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send:
            await offer_prompt_setup(bot, 1, 42, "@0")

        mock_send.assert_not_called()

    async def test_skips_when_marker_present(self) -> None:
        from ccgram.handlers.shell_commands import offer_prompt_setup

        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
        ):
            await offer_prompt_setup(bot, 1, 42, "@0")

        mock_send.assert_not_called()

    async def test_sends_keyboard_when_marker_absent(self) -> None:
        from ccgram.handlers.shell_commands import offer_prompt_setup

        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await offer_prompt_setup(bot, 1, 42, "@0")

        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args
        assert "prompt marker" in call_kwargs[0][2]
        keyboard = call_kwargs[1]["reply_markup"]
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        assert len(buttons) == 2
        assert "Set up" in buttons[0].text
        assert "Skip" in buttons[1].text

    async def test_setup_callback_calls_setup_shell_prompt(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                "ccgram.providers.shell.setup_shell_prompt", new_callable=AsyncMock
            ) as mock_setup,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window", return_value=True
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock())
            await handle_shell_callback(query, 1, f"{CB_SHELL_SETUP}@3", bot, 42)

        mock_setup.assert_awaited_once_with("@3")
        mock_edit.assert_called_once()
        assert "\u2705" in mock_edit.call_args[0][1]
        query.answer.assert_called_once_with("Done")

    async def test_skip_callback_edits_message(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window", return_value=True
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await handle_shell_callback(query, 1, f"{CB_SHELL_SETUP_SKIP}@3", bot, 42)

        mock_edit.assert_called_once()
        assert "Skipped" in mock_edit.call_args[0][1]
        query.answer.assert_called_once_with("Skipped")


class TestLazyMarkerRecovery:
    async def test_raw_command_restores_marker_when_missing(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.start_shell_capture"),
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "ccgram.providers.shell.setup_shell_prompt", new_callable=AsyncMock
            ) as mock_setup,
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

        mock_setup.assert_awaited_once_with("@0")

    async def test_raw_command_skips_setup_when_marker_present(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.start_shell_capture"),
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "ccgram.providers.shell.setup_shell_prompt", new_callable=AsyncMock
            ) as mock_setup,
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

        mock_setup.assert_not_awaited()

    async def test_lazy_recovery_skips_when_user_chose_skip(self) -> None:
        from ccgram.handlers.shell_commands import _marker_setup_skipped

        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        _marker_setup_skipped.add("@0")
        try:
            with (
                patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
                patch(f"{_MOD}.clear_probe_failures"),
                patch(f"{_MOD}.session_manager") as mock_sm,
                patch(f"{_MOD}.tmux_manager") as mock_tm,
                patch(f"{_MOD}.start_shell_capture"),
                patch(
                    "ccgram.providers.shell.setup_shell_prompt",
                    new_callable=AsyncMock,
                ) as mock_setup,
            ):
                mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
                mock_tm.find_window_by_id = AsyncMock(return_value=None)
                mock_tm.capture_pane = AsyncMock(return_value=None)
                await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

            mock_setup.assert_not_awaited()
        finally:
            _marker_setup_skipped.discard("@0")


class TestSkipTracking:
    async def test_skip_callback_does_not_call_setup(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(
                "ccgram.providers.shell.setup_shell_prompt", new_callable=AsyncMock
            ) as mock_setup,
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window", return_value=True
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await handle_shell_callback(query, 1, f"{CB_SHELL_SETUP_SKIP}@0", bot, 42)

        mock_setup.assert_not_awaited()

    async def test_skip_callback_records_window_in_skip_set(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window", return_value=True
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await handle_shell_callback(query, 1, f"{CB_SHELL_SETUP_SKIP}@5", bot, 42)

        assert "@5" in _marker_setup_skipped

    async def test_setup_callback_clears_skip_flag(self) -> None:
        from ccgram.handlers.shell_commands import _marker_setup_skipped

        _marker_setup_skipped.add("@3")
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch("ccgram.providers.shell.setup_shell_prompt", new_callable=AsyncMock),
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window", return_value=True
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock())
            await handle_shell_callback(query, 1, f"{CB_SHELL_SETUP}@3", bot, 42)

        assert "@3" not in _marker_setup_skipped

    async def test_setup_callback_handles_dead_window(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                "ccgram.providers.shell.setup_shell_prompt", new_callable=AsyncMock
            ) as mock_setup,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.callback_helpers.user_owns_window", return_value=True
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            await handle_shell_callback(query, 1, f"{CB_SHELL_SETUP}@99", bot, 42)

        mock_setup.assert_not_awaited()
        assert "no longer exists" in mock_edit.call_args[0][1]
        query.answer.assert_called_once_with("Window gone", show_alert=True)


class TestHasPromptMarker:
    @pytest.mark.parametrize(
        ("capture_value", "expected"),
        [("ccgram:0❯ ", True), ("$ ", False), (None, False)],
        ids=["marker-present", "marker-absent", "capture-none"],
    )
    async def test_has_prompt_marker(
        self, capture_value: str | None, expected: bool
    ) -> None:
        from ccgram.providers.shell import has_prompt_marker

        with patch("ccgram.tmux_manager.tmux_manager") as mock_tm:
            mock_tm.capture_pane = AsyncMock(return_value=capture_value)
            assert await has_prompt_marker("@0") is expected


class TestClearMarkerSkip:
    def test_clear_removes_window_from_skip_set(self) -> None:
        from ccgram.handlers.shell_commands import _marker_setup_skipped

        _marker_setup_skipped.add("@0")
        clear_marker_skip("@0")
        assert "@0" not in _marker_setup_skipped

    def test_clear_nonexistent_no_error(self) -> None:
        clear_marker_skip("@999")

    async def test_after_clear_offer_shows_keyboard(self) -> None:
        from ccgram.handlers.shell_commands import (
            _marker_setup_skipped,
            offer_prompt_setup,
        )

        _marker_setup_skipped.add("@0")
        clear_marker_skip("@0")

        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await offer_prompt_setup(bot, 1, 42, "@0")

        mock_send.assert_called_once()


class TestHasShellPending:
    def test_returns_false_when_empty(self) -> None:
        assert has_shell_pending(-100, 42) is False

    def test_returns_true_when_entry_exists(self) -> None:
        _shell_pending[(-100, 42)] = ("ls", 1)
        assert has_shell_pending(-100, 42) is True

    def test_returns_false_for_different_key(self) -> None:
        _shell_pending[(-100, 42)] = ("ls", 1)
        assert has_shell_pending(-100, 99) is False


class TestDangerousCommandPrefix:
    async def test_dangerous_result_shows_warning_prefix(self) -> None:
        bot = AsyncMock(spec=Bot)
        result = CommandResult(
            command="rm -rf /", explanation="Delete all", is_dangerous=True
        )

        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send:
            await show_command_approval(bot, -100, 42, "@0", result, user_id=1)

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][2]
        assert "\u26a0\ufe0f *Potentially dangerous*" in sent_text
        assert "rm -rf /" in sent_text

    async def test_non_dangerous_result_no_warning_prefix(self) -> None:
        bot = AsyncMock(spec=Bot)
        result = CommandResult(
            command="ls -la", explanation="List files", is_dangerous=False
        )

        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send:
            await show_command_approval(bot, -100, 42, "@0", result, user_id=1)

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][2]
        assert "Potentially dangerous" not in sent_text
        assert "ls -la" in sent_text


class TestDetectShellTools:
    def setup_method(self) -> None:
        import ccgram.handlers.shell_commands as mod

        self._mod = mod
        self._original = mod._cached_shell_tools
        mod._cached_shell_tools = None

    def teardown_method(self) -> None:
        self._mod._cached_shell_tools = self._original

    def test_returns_detected_tools(self) -> None:
        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("fd", "rg") else None

        with patch("shutil.which", side_effect=fake_which):
            from ccgram.handlers.shell_commands import _detect_shell_tools

            result = _detect_shell_tools()

        assert "fd" in result
        assert "rg" in result
        assert "bat" not in result

    def test_cache_populated_and_reused(self) -> None:
        with patch("shutil.which", return_value=None):
            from ccgram.handlers.shell_commands import _detect_shell_tools

            first = _detect_shell_tools()
            second = _detect_shell_tools()

        assert first is second
