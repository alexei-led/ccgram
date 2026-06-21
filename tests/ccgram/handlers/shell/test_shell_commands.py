import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot, CallbackQuery, InlineKeyboardMarkup, Message

from ccgram.handlers.callback_data import (
    CB_SHELL_CANCEL,
    CB_SHELL_CONFIRM_DANGER,
    CB_SHELL_EDIT,
    CB_SHELL_RUN,
)
from ccgram.handlers.shell.shell_commands import (
    _BANG_HINT_TEXT,
    _build_approval_keyboard,
    _cancel_stuck_input,
    _clear_shell_hint_seen,
    _generation_counter,
    _shell_hint_seen,
    _shell_pending,
    clear_shell_pending,
    handle_shell_callback,
    handle_shell_message,
    has_shell_pending,
    show_command_approval,
)
from ccgram.llm.base import CommandResult

_MOD = "ccgram.handlers.shell.shell_commands"
_CTX = "ccgram.handlers.shell.shell_context"


@pytest.fixture(autouse=True)
def _clean_shell_state():
    _shell_pending.clear()
    _generation_counter.clear()
    _shell_hint_seen.clear()
    yield
    _shell_pending.clear()
    _generation_counter.clear()
    _shell_hint_seen.clear()


class TestPendingState:
    def test_clear_removes_entry(self) -> None:
        _shell_pending[(-100, 42)] = ("ls", 1, 0)
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
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
            patch(
                "ccgram.handlers.shell.shell_capture.mark_telegram_command"
            ) as mock_mark,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            await handle_shell_message(bot, 1, 42, "@0", "!ls -la", message)

            mock_send.assert_called_once_with("@0", "ls -la", raw=True)
            mock_mark.assert_called_once()
            args = mock_mark.call_args.args
            assert args[:4] == ("@0", "ls -la", 1, 42)

    async def test_bang_with_space_strips_leading_space(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window"),
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
            patch("ccgram.handlers.shell.shell_capture.mark_telegram_command"),
        ):
            await handle_shell_message(bot, 1, 42, "@0", "! ls", message)

            mock_send.assert_called_once_with("@0", "ls", raw=True)

    async def test_bare_bang_is_ignored(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window") as mock_sm,
        ):
            mock_sm.send_to_window = AsyncMock()
            await handle_shell_message(bot, 1, 42, "@0", "!", message)

            mock_sm.send_to_window.assert_not_called()

    async def test_no_bang_no_llm_sends_raw(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=None),
            patch(f"{_CTX}.view_window"),
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
            patch("ccgram.handlers.shell.shell_capture.mark_telegram_command"),
        ):
            await handle_shell_message(bot, 1, 42, "@0", "find . -name foo", message)

            mock_send.assert_called_once_with("@0", "find . -name foo", raw=True)

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
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock),
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
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
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_CTX}.view_window") as mock_sm,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "do something", message)

            mock_send.assert_called_once()
            assert "LLM request failed" in mock_send.call_args[0][2]
            mock_sm.send_to_window.assert_not_called()

    async def test_llm_config_error_notifies_user(self) -> None:
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", side_effect=ValueError("bad provider")),
            patch(f"{_CTX}.view_window") as mock_sm,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await handle_shell_message(bot, 1, 42, "@0", "do something")

            mock_send.assert_called_once()
            assert "LLM misconfigured" in mock_send.call_args[0][2]
            mock_sm.send_to_window.assert_not_called()

    async def test_send_failure_replies_error(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window") as mock_sm,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(False, "Window not found"))
            mock_tr.resolve_chat_id.return_value = -100

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
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "list files")

            approval_calls = [
                call
                for call in mock_send.call_args_list
                if call.args[2] != _BANG_HINT_TEXT
            ]
            assert len(approval_calls) == 1


class TestHandleShellCallback:
    async def test_run_with_pending_executes_and_clears(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
            patch(
                "ccgram.handlers.shell.shell_capture.mark_telegram_command"
            ) as mock_mark,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_window_for_thread.return_value = "@0"
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            _shell_pending[(-100, 42)] = ("ls -la", 1, 0)

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            query.answer.assert_called_once()
            mock_send.assert_called_once_with("@0", "ls -la", raw=True)
            mock_mark.assert_called_once_with("@0", "ls -la", 1, 42, 0)
            assert _shell_pending.get((-100, 42)) is None

    async def test_run_wrong_user_rejects(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            _shell_pending[(-100, 42)] = ("ls -la", 999, 0)

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            assert "Not your command" in mock_edit.call_args[0][1]

    async def test_confirm_danger_wrong_user_rejects(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            _shell_pending[(-100, 42)] = ("rm -rf /", 999, 0)

            await handle_shell_callback(
                query, 1, f"{CB_SHELL_CONFIRM_DANGER}@0", bot, 42
            )

            assert "Not your command" in mock_edit.call_args[0][1]

    async def test_run_no_window_binding_rejects(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_window_for_thread.return_value = None
            _shell_pending[(-100, 42)] = ("ls -la", 1, 0)

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            assert "No session bound" in mock_edit.call_args[0][1]
            assert _shell_pending.get((-100, 42)) is None

    async def test_run_without_pending_shows_expired(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_tr.resolve_chat_id.return_value = -100

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            mock_edit.assert_called_once()
            assert "expired" in mock_edit.call_args[0][1]

    async def test_cancel_clears_pending(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            _shell_pending[(-100, 42)] = ("rm -rf /", 1, 0)

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
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            _shell_pending[(-100, 42)] = ("grep -r pattern .", 1, 0)

            await handle_shell_callback(query, 1, f"{CB_SHELL_EDIT}@0", bot, 42)

            mock_edit.assert_called_once()
            assert "grep -r pattern ." in mock_edit.call_args[0][1]
            assert _shell_pending.get((-100, 42)) is None

    async def test_edit_without_pending_shows_expired(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_tr.resolve_chat_id.return_value = -100

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
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
            patch("ccgram.handlers.shell.shell_capture.mark_telegram_command"),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_window_for_thread.return_value = "@0"
            _shell_pending[(-100, 42)] = ("rm -rf /tmp/test", 1, 0)

            await handle_shell_callback(
                query, 1, f"{CB_SHELL_CONFIRM_DANGER}@0", bot, 42
            )

            mock_send.assert_called_once_with("@0", "rm -rf /tmp/test", raw=True)
            assert _shell_pending.get((-100, 42)) is None


class TestGatherLlmContext:
    async def test_assembles_cwd_shell_and_tools(self) -> None:
        from ccgram.handlers.shell.shell_commands import gather_llm_context

        with (
            patch(
                "ccgram.providers.shell.detect_pane_shell",
                new_callable=AsyncMock,
                return_value="fish",
            ),
            patch(
                f"{_CTX}._detect_shell_tools",
                return_value="rg (grep replacement)",
            ),
            patch(f"{_CTX}.view_window") as mock_sm,
        ):
            mock_sm.return_value = MagicMock(cwd="/home/user/project")
            ctx = await gather_llm_context("@0")

        assert ctx["cwd"] == "/home/user/project"
        assert ctx["shell"] == "fish"
        assert ctx["shell_tools"] == "rg (grep replacement)"

    async def test_empty_cwd_when_none(self) -> None:
        from ccgram.handlers.shell.shell_commands import gather_llm_context

        with (
            patch(
                "ccgram.providers.shell.detect_pane_shell",
                new_callable=AsyncMock,
                return_value="bash",
            ),
            patch(
                f"{_CTX}._detect_shell_tools",
                return_value="",
            ),
            patch(f"{_CTX}.view_window") as mock_sm,
        ):
            mock_sm.return_value = MagicMock(cwd="")
            ctx = await gather_llm_context("@0")

        assert ctx["cwd"] == ""


class TestCancelStuckInput:
    def _mock_window(self, pane_cmd: str = "fish"):  # noqa: ANN202
        from ccgram.multiplexer.base import WindowRef as TmuxWindow

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
        message.message_id = 7
        result = CommandResult(
            command="ls", explanation="List files", is_dangerous=False
        )

        with patch(f"{_MOD}.safe_reply", new_callable=AsyncMock) as mock_reply:
            await show_command_approval(bot, -100, 42, "@0", result, 1, message)

        mock_reply.assert_called_once()
        assert "`ls`" in mock_reply.call_args[0][1]
        assert _shell_pending[(-100, 42)] == ("ls", 1, 7)

    async def test_message_none_uses_safe_send(self) -> None:
        bot = AsyncMock(spec=Bot)
        result = CommandResult(command="pwd", explanation="", is_dangerous=False)

        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send:
            await show_command_approval(bot, -100, 42, "@0", result, 1, None)

        mock_send.assert_called_once()
        assert "`pwd`" in mock_send.call_args[0][2]
        assert _shell_pending[(-100, 42)] == ("pwd", 1, 0)


class TestLazyMarkerRecovery:
    async def test_raw_command_restores_marker_when_missing(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch("ccgram.handlers.shell.shell_capture.mark_telegram_command"),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

        mock_ensure.assert_awaited_once_with("@0", "lazy")


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

        with patch("ccgram.multiplexer.multiplexer") as mock_tm:
            mock_tm.capture_pane = AsyncMock(return_value=capture_value)
            assert await has_prompt_marker("@0") is expected


class TestHasShellPending:
    def test_returns_false_when_empty(self) -> None:
        assert has_shell_pending(-100, 42) is False

    def test_returns_true_when_entry_exists(self) -> None:
        _shell_pending[(-100, 42)] = ("ls", 1, 0)
        assert has_shell_pending(-100, 42) is True

    def test_returns_false_for_different_key(self) -> None:
        _shell_pending[(-100, 42)] = ("ls", 1, 0)
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
        from ccgram.handlers.shell.shell_context import _detect_shell_tools

        _detect_shell_tools.cache_clear()

    def teardown_method(self) -> None:
        from ccgram.handlers.shell.shell_context import _detect_shell_tools

        _detect_shell_tools.cache_clear()

    def test_returns_detected_tools(self) -> None:
        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("fd", "rg") else None

        with patch("shutil.which", side_effect=fake_which):
            from ccgram.handlers.shell.shell_context import _detect_shell_tools

            result = _detect_shell_tools()

        assert "fd" in result
        assert "rg" in result
        assert "bat" not in result

    def test_cache_populated_and_reused(self) -> None:
        with patch("shutil.which", return_value=None):
            from ccgram.handlers.shell.shell_context import _detect_shell_tools

            first = _detect_shell_tools()
            second = _detect_shell_tools()

        assert first is second


class TestGenerationCounter:
    async def test_stale_generation_dropped(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        call_count = 0

        async def slow_generate(*args, **kwargs):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            return CommandResult(
                command=f"cmd-{call_count}", explanation="", is_dangerous=False
            )

        mock_completer = AsyncMock()
        mock_completer.generate_command = slow_generate

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock),
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock),
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "first command", message)

        assert (-100, 42) in _shell_pending
        assert _shell_pending[(-100, 42)][0] == "cmd-1"

    async def test_generation_counter_increments(self) -> None:
        bot = AsyncMock(spec=Bot)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(command="ls", explanation="", is_dangerous=False)
        )

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock),
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "first")
            assert _generation_counter[(-100, 42)] == 1

            await handle_shell_message(bot, 1, 42, "@0", "second")
            assert _generation_counter[(-100, 42)] == 1


class TestBangHint:
    async def test_hint_sent_once_per_session(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(command="ls", explanation="", is_dangerous=False)
        )

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock),
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "first command", message)
            assert (-100, 42) in _shell_hint_seen
            hint_calls = [
                call
                for call in mock_send.call_args_list
                if call.args[2] == _BANG_HINT_TEXT
            ]
            assert len(hint_calls) == 1

            mock_send.reset_mock()
            _shell_pending.clear()

            await handle_shell_message(bot, 1, 42, "@0", "second command", message)
            hint_calls = [
                call
                for call in mock_send.call_args_list
                if call.args[2] == _BANG_HINT_TEXT
            ]
            assert hint_calls == []

    async def test_hint_not_sent_for_bang_prefix(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch("ccgram.handlers.shell.shell_capture.mark_telegram_command"),
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)

            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

            assert (-100, 42) not in _shell_hint_seen
            hint_calls = [
                call
                for call in mock_send.call_args_list
                if call.args[2:3] == (_BANG_HINT_TEXT,)
            ]
            assert hint_calls == []

    async def test_hint_not_sent_when_no_llm_configured(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=None),
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch("ccgram.handlers.shell.shell_capture.mark_telegram_command"),
        ):
            mock_tr.resolve_chat_id.return_value = -100

            await handle_shell_message(bot, 1, 42, "@0", "anything", message)

            assert (-100, 42) not in _shell_hint_seen

    def test_clear_hint_seen_cleanup(self) -> None:
        _shell_hint_seen.add((-100, 42))
        _clear_shell_hint_seen(-100, 42)
        assert (-100, 42) not in _shell_hint_seen

    def test_clear_hint_seen_idempotent(self) -> None:
        _clear_shell_hint_seen(-999, 999)

    def test_clear_shell_pending_does_not_reset_hint(self) -> None:
        _shell_hint_seen.add((-100, 42))
        clear_shell_pending(-100, 42)
        assert (-100, 42) in _shell_hint_seen


class TestCommandHistoryRecording:
    async def test_llm_path_records_command_history(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(command="ls", explanation="", is_dangerous=False)
        )

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock),
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch("ccgram.handlers.command_history.record_command") as mock_record,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(
                bot, 1, 42, "@0", "list all python files", message
            )

        mock_record.assert_called_once_with(1, 42, "list all python files")


class TestShowCommandApprovalPreventsOverwrite:
    async def test_returns_false_when_slot_occupied(self) -> None:
        bot = AsyncMock(spec=Bot)
        result = CommandResult(command="pwd", explanation="", is_dangerous=False)

        _shell_pending[(-100, 42)] = ("ls", 1, 0)

        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send:
            returned = await show_command_approval(
                bot, -100, 42, "@0", result, user_id=2
            )

        assert returned is False
        mock_send.assert_not_called()
        assert _shell_pending[(-100, 42)] == ("ls", 1, 0)

    async def test_returns_true_when_slot_empty(self) -> None:
        bot = AsyncMock(spec=Bot)
        result = CommandResult(command="pwd", explanation="", is_dangerous=False)

        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock):
            returned = await show_command_approval(
                bot, -100, 42, "@0", result, user_id=1
            )

        assert returned is True
        assert _shell_pending[(-100, 42)] == ("pwd", 1, 0)


class TestTypingAction:
    async def test_immediate_typing_action_on_entry(self) -> None:
        bot = AsyncMock(spec=Bot)
        bot.send_chat_action = AsyncMock()
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch("ccgram.handlers.shell.shell_capture.mark_telegram_command"),
            patch(f"{_MOD}.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)

            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

        assert bot.send_chat_action.await_count >= 1
        call = bot.send_chat_action.await_args_list[0]
        assert call.kwargs["chat_id"] == -100
        assert call.kwargs["message_thread_id"] == 42

    async def test_typing_pulse_refreshes_during_llm(self) -> None:
        from ccgram.handlers.shell import shell_commands as sc

        bot = AsyncMock(spec=Bot)
        bot.send_chat_action = AsyncMock()
        message = AsyncMock(spec=Message)

        async def slow_generate(*args, **kwargs):  # noqa: ARG001
            # 0.25 s with refresh 0.05 s ⇒ at least 4 pulses + initial.
            await asyncio.sleep(0.25)
            return CommandResult(command="ls", explanation="", is_dangerous=False)

        mock_completer = AsyncMock()
        mock_completer.generate_command = slow_generate

        with (
            patch.object(sc, "_TYPING_REFRESH_INTERVAL", 0.05),
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock),
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock),
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "list files", message)

        # immediate fire + ≥3 pulses while LLM was sleeping (timing-tolerant).
        assert bot.send_chat_action.await_count >= 3

    async def test_typing_pulse_cancelled_after_completion(self) -> None:
        from ccgram.handlers.shell import shell_commands as sc

        bot = AsyncMock(spec=Bot)
        bot.send_chat_action = AsyncMock()
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(command="ls", explanation="", is_dangerous=False)
        )

        with (
            patch.object(sc, "_TYPING_REFRESH_INTERVAL", 0.02),
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock),
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock),
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "list files", message)
            count_after_complete = bot.send_chat_action.await_count
            await asyncio.sleep(0.1)

        # No further pulses — task was cancelled.
        assert bot.send_chat_action.await_count == count_after_complete

    async def test_typing_pulse_cancelled_on_llm_error(self) -> None:
        from ccgram.handlers.shell import shell_commands as sc

        bot = AsyncMock(spec=Bot)
        bot.send_chat_action = AsyncMock()
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with (
            patch.object(sc, "_TYPING_REFRESH_INTERVAL", 0.02),
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock),
            patch(
                f"{_MOD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "do something", message)
            count_after_abort = bot.send_chat_action.await_count
            await asyncio.sleep(0.1)

        assert bot.send_chat_action.await_count == count_after_abort

    async def test_typing_action_swallows_telegram_error(self) -> None:
        from telegram.error import TelegramError

        bot = AsyncMock(spec=Bot)
        bot.send_chat_action = AsyncMock(side_effect=TelegramError("bad thread"))
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch("ccgram.handlers.shell.shell_capture.mark_telegram_command"),
            patch(f"{_MOD}.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)

            # Must not raise.
            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)


class TestRunningReaction:
    async def test_handle_shell_message_reacts_running_on_user_message(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)
        message.message_id = 4242
        message.chat = MagicMock()
        message.chat.id = -100

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch(
                "ccgram.handlers.shell.shell_capture.mark_telegram_command"
            ) as mock_mark,
            patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

        mock_react.assert_awaited_once()
        args = mock_react.call_args.args
        assert args[1] == -100
        assert args[2] == 4242
        from ccgram.handlers.reactions import REACT_RUNNING

        assert args[3] == REACT_RUNNING
        # mark_telegram_command receives the user-msg id for completion react
        mock_mark.assert_called_once_with("@0", "ls", 1, 42, 4242)

    async def test_handle_shell_message_no_message_skips_reaction(self) -> None:
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch(
                "ccgram.handlers.shell.shell_capture.mark_telegram_command"
            ) as mock_mark,
            patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            await handle_shell_message(bot, 1, 42, "@0", "!ls", message=None)

        mock_react.assert_not_awaited()
        mock_mark.assert_called_once_with("@0", "ls", 1, 42, 0)

    async def test_run_callback_passes_stored_message_id(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_CTX}.view_window"),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(
                f"{_MOD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch(
                "ccgram.handlers.shell.shell_capture.mark_telegram_command"
            ) as mock_mark,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_window_for_thread.return_value = "@0"
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            _shell_pending[(-100, 42)] = ("uname -a", 1, 9999)

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

        mock_mark.assert_called_once_with("@0", "uname -a", 1, 42, 9999)
