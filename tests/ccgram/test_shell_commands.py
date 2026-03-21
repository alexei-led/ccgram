"""Tests for shell command generation and approval flow."""

from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Bot, CallbackQuery, InlineKeyboardMarkup, Message

from ccgram.handlers.callback_data import (
    CB_SHELL_CANCEL,
    CB_SHELL_CONFIRM_DANGER,
    CB_SHELL_EDIT,
    CB_SHELL_RUN,
)
from ccgram.handlers.shell_commands import (
    _build_approval_keyboard,
    _clear_pending,
    _get_pending,
    _set_pending,
    _shell_pending,
    handle_shell_callback,
    handle_shell_message,
)
from ccgram.llm.base import CommandResult

_MOD = "ccgram.handlers.shell_commands"


class TestPendingState:
    def setup_method(self) -> None:
        _shell_pending.clear()

    def test_set_and_get_roundtrip(self) -> None:
        _set_pending(100, 42, "ls -la", "list files")
        result = _get_pending(100, 42)
        assert result == ("ls -la", "list files")

    def test_get_nonexistent_returns_none(self) -> None:
        assert _get_pending(999, 999) is None

    def test_clear_removes_entry(self) -> None:
        _set_pending(100, 42, "ls", "list")
        _clear_pending(100, 42)
        assert _get_pending(100, 42) is None

    def test_clear_nonexistent_no_error(self) -> None:
        _clear_pending(999, 999)

    def test_overwrite_existing(self) -> None:
        _set_pending(100, 42, "ls", "list")
        _set_pending(100, 42, "pwd", "print dir")
        result = _get_pending(100, 42)
        assert result == ("pwd", "print dir")


class TestBuildApprovalKeyboard:
    def test_non_dangerous_has_run_edit_cancel(self) -> None:
        kb = _build_approval_keyboard("@0", is_dangerous=False)
        assert isinstance(kb, InlineKeyboardMarkup)
        buttons = [btn for row in kb.inline_keyboard for btn in row]
        texts = [btn.text for btn in buttons]
        assert any("Run" in t for t in texts)
        assert any("Edit" in t for t in texts)
        assert any("Cancel" in t for t in texts)

    def test_non_dangerous_callback_data_includes_window_id(self) -> None:
        kb = _build_approval_keyboard("@5", is_dangerous=False)
        buttons = [btn for row in kb.inline_keyboard for btn in row]
        run_btn = next(b for b in buttons if "Run" in b.text)
        assert run_btn.callback_data == f"{CB_SHELL_RUN}@5"

    def test_dangerous_has_confirm_and_cancel_only(self) -> None:
        kb = _build_approval_keyboard("@0", is_dangerous=True)
        buttons = [btn for row in kb.inline_keyboard for btn in row]
        texts = [btn.text for btn in buttons]
        assert any("Confirm" in t for t in texts)
        assert any("Cancel" in t for t in texts)
        assert not any("Edit" in t for t in texts)

    def test_dangerous_callback_data_includes_window_id(self) -> None:
        kb = _build_approval_keyboard("@3", is_dangerous=True)
        buttons = [btn for row in kb.inline_keyboard for btn in row]
        confirm_btn = next(b for b in buttons if "Confirm" in b.text)
        assert confirm_btn.callback_data == f"{CB_SHELL_CONFIRM_DANGER}@3"


class TestHandleShellMessage:
    def setup_method(self) -> None:
        _shell_pending.clear()

    async def test_bang_prefix_sends_raw_command(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.start_shell_capture") as mock_capture,
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            await handle_shell_message(bot, 1, 42, "@0", "!ls -la", message)

            mock_sm.send_to_window.assert_called_once_with("@0", "ls -la")
            mock_capture.assert_called_once_with(bot, 1, 42, "@0")

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

            mock_sm.send_to_window.assert_called_once_with("@0", "ls")

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

            mock_sm.send_to_window.assert_called_once_with("@0", "find . -name foo")

    async def test_no_bang_with_llm_calls_completer(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="find . -name foo", explanation="Search", is_dangerous=False
            )
        )

        mock_ws = MagicMock()
        mock_ws.cwd = "/tmp"

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock),
        ):
            mock_sm.get_window_state.return_value = mock_ws
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(
                bot, 1, 42, "@0", "find files named foo", message
            )

            mock_completer.generate_command.assert_called_once()
            call_kwargs = mock_completer.generate_command.call_args
            assert call_kwargs[0][0] == "find files named foo"

    async def test_llm_error_falls_back_to_raw(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        mock_ws = MagicMock()
        mock_ws.cwd = "/tmp"

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.start_shell_capture"),
        ):
            mock_sm.get_window_state.return_value = mock_ws
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(bot, 1, 42, "@0", "do something", message)

            mock_sm.send_to_window.assert_called_once_with("@0", "do something")

    async def test_send_failure_replies_error(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD}.clear_probe_failures"),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send,
        ):
            mock_sm.send_to_window = AsyncMock(return_value=(False, "Window not found"))
            mock_sm.resolve_chat_id.return_value = -100

            await handle_shell_message(bot, 1, 42, "@0", "!ls", message)

            mock_send.assert_called_once()
            assert "Window not found" in mock_send.call_args[0][2]


class TestHandleShellCallback:
    def setup_method(self) -> None:
        _shell_pending.clear()

    async def test_run_with_pending_executes_and_clears(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock),
            patch(f"{_MOD}.start_shell_capture"),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            _set_pending(-100, 42, "ls -la", "list files")

            await handle_shell_callback(query, 1, f"{CB_SHELL_RUN}@0", bot, 42)

            query.answer.assert_called_once()
            mock_sm.send_to_window.assert_called_once_with("@0", "ls -la")
            assert _get_pending(-100, 42) is None

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
            _set_pending(-100, 42, "rm -rf /", "delete all")

            await handle_shell_callback(query, 1, f"{CB_SHELL_CANCEL}@0", bot, 42)

            query.answer.assert_called_once_with("Cancelled")
            assert _get_pending(-100, 42) is None
            mock_edit.assert_called_once()
            assert "Cancelled" in mock_edit.call_args[0][1]

    async def test_edit_with_pending_shows_copyable_command(self) -> None:
        query = AsyncMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.safe_edit", new_callable=AsyncMock) as mock_edit,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            _set_pending(-100, 42, "grep -r pattern .", "search")

            await handle_shell_callback(query, 1, f"{CB_SHELL_EDIT}@0", bot, 42)

            mock_edit.assert_called_once()
            assert "grep -r pattern ." in mock_edit.call_args[0][1]

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
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            _set_pending(-100, 42, "rm -rf /tmp/test", "delete")

            await handle_shell_callback(
                query, 1, f"{CB_SHELL_CONFIRM_DANGER}@0", bot, 42
            )

            mock_sm.send_to_window.assert_called_once_with("@0", "rm -rf /tmp/test")
            assert _get_pending(-100, 42) is None
