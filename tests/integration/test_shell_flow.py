"""Integration tests for shell provider Telegram → Shell → Telegram flow.

Tests the complete round-trip: command routing, execution, output capture,
and relay back to Telegram. Uses mock bot + mock tmux with real
shell_commands/shell_capture logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot, Message

from ccgram.handlers.shell_capture import reset_shell_monitor_state
from ccgram.handlers.shell_commands import (
    _generation_counter,
    _shell_pending,
    handle_shell_message,
)
from ccgram.llm.base import CommandResult

pytestmark = pytest.mark.integration

_MOD_CMD = "ccgram.handlers.shell_commands"
_MOD_CAP = "ccgram.handlers.shell_capture"

TEST_USER_ID = 1
TEST_THREAD_ID = 42
TEST_CHAT_ID = -100
TEST_WINDOW_ID = "@0"


@pytest.fixture(autouse=True)
def _clean_state():
    _shell_pending.clear()
    _generation_counter.clear()
    reset_shell_monitor_state()
    yield
    _shell_pending.clear()
    _generation_counter.clear()
    reset_shell_monitor_state()


class TestRawCommandFlow:
    @pytest.mark.asyncio()
    async def test_bang_prefix_sends_to_tmux_and_marks_command(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD_CMD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD_CMD}.clear_probe_failures"),
            patch(f"{_MOD_CMD}.session_manager") as mock_sm,
            patch(f"{_MOD_CMD}.tmux_manager") as mock_tm,
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "ccgram.handlers.shell_capture.mark_telegram_command",
            ) as mock_mark,
        ):
            mock_sm.resolve_chat_id.return_value = TEST_CHAT_ID
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)

            await handle_shell_message(
                bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, "!ls -la", message
            )

            mock_sm.send_to_window.assert_called_once_with(
                TEST_WINDOW_ID, "ls -la", raw=True
            )
            mock_mark.assert_called_once_with(
                TEST_WINDOW_ID, "ls -la", TEST_USER_ID, TEST_THREAD_ID
            )

    @pytest.mark.asyncio()
    async def test_raw_command_output_relayed_via_passive_monitor(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 99

        pane = "ccgram:0❯ ls -la\nfile1.txt\nfile2.txt\nccgram:0❯"

        with (
            patch(
                f"{_MOD_CAP}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD_CAP}.session_manager") as mock_sm,
            patch(
                f"{_MOD_CAP}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = TEST_CHAT_ID

            await check_passive_shell_output(
                bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, pane
            )

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][2]
        assert "❯ ls -la" in sent_text
        assert "file1.txt" in sent_text
        assert sent_text.startswith("```\n")

        state = _shell_monitor_state[TEST_WINDOW_ID]
        assert state.msg_id == 99
        assert state.last_command_echo == "ccgram:0❯ ls -la"

    @pytest.mark.asyncio()
    async def test_raw_command_error_shows_exit_indicator(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 77

        pane = "ccgram:0❯ bad-cmd\nbad-cmd: not found\nccgram:127❯"

        with (
            patch(
                f"{_MOD_CAP}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(
                f"{_MOD_CAP}.edit_with_fallback", new_callable=AsyncMock
            ) as mock_edit,
            patch(f"{_MOD_CAP}.session_manager") as mock_sm,
            patch(
                f"{_MOD_CAP}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = TEST_CHAT_ID

            await check_passive_shell_output(
                bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, pane
            )

        assert mock_edit.called
        edit_text = mock_edit.call_args[0][3]
        assert "exit 127" in edit_text
        assert _shell_monitor_state[TEST_WINDOW_ID].exit_code_sent is True


class TestLlmCommandFlow:
    @pytest.mark.asyncio()
    async def test_nl_generates_command_and_shows_approval(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="ls -la", explanation="List files", is_dangerous=False
            )
        )

        with (
            patch(f"{_MOD_CMD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD_CMD}.clear_probe_failures"),
            patch(f"{_MOD_CMD}.get_completer", return_value=mock_completer),
            patch(f"{_MOD_CMD}.session_manager") as mock_sm,
            patch(f"{_MOD_CMD}.tmux_manager") as mock_tm,
            patch(f"{_MOD_CMD}.safe_reply", new_callable=AsyncMock) as mock_reply,
            patch(
                f"{_MOD_CMD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_sm.resolve_chat_id.return_value = TEST_CHAT_ID
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(
                bot,
                TEST_USER_ID,
                TEST_THREAD_ID,
                TEST_WINDOW_ID,
                "list all files",
                message,
            )

        mock_completer.generate_command.assert_called_once()
        mock_reply.assert_called_once()
        reply_text = mock_reply.call_args[0][1]
        assert "`ls -la`" in reply_text
        assert "List files" in reply_text

        assert (TEST_CHAT_ID, TEST_THREAD_ID) in _shell_pending
        assert _shell_pending[(TEST_CHAT_ID, TEST_THREAD_ID)] == (
            "ls -la",
            TEST_USER_ID,
        )

    @pytest.mark.asyncio()
    async def test_no_llm_falls_back_to_raw(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD_CMD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD_CMD}.clear_probe_failures"),
            patch(f"{_MOD_CMD}.get_completer", return_value=None),
            patch(f"{_MOD_CMD}.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.shell_capture.mark_telegram_command",
            ) as mock_mark,
        ):
            mock_sm.resolve_chat_id.return_value = TEST_CHAT_ID
            mock_sm.send_to_window = AsyncMock(return_value=(True, ""))

            await handle_shell_message(
                bot,
                TEST_USER_ID,
                TEST_THREAD_ID,
                TEST_WINDOW_ID,
                "find . -name foo",
                message,
            )

            mock_sm.send_to_window.assert_called_once_with(
                TEST_WINDOW_ID, "find . -name foo", raw=True
            )
            mock_mark.assert_called_once()


class TestErrorRecovery:
    @pytest.mark.asyncio()
    async def test_telegram_command_error_triggers_fix_suggestion(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
            mark_telegram_command,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 88

        mark_telegram_command(TEST_WINDOW_ID, "lss", TEST_USER_ID, TEST_THREAD_ID)

        pane = "ccgram:0❯ lss\nlss: command not found\nccgram:127❯"

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="ls", explanation="Fixed typo", is_dangerous=False
            )
        )

        with (
            patch(
                f"{_MOD_CAP}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD_CAP}.edit_with_fallback", new_callable=AsyncMock),
            patch(f"{_MOD_CAP}.session_manager") as mock_sm,
            patch(
                f"{_MOD_CAP}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
            patch("ccgram.llm.get_completer", return_value=mock_completer),
            patch(
                "ccgram.handlers.shell_commands.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell_commands.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_sm.resolve_chat_id.return_value = TEST_CHAT_ID

            await check_passive_shell_output(
                bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, pane
            )

        mock_completer.generate_command.assert_called_once()
        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][2]
        assert "`ls`" in sent_text

        state = _shell_monitor_state[TEST_WINDOW_ID]
        assert state.telegram_command == ""

    @pytest.mark.asyncio()
    async def test_fix_suggestion_skipped_when_no_llm(self) -> None:
        from ccgram.handlers.shell_capture import (
            check_passive_shell_output,
            mark_telegram_command,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 89

        mark_telegram_command(TEST_WINDOW_ID, "bad", TEST_USER_ID, TEST_THREAD_ID)

        pane = "ccgram:0❯ bad\nbad: not found\nccgram:1❯"

        with (
            patch(
                f"{_MOD_CAP}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD_CAP}.edit_with_fallback", new_callable=AsyncMock),
            patch(f"{_MOD_CAP}.session_manager") as mock_sm,
            patch(
                f"{_MOD_CAP}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
            patch("ccgram.llm.get_completer", return_value=None),
            patch(
                "ccgram.handlers.shell_commands.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_sm.resolve_chat_id.return_value = TEST_CHAT_ID

            await check_passive_shell_output(
                bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, pane
            )

        mock_send.assert_not_called()


class TestPassiveMonitoringRoundTrip:
    @pytest.mark.asyncio()
    async def test_in_progress_then_completed_edits_message(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 100

        pane_in_progress = "ccgram:0❯ slow-cmd\npartial output"
        pane_completed = "ccgram:0❯ slow-cmd\npartial output\nfinal line\nccgram:0❯"

        with (
            patch(
                f"{_MOD_CAP}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD_CAP}.session_manager") as mock_sm,
            patch(
                f"{_MOD_CAP}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane_in_progress,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = TEST_CHAT_ID

            await check_passive_shell_output(
                bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, pane_in_progress
            )

        mock_send.assert_called_once()
        state = _shell_monitor_state[TEST_WINDOW_ID]
        assert state.msg_id == 100
        assert state.exit_code_sent is False

        with (
            patch(
                f"{_MOD_CAP}.rate_limit_send_message",
                new_callable=AsyncMock,
            ),
            patch(
                f"{_MOD_CAP}.edit_with_fallback", new_callable=AsyncMock
            ) as mock_edit,
            patch(f"{_MOD_CAP}.session_manager") as mock_sm2,
            patch(
                f"{_MOD_CAP}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane_completed,
            ),
        ):
            mock_sm2.resolve_chat_id.return_value = TEST_CHAT_ID

            await check_passive_shell_output(
                bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, pane_completed
            )

        assert mock_edit.called
        state = _shell_monitor_state[TEST_WINDOW_ID]
        assert state.msg_id == 100
