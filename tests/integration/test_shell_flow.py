"""Integration tests for shell provider Telegram → Shell → Telegram flow.

Tests the complete round-trip: command routing, execution, output capture,
and relay back to Telegram. Uses mock bot + mock tmux with real
shell_commands/shell_capture logic.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram import Bot, Message

from ccgram.handlers.shell.shell_capture import (
    _shell_monitor_state,
    check_passive_shell_output,
    mark_telegram_command,
    reset_shell_monitor_state,
)
from ccgram.handlers.shell.shell_commands import (
    _generation_counter,
    _shell_pending,
    handle_shell_message,
    show_command_approval,
)
from ccgram.llm.base import CommandResult
from ccgram.multiplexer.base import CaptureResult

pytestmark = pytest.mark.integration

_MOD_CMD = "ccgram.handlers.shell.shell_commands"
_MOD_CAP = "ccgram.handlers.shell.shell_capture"

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


@contextmanager
def _passive_capture(pane: str, *, sent_message_id: int):
    """Patch what ``check_passive_shell_output`` talks to, yield its send/edit mocks.

    The pane text is what the multiplexer would return for the window; the send
    mock stands in for the first relay message and the edit mock for every
    in-place update of it.
    """
    sent = MagicMock()
    sent.message_id = sent_message_id
    with ExitStack() as stack:
        enter = stack.enter_context
        mock_send = enter(
            patch(
                f"{_MOD_CAP}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=sent,
            )
        )
        mock_edit = enter(
            patch(f"{_MOD_CAP}.edit_with_fallback", new_callable=AsyncMock)
        )
        enter(
            patch(
                f"{_MOD_CAP}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            )
        )
        mock_router = enter(patch(f"{_MOD_CAP}.thread_router"))
        mock_router.resolve_chat_id.return_value = TEST_CHAT_ID
        yield mock_send, mock_edit


async def _run_passive(bot, pane: str) -> None:
    await check_passive_shell_output(
        bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, pane
    )


def _fixed_completer(command: str, explanation: str) -> AsyncMock:
    completer = AsyncMock()
    completer.generate_command = AsyncMock(
        return_value=CommandResult(
            command=command, explanation=explanation, is_dangerous=False
        )
    )
    return completer


class TestRawCommandFlow:
    async def test_bang_prefix_sends_to_tmux_and_marks_command(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD_CMD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD_CMD}.lifecycle_strategy.clear_probe_failures"),
            patch("ccgram.handlers.shell.shell_context.view_window"),
            patch(f"{_MOD_CMD}.thread_router") as mock_tr,
            patch(f"{_MOD_CMD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD_CMD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(f"{_MOD_CAP}.mark_telegram_command") as mock_mark,
        ):
            mock_tr.resolve_chat_id.return_value = TEST_CHAT_ID
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value=None)

            await handle_shell_message(
                bot, TEST_USER_ID, TEST_THREAD_ID, TEST_WINDOW_ID, "!ls -la", message
            )

        mock_send.assert_called_once_with(
            TEST_USER_ID, TEST_WINDOW_ID, TEST_THREAD_ID, "ls -la", ANY, raw=True
        )
        assert mock_mark.call_args.args[:4] == (
            TEST_WINDOW_ID,
            "ls -la",
            TEST_USER_ID,
            TEST_THREAD_ID,
        )

    async def test_raw_command_output_relayed_via_passive_monitor(self) -> None:
        bot = AsyncMock(spec=Bot)
        pane = "ccgram:0❯ ls -la\nfile1.txt\nfile2.txt\nccgram:0❯"

        with _passive_capture(pane, sent_message_id=99) as (mock_send, _edit):
            await _run_passive(bot, pane)

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][2]
        assert "❯ ls -la" in sent_text
        assert "file1.txt" in sent_text
        assert sent_text.startswith("```\n")

        state = _shell_monitor_state[TEST_WINDOW_ID]
        assert state.msg_id == 99
        assert state.last_command_echo == "ccgram:0❯ ls -la"

    async def test_raw_command_error_shows_exit_indicator(self) -> None:
        bot = AsyncMock(spec=Bot)
        pane = "ccgram:0❯ bad-cmd\nbad-cmd: not found\nccgram:127❯"

        with _passive_capture(pane, sent_message_id=77) as (_send, mock_edit):
            await _run_passive(bot, pane)

        assert mock_edit.called
        assert "exit 127" in mock_edit.call_args[0][3]
        assert _shell_monitor_state[TEST_WINDOW_ID].exit_code_sent is True


class TestLlmCommandFlow:
    async def test_nl_generates_command_and_shows_approval(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)
        completer = _fixed_completer("ls -la", "List files")

        with (
            patch(f"{_MOD_CMD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD_CMD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD_CMD}.get_completer", return_value=completer),
            patch(f"{_MOD_CMD}.thread_router") as mock_tr,
            patch(f"{_MOD_CMD}.tmux_manager") as mock_tm,
            patch(f"{_MOD_CMD}.safe_reply", new_callable=AsyncMock) as mock_reply,
            patch(
                f"{_MOD_CMD}.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
        ):
            mock_tr.resolve_chat_id.return_value = TEST_CHAT_ID
            mock_tm.capture_pane = AsyncMock(return_value="$ ")

            await handle_shell_message(
                bot,
                TEST_USER_ID,
                TEST_THREAD_ID,
                TEST_WINDOW_ID,
                "list all files",
                message,
            )

        completer.generate_command.assert_called_once()
        mock_reply.assert_called_once()
        reply_text = mock_reply.call_args[0][1]
        assert "`ls -la`" in reply_text
        assert "List files" in reply_text

        pending = _shell_pending[(TEST_CHAT_ID, TEST_THREAD_ID)]
        assert pending[0] == "ls -la"
        assert pending[1] == TEST_USER_ID

    async def test_no_llm_falls_back_to_raw(self) -> None:
        bot = AsyncMock(spec=Bot)
        message = AsyncMock(spec=Message)

        with (
            patch(f"{_MOD_CMD}.enqueue_status_update", new_callable=AsyncMock),
            patch(f"{_MOD_CMD}.lifecycle_strategy.clear_probe_failures"),
            patch(f"{_MOD_CMD}.get_completer", return_value=None),
            patch("ccgram.handlers.shell.shell_context.view_window"),
            patch(f"{_MOD_CMD}.thread_router") as mock_tr,
            patch(
                f"{_MOD_CMD}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
            patch(f"{_MOD_CAP}.mark_telegram_command") as mock_mark,
        ):
            mock_tr.resolve_chat_id.return_value = TEST_CHAT_ID

            await handle_shell_message(
                bot,
                TEST_USER_ID,
                TEST_THREAD_ID,
                TEST_WINDOW_ID,
                "find . -name foo",
                message,
            )

        mock_send.assert_called_once_with(
            TEST_USER_ID,
            TEST_WINDOW_ID,
            TEST_THREAD_ID,
            "find . -name foo",
            TEST_CHAT_ID,
            raw=True,
        )
        mock_mark.assert_called_once()


class TestErrorRecovery:
    async def test_telegram_command_error_triggers_fix_suggestion(self) -> None:
        bot = AsyncMock(spec=Bot)
        mark_telegram_command(TEST_WINDOW_ID, "lss", TEST_USER_ID, TEST_THREAD_ID)
        pane = "ccgram:0❯ lss\nlss: command not found\nccgram:127❯"
        completer = _fixed_completer("ls", "Fixed typo")

        with (
            _passive_capture(pane, sent_message_id=88),
            patch("ccgram.llm.get_completer", return_value=completer),
            patch(
                "ccgram.handlers.shell.shell_context.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(f"{_MOD_CMD}.safe_send", new_callable=AsyncMock) as mock_send,
            patch(f"{_MOD_CAP}._approval_callback", new=show_command_approval),
        ):
            await _run_passive(bot, pane)

        completer.generate_command.assert_called_once()
        mock_send.assert_called_once()
        assert "`ls`" in mock_send.call_args[0][2]
        assert _shell_monitor_state[TEST_WINDOW_ID].telegram_command == ""

    async def test_fix_suggestion_skipped_when_no_llm(self) -> None:
        bot = AsyncMock(spec=Bot)
        mark_telegram_command(TEST_WINDOW_ID, "bad", TEST_USER_ID, TEST_THREAD_ID)
        pane = "ccgram:0❯ bad\nbad: not found\nccgram:1❯"

        with (
            _passive_capture(pane, sent_message_id=89),
            patch("ccgram.llm.get_completer", return_value=None),
            patch(f"{_MOD_CMD}.safe_send", new_callable=AsyncMock) as mock_send,
        ):
            await _run_passive(bot, pane)

        mock_send.assert_not_called()


class TestPassiveMonitoringRoundTrip:
    async def test_in_progress_then_completed_edits_same_message(self) -> None:
        bot = AsyncMock(spec=Bot)
        pane_in_progress = "ccgram:0❯ slow-cmd\npartial output"
        pane_completed = "ccgram:0❯ slow-cmd\npartial output\nfinal line\nccgram:0❯"

        with _passive_capture(pane_in_progress, sent_message_id=100) as (mock_send, _):
            await _run_passive(bot, pane_in_progress)

        mock_send.assert_called_once()
        state = _shell_monitor_state[TEST_WINDOW_ID]
        assert state.msg_id == 100
        assert state.exit_code_sent is False

        with _passive_capture(pane_completed, sent_message_id=100) as (_, mock_edit):
            await _run_passive(bot, pane_completed)

        assert mock_edit.called
        assert _shell_monitor_state[TEST_WINDOW_ID].msg_id == 100
