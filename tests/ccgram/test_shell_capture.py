"""Tests for shell output capture and relay."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot

from ccgram.handlers.shell_capture import (
    _CaptureState,
    _OUTPUT_LIMIT,
    _extract_command_output,
    _extract_new_output,
    _maybe_suggest_fix,
    _poll_once,
    _shell_capture_tasks,
    cancel_shell_capture,
    start_shell_capture,
    strip_terminal_glyphs,
)

_MOD = "ccgram.handlers.shell_capture"


class TestStripTerminalGlyphs:
    def test_strips_nerd_font_glyphs(self) -> None:
        assert strip_terminal_glyphs("\ue0b0 hello") == " hello"

    def test_strips_pua_supplement(self) -> None:
        assert strip_terminal_glyphs("\U000f0001 icon") == " icon"

    def test_preserves_normal_text(self) -> None:
        assert strip_terminal_glyphs("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert strip_terminal_glyphs("") == ""


class TestExtractCommandOutput:
    @pytest.mark.parametrize(
        ("pane", "expected_text", "expected_code"),
        [
            (
                "ccgram:0❯ ls\nfile1.txt\nfile2.txt\nccgram:0❯",
                "file1.txt\nfile2.txt",
                0,
            ),
            (
                "ccgram:0❯ bad-cmd\nerror: not found\nccgram:127❯",
                "error: not found",
                127,
            ),
            ("ccgram:0❯ true\nccgram:0❯", "", 0),
            ("ccgram:0❯", "", 0),
        ],
        ids=["success", "failure-127", "no-output", "bare-prompt"],
    )
    def test_marker_extraction(
        self, pane: str, expected_text: str, expected_code: int
    ) -> None:
        result = _extract_command_output([], pane)
        assert result.text == expected_text
        assert result.exit_code == expected_code

    def test_no_markers_falls_back_to_baseline(self) -> None:
        current = "$ ls\nfile1.txt"
        result = _extract_command_output([], current)
        assert result.text == "$ ls\nfile1.txt"
        assert result.exit_code is None

    def test_no_markers_with_baseline(self) -> None:
        baseline = ["old content", "$ "]
        current = "old content\n$ ls\nfile1.txt"
        result = _extract_command_output(baseline, current)
        assert "file1.txt" in result.text
        assert result.exit_code is None

    def test_empty_current(self) -> None:
        result = _extract_command_output([], "")
        assert result.text == ""
        assert result.exit_code is None

    def test_multiline_output_with_markers(self) -> None:
        current = (
            "ccgram:0❯ find . -name '*.py'\n"
            "./src/main.py\n"
            "./src/utils.py\n"
            "./tests/test_main.py\n"
            "ccgram:0❯"
        )
        result = _extract_command_output([], current)
        assert result.exit_code == 0
        assert "./src/main.py" in result.text
        assert "./tests/test_main.py" in result.text

    def test_command_still_running_no_end_marker(self) -> None:
        current = "ccgram:0❯ long-cmd\npartial output line 1\npartial output line 2"
        result = _extract_command_output([], current)
        assert result.exit_code is None


class TestExtractNewOutput:
    def test_empty_current(self) -> None:
        assert _extract_new_output(["$ "], "") == ""

    def test_empty_baseline(self) -> None:
        assert _extract_new_output([], "output\nlines") == "output\nlines"

    def test_single_line_baseline(self) -> None:
        result = _extract_new_output(["$ "], "$ ls\nfile.txt")
        assert "file.txt" in result

    def test_multi_line_overlap(self) -> None:
        baseline = ["line1", "line2", "$ "]
        current = "line1\nline2\n$ ls\nfile.txt"
        result = _extract_new_output(baseline, current)
        assert "file.txt" in result

    def test_no_overlap(self) -> None:
        baseline = ["completely", "different", "$ "]
        current = "new content\nhere"
        result = _extract_new_output(baseline, current)
        assert "new content" in result

    def test_baseline_longer_than_current(self) -> None:
        baseline = ["a", "b", "c", "d", "$ "]
        current = "x"
        result = _extract_new_output(baseline, current)
        assert result == "x"


class TestPollOnce:
    async def test_window_not_found_stops(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=None)

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is True

    async def test_capture_returns_none_continues(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()
        mock_window = MagicMock()

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value=None)

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is False

    async def test_new_output_sends_message(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()
        mock_window = MagicMock()
        mock_sent = MagicMock()
        mock_sent.message_id = 123

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="$ ls\nfile1.txt\nfile2.txt")

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert state.msg_id == 123
        assert "file1.txt" in state.last_output
        assert should_stop is False
        mock_send.assert_called_once()

    async def test_changed_output_edits_message(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(msg_id=123, last_output="old output")
        mock_window = MagicMock()

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="new output here")

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert state.msg_id == 123
        assert state.last_output == "new output here"
        assert should_stop is False
        mock_edit.assert_called_once_with(bot, -100, 123, "new output here")

    async def test_long_output_truncated(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()
        mock_window = MagicMock()
        long_text = "x" * 5000

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=MagicMock(message_id=1),
            ) as mock_send,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value=long_text)

            await _poll_once(bot, -100, 42, "@0", state)

        sent_text = mock_send.call_args[0][2]
        assert len(sent_text) <= _OUTPUT_LIMIT + 10
        assert sent_text.startswith("\u2026 ")

    async def test_unchanged_output_increments_stable(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(last_output="$ ls\nfile1.txt")
        mock_window = MagicMock()

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="$ ls\nfile1.txt")

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert state.stable_count == 1
        assert should_stop is False

    async def test_unchanged_output_reaches_threshold_stops(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            last_output="$ ls\nfile1.txt",
            stable_count=1,
        )
        mock_window = MagicMock()

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="$ ls\nfile1.txt")

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is True

    async def test_prompt_marker_stops_immediately(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()
        mock_window = MagicMock()
        mock_sent = MagicMock()
        mock_sent.message_id = 1

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(
                return_value="ccgram:0❯ ls\nfile1.txt\nccgram:0❯"
            )

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is True
        assert state.exit_code == 0
        assert state.last_output == "file1.txt"

    async def test_prompt_marker_with_error_exit_code(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()
        mock_window = MagicMock()
        mock_sent = MagicMock()
        mock_sent.message_id = 1

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(
                return_value="ccgram:0❯ bad\nerror msg\nccgram:127❯"
            )

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is True
        assert state.exit_code == 127
        assert state.last_output == "error msg"

    async def test_empty_output_continues(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()
        mock_window = MagicMock()

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="  \n  \n  ")

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is False

    async def test_new_output_resets_stable_count(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            msg_id=1,
            last_output="old",
            stable_count=1,
        )
        mock_window = MagicMock()

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="new output")

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert state.stable_count == 0
        assert state.last_output == "new output"
        assert should_stop is False


class TestMaybeSuggestFix:
    async def test_noop_for_exit_code_zero(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(exit_code=0, msg_id=1, last_output="ok", command="ls")
        with patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit:
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)
        mock_edit.assert_not_called()

    async def test_noop_for_exit_code_none(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(exit_code=None, msg_id=1, last_output="ok", command="ls")
        with patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit:
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)
        mock_edit.assert_not_called()

    async def test_edits_error_prefix_and_calls_llm(self) -> None:
        from ccgram.llm.base import CommandResult

        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            exit_code=1, msg_id=99, last_output="err", command="bad-cmd"
        )

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="good-cmd", explanation="fix", is_dangerous=False
            )
        )

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch("ccgram.llm.get_completer", return_value=mock_completer),
            patch(
                "ccgram.handlers.shell_commands._gather_llm_context",
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell_commands._show_command_approval",
                new_callable=AsyncMock,
            ) as mock_approval,
        ):
            mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

        mock_edit.assert_called_once()
        assert "\u274c exit 1" in mock_edit.call_args[0][3]
        mock_approval.assert_awaited_once()

    async def test_skips_when_same_command_suggested(self) -> None:
        from ccgram.llm.base import CommandResult

        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            exit_code=1, msg_id=99, last_output="err", command="bad-cmd"
        )

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="bad-cmd", explanation="same", is_dangerous=False
            )
        )

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch("ccgram.llm.get_completer", return_value=mock_completer),
            patch(
                "ccgram.handlers.shell_commands._gather_llm_context",
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell_commands._show_command_approval",
                new_callable=AsyncMock,
            ) as mock_approval,
        ):
            mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

        mock_approval.assert_not_awaited()

    async def test_skips_when_pending_already_set(self) -> None:
        from ccgram.handlers.shell_commands import _shell_pending
        from ccgram.llm.base import CommandResult

        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            exit_code=1, msg_id=99, last_output="err", command="bad-cmd"
        )

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="good-cmd", explanation="fix", is_dangerous=False
            )
        )

        _shell_pending[(-100, 42)] = ("new-user-cmd", 1)
        try:
            with (
                patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
                patch(f"{_MOD}.session_manager") as mock_sm,
                patch("ccgram.llm.get_completer", return_value=mock_completer),
                patch(
                    "ccgram.handlers.shell_commands._gather_llm_context",
                    return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
                ),
                patch(
                    "ccgram.handlers.shell_commands._show_command_approval",
                    new_callable=AsyncMock,
                ) as mock_approval,
            ):
                mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
                await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

            mock_approval.assert_not_awaited()
            assert _shell_pending[(-100, 42)] == ("new-user-cmd", 1)
        finally:
            _shell_pending.pop((-100, 42), None)


class TestStartShellCapture:
    def setup_method(self) -> None:
        _shell_capture_tasks.clear()

    def teardown_method(self) -> None:
        for task in _shell_capture_tasks.values():
            if not task.done():
                task.cancel()
        _shell_capture_tasks.clear()

    async def test_start_creates_task(self) -> None:
        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}._capture_shell_output", new_callable=AsyncMock),
            patch(f"{_MOD}.task_done_callback"),
        ):
            start_shell_capture(bot, 1, 42, "@0")

        assert (1, 42) in _shell_capture_tasks
        task = _shell_capture_tasks[(1, 42)]
        assert isinstance(task, asyncio.Task)
        task.cancel()

    async def test_start_cancels_existing(self) -> None:
        bot = AsyncMock(spec=Bot)
        old_task = asyncio.create_task(asyncio.sleep(100))
        _shell_capture_tasks[(1, 42)] = old_task

        with (
            patch(f"{_MOD}._capture_shell_output", new_callable=AsyncMock),
            patch(f"{_MOD}.task_done_callback"),
        ):
            start_shell_capture(bot, 1, 42, "@0")

        await asyncio.sleep(0)
        assert old_task.cancelled()
        new_task = _shell_capture_tasks[(1, 42)]
        assert new_task is not old_task
        new_task.cancel()


class TestCancelShellCapture:
    def setup_method(self) -> None:
        _shell_capture_tasks.clear()

    async def test_cancel_stops_task(self) -> None:
        task = asyncio.create_task(asyncio.sleep(100))
        _shell_capture_tasks[(1, 42)] = task

        cancel_shell_capture(1, 42)

        await asyncio.sleep(0)
        assert task.cancelled()
        assert (1, 42) not in _shell_capture_tasks

    async def test_cancel_nonexistent_no_error(self) -> None:
        cancel_shell_capture(999, 999)
