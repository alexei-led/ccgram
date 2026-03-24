"""Tests for shell output capture and relay."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot

from ccgram.handlers.shell_capture import (
    _CaptureState,
    _extract_command_output,
    _extract_new_output,
    _looks_like_info_bar,
    _maybe_suggest_fix,
    _poll_once,
    _shell_capture_tasks,
    _strip_trailing_prompt,
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

            with patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=None,
            ):
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

            with patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value="$ ls\nfile1.txt\nfile2.txt",
            ):
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
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value="new output here",
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert state.msg_id == 123
        assert state.last_output == "new output here"
        assert should_stop is False
        mock_edit.assert_called_once_with(bot, -100, 123, "```\nnew output here\n```")

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
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=long_text,
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)

            await _poll_once(bot, -100, 42, "@0", state)

        sent_text = mock_send.call_args[0][2]
        assert sent_text.startswith("```\n\u2026 ")
        assert sent_text.endswith("\n```")

    async def test_unchanged_output_increments_stable(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(last_output="$ ls\nfile1.txt")
        mock_window = MagicMock()

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value="$ ls\nfile1.txt",
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)

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

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value="$ ls\nfile1.txt",
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is True

    async def test_prompt_marker_stops_immediately(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()
        mock_window = MagicMock()
        mock_sent = MagicMock()
        mock_sent.message_id = 1
        pane = "ccgram:0❯ ls\nfile1.txt\nccgram:0❯"

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)

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
        pane = "ccgram:0❯ bad\nerror msg\nccgram:127❯"

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is True
        assert state.exit_code == 127
        assert state.last_output == "error msg"

    async def test_empty_output_continues(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState()
        mock_window = MagicMock()

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value="  \n  \n  ",
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)

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
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value="new output",
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)

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
                "ccgram.handlers.shell_commands.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell_commands.show_command_approval",
                new_callable=AsyncMock,
            ) as mock_approval,
        ):
            mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

        mock_edit.assert_called_once()
        assert "\u274c exit 1" in mock_edit.call_args[0][3]
        mock_approval.assert_awaited_once()
        approval_args = mock_approval.call_args[0]
        assert approval_args[4].command == "good-cmd"

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
                "ccgram.handlers.shell_commands.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell_commands.show_command_approval",
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
                    "ccgram.handlers.shell_commands.gather_llm_context",
                    new_callable=AsyncMock,
                    return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
                ),
                patch(
                    "ccgram.handlers.shell_commands.show_command_approval",
                    new_callable=AsyncMock,
                    return_value=False,
                ) as mock_approval,
            ):
                mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
                await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

            mock_approval.assert_awaited_once()
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


class TestStripTrailingPrompt:
    @pytest.mark.parametrize(
        ("lines", "expected"),
        [
            ([], []),
            (["output line", "$ "], ["output line"]),
            (["output line", "% "], ["output line"]),
            (["output line", "> "], ["output line"]),
            (["output line", "# "], ["output line"]),
            (["output line", "\ue0b0❮ "], ["output line"]),
            (["output line", "", "$ "], ["output line"]),
            (["~/projects/foo · main", "$ "], []),
            (["normal output text"], ["normal output text"]),
            (
                ["output", "this is a long line with > embedded in it"],
                ["output", "this is a long line with > embedded in it"],
            ),
        ],
        ids=[
            "empty-input",
            "dollar-prompt",
            "percent-prompt",
            "angle-prompt",
            "hash-prompt",
            "glyph-chevron-prompt",
            "blank-line-before-prompt",
            "info-bar-plus-prompt",
            "no-prompt-not-stripped",
            "long-line-with-prompt-char-not-stripped",
        ],
    )
    def test_strip_trailing_prompt(self, lines: list[str], expected: list[str]) -> None:
        assert _strip_trailing_prompt(lines.copy()) == expected


class TestLooksLikeInfoBar:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("~/projects/foo", True),
            ("main · feature-branch", True),
            ("\ue0b0 branch-name", True),
            ("normal output text", False),
            ("", False),
        ],
        ids=["tilde-slash", "middle-dot", "nerd-font-glyph", "normal-text", "empty"],
    )
    def test_looks_like_info_bar(self, line: str, expected: bool) -> None:
        assert _looks_like_info_bar(line) is expected


class TestMaybeSuggestFixErrorPaths:
    async def test_msg_id_none_skips_edit_but_calls_llm(self) -> None:
        from ccgram.llm.base import CommandResult

        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            exit_code=1, msg_id=None, last_output="err", command="bad-cmd"
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
                "ccgram.handlers.shell_commands.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell_commands.show_command_approval",
                new_callable=AsyncMock,
            ) as mock_approval,
        ):
            mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

        mock_edit.assert_not_called()
        mock_approval.assert_awaited_once()

    async def test_import_error_skips_suggestion(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            exit_code=1, msg_id=99, last_output="err", command="bad-cmd"
        )

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch("ccgram.llm.get_completer", side_effect=ImportError("no module")),
            patch(
                "ccgram.handlers.shell_commands.show_command_approval",
                new_callable=AsyncMock,
            ) as mock_approval,
        ):
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

        mock_approval.assert_not_awaited()

    async def test_generate_command_runtime_error_no_fix(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            exit_code=1, msg_id=99, last_output="err", command="bad-cmd"
        )

        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            side_effect=RuntimeError("API down")
        )

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch("ccgram.llm.get_completer", return_value=mock_completer),
            patch(
                "ccgram.handlers.shell_commands.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell_commands.show_command_approval",
                new_callable=AsyncMock,
            ) as mock_approval,
        ):
            mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

        mock_approval.assert_not_awaited()


class TestClassifyError:
    @pytest.mark.parametrize(
        ("exit_code", "output", "expected_substring"),
        [
            (127, "", "command not found"),
            (0, "bash: foo: command not found", "command not found"),
            (126, "", "permission denied"),
            (1, "Permission denied", "permission denied"),
            (2, "syntax error near unexpected token", "syntax error"),
            (1, "parse error: expected end", "syntax error"),
            (1, "No such file or directory", "file/directory not found"),
            (1, "invalid option -- 'z'", "invalid option"),
            (1, "unrecognized option '--foo'", "invalid option"),
            (1, "some generic error", ""),
            (None, "", ""),
        ],
        ids=[
            "exit-127",
            "command-not-found-text",
            "exit-126",
            "permission-denied-text",
            "syntax-error-text",
            "parse-error-text",
            "no-such-file",
            "invalid-option",
            "unrecognized-option",
            "generic-error",
            "none-exit-code",
        ],
    )
    def test_classify_error(
        self, exit_code: int | None, output: str, expected_substring: str
    ) -> None:
        from ccgram.handlers.shell_capture import _classify_error

        result = _classify_error(exit_code, output)
        if expected_substring:
            assert expected_substring in result
        else:
            assert result == ""


class TestUpdateErrorMessage:
    async def test_formats_with_code_fence(self) -> None:
        from ccgram.handlers.shell_capture import _update_error_message

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit:
            await _update_error_message(bot, -100, 99, 1, "some error output")

        formatted = mock_edit.call_args[0][3]
        assert formatted.startswith("\u274c exit 1\n```\n")
        assert formatted.endswith("\n```")
        assert "some error output" in formatted

    async def test_escapes_backticks_in_output(self) -> None:
        from ccgram.handlers.shell_capture import _update_error_message

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit:
            await _update_error_message(bot, -100, 99, 1, "has ``` backticks")

        formatted = mock_edit.call_args[0][3]
        body = formatted.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert "```" not in body


class TestMaybeSuggestFixNoLlm:
    async def test_no_completer_skips_suggestion(self) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            exit_code=1, msg_id=99, last_output="err", command="bad-cmd"
        )

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch("ccgram.llm.get_completer", return_value=None),
            patch(
                "ccgram.handlers.shell_commands.show_command_approval",
                new_callable=AsyncMock,
            ) as mock_approval,
        ):
            await _maybe_suggest_fix(bot, 1, -100, 42, "@0", state)

        mock_edit.assert_awaited_once()
        mock_approval.assert_not_awaited()


class TestRelayOutputBackticks:
    async def test_triple_backticks_escaped_in_relay(self) -> None:
        from ccgram.handlers.shell_capture import _relay_output

        bot = AsyncMock(spec=Bot)

        mock_sent = MagicMock()
        mock_sent.message_id = 42

        with patch(
            f"{_MOD}.rate_limit_send_message",
            new_callable=AsyncMock,
            return_value=mock_sent,
        ) as mock_send:
            await _relay_output(bot, -100, 42, "output has ``` backticks")

        formatted = mock_send.call_args[0][2]
        inner = formatted.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert "```" not in inner
        assert "` ` `" in inner

    async def test_relay_skips_whitespace_only_output(self) -> None:
        from ccgram.handlers.shell_capture import _relay_output

        bot = AsyncMock(spec=Bot)

        with patch(
            f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock
        ) as mock_send:
            await _relay_output(bot, -100, 42, "   \n  \n  ")

        mock_send.assert_not_called()


# ── Passive shell output monitoring tests ──────────────────────────────


class TestFindCommandEcho:
    def test_finds_echo_above_bare_prompt(self) -> None:
        from ccgram.handlers.shell_capture import _find_command_echo

        lines = ["ccgram:0❯ ls", "file1.txt", "ccgram:0❯"]
        assert _find_command_echo(lines) == ("ccgram:0❯ ls", 0)

    def test_returns_none_for_idle(self) -> None:
        from ccgram.handlers.shell_capture import _find_command_echo

        lines = ["ccgram:0❯"]
        assert _find_command_echo(lines) is None

    def test_returns_none_for_no_markers(self) -> None:
        from ccgram.handlers.shell_capture import _find_command_echo

        lines = ["$ ls", "file.txt"]
        assert _find_command_echo(lines) is None

    def test_finds_last_command(self) -> None:
        from ccgram.handlers.shell_capture import _find_command_echo

        lines = [
            "ccgram:0❯ ls",
            "file1.txt",
            "ccgram:0❯ pwd",
            "/home",
            "ccgram:0❯",
        ]
        assert _find_command_echo(lines) == ("ccgram:0❯ pwd", 2)


class TestFindInProgress:
    def test_finds_running_command(self) -> None:
        from ccgram.handlers.shell_capture import _find_in_progress

        lines = ["ccgram:0❯ tail -f log", "line1", "line2"]
        result = _find_in_progress(lines)
        assert result is not None
        assert result.command_echo == "ccgram:0❯ tail -f log"
        assert result.echo_index == 0
        assert result.text == "line1\nline2"
        assert result.exit_code is None

    def test_returns_none_for_bare_prompt(self) -> None:
        from ccgram.handlers.shell_capture import _find_in_progress

        lines = ["ccgram:0❯"]
        assert _find_in_progress(lines) is None

    def test_empty_output_in_progress(self) -> None:
        from ccgram.handlers.shell_capture import _find_in_progress

        lines = ["ccgram:0❯ slow-cmd"]
        result = _find_in_progress(lines)
        assert result is not None
        assert result.text == ""


class TestExtractPassiveOutput:
    @pytest.mark.parametrize(
        ("pane", "echo", "expected_text", "expected_code"),
        [
            (
                "ccgram:0❯ ls\nfile1.txt\nfile2.txt\nccgram:0❯",
                "ccgram:0❯ ls",
                "file1.txt\nfile2.txt",
                0,
            ),
            (
                "ccgram:0❯ bad-cmd\nerror: not found\nccgram:127❯",
                "ccgram:0❯ bad-cmd",
                "error: not found",
                127,
            ),
            (
                "ccgram:0❯ true\nccgram:0❯",
                "ccgram:0❯ true",
                "",
                0,
            ),
        ],
        ids=["success", "failure-127", "no-output"],
    )
    def test_completed_commands(
        self,
        pane: str,
        echo: str,
        expected_text: str,
        expected_code: int,
    ) -> None:
        from ccgram.handlers.shell_capture import _extract_passive_output

        result = _extract_passive_output(pane)
        assert result is not None
        assert result.command_echo == echo
        assert result.text == expected_text
        assert result.exit_code == expected_code

    def test_idle_returns_none(self) -> None:
        from ccgram.handlers.shell_capture import _extract_passive_output

        assert _extract_passive_output("ccgram:0❯") is None

    def test_no_markers_returns_none(self) -> None:
        from ccgram.handlers.shell_capture import _extract_passive_output

        assert _extract_passive_output("$ ls\nfile.txt") is None

    def test_empty_returns_none(self) -> None:
        from ccgram.handlers.shell_capture import _extract_passive_output

        assert _extract_passive_output("") is None

    def test_in_progress_command(self) -> None:
        from ccgram.handlers.shell_capture import _extract_passive_output

        pane = "ccgram:0❯ tail -f log\nline1\nline2"
        result = _extract_passive_output(pane)
        assert result is not None
        assert result.command_echo == "ccgram:0❯ tail -f log"
        assert result.text == "line1\nline2"
        assert result.exit_code is None


@pytest.fixture()
def _clean_monitor_state():
    from ccgram.handlers.shell_capture import reset_shell_monitor_state

    reset_shell_monitor_state()
    yield
    reset_shell_monitor_state()


@pytest.mark.usefixtures("_clean_monitor_state")
class TestCheckPassiveShellOutput:
    @pytest.mark.asyncio()
    async def test_skips_when_no_markers(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock) as m:
            await check_passive_shell_output(bot, 1, 42, "@0", "$ ls\nfile.txt")
        m.assert_not_called()
        assert (
            "@0" not in _shell_monitor_state
            or _shell_monitor_state["@0"].msg_id is None
        )

    @pytest.mark.asyncio()
    async def test_skips_when_active_capture_running(self) -> None:
        from ccgram.handlers.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        task = MagicMock()
        task.done.return_value = False
        _shell_capture_tasks[(1, 42)] = task
        try:
            with patch(f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock) as m:
                await check_passive_shell_output(
                    bot, 1, 42, "@0", "ccgram:0❯ ls\nfile.txt\nccgram:0❯"
                )
            m.assert_not_called()
        finally:
            _shell_capture_tasks.pop((1, 42), None)

    @pytest.mark.asyncio()
    async def test_relays_completed_command(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 99

        pane = "ccgram:0❯ ls\nfile1.txt\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        mock_send.assert_called_once()
        state = _shell_monitor_state["@0"]
        assert state.msg_id == 99
        assert state.last_command_echo == "ccgram:0❯ ls"

    @pytest.mark.asyncio()
    async def test_skips_unchanged_content(self) -> None:
        from ccgram.handlers.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 99
        pane = "ccgram:0❯ ls\nfile1.txt\nccgram:0❯"

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        # Second call with same content — should not relay again
        with (
            patch(
                f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock
            ) as mock_send2,
            patch(f"{_MOD}.session_manager") as mock_sm2,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm2.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        mock_send2.assert_not_called()

    @pytest.mark.asyncio()
    async def test_error_indicator_for_nonzero_exit(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 77

        pane = "ccgram:0❯ bad-cmd\nerror: not found\nccgram:127❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        assert mock_edit.called
        state = _shell_monitor_state["@0"]
        assert state.exit_code_sent is True

    @pytest.mark.asyncio()
    async def test_new_command_resets_state(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 50

        pane1 = "ccgram:0❯ ls\nfile.txt\nccgram:0❯"
        pane2 = "ccgram:0❯ pwd\n/home\nccgram:0❯"

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane1,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane1)
            assert _shell_monitor_state["@0"].last_command_echo == "ccgram:0❯ ls"

        mock_sent2 = MagicMock()
        mock_sent2.message_id = 51

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent2,
            ),
            patch(f"{_MOD}.session_manager") as mock_sm2,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane2,
            ),
        ):
            mock_sm2.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane2)

        state = _shell_monitor_state["@0"]
        assert state.last_command_echo == "ccgram:0❯ pwd"
        assert state.msg_id == mock_sent2.message_id

    @pytest.mark.asyncio()
    async def test_long_output_with_scrollback(self) -> None:
        """Command echo scrolled off visible pane — scrollback finds it."""
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 88

        # Visible pane: only output tail + bare prompt (echo scrolled off)
        visible = "\n".join([f"file{i}.txt" for i in range(20)] + ["ccgram:0❯"])
        # Scrollback: includes the command echo at the top
        scrollback = "ccgram:0❯ ls -al\n" + visible

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=scrollback,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", visible)

        mock_send.assert_called_once()
        state = _shell_monitor_state["@0"]
        assert state.msg_id == 88
        assert state.last_command_echo == "ccgram:0❯ ls -al"


class TestClearShellMonitorState:
    def test_clear_removes_state(self) -> None:
        from ccgram.handlers.shell_capture import (
            _ShellMonitorState,
            _shell_monitor_state,
            clear_shell_monitor_state,
        )

        _shell_monitor_state["@5"] = _ShellMonitorState(last_command_echo="test")
        clear_shell_monitor_state("@5")
        assert "@5" not in _shell_monitor_state

    def test_clear_nonexistent_is_noop(self) -> None:
        from ccgram.handlers.shell_capture import clear_shell_monitor_state

        clear_shell_monitor_state("@99")

    def test_reset_clears_all(self) -> None:
        from ccgram.handlers.shell_capture import (
            _ShellMonitorState,
            _shell_monitor_state,
            reset_shell_monitor_state,
        )

        _shell_monitor_state["@1"] = _ShellMonitorState()
        _shell_monitor_state["@2"] = _ShellMonitorState()
        reset_shell_monitor_state()
        assert len(_shell_monitor_state) == 0


@pytest.mark.usefixtures("_clean_monitor_state")
class TestPassiveEdgeCases:
    @pytest.mark.asyncio()
    async def test_same_command_rerun_creates_new_message(self) -> None:
        """Running `ls` twice should relay both — different echo_index."""
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent1 = MagicMock()
        mock_sent1.message_id = 60
        mock_sent2 = MagicMock()
        mock_sent2.message_id = 61

        # First ls at line 0
        pane1 = "ccgram:0❯ ls\nfile.txt\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent1,
            ),
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane1,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane1)

        assert _shell_monitor_state["@0"].msg_id == 60

        # Second ls — same text but at a different line index (scrolled down)
        pane2 = "ccgram:0❯ ls\nfile.txt\nccgram:0❯ ls\nfile.txt\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent2,
            ),
            patch(f"{_MOD}.session_manager") as mock_sm2,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane2,
            ),
        ):
            mock_sm2.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane2)

        # Should have reset to a new message (different echo_index)
        assert _shell_monitor_state["@0"].msg_id == 61

    @pytest.mark.asyncio()
    async def test_scroll_out_preserves_in_progress(self) -> None:
        """Marker scrolling out of tail should not reset in-progress state."""
        from ccgram.handlers.shell_capture import (
            _ShellMonitorState,
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)

        # Simulate in-progress tracking (command was detected earlier)
        _shell_monitor_state["@0"] = _ShellMonitorState(
            last_command_echo="ccgram:0❯ long-cmd",
            last_echo_index=0,
            msg_id=70,
            last_output="partial",
        )

        # Pane with no markers in tail (scrolled out) — should preserve state
        no_marker_pane = "\n".join([f"output line {i}" for i in range(20)])
        with patch(f"{_MOD}.session_manager") as mock_sm:
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", no_marker_pane)

        # State should be preserved, not reset
        state = _shell_monitor_state["@0"]
        assert state.last_command_echo == "ccgram:0❯ long-cmd"
        assert state.msg_id == 70

    @pytest.mark.asyncio()
    async def test_active_capture_fast_forwards_passive(self) -> None:
        """After active capture completes, passive state should be fast-forwarded."""
        from ccgram.handlers.shell_capture import _fast_forward_passive_state

        pane = "ccgram:0❯ ls\nfile.txt\nccgram:0❯"
        with patch(
            f"{_MOD}._capture_with_scrollback",
            new_callable=AsyncMock,
            return_value=pane,
        ):
            await _fast_forward_passive_state("@0")

        from ccgram.handlers.shell_capture import _shell_monitor_state

        state = _shell_monitor_state["@0"]
        assert state.last_command_echo == "ccgram:0❯ ls"
        assert state.last_echo_index == 0
        assert state.last_text_hash == hash(pane)


class TestCommandFromEcho:
    def test_extracts_command_text(self) -> None:
        from ccgram.handlers.shell_capture import _command_from_echo

        assert _command_from_echo("ccgram:0❯ ls -al") == "ls -al"

    def test_strips_whitespace(self) -> None:
        from ccgram.handlers.shell_capture import _command_from_echo

        assert _command_from_echo("ccgram:0❯ echo hi   ") == "echo hi"

    def test_error_exit_code(self) -> None:
        from ccgram.handlers.shell_capture import _command_from_echo

        assert _command_from_echo("ccgram:127❯ bad-cmd") == "bad-cmd"

    def test_non_matching_returns_input(self) -> None:
        from ccgram.handlers.shell_capture import _command_from_echo

        assert _command_from_echo("$ ls") == "$ ls"


class TestHasMarkersInTail:
    def test_marker_at_end(self) -> None:
        from ccgram.handlers.shell_capture import _has_markers_in_tail

        text = "file1.txt\nfile2.txt\nccgram:0❯"
        assert _has_markers_in_tail(text) is True

    def test_no_markers(self) -> None:
        from ccgram.handlers.shell_capture import _has_markers_in_tail

        text = "file1.txt\nfile2.txt\n$ "
        assert _has_markers_in_tail(text) is False

    def test_marker_with_leading_whitespace(self) -> None:
        from ccgram.handlers.shell_capture import _has_markers_in_tail

        text = "line1\n                    ccgram:0❯"
        assert _has_markers_in_tail(text) is True

    def test_marker_with_command(self) -> None:
        from ccgram.handlers.shell_capture import _has_markers_in_tail

        text = "output\nccgram:0❯ ls -al"
        assert _has_markers_in_tail(text) is True


@pytest.mark.usefixtures("_clean_monitor_state")
class TestPassiveRelayFormatting:
    """Test that passive output includes the command header."""

    @pytest.mark.asyncio()
    async def test_output_includes_command_header(self) -> None:
        from ccgram.handlers.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 200

        pane = "ccgram:0❯ echo hi\nhello\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        sent_text = mock_send.call_args[0][2]
        assert "❯ echo hi" in sent_text
        assert "hello" in sent_text
        assert sent_text.startswith("```\n")

    @pytest.mark.asyncio()
    async def test_multiline_output_formatted(self) -> None:
        from ccgram.handlers.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 201

        pane = "ccgram:0❯ seq 1 3\n1\n2\n3\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        sent_text = mock_send.call_args[0][2]
        assert "❯ seq 1 3" in sent_text
        assert "1\n2\n3" in sent_text

    @pytest.mark.asyncio()
    async def test_error_command_shows_exit_indicator(self) -> None:
        from ccgram.handlers.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 202

        pane = "ccgram:0❯ bad-cmd\nbad-cmd: not found\nccgram:127❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=pane,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        assert _shell_monitor_state["@0"].exit_code_sent is True
        assert mock_edit.called
        edit_text = mock_edit.call_args[0][3]  # (bot, chat_id, msg_id, text)
        assert "exit 127" in edit_text


class TestCaptureWithScrollback:
    @pytest.mark.asyncio()
    async def test_returns_text_on_success(self) -> None:
        from ccgram.handlers.shell_capture import _capture_with_scrollback

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"line1\nline2\n", b"")
            mock_exec.return_value = mock_proc
            result = await _capture_with_scrollback("@4")

        assert result == "line1\nline2"

    @pytest.mark.asyncio()
    async def test_returns_none_on_empty(self) -> None:
        from ccgram.handlers.shell_capture import _capture_with_scrollback

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"  \n  \n", b"")
            mock_exec.return_value = mock_proc
            result = await _capture_with_scrollback("@4")

        assert result is None

    @pytest.mark.asyncio()
    async def test_uses_correct_tmux_flags(self) -> None:
        from ccgram.handlers.shell_capture import _capture_with_scrollback

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"output", b"")
            mock_exec.return_value = mock_proc
            await _capture_with_scrollback("@4", history=100)

        args = mock_exec.call_args[0]
        assert "tmux" in args
        assert "capture-pane" in args
        assert "-J" in args
        assert "-S" in args
        assert "-100" in args
        assert "@4" in args
