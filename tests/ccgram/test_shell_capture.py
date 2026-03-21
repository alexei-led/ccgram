"""Tests for shell output capture and relay."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Bot

from ccgram.handlers.shell_capture import (
    _CaptureState,
    _OUTPUT_LIMIT,
    _poll_once,
    _shell_capture_tasks,
    cancel_shell_capture,
    start_shell_capture,
)

_MOD = "ccgram.handlers.shell_capture"


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

    async def test_unchanged_output_differs_from_baseline_increments_stable(
        self,
    ) -> None:
        bot = AsyncMock(spec=Bot)
        state = _CaptureState(
            last_output="$ ls\nfile1.txt",
            baseline_hash=hash("something else"),
        )
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
            baseline_hash=hash("something else"),
        )
        mock_window = MagicMock()

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="$ ls\nfile1.txt")

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert should_stop is True

    async def test_unchanged_output_same_as_baseline_continues(self) -> None:
        bot = AsyncMock(spec=Bot)
        baseline_content = "initial content"
        state = _CaptureState(
            last_output=baseline_content,
            baseline_hash=hash(baseline_content),
        )
        mock_window = MagicMock()

        with patch(f"{_MOD}.tmux_manager") as mock_tm:
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value=baseline_content)

            should_stop = await _poll_once(bot, -100, 42, "@0", state)

        assert state.stable_count == 0
        assert should_stop is False

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
            baseline_hash=hash("baseline"),
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
