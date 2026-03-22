"""Integration tests for shell provider — real asyncio task lifecycle.

Tests the full shell capture loop and task management with real asyncio
tasks but mocked tmux and Telegram APIs. No LLM required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.shell_capture import (
    _capture_shell_output,
    _shell_capture_tasks,
    cancel_shell_capture,
    start_shell_capture,
)

pytestmark = pytest.mark.integration

_MOD = "ccgram.handlers.shell_capture"


@pytest.fixture(autouse=True)
def _clean_capture_tasks():
    """Ensure capture task registry is clean before and after each test."""
    _shell_capture_tasks.clear()
    yield
    for task in _shell_capture_tasks.values():
        if not task.done():
            task.cancel()
    _shell_capture_tasks.clear()


class TestShellCaptureLoop:
    """Full _capture_shell_output loop with mocked tmux."""

    async def test_stops_on_output_stability(self) -> None:
        """Capture exits when output stabilizes (2 consecutive identical polls)."""
        bot = AsyncMock()
        call_count = 0
        outputs = [
            "$ ",  # baseline
            "$ ls\nfile1.txt",  # poll 1: new output → relay
            "$ ls\nfile1.txt",  # poll 2: same → stable_count=1
            "$ ls\nfile1.txt",  # poll 3: same → stable_count=2 → stop
        ]

        async def fake_capture(window_id):
            nonlocal call_count
            idx = min(call_count, len(outputs) - 1)
            call_count += 1
            return outputs[idx]

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=MagicMock(message_id=42),
            ) as mock_send,
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_tm.capture_pane = AsyncMock(side_effect=fake_capture)
            mock_sm.resolve_chat_id.return_value = -100

            await _capture_shell_output(bot, 1, 42, "@0")

        assert call_count >= 3
        mock_send.assert_called_once()

    async def test_stops_when_window_disappears(self) -> None:
        """Capture exits when tmux window is no longer found."""
        bot = AsyncMock()
        find_count = 0

        async def fake_find(window_id):
            nonlocal find_count
            find_count += 1
            return MagicMock() if find_count <= 2 else None

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_tm.find_window_by_id = AsyncMock(side_effect=fake_find)
            mock_tm.capture_pane = AsyncMock(return_value="baseline")
            mock_sm.resolve_chat_id.return_value = -100

            await _capture_shell_output(bot, 1, 42, "@0")

        assert find_count >= 2

    async def test_streaming_updates_edit_in_place(self) -> None:
        """First output creates a message, subsequent changes edit it."""
        bot = AsyncMock()
        call_count = 0

        async def fake_capture(window_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "baseline"
            if call_count == 2:
                return "output v1"
            if call_count == 3:
                return "output v2"
            return "output v2"  # stabilize

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=MagicMock(message_id=42),
            ) as mock_send,
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_tm.capture_pane = AsyncMock(side_effect=fake_capture)
            mock_sm.resolve_chat_id.return_value = -100

            await _capture_shell_output(bot, 1, 42, "@0")

        mock_send.assert_called_once()
        assert mock_edit.call_count >= 1

    async def test_task_cleans_up_on_completion(self) -> None:
        """Capture task removes itself from registry when done."""
        bot = AsyncMock()

        with (
            patch(f"{_MOD}.tmux_manager") as mock_tm,
            patch(f"{_MOD}.session_manager") as mock_sm,
            patch(f"{_MOD}.task_done_callback"),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            # Window disappears immediately → loop exits fast
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            mock_tm.capture_pane = AsyncMock(return_value="")
            mock_sm.resolve_chat_id.return_value = -100

            start_shell_capture(bot, 1, 42, "@0")
            task = _shell_capture_tasks.get((1, 42))
            assert task is not None

            # Wait for the task to complete
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                task.cancel()

        assert (1, 42) not in _shell_capture_tasks


class TestShellCaptureTaskLifecycle:
    """Task start/cancel/replace with real asyncio tasks."""

    async def test_start_cancel_roundtrip(self) -> None:
        bot = AsyncMock()

        async def long_capture(b, u, t, w):
            await asyncio.sleep(100)

        with (
            patch(f"{_MOD}._capture_shell_output", side_effect=long_capture),
            patch(f"{_MOD}.task_done_callback"),
        ):
            start_shell_capture(bot, 1, 42, "@0")
            assert (1, 42) in _shell_capture_tasks
            task = _shell_capture_tasks[(1, 42)]

            cancel_shell_capture(1, 42)
            await asyncio.sleep(0)

            assert task.cancelled()
            assert (1, 42) not in _shell_capture_tasks

    async def test_replace_cancels_previous(self) -> None:
        bot = AsyncMock()

        async def long_capture(b, u, t, w):
            await asyncio.sleep(100)

        with (
            patch(f"{_MOD}._capture_shell_output", side_effect=long_capture),
            patch(f"{_MOD}.task_done_callback"),
        ):
            start_shell_capture(bot, 1, 42, "@0")
            first_task = _shell_capture_tasks[(1, 42)]

            start_shell_capture(bot, 1, 42, "@1")
            await asyncio.sleep(0)

            assert first_task.cancelled()
            second_task = _shell_capture_tasks[(1, 42)]
            assert second_task is not first_task
            second_task.cancel()

    async def test_concurrent_users_independent(self) -> None:
        bot = AsyncMock()

        async def long_capture(b, u, t, w):
            await asyncio.sleep(100)

        with (
            patch(f"{_MOD}._capture_shell_output", side_effect=long_capture),
            patch(f"{_MOD}.task_done_callback"),
        ):
            start_shell_capture(bot, 1, 42, "@0")
            start_shell_capture(bot, 2, 99, "@1")

            assert (1, 42) in _shell_capture_tasks
            assert (2, 99) in _shell_capture_tasks

            cancel_shell_capture(1, 42)
            await asyncio.sleep(0)

            assert (1, 42) not in _shell_capture_tasks
            assert (2, 99) in _shell_capture_tasks

            _shell_capture_tasks[(2, 99)].cancel()
