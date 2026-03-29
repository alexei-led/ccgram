"""Tests for resilience and crash-prevention fixes.

Covers: background loop catch-all, dead worker respawn, session_map corruption
guard, pyte feed/resize guards, rate_limit_send locking, JSONL malformed entry
handling, probe failure clearing, poll state cleanup, shutdown notification
lifecycle.
"""

import asyncio
import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from telegram.error import TelegramError


class TestScreenBufferResilience:
    def test_feed_malformed_ansi_does_not_raise(self):
        from ccgram.screen_buffer import ScreenBuffer

        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("normal text")
        buf.feed("\x1b[9999;9999H")
        buf.feed("\x1b[?????m")
        assert isinstance(buf.rendered_text, str)

    def test_resize_zero_dimensions_ignored(self):
        from ccgram.screen_buffer import ScreenBuffer

        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("hello")
        buf.resize(0, 0)
        assert buf.columns == 40
        assert buf.rows == 5

    def test_resize_negative_dimensions_ignored(self):
        from ccgram.screen_buffer import ScreenBuffer

        buf = ScreenBuffer(columns=40, rows=5)
        buf.resize(-1, -1)
        assert buf.columns == 40

    def test_resize_valid_dimensions_works(self):
        from ccgram.screen_buffer import ScreenBuffer

        buf = ScreenBuffer(columns=40, rows=5)
        buf.resize(80, 24)
        assert buf.columns == 80
        assert buf.rows == 24


class TestJsonlMalformedEntries:
    def test_non_dict_message_field_skipped_in_parse(self):
        from ccgram.providers._jsonl import parse_jsonl_entries

        entries = [
            {"type": "assistant", "message": "not a dict"},
            {"type": "assistant", "message": {"content": "valid text"}},
            {"type": "assistant", "message": None},
            {"type": "assistant", "message": 42},
        ]
        messages, _ = parse_jsonl_entries(entries, {})
        assert len(messages) == 1
        assert messages[0].text == "valid text"

    def test_non_dict_message_field_skipped_in_history(self):
        from ccgram.providers._jsonl import parse_jsonl_history_entry

        assert (
            parse_jsonl_history_entry({"type": "assistant", "message": "string"})
            is None
        )
        assert parse_jsonl_history_entry({"type": "assistant", "message": None}) is None
        assert parse_jsonl_history_entry({"type": "assistant", "message": 42}) is None

    def test_valid_entry_still_parsed(self):
        from ccgram.providers._jsonl import parse_jsonl_history_entry

        entry = {"type": "assistant", "message": {"content": "hello world"}}
        result = parse_jsonl_history_entry(entry)
        assert result is not None
        assert result.text == "hello world"


class TestSessionMapCorruptionGuard:
    def test_corrupted_json_backed_up(self, tmp_path: Path):
        from ccgram.hook import _update_session_map

        map_file = tmp_path / "session_map.json"
        map_file.write_text("{invalid json")
        with (
            patch("ccgram.utils.ccgram_dir", return_value=tmp_path),
            patch("ccgram.utils.atomic_write_json") as mock_write,
        ):
            _update_session_map(
                session_window_key="ccgram:@0",
                session_id="test-sid",
                cwd="/tmp",
                window_name="test",
                transcript_path="/tmp/t.jsonl",
                tmux_session_name="ccgram",
            )

        backup = tmp_path / "session_map.json.corrupt"
        assert backup.exists()
        assert backup.read_text() == "{invalid json"
        written_data = mock_write.call_args[0][1]
        assert "ccgram:@0" in written_data

    def test_valid_json_preserved_on_update(self, tmp_path: Path):
        from ccgram.hook import _update_session_map

        map_file = tmp_path / "session_map.json"
        existing = {"ccgram:@5": {"session_id": "old", "cwd": "/old"}}
        map_file.write_text(json.dumps(existing))

        with (
            patch("ccgram.utils.ccgram_dir", return_value=tmp_path),
            patch("ccgram.utils.atomic_write_json") as mock_write,
        ):
            _update_session_map(
                session_window_key="ccgram:@0",
                session_id="new-sid",
                cwd="/new",
                window_name="new",
                transcript_path="/tmp/t.jsonl",
                tmux_session_name="ccgram",
            )

        written_data = mock_write.call_args[0][1]
        assert "ccgram:@5" in written_data
        assert "ccgram:@0" in written_data


class TestDeadWorkerRespawn:
    async def test_dead_worker_is_respawned(self):
        from ccgram.handlers.message_queue import (
            _message_queues,
            _queue_locks,
            _queue_workers,
            get_or_create_queue,
        )

        bot = MagicMock()
        user_id = 99999

        _message_queues.pop(user_id, None)
        _queue_locks.pop(user_id, None)
        _queue_workers.pop(user_id, None)

        queue = get_or_create_queue(bot, user_id)
        assert user_id in _queue_workers
        first_worker = _queue_workers[user_id]

        first_worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first_worker
        assert first_worker.done()

        queue2 = get_or_create_queue(bot, user_id)
        assert queue2 is queue
        second_worker = _queue_workers[user_id]
        assert second_worker is not first_worker
        assert not second_worker.done()

        second_worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await second_worker
        _message_queues.pop(user_id, None)
        _queue_locks.pop(user_id, None)
        _queue_workers.pop(user_id, None)


class TestRateLimitSendLocking:
    async def test_concurrent_sends_serialized(self):
        import time

        from ccgram.handlers.message_sender import (
            MESSAGE_SEND_INTERVAL,
            _last_send_time,
            _rate_limit_locks,
            rate_limit_send,
        )

        chat_id = 88888
        _last_send_time.pop(chat_id, None)
        _rate_limit_locks.pop(chat_id, None)

        timestamps: list[float] = []

        async def timed_send():
            await rate_limit_send(chat_id)
            timestamps.append(time.monotonic())

        # Seed the rate limiter so subsequent calls must wait
        await rate_limit_send(chat_id)
        t0 = time.monotonic()

        # Launch 2 concurrent senders — they should be serialized by the lock
        tasks = [asyncio.create_task(timed_send()) for _ in range(2)]
        await asyncio.gather(*tasks)

        assert len(timestamps) == 2
        # Each send should be spaced at least MESSAGE_SEND_INTERVAL apart from seed
        for ts in timestamps:
            assert (
                ts - t0 >= MESSAGE_SEND_INTERVAL * 0.8
            )  # 80% tolerance for scheduling jitter

        _last_send_time.pop(chat_id, None)
        _rate_limit_locks.pop(chat_id, None)


class TestCallbackErrorWidened:
    def test_callback_error_catches_programming_errors(self):
        from ccgram.session_monitor import _CallbackError

        assert issubclass(KeyError, _CallbackError)
        assert issubclass(TypeError, _CallbackError)
        assert issubclass(AttributeError, _CallbackError)
        assert issubclass(IndexError, _CallbackError)
        assert issubclass(TelegramError, _CallbackError)


class TestProbeFailureClearing:
    def test_clear_probe_failures_resets_counter(self):
        from ccgram.handlers.polling_coordinator import _get_window_state
        from ccgram.handlers.polling_strategies import (
            clear_probe_failures,
            terminal_strategy,
        )

        _window_poll_state = terminal_strategy._states
        ws = _get_window_state("@test-probe")
        ws.probe_failures = 5

        clear_probe_failures("@test-probe")

        ws2 = _window_poll_state.get("@test-probe")
        assert ws2 is not None
        assert ws2.probe_failures == 0

        _window_poll_state.pop("@test-probe", None)


class TestPollStateCleanup:
    def test_clear_window_poll_state_removes_entry(self):
        from ccgram.handlers.polling_coordinator import _get_window_state
        from ccgram.handlers.polling_strategies import (
            clear_window_poll_state,
            terminal_strategy,
        )

        _window_poll_state = terminal_strategy._states
        _get_window_state("@cleanup-test")
        assert "@cleanup-test" in _window_poll_state

        clear_window_poll_state("@cleanup-test")
        assert "@cleanup-test" not in _window_poll_state


class TestGlobalExceptionHandler:
    def test_handler_logs_exception(self):
        from ccgram.bot import _global_exception_handler

        loop = MagicMock()
        error = ValueError("test error")
        context = {"exception": error, "message": "test context"}

        with patch("ccgram.bot.logger") as mock_logger:
            _global_exception_handler(loop, context)

        mock_logger.error.assert_called_once()
        assert "test context" in str(mock_logger.error.call_args)


class TestShutdownNotificationLifecycle:
    async def test_post_stop_sends_notification(self):
        from ccgram.bot import post_stop

        application = MagicMock()
        application.bot = AsyncMock()

        with (
            patch(
                "ccgram.bot._send_shutdown_notification", new_callable=AsyncMock
            ) as mock_send,
        ):
            await post_stop(application)

        mock_send.assert_awaited_once_with(application)

    async def test_post_shutdown_does_not_send_notification(self):
        from ccgram.bot import post_shutdown

        application = MagicMock()

        with (
            patch(
                "ccgram.bot._send_shutdown_notification", new_callable=AsyncMock
            ) as mock_send,
            patch("ccgram.bot._status_poll_task", None),
            patch("ccgram.bot.session_monitor", None),
            patch("ccgram.bot.session_manager"),
            patch("ccgram.bot.shutdown_workers", new_callable=AsyncMock),
        ):
            await post_shutdown(application)

        mock_send.assert_not_awaited()


class TestShellDetectionSafety:
    async def test_script_shell_not_interactive(self):
        from ccgram.providers.shell import _is_interactive_shell

        mock_tmux = MagicMock()
        mock_window = MagicMock()
        mock_window.pane_tty = "/dev/ttys005"
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)

        with (
            patch("ccgram.tmux_manager.tmux_manager", mock_tmux),
            patch(
                "ccgram.providers.process_detection.get_foreground_args",
                new_callable=AsyncMock,
                return_value=("bash ./scripts/restart.sh run", 0),
            ),
        ):
            assert await _is_interactive_shell("@0") is False

    async def test_idle_shell_is_interactive(self):
        from ccgram.providers.shell import _is_interactive_shell

        mock_tmux = MagicMock()
        mock_window = MagicMock()
        mock_window.pane_tty = "/dev/ttys005"
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)

        with (
            patch("ccgram.tmux_manager.tmux_manager", mock_tmux),
            patch(
                "ccgram.providers.process_detection.get_foreground_args",
                new_callable=AsyncMock,
                return_value=("-bash", 1234),
            ),
        ):
            assert await _is_interactive_shell("@0") is True

    async def test_setup_shell_prompt_skips_own_window(self):
        from ccgram.providers.shell import setup_shell_prompt

        with (
            patch("ccgram.config.config") as mock_config,
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
            ) as mock_has_marker,
            patch(
                "ccgram.providers.shell._is_interactive_shell",
                new_callable=AsyncMock,
            ),
        ):
            mock_config.own_window_id = "@99"
            await setup_shell_prompt("@99", clear=False)

        mock_has_marker.assert_not_awaited()

    async def test_setup_shell_prompt_skips_non_interactive(self):
        from ccgram.providers.shell import setup_shell_prompt

        with (
            patch("ccgram.config.config") as mock_config,
            patch(
                "ccgram.providers.shell._is_interactive_shell",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "ccgram.providers.shell.has_prompt_marker",
                new_callable=AsyncMock,
            ) as mock_has_marker,
        ):
            mock_config.own_window_id = None
            await setup_shell_prompt("@5", clear=False)

        mock_has_marker.assert_not_awaited()
