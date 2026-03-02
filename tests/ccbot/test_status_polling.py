"""Tests for status polling: shell detection, autoclose timers, rename sync,
activity heuristic, and startup timeout."""

import time

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, TelegramError

from conftest import make_mock_provider

from ccbot.handlers.status_polling import (
    _autoclose_timers,
    _cancel_idle_clear_timer,
    _check_autoclose_timers,
    _check_idle_clear_timers,
    _check_transcript_activity,
    _clear_autoclose_if_active,
    _has_seen_status,
    _idle_clear_timers,
    _idle_status_cleared,
    _MAX_PROBE_FAILURES,
    _probe_failures,
    _probe_topic_existence,
    _prune_stale_state,
    _start_autoclose_timer,
    _start_idle_clear_timer,
    _startup_times,
    clear_autoclose_timer,
    is_shell_prompt,
    reset_autoclose_state,
    reset_idle_clear_state,
    reset_probe_failures_state,
    reset_seen_status_state,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_autoclose_state()
    reset_seen_status_state()
    reset_idle_clear_state()
    reset_probe_failures_state()
    yield
    reset_autoclose_state()
    reset_seen_status_state()
    reset_probe_failures_state()
    reset_idle_clear_state()


class TestIsShellPrompt:
    @pytest.mark.parametrize(
        "cmd",
        ["bash", "zsh", "fish", "sh", "/usr/bin/zsh", "  bash  ", "dash", "ksh"],
    )
    def test_shell_detected(self, cmd: str) -> None:
        assert is_shell_prompt(cmd) is True

    @pytest.mark.parametrize("cmd", ["node", "claude", "npx", ""])
    def test_non_shell_rejected(self, cmd: str) -> None:
        assert is_shell_prompt(cmd) is False


class TestAutocloseTimers:
    def test_start_timer(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        assert _autoclose_timers[(1, 42)] == ("done", 100.0)

    def test_start_timer_preserves_existing_same_state(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _start_autoclose_timer(1, 42, "done", 200.0)
        assert _autoclose_timers[(1, 42)] == ("done", 100.0)

    def test_start_timer_resets_on_state_change(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _start_autoclose_timer(1, 42, "dead", 200.0)
        assert _autoclose_timers[(1, 42)] == ("dead", 200.0)

    def test_clear_on_active(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _clear_autoclose_if_active(1, 42)
        assert (1, 42) not in _autoclose_timers

    def test_clear_timer(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        clear_autoclose_timer(1, 42)
        assert (1, 42) not in _autoclose_timers

    def test_clear_nonexistent_is_noop(self) -> None:
        clear_autoclose_timer(1, 42)

    @pytest.mark.parametrize(
        ("state", "minutes", "elapsed"),
        [("done", 30, 30 * 60 + 1), ("dead", 10, 10 * 60 + 1)],
        ids=["done", "dead"],
    )
    async def test_check_expired(
        self, state: str, minutes: int, elapsed: float
    ) -> None:
        _start_autoclose_timer(1, 42, state, 0.0)
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.config") as mock_config,
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = minutes
            mock_time.monotonic.return_value = elapsed
            mock_sm.resolve_chat_id.return_value = -100
            await _check_autoclose_timers(bot)
        bot.close_forum_topic.assert_called_once_with(
            chat_id=-100, message_thread_id=42
        )
        assert (1, 42) not in _autoclose_timers

    async def test_check_not_expired_yet(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.config") as mock_config,
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = 10
            mock_time.monotonic.return_value = 29 * 60
            await _check_autoclose_timers(bot)
        bot.close_forum_topic.assert_not_called()
        assert (1, 42) in _autoclose_timers

    async def test_check_disabled_when_zero(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.config") as mock_config,
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 0
            mock_config.autoclose_dead_minutes = 0
            mock_time.monotonic.return_value = 999999
            await _check_autoclose_timers(bot)
        bot.close_forum_topic.assert_not_called()

    async def test_check_telegram_error_handled(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock()
        bot.close_forum_topic.side_effect = TelegramError("fail")
        with (
            patch("ccbot.handlers.status_polling.config") as mock_config,
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = 10
            mock_time.monotonic.return_value = 30 * 60 + 1
            mock_sm.resolve_chat_id.return_value = -100
            await _check_autoclose_timers(bot)
        assert (1, 42) not in _autoclose_timers


class TestTranscriptActivityHeuristic:
    def test_active_when_recent_transcript(self) -> None:
        now = time.monotonic()
        mock_monitor = MagicMock()
        mock_monitor.get_last_activity.return_value = now - 5.0
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccbot.handlers.status_polling.get_active_monitor",
                return_value=mock_monitor,
            ),
        ):
            mock_sm.get_session_id_for_window.return_value = "sess-123"
            result = _check_transcript_activity("@0", now)
        assert result is True
        assert "@0" in _has_seen_status

    def test_inactive_when_stale_transcript(self) -> None:
        now = time.monotonic()
        mock_monitor = MagicMock()
        mock_monitor.get_last_activity.return_value = now - 20.0
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccbot.handlers.status_polling.get_active_monitor",
                return_value=mock_monitor,
            ),
        ):
            mock_sm.get_session_id_for_window.return_value = "sess-123"
            result = _check_transcript_activity("@0", now)
        assert result is False
        assert "@0" not in _has_seen_status

    def test_inactive_when_no_session(self) -> None:
        now = time.monotonic()
        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.get_session_id_for_window.return_value = None
            result = _check_transcript_activity("@0", now)
        assert result is False

    def test_inactive_when_no_monitor(self) -> None:
        now = time.monotonic()
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccbot.handlers.status_polling.get_active_monitor",
                return_value=None,
            ),
        ):
            mock_sm.get_session_id_for_window.return_value = "sess-123"
            result = _check_transcript_activity("@0", now)
        assert result is False

    def test_clears_startup_timer_on_activity(self) -> None:
        now = time.monotonic()
        _startup_times["@0"] = now - 15.0
        mock_monitor = MagicMock()
        mock_monitor.get_last_activity.return_value = now - 3.0
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccbot.handlers.status_polling.get_active_monitor",
                return_value=mock_monitor,
            ),
        ):
            mock_sm.get_session_id_for_window.return_value = "sess-123"
            result = _check_transcript_activity("@0", now)
        assert result is True
        assert "@0" not in _startup_times


class TestStartupTimeout:
    async def test_first_poll_records_startup_time(self) -> None:
        from ccbot.handlers.status_polling import _handle_no_status

        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch("ccbot.handlers.status_polling._send_typing_throttled"),
            patch(
                "ccbot.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1000.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "node", "normal")
        assert "@0" in _startup_times

    async def test_startup_timeout_transitions_to_idle(self) -> None:
        from ccbot.handlers.status_polling import _handle_no_status

        bot = AsyncMock()
        _startup_times["@0"] = 1000.0
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji") as mock_emoji,
            patch("ccbot.handlers.status_polling.enqueue_status_update"),
            patch(
                "ccbot.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1000.0 + 31.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "node", "normal")
        assert "@0" in _has_seen_status
        assert "@0" not in _startup_times
        mock_emoji.assert_called_once_with(bot, -100, 42, "idle", "project")

    async def test_startup_grace_period_sends_typing(self) -> None:
        from ccbot.handlers.status_polling import _handle_no_status

        bot = AsyncMock()
        _startup_times["@0"] = 1000.0
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji") as mock_emoji,
            patch(
                "ccbot.handlers.status_polling._send_typing_throttled"
            ) as mock_typing,
            patch(
                "ccbot.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1010.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "node", "normal")
        mock_typing.assert_called_once_with(bot, 1, 42)
        mock_emoji.assert_called_once_with(bot, -100, 42, "active", "project")
        assert "@0" not in _has_seen_status


class TestParseWithPyte:
    """Tests for pyte-based screen parsing integration."""

    def setup_method(self) -> None:
        from ccbot.handlers.status_polling import reset_screen_buffer_state

        reset_screen_buffer_state()

    def teardown_method(self) -> None:
        from ccbot.handlers.status_polling import reset_screen_buffer_state

        reset_screen_buffer_state()

    def test_detects_spinner_status(self) -> None:
        from ccbot.handlers.status_polling import _parse_with_pyte

        sep = "─" * 30
        pane_text = f"Some output\n✻ Reading file src/main.py\n{sep}\n"
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.raw_text == "Reading file src/main.py"
        assert result.display_label == "\U0001f4d6 reading\u2026"
        assert result.is_interactive is False

    def test_detects_braille_spinner(self) -> None:
        from ccbot.handlers.status_polling import _parse_with_pyte

        sep = "─" * 30
        pane_text = f"Output\n⠋ Thinking about things\n{sep}\n"
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.raw_text == "Thinking about things"
        assert result.is_interactive is False

    def test_detects_interactive_ui(self) -> None:
        from ccbot.handlers.status_polling import _parse_with_pyte

        pane_text = (
            "  Would you like to proceed?\n"
            "  ─────────────────────────────────\n"
            "  Yes     No\n"
            "  ─────────────────────────────────\n"
            "  ctrl-g to edit in vim\n"
        )
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.is_interactive is True
        assert result.ui_type == "ExitPlanMode"

    def test_returns_none_for_plain_text(self) -> None:
        from ccbot.handlers.status_polling import _parse_with_pyte

        pane_text = "$ echo hello\nhello\n$\n"
        result = _parse_with_pyte("@0", pane_text)
        assert result is None

    def test_screen_buffer_cached_per_window(self) -> None:
        from ccbot.handlers.status_polling import _parse_with_pyte, _screen_buffers

        sep = "─" * 30
        pane_text = f"Output\n✻ Working\n{sep}\n"
        _parse_with_pyte("@0", pane_text)
        assert "@0" in _screen_buffers

        _parse_with_pyte("@1", pane_text)
        assert "@1" in _screen_buffers
        assert "@0" in _screen_buffers

    def test_interactive_takes_precedence_over_status(self) -> None:
        from ccbot.handlers.status_polling import _parse_with_pyte

        sep = "─" * 30
        pane_text = (
            f"✻ Working on task\n{sep}\n"
            "  Do you want to proceed?\n"
            "  Allow write to /tmp/foo\n"
            "  Esc to cancel\n"
        )
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.is_interactive is True
        assert result.ui_type == "PermissionPrompt"


class TestPyteFallbackInUpdateStatus:
    """Tests that update_status_message falls back to regex when pyte returns None."""

    async def test_falls_back_to_provider_when_pyte_returns_none(self) -> None:
        with (
            patch("ccbot.handlers.status_polling.tmux_manager") as mock_tm,
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch("ccbot.handlers.status_polling.enqueue_status_update"),
            patch(
                "ccbot.handlers.status_polling.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccbot.handlers.status_polling.get_provider_for_window",
                return_value=make_mock_provider(has_status=True),
            ) as mock_get_provider,
            patch(
                "ccbot.handlers.status_polling._parse_with_pyte",
                return_value=None,
            ),
        ):
            from ccbot.handlers.status_polling import update_status_message

            mock_window = MagicMock()
            mock_window.window_id = "@0"
            mock_window.window_name = "project"
            mock_window.pane_current_command = "node"
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tm.get_pane_title = AsyncMock(return_value="")
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            mock_sm.get_notification_mode.return_value = "normal"

            bot = AsyncMock()
            await update_status_message(bot, 1, "@0", thread_id=42)

            # Provider regex parsing was called as fallback
            mock_get_provider.return_value.parse_terminal_status.assert_called_once()

    async def test_uses_pyte_result_when_available(self) -> None:
        from ccbot.providers.base import StatusUpdate

        pyte_status = StatusUpdate(
            raw_text="Reading file",
            display_label="\U0001f4d6 reading\u2026",
        )
        with (
            patch("ccbot.handlers.status_polling.tmux_manager") as mock_tm,
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccbot.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
            patch(
                "ccbot.handlers.status_polling.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccbot.handlers.status_polling.get_provider_for_window",
                return_value=make_mock_provider(has_status=True),
            ) as mock_get_provider,
            patch(
                "ccbot.handlers.status_polling._parse_with_pyte",
                return_value=pyte_status,
            ),
        ):
            from ccbot.handlers.status_polling import update_status_message

            mock_window = MagicMock()
            mock_window.window_id = "@0"
            mock_window.window_name = "project"
            mock_window.pane_current_command = "node"
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tm.get_pane_title = AsyncMock(return_value="")
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            mock_sm.get_notification_mode.return_value = "normal"

            bot = AsyncMock()
            await update_status_message(bot, 1, "@0", thread_id=42)

            # Provider regex parsing was NOT called (pyte succeeded)
            mock_get_provider.return_value.parse_terminal_status.assert_not_called()
            # Status was enqueued using pyte result
            mock_enqueue.assert_called_once()
            call_args = mock_enqueue.call_args
            assert call_args[0][3] == "\U0001f4d6 reading\u2026"


class TestIdleClearTimers:
    def test_start_timer(self) -> None:
        with patch("ccbot.handlers.status_polling.time") as mock_time:
            mock_time.monotonic.return_value = 500.0
            _start_idle_clear_timer(1, 42, "@0")
        assert (1, 42) in _idle_clear_timers
        assert _idle_clear_timers[(1, 42)] == ("@0", 500.0)

    def test_start_timer_does_not_overwrite_existing(self) -> None:
        with patch("ccbot.handlers.status_polling.time") as mock_time:
            mock_time.monotonic.return_value = 500.0
            _start_idle_clear_timer(1, 42, "@0")
            mock_time.monotonic.return_value = 600.0
            _start_idle_clear_timer(1, 42, "@0")
        assert _idle_clear_timers[(1, 42)] == ("@0", 500.0)

    def test_cancel_timer(self) -> None:
        _idle_clear_timers[(1, 42)] = ("@0", 500.0)
        _cancel_idle_clear_timer(1, 42)
        assert (1, 42) not in _idle_clear_timers

    def test_cancel_nonexistent_is_noop(self) -> None:
        _cancel_idle_clear_timer(1, 42)

    async def test_check_expired_enqueues_clear(self) -> None:
        from ccbot.handlers.status_polling import _IDLE_CLEAR_DELAY

        _idle_clear_timers[(1, 42)] = ("@0", 0.0)
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.time") as mock_time,
            patch(
                "ccbot.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
        ):
            mock_time.monotonic.return_value = _IDLE_CLEAR_DELAY + 1.0
            await _check_idle_clear_timers(bot)
        mock_enqueue.assert_called_once_with(bot, 1, "@0", None, thread_id=42)
        assert (1, 42) not in _idle_clear_timers
        assert "@0" in _idle_status_cleared

    async def test_check_not_expired_yet(self) -> None:
        from ccbot.handlers.status_polling import _IDLE_CLEAR_DELAY

        _idle_clear_timers[(1, 42)] = ("@0", 0.0)
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.time") as mock_time,
            patch(
                "ccbot.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
        ):
            mock_time.monotonic.return_value = _IDLE_CLEAR_DELAY - 1.0
            await _check_idle_clear_timers(bot)
        mock_enqueue.assert_not_called()
        assert (1, 42) in _idle_clear_timers

    async def test_check_empty_is_noop(self) -> None:
        bot = AsyncMock()
        with patch(
            "ccbot.handlers.status_polling.enqueue_status_update"
        ) as mock_enqueue:
            await _check_idle_clear_timers(bot)
        mock_enqueue.assert_not_called()


class TestClearSeenStatus:
    def test_clears_idle_status_cleared(self) -> None:
        from ccbot.handlers.status_polling import clear_seen_status

        _idle_status_cleared.add("@0")
        _has_seen_status.add("@0")
        _startup_times["@0"] = 100.0
        clear_seen_status("@0")
        assert "@0" not in _idle_status_cleared
        assert "@0" not in _has_seen_status
        assert "@0" not in _startup_times


class TestTransitionToIdle:
    async def test_starts_idle_clear_timer_and_sends_idle_text(self) -> None:
        from ccbot.handlers.callback_data import IDLE_STATUS_TEXT
        from ccbot.handlers.status_polling import _transition_to_idle

        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccbot.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 100.0
            await _transition_to_idle(bot, 1, "@0", 42, -100, "project", "normal")
        mock_enqueue.assert_called_once()
        assert mock_enqueue.call_args[0][3] == IDLE_STATUS_TEXT
        assert mock_enqueue.call_args[1]["thread_id"] == 42
        assert (1, 42) in _idle_clear_timers

    async def test_skips_when_already_cleared(self) -> None:
        from ccbot.handlers.status_polling import _transition_to_idle

        _idle_status_cleared.add("@0")
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccbot.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
        ):
            await _transition_to_idle(bot, 1, "@0", 42, -100, "project", "normal")
        mock_enqueue.assert_not_called()
        assert (1, 42) not in _idle_clear_timers

    @pytest.mark.parametrize("mode", ["muted", "errors_only"])
    async def test_suppressed_mode_clears_status_no_timer(self, mode: str) -> None:
        from ccbot.handlers.status_polling import _transition_to_idle

        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccbot.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
        ):
            await _transition_to_idle(bot, 1, "@0", 42, -100, "project", mode)
        mock_enqueue.assert_called_once_with(bot, 1, "@0", None, thread_id=42)
        assert (1, 42) not in _idle_clear_timers


class TestShellPromptClearsStatus:
    async def test_shell_prompt_enqueues_status_clear(self) -> None:
        from ccbot.handlers.status_polling import _handle_no_status

        _has_seen_status.add("@0")
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccbot.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
            patch(
                "ccbot.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1000.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "bash", "normal")
        mock_enqueue.assert_called_once_with(bot, 1, "@0", None, thread_id=42)

    async def test_shell_prompt_cancels_idle_timer(self) -> None:
        from ccbot.handlers.status_polling import _handle_no_status

        _has_seen_status.add("@0")
        _idle_clear_timers[(1, 42)] = ("@0", 500.0)
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch("ccbot.handlers.status_polling.enqueue_status_update"),
            patch(
                "ccbot.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccbot.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1000.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "fish", "normal")
        assert (1, 42) not in _idle_clear_timers


class TestActiveStatusCancelsIdleTimer:
    async def test_status_line_cancels_idle_timer(self) -> None:
        from ccbot.providers.base import StatusUpdate

        from ccbot.handlers.status_polling import update_status_message

        _idle_clear_timers[(1, 42)] = ("@0", 500.0)
        _idle_status_cleared.add("@0")

        pyte_status = StatusUpdate(
            raw_text="Working on task",
            display_label="\u2699\ufe0f working\u2026",
        )
        with (
            patch("ccbot.handlers.status_polling.tmux_manager") as mock_tm,
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch("ccbot.handlers.status_polling.enqueue_status_update"),
            patch("ccbot.handlers.status_polling._send_typing_throttled"),
            patch(
                "ccbot.handlers.status_polling.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccbot.handlers.status_polling.get_provider_for_window",
                return_value=make_mock_provider(has_status=True),
            ),
            patch(
                "ccbot.handlers.status_polling._parse_with_pyte",
                return_value=pyte_status,
            ),
        ):
            mock_window = MagicMock()
            mock_window.window_id = "@0"
            mock_window.window_name = "project"
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="output")
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            mock_sm.get_notification_mode.return_value = "normal"

            bot = AsyncMock()
            await update_status_message(bot, 1, "@0", thread_id=42)

        assert (1, 42) not in _idle_clear_timers
        assert "@0" not in _idle_status_cleared

    async def test_transcript_activity_cancels_idle_timer(self) -> None:
        from ccbot.handlers.status_polling import _handle_no_status

        _idle_clear_timers[(1, 42)] = ("@0", 500.0)
        bot = AsyncMock()
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.update_topic_emoji"),
            patch("ccbot.handlers.status_polling._send_typing_throttled"),
            patch(
                "ccbot.handlers.status_polling._check_transcript_activity",
                return_value=True,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "node", "normal")
        assert (1, 42) not in _idle_clear_timers


class TestProbeFailures:
    async def test_probe_skips_suspended_windows(self) -> None:
        _probe_failures["@5"] = _MAX_PROBE_FAILURES
        bot = AsyncMock()
        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            await _probe_topic_existence(bot)
        bot.unpin_all_forum_topic_messages.assert_not_called()

    async def test_probe_success_resets_counter(self) -> None:
        _probe_failures["@5"] = 2
        bot = AsyncMock()
        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_sm.resolve_chat_id.return_value = -100
            await _probe_topic_existence(bot)
        assert "@5" not in _probe_failures
        bot.unpin_all_forum_topic_messages.assert_called_once_with(
            chat_id=-100, message_thread_id=42
        )

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(TelegramError("Timed out"), id="telegram-error"),
            pytest.param(BadRequest("Permission denied"), id="bad-request-other"),
        ],
    )
    async def test_probe_error_increments_counter(self, exc: TelegramError) -> None:
        bot = AsyncMock()
        bot.unpin_all_forum_topic_messages.side_effect = exc
        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_sm.resolve_chat_id.return_value = -100
            await _probe_topic_existence(bot)
        assert _probe_failures["@5"] == 1

    async def test_probe_suspends_after_max_failures(self) -> None:
        bot = AsyncMock()
        bot.unpin_all_forum_topic_messages.side_effect = TelegramError("Timed out")
        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_sm.resolve_chat_id.return_value = -100
            for _ in range(_MAX_PROBE_FAILURES + 1):
                await _probe_topic_existence(bot)
        assert bot.unpin_all_forum_topic_messages.call_count == _MAX_PROBE_FAILURES
        assert _probe_failures["@5"] == _MAX_PROBE_FAILURES

    @pytest.mark.parametrize(
        "window_alive",
        [
            pytest.param(True, id="window-alive"),
            pytest.param(False, id="window-already-gone"),
        ],
    )
    async def test_topic_deleted_cleans_up(self, window_alive: bool) -> None:
        _probe_failures["@5"] = 1
        bot = AsyncMock()
        bot.unpin_all_forum_topic_messages.side_effect = BadRequest("Topic_id_invalid")
        mock_window = MagicMock()
        mock_window.window_id = "@5"
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch("ccbot.handlers.status_polling.tmux_manager") as mock_tm,
            patch(
                "ccbot.handlers.status_polling.clear_topic_state",
                new_callable=AsyncMock,
            ) as mock_cleanup,
        ):
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(
                return_value=mock_window if window_alive else None
            )
            mock_tm.kill_window = AsyncMock()
            await _probe_topic_existence(bot)
        if window_alive:
            mock_tm.kill_window.assert_called_once_with("@5")
        else:
            mock_tm.kill_window.assert_not_called()
        mock_cleanup.assert_called_once_with(1, 42, bot, window_id="@5")
        mock_sm.unbind_thread.assert_called_once_with(1, 42)
        assert "@5" not in _probe_failures


class TestPruneStaleStatePolling:
    async def test_calls_sync_and_prune(self) -> None:
        mock_win = MagicMock()
        mock_win.window_id = "@1"
        mock_win.window_name = "proj"
        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.sync_display_names.return_value = False
            mock_sm.prune_stale_state.return_value = False
            await _prune_stale_state([mock_win])
        mock_sm.sync_display_names.assert_called_once_with([("@1", "proj")])
        mock_sm.prune_stale_state.assert_called_once_with({"@1"})

    async def test_empty_window_list(self) -> None:
        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.sync_display_names.return_value = False
            mock_sm.prune_stale_state.return_value = False
            await _prune_stale_state([])
        mock_sm.sync_display_names.assert_called_once_with([])
        mock_sm.prune_stale_state.assert_called_once_with(set())


class TestMaybeDiscoverTranscript:
    async def test_skips_when_session_id_already_set(self) -> None:
        from ccbot.handlers.status_polling import _maybe_discover_transcript

        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.window_states = {
                "@7": MagicMock(session_id="existing-id", cwd="/proj")
            }
            await _maybe_discover_transcript("@7")
        # Should not call register
        mock_sm.register_hookless_session.assert_not_called()

    async def test_skips_when_no_cwd(self) -> None:
        from ccbot.handlers.status_polling import _maybe_discover_transcript

        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.window_states = {"@7": MagicMock(session_id="", cwd="")}
            await _maybe_discover_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_skips_when_provider_has_hooks(self) -> None:
        from ccbot.handlers.status_polling import _maybe_discover_transcript

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = True
        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccbot.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
        ):
            mock_sm.window_states = {"@7": MagicMock(session_id="", cwd="/proj")}
            await _maybe_discover_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_skips_when_window_not_tracked(self) -> None:
        from ccbot.handlers.status_polling import _maybe_discover_transcript

        with patch("ccbot.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.window_states = {}
            await _maybe_discover_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_registers_when_transcript_found(self) -> None:
        from ccbot.handlers.status_polling import _maybe_discover_transcript
        from ccbot.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccbot:@7",
        )
        mock_provider.discover_transcript.return_value = event

        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccbot.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccbot.handlers.status_polling.config") as mock_config,
        ):
            mock_sm.window_states = {"@7": MagicMock(session_id="", cwd="/my/project")}
            mock_config.tmux_session_name = "ccbot"
            await _maybe_discover_transcript("@7")

        mock_sm.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            provider_name="codex",
        )
        mock_sm.write_hookless_session_map.assert_called_once_with(
            window_id="@7",
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            provider_name="codex",
        )

    async def test_noop_when_discovery_returns_none(self) -> None:
        from ccbot.handlers.status_polling import _maybe_discover_transcript

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.discover_transcript.return_value = None

        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccbot.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccbot.handlers.status_polling.config") as mock_config,
        ):
            mock_sm.window_states = {"@7": MagicMock(session_id="", cwd="/proj")}
            mock_config.tmux_session_name = "ccbot"
            await _maybe_discover_transcript("@7")

        mock_sm.register_hookless_session.assert_not_called()
        mock_sm.write_hookless_session_map.assert_not_called()

    async def test_session_map_write_runs_in_background_thread(self) -> None:
        """Regression: write_hookless_session_map must run in a thread (flock)."""
        from ccbot.handlers.status_polling import _maybe_discover_transcript
        from ccbot.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccbot:@7",
        )
        mock_provider.discover_transcript.return_value = event

        with (
            patch("ccbot.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccbot.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccbot.handlers.status_polling.config") as mock_config,
            patch("ccbot.handlers.status_polling.asyncio") as mock_asyncio,
        ):
            mock_sm.window_states = {"@7": MagicMock(session_id="", cwd="/my/project")}
            mock_config.tmux_session_name = "ccbot"
            mock_asyncio.to_thread = AsyncMock(side_effect=[event, None])
            await _maybe_discover_transcript("@7")

        # discover_transcript in thread, write_hookless_session_map in thread
        assert mock_asyncio.to_thread.call_count == 2
        discover_call = mock_asyncio.to_thread.call_args_list[0]
        assert discover_call.args[0] == mock_provider.discover_transcript
        write_call = mock_asyncio.to_thread.call_args_list[1]
        assert write_call.args[0] == mock_sm.write_hookless_session_map
        # register_hookless_session called directly on event loop (not via to_thread)
        mock_sm.register_hookless_session.assert_called_once()
