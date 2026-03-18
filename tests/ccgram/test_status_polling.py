"""Tests for status polling: shell detection, autoclose timers, rename sync,
activity heuristic, and startup timeout."""

import time

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot
from telegram.error import BadRequest, TelegramError

from conftest import make_mock_provider

from ccgram.handlers.status_polling import (
    _check_autoclose_timers,
    _check_transcript_activity,
    _clear_autoclose_if_active,
    _dead_notified,
    _get_window_state,
    _handle_dead_window_notification,
    _MAX_PROBE_FAILURES,
    _probe_topic_existence,
    _prune_stale_state,
    _start_autoclose_timer,
    _topic_poll_state,
    _window_poll_state,
    clear_autoclose_timer,
    is_shell_prompt,
)


# Helpers for readable assertions on dataclass-based state
def _has_autoclose(user_id: int, thread_id: int) -> bool:
    ts = _topic_poll_state.get((user_id, thread_id))
    return ts is not None and ts.autoclose is not None


def _get_autoclose(user_id: int, thread_id: int) -> tuple[str, float] | None:
    ts = _topic_poll_state.get((user_id, thread_id))
    return ts.autoclose if ts else None


@pytest.fixture(autouse=True)
def _reset():
    _window_poll_state.clear()
    _topic_poll_state.clear()
    _dead_notified.clear()
    yield
    _window_poll_state.clear()
    _topic_poll_state.clear()
    _dead_notified.clear()


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
        assert _get_autoclose(1, 42) == ("done", 100.0)

    def test_start_timer_preserves_existing_same_state(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _start_autoclose_timer(1, 42, "done", 200.0)
        assert _get_autoclose(1, 42) == ("done", 100.0)

    def test_start_timer_resets_on_state_change(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _start_autoclose_timer(1, 42, "dead", 200.0)
        assert _get_autoclose(1, 42) == ("dead", 200.0)

    def test_clear_on_active(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _clear_autoclose_if_active(1, 42)
        assert not _has_autoclose(1, 42)

    def test_clear_timer(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        clear_autoclose_timer(1, 42)
        assert not _has_autoclose(1, 42)

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
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = minutes
            mock_time.monotonic.return_value = elapsed
            mock_sm.resolve_chat_id.return_value = -100
            await _check_autoclose_timers(bot)
        bot.close_forum_topic.assert_called_once_with(
            chat_id=-100, message_thread_id=42
        )
        assert not _has_autoclose(1, 42)

    async def test_check_not_expired_yet(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = 10
            mock_time.monotonic.return_value = 29 * 60
            await _check_autoclose_timers(bot)
        bot.close_forum_topic.assert_not_called()
        assert _has_autoclose(1, 42)

    async def test_check_disabled_when_zero(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 0
            mock_config.autoclose_dead_minutes = 0
            mock_time.monotonic.return_value = 999999
            await _check_autoclose_timers(bot)
        bot.close_forum_topic.assert_not_called()

    async def test_check_telegram_error_handled(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock(spec=Bot)
        bot.close_forum_topic.side_effect = TelegramError("fail")
        with (
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = 10
            mock_time.monotonic.return_value = 30 * 60 + 1
            mock_sm.resolve_chat_id.return_value = -100
            await _check_autoclose_timers(bot)
        assert not _has_autoclose(1, 42)


class TestTranscriptActivityHeuristic:
    def test_active_when_recent_transcript(self) -> None:
        now = time.monotonic()
        mock_monitor = MagicMock()
        mock_monitor.get_last_activity.return_value = now - 5.0
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_active_monitor",
                return_value=mock_monitor,
            ),
        ):
            mock_sm.get_session_id_for_window.return_value = "sess-123"
            result = _check_transcript_activity("@0", now)
        assert result is True
        assert _window_poll_state.get("@0") and _window_poll_state["@0"].has_seen_status

    def test_inactive_when_stale_transcript(self) -> None:
        now = time.monotonic()
        mock_monitor = MagicMock()
        mock_monitor.get_last_activity.return_value = now - 20.0
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_active_monitor",
                return_value=mock_monitor,
            ),
        ):
            mock_sm.get_session_id_for_window.return_value = "sess-123"
            result = _check_transcript_activity("@0", now)
        assert result is False
        assert not (
            _window_poll_state.get("@0") and _window_poll_state["@0"].has_seen_status
        )

    def test_inactive_when_no_session(self) -> None:
        now = time.monotonic()
        with patch("ccgram.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.get_session_id_for_window.return_value = None
            result = _check_transcript_activity("@0", now)
        assert result is False

    def test_inactive_when_no_monitor(self) -> None:
        now = time.monotonic()
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_active_monitor",
                return_value=None,
            ),
        ):
            mock_sm.get_session_id_for_window.return_value = "sess-123"
            result = _check_transcript_activity("@0", now)
        assert result is False

    def test_clears_startup_timer_on_activity(self) -> None:
        now = time.monotonic()

        _get_window_state("@0").startup_time = now - 15.0
        mock_monitor = MagicMock()
        mock_monitor.get_last_activity.return_value = now - 3.0
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_active_monitor",
                return_value=mock_monitor,
            ),
        ):
            mock_sm.get_session_id_for_window.return_value = "sess-123"
            result = _check_transcript_activity("@0", now)
        assert result is True
        assert (
            _window_poll_state.get("@0") is None
            or _window_poll_state["@0"].startup_time is None
        )


class TestStartupTimeout:
    async def test_first_poll_records_startup_time(self) -> None:
        from ccgram.handlers.status_polling import _handle_no_status

        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.update_topic_emoji"),
            patch("ccgram.handlers.status_polling._send_typing_throttled"),
            patch(
                "ccgram.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1000.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "node", "normal")
        assert (
            _window_poll_state.get("@0") is not None
            and _window_poll_state["@0"].startup_time is not None
        )

    async def test_startup_timeout_transitions_to_idle(self) -> None:
        from ccgram.handlers.status_polling import _handle_no_status

        bot = AsyncMock(spec=Bot)
        _get_window_state("@0").startup_time = 1000.0
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.update_topic_emoji") as mock_emoji,
            patch("ccgram.handlers.status_polling.enqueue_status_update"),
            patch(
                "ccgram.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1000.0 + 31.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "node", "normal")
        assert _window_poll_state.get("@0") and _window_poll_state["@0"].has_seen_status
        assert (
            _window_poll_state.get("@0") is None
            or _window_poll_state["@0"].startup_time is None
        )
        mock_emoji.assert_called_once_with(bot, -100, 42, "idle", "project")

    async def test_startup_grace_period_sends_typing(self) -> None:
        from ccgram.handlers.status_polling import _handle_no_status

        bot = AsyncMock(spec=Bot)
        _get_window_state("@0").startup_time = 1000.0
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.update_topic_emoji") as mock_emoji,
            patch(
                "ccgram.handlers.status_polling._send_typing_throttled"
            ) as mock_typing,
            patch(
                "ccgram.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1010.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "node", "normal")
        mock_typing.assert_called_once_with(bot, 1, 42)
        mock_emoji.assert_called_once_with(bot, -100, 42, "active", "project")
        assert not (
            _window_poll_state.get("@0") and _window_poll_state["@0"].has_seen_status
        )


class TestParseWithPyte:
    """Tests for pyte-based screen parsing integration."""

    def setup_method(self) -> None:
        from ccgram.handlers.status_polling import reset_screen_buffer_state

        reset_screen_buffer_state()

    def teardown_method(self) -> None:
        from ccgram.handlers.status_polling import reset_screen_buffer_state

        reset_screen_buffer_state()

    def test_detects_spinner_status(self) -> None:
        from ccgram.handlers.status_polling import _parse_with_pyte

        sep = "─" * 30
        pane_text = f"Some output\n✻ Reading file src/main.py\n{sep}\n"
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.raw_text == "Reading file src/main.py"
        assert result.display_label == "\U0001f4d6 reading\u2026"
        assert result.is_interactive is False

    def test_detects_braille_spinner(self) -> None:
        from ccgram.handlers.status_polling import _parse_with_pyte

        sep = "─" * 30
        pane_text = f"Output\n⠋ Thinking about things\n{sep}\n"
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.raw_text == "Thinking about things"
        assert result.is_interactive is False

    def test_detects_interactive_ui(self) -> None:
        from ccgram.handlers.status_polling import _parse_with_pyte

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
        from ccgram.handlers.status_polling import _parse_with_pyte

        pane_text = "$ echo hello\nhello\n$\n"
        result = _parse_with_pyte("@0", pane_text)
        assert result is None

    def test_screen_buffer_cached_per_window(self) -> None:
        from ccgram.handlers.status_polling import _parse_with_pyte

        sep = "─" * 30
        pane_text = f"Output\n✻ Working\n{sep}\n"
        _parse_with_pyte("@0", pane_text)
        assert _window_poll_state.get("@0") is not None
        assert _window_poll_state["@0"].screen_buffer is not None

        _parse_with_pyte("@1", pane_text)
        assert _window_poll_state.get("@1") is not None
        assert _window_poll_state["@1"].screen_buffer is not None
        assert _window_poll_state["@0"].screen_buffer is not None

    def test_interactive_takes_precedence_over_status(self) -> None:
        from ccgram.handlers.status_polling import _parse_with_pyte

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
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tm,
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.update_topic_emoji"),
            patch("ccgram.handlers.status_polling.enqueue_status_update"),
            patch(
                "ccgram.handlers.status_polling.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=make_mock_provider(has_status=True),
            ) as mock_get_provider,
            patch(
                "ccgram.handlers.status_polling._parse_with_pyte",
                return_value=None,
            ),
        ):
            from ccgram.handlers.status_polling import update_status_message

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

            bot = AsyncMock(spec=Bot)
            await update_status_message(bot, 1, "@0", thread_id=42)

            # Provider regex parsing was called as fallback
            mock_get_provider.return_value.parse_terminal_status.assert_called_once()

    async def test_uses_pyte_result_when_available(self) -> None:
        from ccgram.providers.base import StatusUpdate

        pyte_status = StatusUpdate(
            raw_text="Reading file",
            display_label="\U0001f4d6 reading\u2026",
        )
        with (
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tm,
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccgram.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.status_polling.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=make_mock_provider(has_status=True),
            ) as mock_get_provider,
            patch(
                "ccgram.handlers.status_polling._parse_with_pyte",
                return_value=pyte_status,
            ),
        ):
            from ccgram.handlers.status_polling import update_status_message

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

            bot = AsyncMock(spec=Bot)
            await update_status_message(bot, 1, "@0", thread_id=42)

            # Provider regex parsing was NOT called (pyte succeeded)
            mock_get_provider.return_value.parse_terminal_status.assert_not_called()
            # Status was enqueued using pyte result
            mock_enqueue.assert_called_once()
            call_args = mock_enqueue.call_args
            assert call_args[0][3] == "\U0001f4d6 reading\u2026"


class TestClearSeenStatus:
    def test_clears_seen_status_and_startup(self) -> None:
        from ccgram.handlers.status_polling import clear_seen_status

        _get_window_state("@0").has_seen_status = True
        _get_window_state("@0").startup_time = 100.0
        clear_seen_status("@0")
        assert not (
            _window_poll_state.get("@0") and _window_poll_state["@0"].has_seen_status
        )
        assert (
            _window_poll_state.get("@0") is None
            or _window_poll_state["@0"].startup_time is None
        )


class TestTransitionToIdle:
    async def test_sends_idle_text(self) -> None:
        from ccgram.handlers.callback_data import IDLE_STATUS_TEXT
        from ccgram.handlers.status_polling import _transition_to_idle

        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccgram.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 100.0
            await _transition_to_idle(bot, 1, "@0", 42, -100, "project", "normal")
        mock_enqueue.assert_called_once()
        assert mock_enqueue.call_args[0][3] == IDLE_STATUS_TEXT
        assert mock_enqueue.call_args[1]["thread_id"] == 42

    @pytest.mark.parametrize("mode", ["muted", "errors_only"])
    async def test_suppressed_mode_clears_status_no_timer(self, mode: str) -> None:
        from ccgram.handlers.status_polling import _transition_to_idle

        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccgram.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
        ):
            await _transition_to_idle(bot, 1, "@0", 42, -100, "project", mode)
        mock_enqueue.assert_called_once_with(bot, 1, "@0", None, thread_id=42)


class TestShellPromptClearsStatus:
    async def test_shell_prompt_enqueues_status_clear(self) -> None:
        from ccgram.handlers.status_polling import _handle_no_status

        _get_window_state("@0").has_seen_status = True
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccgram.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1000.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            await _handle_no_status(bot, 1, "@0", 42, "bash", "normal")
        mock_enqueue.assert_called_once_with(bot, 1, "@0", None, thread_id=42)

    async def test_hookless_shell_prompt_keeps_idle_status(self) -> None:
        from ccgram.handlers.callback_data import IDLE_STATUS_TEXT
        from ccgram.handlers.status_polling import _handle_no_status

        _get_window_state("@0").has_seen_status = True
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.update_topic_emoji"),
            patch(
                "ccgram.handlers.status_polling.enqueue_status_update"
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.status_polling._check_transcript_activity",
                return_value=False,
            ),
            patch("ccgram.handlers.status_polling.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 1000.0
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "project"
            mock_sm.get_window_state.return_value = MagicMock(provider_name="codex")
            await _handle_no_status(bot, 1, "@0", 42, "bash", "normal")

        mock_enqueue.assert_called_once_with(
            bot, 1, "@0", IDLE_STATUS_TEXT, thread_id=42
        )
        assert not _has_autoclose(1, 42)


class TestProbeFailures:
    async def test_probe_skips_suspended_windows(self) -> None:
        _get_window_state("@5").probe_failures = _MAX_PROBE_FAILURES
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            await _probe_topic_existence(bot)
        bot.unpin_all_forum_topic_messages.assert_not_called()

    async def test_probe_success_resets_counter(self) -> None:
        _get_window_state("@5").probe_failures = 2
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_sm.resolve_chat_id.return_value = -100
            await _probe_topic_existence(bot)
        assert (
            _window_poll_state.get("@5") is None
            or _window_poll_state["@5"].probe_failures == 0
        )
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
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages.side_effect = exc
        with patch("ccgram.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_sm.resolve_chat_id.return_value = -100
            await _probe_topic_existence(bot)
        assert _window_poll_state["@5"].probe_failures == 1

    async def test_probe_suspends_after_max_failures(self) -> None:
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages.side_effect = TelegramError("Timed out")
        with patch("ccgram.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_sm.resolve_chat_id.return_value = -100
            for _ in range(_MAX_PROBE_FAILURES + 1):
                await _probe_topic_existence(bot)
        assert bot.unpin_all_forum_topic_messages.call_count == _MAX_PROBE_FAILURES
        assert _window_poll_state["@5"].probe_failures == _MAX_PROBE_FAILURES

    @pytest.mark.parametrize(
        "window_alive",
        [
            pytest.param(True, id="window-alive"),
            pytest.param(False, id="window-already-gone"),
        ],
    )
    async def test_topic_deleted_cleans_up(self, window_alive: bool) -> None:
        _get_window_state("@5").probe_failures = 1
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages.side_effect = BadRequest("Topic_id_invalid")
        mock_window = MagicMock()
        mock_window.window_id = "@5"
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.status_polling.clear_topic_state",
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
        assert (
            _window_poll_state.get("@5") is None
            or _window_poll_state["@5"].probe_failures == 0
        )


class TestPruneStaleStatePolling:
    async def test_calls_sync_and_prune(self) -> None:
        mock_win = MagicMock()
        mock_win.window_id = "@1"
        mock_win.window_name = "proj"
        with patch("ccgram.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.sync_display_names.return_value = False
            mock_sm.prune_stale_state.return_value = False
            await _prune_stale_state([mock_win])
        mock_sm.sync_display_names.assert_called_once_with([("@1", "proj")])
        mock_sm.prune_stale_state.assert_called_once_with({"@1"})

    async def test_empty_window_list(self) -> None:
        with patch("ccgram.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.sync_display_names.return_value = False
            mock_sm.prune_stale_state.return_value = False
            await _prune_stale_state([])
        mock_sm.sync_display_names.assert_called_once_with([])
        mock_sm.prune_stale_state.assert_called_once_with(set())


class TestMaybeDiscoverTranscript:
    async def test_noop_when_discovered_session_matches_current(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        mock_provider.discover_transcript.return_value = SessionStartEvent(
            session_id="existing-id",
            cwd="/proj",
            transcript_path="/path/existing.jsonl",
            window_key="ccgram:@7",
        )

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
            patch("ccgram.handlers.status_polling.config") as mock_config,
        ):
            mock_sm.window_states = {
                "@7": MagicMock(
                    session_id="existing-id",
                    cwd="/proj",
                    transcript_path="/path/existing.jsonl",
                    provider_name="codex",
                )
            }
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_config.tmux_session_name = "ccgram"
            await _maybe_discover_transcript("@7")

        mock_sm.register_hookless_session.assert_not_called()
        mock_sm.write_hookless_session_map.assert_not_called()

    async def test_skips_when_no_cwd_and_no_tmux_window(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
        ):
            mock_sm.window_states = {"@7": MagicMock(session_id="", cwd="")}
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            await _maybe_discover_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_falls_back_to_tmux_cwd_when_state_cwd_empty(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-xyz",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        mock_state = MagicMock(session_id="", cwd="", provider_name="codex")
        mock_window = MagicMock(cwd="/my/project", pane_current_command="bun")

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.status_polling.config") as mock_config,
        ):
            mock_sm.window_states = {"@7": mock_state}
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_config.tmux_session_name = "ccgram"
            await _maybe_discover_transcript("@7")

        mock_sm.set_window_provider.assert_called_once_with(
            "@7", "codex", cwd="/my/project"
        )
        mock_sm.register_hookless_session.assert_called_once()

    async def test_skips_when_provider_has_hooks(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = True
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
        ):
            mock_sm.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="claude")
            }
            await _maybe_discover_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_skips_when_window_not_tracked(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript

        with patch("ccgram.handlers.status_polling.session_manager") as mock_sm:
            mock_sm.window_states = {}
            await _maybe_discover_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_registers_when_transcript_found(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
        ):
            mock_sm.window_states = {
                "@7": MagicMock(session_id="", cwd="/my/project", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_window = MagicMock(pane_current_command="bun")
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
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

    async def test_updates_when_new_session_discovered_for_same_window(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-new",
            cwd="/my/project",
            transcript_path="/path/to/new.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
        ):
            mock_sm.window_states = {
                "@7": MagicMock(
                    session_id="uuid-old",
                    cwd="/my/project",
                    transcript_path="/path/to/old.jsonl",
                    provider_name="codex",
                )
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await _maybe_discover_transcript("@7")

        mock_sm.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="uuid-new",
            cwd="/my/project",
            transcript_path="/path/to/new.jsonl",
            provider_name="codex",
        )

    async def test_noop_when_discovery_returns_none(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.discover_transcript.return_value = None

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
        ):
            mock_sm.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await _maybe_discover_transcript("@7")

        mock_sm.register_hookless_session.assert_not_called()
        mock_sm.write_hookless_session_map.assert_not_called()

    async def test_session_map_write_runs_in_background_thread(self) -> None:
        """Regression: write_hookless_session_map must run in a thread (flock)."""
        from ccgram.handlers.status_polling import _maybe_discover_transcript
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.asyncio") as mock_asyncio,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
        ):
            mock_sm.window_states = {
                "@7": MagicMock(session_id="", cwd="/my/project", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_window = MagicMock(pane_current_command="bun")
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_asyncio.to_thread = AsyncMock(side_effect=[event, None])
            await _maybe_discover_transcript("@7")

        # discover_transcript in thread, write_hookless_session_map in thread
        assert mock_asyncio.to_thread.call_count == 2
        discover_call = mock_asyncio.to_thread.call_args_list[0]
        assert discover_call.args[0] == mock_provider.discover_transcript
        write_call = mock_asyncio.to_thread.call_args_list[1]
        assert write_call.args[0] == mock_sm.write_hookless_session_map
        mock_sm.register_hookless_session.assert_called_once()

    async def test_tries_hookless_providers_when_provider_name_empty(self) -> None:
        """When provider_name is empty (detection failed), try all hookless providers."""
        from ccgram.handlers.status_polling import _maybe_discover_transcript
        from ccgram.providers.base import SessionStartEvent

        event = SessionStartEvent(
            session_id="uuid-found",
            cwd="/proj",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )

        mock_codex = MagicMock()
        mock_codex.capabilities.supports_hook = False
        mock_codex.capabilities.name = "codex"
        mock_codex.discover_transcript.return_value = event

        mock_gemini = MagicMock()
        mock_gemini.capabilities.supports_hook = False
        mock_gemini.capabilities.name = "gemini"
        mock_gemini.discover_transcript.return_value = None

        mock_claude = MagicMock()
        mock_claude.capabilities.supports_hook = True
        mock_claude.capabilities.name = "claude"

        mock_registry = MagicMock()
        mock_registry.provider_names.return_value = ["claude", "codex", "gemini"]

        def mock_get(name: str) -> MagicMock:
            return {"claude": mock_claude, "codex": mock_codex, "gemini": mock_gemini}[
                name
            ]

        mock_registry.get = mock_get

        mock_window = MagicMock(pane_current_command="bun")

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.providers.registry", mock_registry),
        ):
            mock_sm.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await _maybe_discover_transcript("@7")

        mock_sm.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="uuid-found",
            cwd="/proj",
            transcript_path="/path/to/transcript.jsonl",
            provider_name="codex",
        )

    async def test_skips_hookless_fallback_when_pane_is_shell(self) -> None:
        """When provider_name is empty and pane is a shell, skip discovery."""
        from ccgram.handlers.status_polling import _maybe_discover_transcript

        mock_window = MagicMock(pane_current_command="bash")

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
        ):
            mock_sm.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="")
            }
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            await _maybe_discover_transcript("@7")

        mock_sm.register_hookless_session.assert_not_called()

    async def test_passes_max_age_zero_when_pane_is_alive(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        mock_provider.discover_transcript.return_value = None

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
            patch("ccgram.handlers.status_polling.asyncio") as mock_asyncio,
        ):
            mock_sm.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_asyncio.to_thread = AsyncMock(return_value=None)
            await _maybe_discover_transcript("@7")

        discover_call = mock_asyncio.to_thread.call_args_list[0]
        assert discover_call.args[0] == mock_provider.discover_transcript
        assert discover_call.kwargs["max_age"] == 0

    async def test_passes_max_age_none_when_pane_not_alive(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.name = "codex"
        mock_provider.discover_transcript.return_value = None

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
            patch("ccgram.handlers.status_polling.asyncio") as mock_asyncio,
        ):
            mock_sm.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            mock_asyncio.to_thread = AsyncMock(return_value=None)
            await _maybe_discover_transcript("@7")

        discover_call = mock_asyncio.to_thread.call_args_list[0]
        assert discover_call.args[0] == mock_provider.discover_transcript
        assert discover_call.kwargs["max_age"] is None

    async def test_rebinds_stale_codex_window_to_gemini_from_pane_title(self) -> None:
        from ccgram.handlers.status_polling import _maybe_discover_transcript
        from ccgram.providers.base import SessionStartEvent

        mock_codex = MagicMock()
        mock_codex.capabilities.supports_hook = False
        mock_codex.capabilities.name = "codex"
        mock_codex.discover_transcript.return_value = None

        gemini_event = SessionStartEvent(
            session_id="gemini-uuid",
            cwd="/Users/alexei/Workspace/ccgram",
            transcript_path="/Users/alexei/.gemini/tmp/ccgram/chats/session.json",
            window_key="ccgram:@7",
        )
        mock_gemini = MagicMock()
        mock_gemini.capabilities.supports_hook = False
        mock_gemini.capabilities.name = "gemini"
        mock_gemini.discover_transcript.return_value = gemini_event

        mock_state = MagicMock(
            session_id="old-codex-id",
            cwd="/Users/alexei",
            transcript_path="/Users/alexei/.codex/sessions/old.jsonl",
            provider_name="codex",
        )

        def _provider_for_window(_: str) -> MagicMock:
            if mock_state.provider_name == "gemini":
                return mock_gemini
            return mock_codex

        def _set_window_provider(
            window_id: str, provider_name: str, *, cwd: str | None = None
        ) -> None:
            assert window_id == "@7"
            mock_state.provider_name = provider_name
            if cwd:
                mock_state.cwd = cwd

        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.get_provider_for_window",
                side_effect=_provider_for_window,
            ),
            patch(
                "ccgram.handlers.status_polling.detect_provider_from_command",
                return_value="",
            ),
            patch("ccgram.handlers.status_polling.config") as mock_config,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tmux,
        ):
            mock_sm.window_states = {"@7": mock_state}
            mock_sm.set_window_provider.side_effect = _set_window_provider
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(
                    pane_current_command="bun",
                    cwd="/Users/alexei/Workspace/ccgram",
                )
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="◇  Ready (ccbot)")
            mock_config.tmux_session_name = "ccgram"
            await _maybe_discover_transcript("@7")

        mock_codex.discover_transcript.assert_not_called()
        mock_gemini.discover_transcript.assert_called_once()
        mock_sm.set_window_provider.assert_called_once_with(
            "@7",
            "gemini",
            cwd="/Users/alexei/Workspace/ccgram",
        )
        mock_sm.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="gemini-uuid",
            cwd="/Users/alexei/Workspace/ccgram",
            transcript_path="/Users/alexei/.gemini/tmp/ccgram/chats/session.json",
            provider_name="gemini",
        )


class TestDeadWindowNotification:
    async def test_marks_notified_even_when_send_fails(self) -> None:
        """When rate_limit_send_message returns None (send fails),
        _dead_notified is still populated so we don't retry forever."""
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "ccgram.handlers.status_polling.update_topic_emoji",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.status_polling.build_recovery_keyboard",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.status_polling.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "test"
            mock_sm.get_window_state.return_value = MagicMock(cwd="/proj")
            await _handle_dead_window_notification(bot, 1, 42, "@5")

        assert (1, 42, "@5") in _dead_notified

    async def test_no_retry_after_failed_send(self) -> None:
        """Second call is a no-op because _dead_notified was set on first call."""
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.status_polling.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_send,
            patch(
                "ccgram.handlers.status_polling.update_topic_emoji",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.status_polling.build_recovery_keyboard",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.status_polling.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_display_name.return_value = "test"
            mock_sm.get_window_state.return_value = MagicMock(cwd="/proj")
            await _handle_dead_window_notification(bot, 1, 42, "@5")
            await _handle_dead_window_notification(bot, 1, 42, "@5")

        mock_send.assert_called_once()

    @pytest.mark.parametrize(
        "error_msg",
        [
            pytest.param("Message thread not found", id="capitalized"),
            pytest.param("message thread not found", id="lowercase"),
            pytest.param("Bad Request: Thread not found", id="thread-variant"),
        ],
    )
    async def test_probe_cleans_up_on_thread_not_found(self, error_msg: str) -> None:
        """Probe handles thread-not-found variants by cleaning up the binding."""
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages.side_effect = BadRequest(error_msg)
        mock_window = MagicMock()
        mock_window.window_id = "@5"
        with (
            patch("ccgram.handlers.status_polling.session_manager") as mock_sm,
            patch("ccgram.handlers.status_polling.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.status_polling.clear_topic_state",
                new_callable=AsyncMock,
            ) as mock_cleanup,
        ):
            mock_sm.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_sm.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.kill_window = AsyncMock()
            await _probe_topic_existence(bot)

        mock_tm.kill_window.assert_called_once_with("@5")
        mock_cleanup.assert_called_once_with(1, 42, bot, window_id="@5")
        mock_sm.unbind_thread.assert_called_once_with(1, 42)
