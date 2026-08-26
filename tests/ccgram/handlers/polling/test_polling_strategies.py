"""Tests for polling strategy classes: state management, RC debounce,
autoclose timers, pane alerts, probe failures, and content-hash caching."""

import time
from unittest.mock import patch

import pytest

from ccgram.handlers.polling.polling_state import (
    InteractiveUIStrategy,
    TerminalPollState,
    TerminalScreenBuffer,
    TopicLifecycleStrategy,
    _strip_hook_runner_noise,
)
from ccgram.handlers.polling.polling_types import (
    MAX_PROBE_FAILURES,
    RC_DEBOUNCE_SECONDS,
    STARTUP_TIMEOUT,
    TYPING_INTERVAL,
    TopicPollState,
    WindowPollState,
)


class TestStripHookRunnerNoise:
    def test_drops_hook_runner_log_lines(self):
        pane = (
            "Real agent output line 1\n"
            "2026-05-15 14:12:27 [debug    ] Processing hook event from stdin\n"
            "2026-05-15 14:12:27 [debug    ] tmux key=ccgram:@10766, window_name=g\n"
            "2026-05-15 14:12:27 [info     ] Updated session_map: ccgram:@10766\n"
            "Real agent output line 2"
        )
        result = _strip_hook_runner_noise(pane)
        assert result == "Real agent output line 1\nReal agent output line 2"

    def test_preserves_pane_without_noise(self):
        pane = "❯ ls -la\ntotal 0\ndrwxr-xr-x  2 user  staff  64 May 15 .\n"
        assert _strip_hook_runner_noise(pane) is pane

    def test_does_not_strip_agent_text_with_brackets(self):
        pane = "Build [debug] succeeded\n[info] not a timestamped log line"
        assert _strip_hook_runner_noise(pane) == pane

    def test_strips_with_leading_ansi_color(self):
        pane = "\x1b[0m2026-05-16 12:00:34 [error    ] boom\nkept line"
        assert _strip_hook_runner_noise(pane) == "kept line"


class TestTerminalScreenBuffer:
    def setup_method(self):
        self.poll_state = TerminalPollState()
        self.strategy = TerminalScreenBuffer(self.poll_state)

    def test_clear_screen_buffer(self):
        ws = self.poll_state.get_state("@0")
        ws.last_pane_hash = 12345
        ws.last_rendered_text = "some text"
        self.strategy.clear_screen_buffer("@0")
        assert ws.screen_buffer is None
        assert ws.last_pane_hash is None
        assert ws.last_rendered_text is None

    def test_reset_screen_buffer_state(self):
        ws = self.poll_state.get_state("@0")
        ws.rc_active = True
        ws.last_pane_hash = 999
        self.strategy.reset_screen_buffer_state()
        assert not ws.rc_active
        assert ws.last_pane_hash is None

    def test_is_rc_active_default_false(self):
        assert not self.strategy.is_rc_active("@0")

    def test_is_rc_active_when_set(self):
        ws = self.poll_state.get_state("@0")
        ws.rc_active = True
        assert self.strategy.is_rc_active("@0")

    def test_update_rc_state_on(self):
        ws = WindowPollState()
        self.strategy.update_rc_state(ws, True)
        assert ws.rc_active
        assert ws.rc_off_since is None

    def test_update_rc_state_stays_off_when_never_active(self):
        ws = WindowPollState()
        self.strategy.update_rc_state(ws, False)
        assert not ws.rc_active
        assert ws.rc_off_since is None

    def test_update_rc_state_debounce_start(self):
        ws = WindowPollState(rc_active=True)
        self.strategy.update_rc_state(ws, False)
        assert ws.rc_active
        assert ws.rc_off_since is not None

    def test_update_rc_state_holds_badge_within_debounce(self):
        off_since = time.monotonic() - RC_DEBOUNCE_SECONDS / 2
        ws = WindowPollState(rc_active=True, rc_off_since=off_since)
        self.strategy.update_rc_state(ws, False)
        assert ws.rc_active
        assert ws.rc_off_since == off_since

    def test_update_rc_state_debounce_completes(self):
        ws = WindowPollState(rc_active=True)
        ws.rc_off_since = time.monotonic() - RC_DEBOUNCE_SECONDS - 1
        self.strategy.update_rc_state(ws, False)
        assert not ws.rc_active
        assert ws.rc_off_since is None

    def test_update_rc_state_debounce_reset_on_redetect(self):
        ws = WindowPollState(rc_active=True, rc_off_since=time.monotonic())
        self.strategy.update_rc_state(ws, True)
        assert ws.rc_active
        assert ws.rc_off_since is None

    def _claude_plan_pane(self) -> str:
        sep = "─" * 30
        return (
            "  Would you like to proceed?\n"
            f"  {sep}\n"
            "  Yes     No\n"
            f"  {sep}\n"
            "  ctrl-g to edit in vim\n"
        )

    def test_claude_chrome_default_detects_interactive(self):
        result = self.strategy.parse_with_pyte("@0", self._claude_plan_pane(), 200, 50)
        assert result is not None
        assert result.is_interactive is True

    def test_non_claude_chrome_skips_claude_ui_returns_none(self):
        result = self.strategy.parse_with_pyte(
            "@0", self._claude_plan_pane(), 200, 50, parse_claude_chrome=False
        )
        assert result is None

    def test_non_claude_chrome_still_populates_rendered_text(self):
        self.strategy.parse_with_pyte(
            "@0",
            "Gemini is thinking…\nsome output",
            200,
            50,
            parse_claude_chrome=False,
        )
        assert "Gemini is thinking" in self.strategy.get_rendered_text("@0", "")

    def test_non_claude_chrome_still_updates_rc_state(self):
        sep = "─" * 30
        pane = f"agent output\r\n{sep}\r\n❯ \r\n{sep}\r\n  Remote Control active\r\n"
        result = self.strategy.parse_with_pyte(
            "@0", pane, 80, 10, parse_claude_chrome=False
        )
        assert result is None
        assert self.strategy.is_rc_active("@0")

    def test_parse_with_pyte_content_hash_cache(self):
        ws = self.poll_state.get_state("@0")
        ws.last_pane_hash = hash(("same text", 200, 50))
        ws.last_pyte_result = None
        result = self.strategy.parse_with_pyte("@0", "same text", 200, 50)
        assert result is None

    def test_parse_with_pyte_invalid_dimensions_defaults(self):
        with patch(
            "ccgram.handlers.polling.polling_state.TerminalScreenBuffer.get_screen_buffer"
        ) as mock_buf:
            mock_buf.return_value.rendered_text = ""
            mock_buf.return_value.display = []
            with (
                patch(
                    "ccgram.terminal_parser.detect_remote_control", return_value=False
                ),
                patch("ccgram.terminal_parser.parse_from_screen", return_value=None),
                patch(
                    "ccgram.terminal_parser.parse_status_from_screen", return_value=None
                ),
            ):
                self.strategy.parse_with_pyte("@0", "text", 0, 0)
                mock_buf.assert_called_with("@0", 200, 50)

    @pytest.mark.parametrize(
        ("count", "ttl_offset", "expected"),
        [
            pytest.param(1, 10.0, True, id="fresh-single-pane"),
            pytest.param(3, 10.0, False, id="fresh-multi-pane"),
            pytest.param(1, -1.0, False, id="expired-single-pane"),
        ],
    )
    def test_is_single_pane_cached(self, count, ttl_offset, expected):
        self.poll_state.get_state("@0").pane_count_cache = (
            count,
            time.monotonic() + ttl_offset,
        )
        assert self.strategy.is_single_pane_cached("@0") is expected

    def test_is_single_pane_cached_false_no_cache(self):
        assert not self.strategy.is_single_pane_cached("@0")

    def test_get_rendered_text_returns_cached(self):
        ws = self.poll_state.get_state("@0")
        ws.last_rendered_text = "cached"
        assert self.strategy.get_rendered_text("@0", "fallback") == "cached"

    def test_get_rendered_text_returns_fallback(self):
        assert self.strategy.get_rendered_text("@0", "fallback") == "fallback"


class TestTerminalPollState:
    def setup_method(self):
        self.strategy = TerminalPollState()

    def test_get_state_creates_new(self):
        ws = self.strategy.get_state("@0")
        assert isinstance(ws, WindowPollState)
        assert not ws.has_seen_status

    def test_get_state_returns_same_instance(self):
        ws1 = self.strategy.get_state("@0")
        ws2 = self.strategy.get_state("@0")
        assert ws1 is ws2

    def test_clear_state_removes(self):
        self.strategy.get_state("@0")
        self.strategy.clear_state("@0")
        assert "@0" not in self.strategy._states

    def test_reset_probe_failures(self):
        ws = self.strategy.get_state("@0")
        ws.probe_failures = 5
        self.strategy.reset_probe_failures("@0")
        assert ws.probe_failures == 0

    def test_clear_seen_status(self):
        ws = self.strategy.get_state("@0")
        ws.has_seen_status = True
        ws.startup_time = 123.0
        self.strategy.clear_seen_status("@0")
        assert not ws.has_seen_status
        assert ws.startup_time is None

    def test_set_unbound_timer(self):
        self.strategy.set_unbound_timer("@0", 42.0)
        ws = self.strategy.get_state("@0")
        assert ws.unbound_timer == 42.0

    def test_clear_unbound_timer(self):
        ws = self.strategy.get_state("@0")
        ws.unbound_timer = 42.0
        self.strategy.clear_unbound_timer("@0")
        assert ws.unbound_timer is None

    def test_reset_all_probe_failures(self):
        self.strategy.get_state("@0").probe_failures = 3
        self.strategy.get_state("@1").probe_failures = 7
        self.strategy.reset_all_probe_failures()
        assert self.strategy.get_state("@0").probe_failures == 0
        assert self.strategy.get_state("@1").probe_failures == 0

    def test_reset_all_seen_status(self):
        self.strategy.get_state("@0").has_seen_status = True
        self.strategy.get_state("@0").startup_time = 1.0
        self.strategy.get_state("@1").has_seen_status = True
        self.strategy.reset_all_seen_status()
        assert not self.strategy.get_state("@0").has_seen_status
        assert self.strategy.get_state("@0").startup_time is None
        assert not self.strategy.get_state("@1").has_seen_status

    def test_reset_all_unbound_timers(self):
        self.strategy.get_state("@0").unbound_timer = 1.0
        self.strategy.get_state("@1").unbound_timer = 2.0
        self.strategy.reset_all_unbound_timers()
        assert self.strategy.get_state("@0").unbound_timer is None
        assert self.strategy.get_state("@1").unbound_timer is None

    def test_mark_seen_status(self):
        ws = self.strategy.get_state("@0")
        ws.startup_time = 123.0
        assert not ws.has_seen_status
        self.strategy.mark_seen_status("@0")
        assert ws.has_seen_status
        assert ws.startup_time is None

    def test_mark_seen_status_creates_state(self):
        self.strategy.mark_seen_status("@new")
        ws = self.strategy.get_state("@new")
        assert ws.has_seen_status
        assert ws.startup_time is None

    def test_is_recently_active_true(self):
        now = time.monotonic()
        assert self.strategy.is_recently_active("@0", now - 1.0)
        assert self.strategy.get_state("@0").has_seen_status

    def test_is_recently_active_false_when_stale(self):
        assert not self.strategy.is_recently_active("@0", time.monotonic() - 60.0)

    def test_is_recently_active_false_when_none(self):
        assert not self.strategy.is_recently_active("@0", None)

    def test_is_startup_expired_true(self):
        ws = self.strategy.get_state("@0")
        ws.startup_time = time.monotonic() - STARTUP_TIMEOUT - 1.0
        assert self.strategy.is_startup_expired("@0")

    def test_is_startup_expired_false_within_grace(self):
        ws = self.strategy.get_state("@0")
        ws.startup_time = time.monotonic()
        assert not self.strategy.is_startup_expired("@0")

    def test_is_startup_expired_false_no_startup(self):
        assert not self.strategy.is_startup_expired("@0")

    @pytest.mark.parametrize(
        "clear",
        [
            "clear_state",
            "reset_probe_failures",
            "clear_seen_status",
            "clear_unbound_timer",
        ],
    )
    def test_clearing_an_unknown_window_does_not_materialise_state(self, clear):
        getattr(self.strategy, clear)("@999")
        assert "@999" not in self.strategy._states


class TestInteractiveUIStrategy:
    def setup_method(self):
        self.poll_state = TerminalPollState()
        self.screen_buffer = TerminalScreenBuffer(self.poll_state)
        self.strategy = InteractiveUIStrategy()

    def test_has_pane_alert_false_by_default(self):
        assert not self.strategy.has_pane_alert("%0")

    def test_has_pane_alert_true_when_set(self):
        self.strategy._pane_alert_hashes["%0"] = ("prompt", 0.0, "@0")
        assert self.strategy.has_pane_alert("%0")

    def test_clear_pane_alerts_for_window(self):
        self.strategy._pane_alert_hashes["%0"] = ("prompt", 0.0, "@0")
        self.strategy._pane_alert_hashes["%1"] = ("prompt", 0.0, "@0")
        self.strategy._pane_alert_hashes["%2"] = ("prompt", 0.0, "@1")
        self.strategy.clear_pane_alerts("@0")
        assert "%0" not in self.strategy._pane_alert_hashes
        assert "%1" not in self.strategy._pane_alert_hashes
        assert "%2" in self.strategy._pane_alert_hashes

    def test_clear_pane_alerts_empty_is_noop(self):
        self.strategy.clear_pane_alerts("@0")


class TestTopicLifecycleStrategy:
    def setup_method(self):
        self.poll_state = TerminalPollState()
        self.strategy = TopicLifecycleStrategy(self.poll_state)

    def test_get_state_creates_new(self):
        ts = self.strategy.get_state(1, 42)
        assert isinstance(ts, TopicPollState)
        assert ts.autoclose is None

    def test_clear_state(self):
        self.strategy.get_state(1, 42)
        self.strategy.clear_state(1, 42)
        assert (1, 42) not in self.strategy._states

    def test_start_autoclose_timer(self):
        self.strategy.start_autoclose_timer(1, 42, "done", 100.0)
        ts = self.strategy.get_state(1, 42)
        assert ts.autoclose == ("done", 100.0)

    def test_start_autoclose_timer_does_not_overwrite_same_state(self):
        self.strategy.start_autoclose_timer(1, 42, "done", 100.0)
        self.strategy.start_autoclose_timer(1, 42, "done", 200.0)
        ts = self.strategy.get_state(1, 42)
        assert ts.autoclose == ("done", 100.0)

    def test_start_autoclose_timer_overwrites_different_state(self):
        self.strategy.start_autoclose_timer(1, 42, "done", 100.0)
        self.strategy.start_autoclose_timer(1, 42, "dead", 200.0)
        ts = self.strategy.get_state(1, 42)
        assert ts.autoclose == ("dead", 200.0)

    def test_clear_autoclose_timer_when_active(self):
        self.strategy.start_autoclose_timer(1, 42, "done", 100.0)
        self.strategy.clear_autoclose_timer(1, 42)
        ts = self.strategy.get_state(1, 42)
        assert ts.autoclose is None

    def test_clear_autoclose_timer_nonexistent(self):
        self.strategy.clear_autoclose_timer(1, 42)

    def test_clear_dead_notification(self):
        self.strategy._dead_notified.add((1, 42, "@0"))
        self.strategy._dead_notified.add((1, 42, "@1"))
        self.strategy._dead_notified.add((2, 43, "@0"))
        self.strategy.clear_dead_notification(1, 42)
        assert (1, 42, "@0") not in self.strategy._dead_notified
        assert (1, 42, "@1") not in self.strategy._dead_notified
        assert (2, 43, "@0") in self.strategy._dead_notified

    def test_reset_dead_notification_state(self):
        self.strategy._dead_notified.add((1, 42, "@0"))
        self.strategy.reset_dead_notification_state()
        assert len(self.strategy._dead_notified) == 0

    def test_clear_probe_failures(self):
        ws = self.poll_state.get_state("@0")
        ws.probe_failures = 5
        self.strategy.clear_probe_failures("@0")
        assert ws.probe_failures == 0

    def test_record_probe_failure_increments(self):
        count = self.strategy.record_probe_failure("@0")
        assert count == 1
        count = self.strategy.record_probe_failure("@0")
        assert count == 2

    def test_record_probe_failure_logs_at_threshold(self):
        for _ in range(MAX_PROBE_FAILURES - 1):
            self.strategy.record_probe_failure("@0")
        with patch("ccgram.handlers.polling.polling_state.logger") as mock_logger:
            self.strategy.record_probe_failure("@0")
            mock_logger.info.assert_called_once()

    def test_clear_typing_state(self):
        ts = self.strategy.get_state(1, 42)
        ts.last_typing_sent = 123.0
        self.strategy.clear_typing_state(1, 42)
        assert ts.last_typing_sent is None

    def test_reset_autoclose_state_clears_all(self):
        self.strategy.start_autoclose_timer(1, 42, "done", 100.0)
        self.strategy.start_autoclose_timer(2, 43, "dead", 200.0)
        ws = self.poll_state.get_state("@0")
        ws.unbound_timer = 50.0
        self.strategy.reset_autoclose_state()
        for ts in self.strategy._states.values():
            assert ts.autoclose is None
        assert ws.unbound_timer is None

    def test_is_typing_throttled_true(self):
        ts = self.strategy.get_state(1, 42)
        ts.last_typing_sent = time.monotonic()
        assert self.strategy.is_typing_throttled(1, 42)

    def test_is_typing_throttled_false_past_interval(self):
        ts = self.strategy.get_state(1, 42)
        ts.last_typing_sent = time.monotonic() - TYPING_INTERVAL - 1.0
        assert not self.strategy.is_typing_throttled(1, 42)

    def test_is_typing_throttled_false_no_state(self):
        assert not self.strategy.is_typing_throttled(1, 42)

    def test_should_skip_probe_true(self):
        ws = self.poll_state.get_state("@0")
        ws.probe_failures = MAX_PROBE_FAILURES
        assert self.strategy.should_skip_probe("@0")

    def test_should_skip_probe_false(self):
        ws = self.poll_state.get_state("@0")
        ws.probe_failures = MAX_PROBE_FAILURES - 1
        assert not self.strategy.should_skip_probe("@0")
