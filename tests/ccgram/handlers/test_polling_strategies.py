"""Tests for polling strategy classes: state management, RC debounce,
autoclose timers, pane alerts, probe failures, and content-hash caching."""

import time
from unittest.mock import patch

import pytest

from ccgram.handlers.polling_strategies import (
    InteractiveUIStrategy,
    TerminalStatusStrategy,
    TopicLifecycleStrategy,
    TopicPollState,
    WindowPollState,
    _MAX_PROBE_FAILURES,
    _RC_DEBOUNCE_SECONDS,
    is_shell_prompt,
)


class TestTerminalStatusStrategy:
    def setup_method(self):
        self.strategy = TerminalStatusStrategy()

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

    def test_clear_state_nonexistent_is_noop(self):
        self.strategy.clear_state("@999")

    def test_clear_screen_buffer(self):
        ws = self.strategy.get_state("@0")
        ws.last_pane_hash = 12345
        ws.last_rendered_text = "some text"
        self.strategy.clear_screen_buffer("@0")
        assert ws.screen_buffer is None
        assert ws.last_pane_hash == 0
        assert ws.last_rendered_text is None

    def test_reset_screen_buffer_state(self):
        ws = self.strategy.get_state("@0")
        ws.rc_active = True
        ws.last_pane_hash = 999
        self.strategy.reset_screen_buffer_state()
        assert not ws.rc_active
        assert ws.last_pane_hash == 0

    def test_is_rc_active_default_false(self):
        assert not self.strategy.is_rc_active("@0")

    def test_is_rc_active_when_set(self):
        ws = self.strategy.get_state("@0")
        ws.rc_active = True
        assert self.strategy.is_rc_active("@0")

    def test_update_rc_state_on(self):
        ws = WindowPollState()
        self.strategy.update_rc_state(ws, True)
        assert ws.rc_active
        assert ws.rc_off_since is None

    def test_update_rc_state_debounce_start(self):
        ws = WindowPollState(rc_active=True)
        self.strategy.update_rc_state(ws, False)
        assert ws.rc_active
        assert ws.rc_off_since is not None

    def test_update_rc_state_debounce_completes(self):
        ws = WindowPollState(rc_active=True)
        ws.rc_off_since = time.monotonic() - _RC_DEBOUNCE_SECONDS - 1
        self.strategy.update_rc_state(ws, False)
        assert not ws.rc_active
        assert ws.rc_off_since is None

    def test_update_rc_state_debounce_reset_on_redetect(self):
        ws = WindowPollState(rc_active=True, rc_off_since=time.monotonic())
        self.strategy.update_rc_state(ws, True)
        assert ws.rc_active
        assert ws.rc_off_since is None

    def test_parse_with_pyte_content_hash_cache(self):
        ws = self.strategy.get_state("@0")
        ws.last_pane_hash = hash(("same text", 200, 50))
        ws.last_pyte_result = None
        result = self.strategy.parse_with_pyte("@0", "same text", 200, 50)
        assert result is None

    def test_parse_with_pyte_invalid_dimensions_defaults(self):
        with patch(
            "ccgram.handlers.polling_strategies.TerminalStatusStrategy.get_screen_buffer"
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


class TestInteractiveUIStrategy:
    def setup_method(self):
        self.terminal = TerminalStatusStrategy()
        self.strategy = InteractiveUIStrategy(self.terminal)

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
        self.terminal = TerminalStatusStrategy()
        self.strategy = TopicLifecycleStrategy(self.terminal)

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
        ws = self.terminal.get_state("@0")
        ws.probe_failures = 5
        self.strategy.clear_probe_failures("@0")
        assert ws.probe_failures == 0

    def test_record_probe_failure_increments(self):
        count = self.strategy.record_probe_failure("@0")
        assert count == 1
        count = self.strategy.record_probe_failure("@0")
        assert count == 2

    def test_record_probe_failure_logs_at_threshold(self):
        for _ in range(_MAX_PROBE_FAILURES - 1):
            self.strategy.record_probe_failure("@0")
        with patch("ccgram.handlers.polling_strategies.logger") as mock_logger:
            self.strategy.record_probe_failure("@0")
            mock_logger.info.assert_called_once()

    def test_clear_typing_state(self):
        ts = self.strategy.get_state(1, 42)
        ts.last_typing_sent = 123.0
        self.strategy.clear_typing_state(1, 42)
        assert ts.last_typing_sent is None

    def test_clear_seen_status(self):
        ws = self.terminal.get_state("@0")
        ws.has_seen_status = True
        ws.startup_time = 100.0
        self.strategy.clear_seen_status("@0")
        assert not ws.has_seen_status
        assert ws.startup_time is None

    def test_reset_autoclose_state_clears_all(self):
        self.strategy.start_autoclose_timer(1, 42, "done", 100.0)
        self.strategy.start_autoclose_timer(2, 43, "dead", 200.0)
        ws = self.terminal.get_state("@0")
        ws.unbound_timer = 50.0
        self.strategy.reset_autoclose_state()
        for ts in self.strategy._states.values():
            assert ts.autoclose is None
        assert ws.unbound_timer is None


class TestIsShellPrompt:
    @pytest.mark.parametrize("cmd", ["bash", "zsh", "fish", "sh", "dash"])
    def test_shell_commands_detected(self, cmd):
        assert is_shell_prompt(cmd)

    @pytest.mark.parametrize("cmd", ["/usr/bin/bash", "/bin/zsh"])
    def test_full_path_shell_detected(self, cmd):
        assert is_shell_prompt(cmd)

    @pytest.mark.parametrize("cmd", ["claude", "codex", "python", "node"])
    def test_non_shell_not_detected(self, cmd):
        assert not is_shell_prompt(cmd)
