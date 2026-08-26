import time

import pytest

from ccgram.handlers.polling.polling_types import (
    STARTUP_TIMEOUT,
    TickContext,
    TickDecision,
)
from ccgram.handlers.polling.window_tick.decide import (
    build_status_line,
    decide_tick,
    is_shell_prompt,
)
from ccgram.providers.base import StatusUpdate


def _make_ctx(
    *,
    window_id: str = "@0",
    resolved_status_text: str | None = None,
    is_shell_prompt: bool = False,
    has_seen_status: bool = False,
    is_recently_active: bool = False,
    startup_time: float | None = None,
    is_dead_window: bool = False,
    supports_hook: bool = True,
    quiet_settled: bool = False,
) -> TickContext:
    return TickContext(
        window_id=window_id,
        resolved_status_text=resolved_status_text,
        is_shell_prompt=is_shell_prompt,
        has_seen_status=has_seen_status,
        is_recently_active=is_recently_active,
        startup_time=startup_time,
        is_dead_window=is_dead_window,
        supports_hook=supports_hook,
        quiet_settled=quiet_settled,
    )


class TestDecideTickActiveStatus:
    def test_resolved_status_yields_active_with_text(self):
        decision = decide_tick(_make_ctx(resolved_status_text="Working..."))
        assert decision.transition == "active"
        assert decision.send_status is True
        assert decision.status_text == "Working..."
        assert decision.show_recovery is False

    def test_recently_active_alone_yields_active_no_status(self):
        decision = decide_tick(_make_ctx(is_recently_active=True))
        assert decision.transition == "active"
        assert decision.send_status is False
        assert decision.status_text is None

    def test_empty_status_text_is_not_treated_as_a_status(self):
        decision = decide_tick(_make_ctx(resolved_status_text="", has_seen_status=True))
        assert decision.transition == "idle"
        assert decision.send_status is True  # genuine idle: bubble wanted
        assert decision.transition == "idle"


class TestDecideTickShellPrompt:
    def test_hook_provider_yields_done(self):
        decision = decide_tick(_make_ctx(is_shell_prompt=True, supports_hook=True))
        assert decision.transition == "done"

    def test_no_hook_provider_yields_idle(self):
        decision = decide_tick(_make_ctx(is_shell_prompt=True, supports_hook=False))
        assert decision.transition == "idle"
        assert decision.send_status is True  # hookless end of turn: bubble


class TestDecideTickIdleAndStarting:
    def test_seen_status_with_no_signal_yields_idle(self):
        decision = decide_tick(_make_ctx(has_seen_status=True))
        assert decision.transition == "idle"
        assert decision.send_status is True

    def test_no_signal_no_startup_yields_starting(self):
        assert decide_tick(_make_ctx(startup_time=None)).transition == "starting"

    def test_startup_within_grace_period_yields_starting(self):
        ctx = _make_ctx(startup_time=time.monotonic())
        assert decide_tick(ctx).transition == "starting"

    def test_startup_expired_yields_idle(self):
        ctx = _make_ctx(startup_time=time.monotonic() - STARTUP_TIMEOUT - 1.0)
        assert decide_tick(ctx).transition == "idle"


class TestDecideTickPrecedence:
    """The kernel is a strict ladder: dead > status > active > shell > seen."""

    def test_dead_window_yields_recovery(self):
        decision = decide_tick(_make_ctx(is_dead_window=True))
        assert decision.show_recovery is True
        assert decision.transition is None
        assert decision.send_status is False

    def test_dead_window_overrides_every_other_signal(self):
        decision = decide_tick(
            _make_ctx(
                is_dead_window=True,
                resolved_status_text="Working",
                is_recently_active=True,
                is_shell_prompt=True,
                has_seen_status=True,
            )
        )
        assert decision.show_recovery is True
        assert decision.transition is None

    @pytest.mark.parametrize(
        "loser",
        [
            pytest.param({"is_recently_active": True}, id="recently-active"),
            pytest.param({"is_shell_prompt": True}, id="shell-prompt"),
            pytest.param({"has_seen_status": True}, id="seen-status"),
        ],
    )
    def test_resolved_status_beats_every_lower_signal(self, loser):
        decision = decide_tick(_make_ctx(resolved_status_text="Working", **loser))
        assert decision.transition == "active"
        assert decision.send_status is True

    def test_recently_active_beats_shell_prompt(self):
        ctx = _make_ctx(is_recently_active=True, is_shell_prompt=True)
        assert decide_tick(ctx).transition == "active"

    def test_shell_prompt_beats_seen_status(self):
        ctx = _make_ctx(is_shell_prompt=True, has_seen_status=True, supports_hook=True)
        assert decide_tick(ctx).transition == "done"


class TestTickDecisionDefaults:
    def test_default_decision_is_a_no_op(self):
        decision = TickDecision()
        assert decision.send_status is False
        assert decision.status_text is None
        assert decision.transition is None
        assert decision.show_recovery is False


class TestBuildStatusLine:
    def test_none_status_returns_none(self):
        assert build_status_line(None) is None

    def test_interactive_status_returns_none(self):
        status = StatusUpdate(
            raw_text="Permission?", display_label="", is_interactive=True
        )
        assert build_status_line(status) is None

    def test_multiline_passes_through_unchanged(self):
        status = StatusUpdate(raw_text="line1\nline2", display_label="")
        assert build_status_line(status) == "line1\nline2"

    def test_single_line_gets_emoji_prefix(self):
        result = build_status_line(StatusUpdate(raw_text="Working", display_label=""))
        assert result is not None
        assert result.endswith(" Working")
        assert result != "Working"


class TestIsShellPrompt:
    @pytest.mark.parametrize(
        "command,expected",
        [
            ("bash", True),
            ("zsh", True),
            ("fish", True),
            ("sh", True),
            ("dash", True),
            ("ksh", True),
            ("tcsh", True),
            ("csh", True),
            ("/bin/bash", True),
            ("/usr/local/bin/zsh", True),
            ("  bash  ", True),
            ("claude", False),
            ("codex", False),
            ("python3", False),
            ("node", False),
            ("npx", False),
            ("bashful", False),
            ("", False),
        ],
    )
    def test_classification(self, command, expected):
        assert is_shell_prompt(command) is expected


class TestQuietSettledLatch:
    """Greptile P1 on #184: quiet settle must not re-enable Ready next tick."""

    def test_quiet_settled_window_stays_quiet_on_later_ticks(self):
        ctx = _make_ctx(has_seen_status=False, quiet_settled=True)
        decision = decide_tick(ctx)
        assert decision.transition == "idle"
        assert decision.send_status is False

    def test_quiet_settled_window_with_real_status_goes_active(self):
        ctx = _make_ctx(resolved_status_text="Working", quiet_settled=True)
        decision = decide_tick(ctx)
        assert decision.transition == "active"
        assert decision.send_status is True
