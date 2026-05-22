"""Tests for the pure decide_tick function in window_tick."""

import time

from ccgram.handlers.polling.polling_types import TickContext, TickDecision
from ccgram.handlers.polling.window_tick import decide_tick


def _ctx(
    window_id: str = "@1",
    resolved_status_text: str | None = None,
    is_shell_prompt: bool = False,
    has_seen_status: bool = False,
    is_recently_active: bool = False,
    startup_time: float | None = None,
    is_dead_window: bool = False,
    supports_hook: bool = True,
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
    )


def test_dead_window_returns_show_recovery():
    ctx = _ctx(is_dead_window=True)
    decision = decide_tick(ctx)
    assert decision.show_recovery is True
    assert decision.transition is None
    assert decision.send_status is False


def test_dead_window_ignores_other_fields():
    ctx = _ctx(
        is_dead_window=True, resolved_status_text="Working...", has_seen_status=True
    )
    decision = decide_tick(ctx)
    assert decision.show_recovery is True


def test_active_status_text_sends_status():
    ctx = _ctx(resolved_status_text="⚙ Compiling…")
    decision = decide_tick(ctx)
    assert decision.send_status is True
    assert decision.status_text == "⚙ Compiling…"
    assert decision.transition == "active"
    assert decision.show_recovery is False


def test_recently_active_transitions_active_without_status():
    ctx = _ctx(is_recently_active=True)
    decision = decide_tick(ctx)
    assert decision.transition == "active"
    assert decision.send_status is False


def test_recently_active_takes_priority_over_shell_prompt():
    ctx = _ctx(is_recently_active=True, is_shell_prompt=True)
    decision = decide_tick(ctx)
    assert decision.transition == "active"


def test_shell_prompt_no_hook_transitions_idle():
    ctx = _ctx(is_shell_prompt=True, supports_hook=False)
    decision = decide_tick(ctx)
    assert decision.transition == "idle"
    assert decision.send_status is False


def test_shell_prompt_with_hook_transitions_done():
    ctx = _ctx(is_shell_prompt=True, supports_hook=True)
    decision = decide_tick(ctx)
    assert decision.transition == "done"


def test_has_seen_status_transitions_idle():
    ctx = _ctx(has_seen_status=True)
    decision = decide_tick(ctx)
    assert decision.transition == "idle"
    assert decision.send_status is False


def test_no_startup_time_transitions_starting():
    ctx = _ctx(startup_time=None)
    decision = decide_tick(ctx)
    assert decision.transition == "starting"


def test_recent_startup_time_transitions_starting():
    ctx = _ctx(startup_time=time.monotonic() - 5.0)
    decision = decide_tick(ctx)
    assert decision.transition == "starting"


def test_expired_startup_time_transitions_idle():
    ctx = _ctx(startup_time=time.monotonic() - 60.0)
    decision = decide_tick(ctx)
    assert decision.transition == "idle"


def test_status_text_takes_priority_over_recently_active():
    ctx = _ctx(resolved_status_text="Running tests", is_recently_active=True)
    decision = decide_tick(ctx)
    assert decision.send_status is True
    assert decision.status_text == "Running tests"
    assert decision.transition == "active"


def test_status_text_takes_priority_over_shell_prompt():
    ctx = _ctx(resolved_status_text="Working", is_shell_prompt=True)
    decision = decide_tick(ctx)
    assert decision.send_status is True
    assert decision.transition == "active"


def test_tick_decision_defaults_are_no_op():
    d = TickDecision()
    assert d.send_status is False
    assert d.status_text is None
    assert d.transition is None
    assert d.show_recovery is False
