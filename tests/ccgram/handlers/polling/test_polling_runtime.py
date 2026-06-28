"""Fitness tests for PollingRuntime (Task 4).

Two invariants:
1. ``PollingRuntime.create()`` builds a fully isolated bundle — strategy
   instances are distinct from the default runtime's singletons.
2. ``runtime.reset_window()`` clears only the isolated runtime's state;
   the default singleton runtime is untouched.
"""

from __future__ import annotations

import pytest

from ccgram.handlers.polling.polling_runtime import PollingRuntime, get_default_runtime
from ccgram.topic_state_registry import topic_state


@pytest.fixture(autouse=True)
def _reset_topic_state_registry():
    """Snapshot/restore topic_state._cleanups so create() registrations don't leak."""
    snapshot = {scope: list(bucket) for scope, bucket in topic_state._cleanups.items()}
    yield
    for scope, bucket in topic_state._cleanups.items():
        bucket[:] = snapshot[scope]


class TestPollingRuntimeIsolation:
    def test_create_returns_polling_runtime(self):
        rt = PollingRuntime.create()
        assert isinstance(rt, PollingRuntime)

    def test_create_instances_are_distinct_from_default(self):
        rt = PollingRuntime.create()
        default = get_default_runtime()
        assert rt.poll_state is not default.poll_state
        assert rt.screen_buffer is not default.screen_buffer
        assert rt.interactive is not default.interactive
        assert rt.lifecycle is not default.lifecycle
        assert rt.pane_status is not default.pane_status

    def test_two_creates_produce_distinct_instances(self):
        rt1 = PollingRuntime.create()
        rt2 = PollingRuntime.create()
        assert rt1.poll_state is not rt2.poll_state
        assert rt1.lifecycle is not rt2.lifecycle

    def test_reset_window_clears_isolated_state(self):
        rt = PollingRuntime.create()
        window_id = "@test-isolated"
        # Prime some state in the isolated runtime.
        rt.poll_state.mark_seen_status(window_id)
        assert rt.poll_state.check_seen_status(window_id)

        rt.reset_window(window_id)

        assert not rt.poll_state.check_seen_status(window_id)

    def test_reset_window_does_not_affect_default_runtime(self):
        rt = PollingRuntime.create()
        default = get_default_runtime()
        window_id = "@test-default-untouched"

        default.poll_state.mark_seen_status(window_id)
        assert default.poll_state.check_seen_status(window_id)

        try:
            rt.reset_window(window_id)
            assert default.poll_state.check_seen_status(window_id)
        finally:
            default.poll_state.clear_seen_status(window_id)

    def test_isolated_lifecycle_state_independent_of_default(self):
        rt = PollingRuntime.create()
        default = get_default_runtime()
        user_id, thread_id, wid = 1, 100, "@iso-dead"

        rt.lifecycle.mark_dead_notified(user_id, thread_id, wid)
        assert rt.lifecycle.is_dead_notified(user_id, thread_id, wid)
        assert not default.lifecycle.is_dead_notified(user_id, thread_id, wid)

    def test_get_default_runtime_returns_same_object(self):
        """get_default_runtime() is idempotent — same object every call."""
        assert get_default_runtime() is get_default_runtime()

    def test_default_runtime_wraps_module_singletons(self):
        """Default runtime attributes ARE the polling_state singletons."""
        from ccgram.handlers.polling.polling_state import (
            interactive_strategy,
            lifecycle_strategy,
            pane_status_strategy,
            terminal_poll_state,
            terminal_screen_buffer,
        )

        default = get_default_runtime()
        assert default.poll_state is terminal_poll_state
        assert default.screen_buffer is terminal_screen_buffer
        assert default.interactive is interactive_strategy
        assert default.lifecycle is lifecycle_strategy
        assert default.pane_status is pane_status_strategy
