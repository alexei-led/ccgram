"""Tests for the backend-neutral topic-mapping projection (Task 10).

Covers:
* ``is_agent_topic_window`` — the capability-gated discovery filter that decides
  whether a multiplexer window surfaces as its own Telegram topic.
* per-session inbound routing and opaque target binding for herdr, where
  ``window_id`` is an opaque durable session target ("topic = agent session").
"""

from __future__ import annotations

import pytest

from ccgram.multiplexer.base import MultiplexerCapabilities, WindowRef
from ccgram.multiplexer.topic_mapping import (
    format_agent_topic_prefix,
    is_agent_topic_window,
)
from ccgram.session import SessionManager
from ccgram.session_resolver import session_resolver
from ccgram.thread_router import thread_router
from ccgram.window_state_store import WindowState, window_store

# tmux-like: no native agent status → every window is a topic.
TMUX_CAPS = MultiplexerCapabilities(
    name="tmux",
    ids_stable_across_restart=True,
    exposes_pane_tty=True,
    native_agent_status=False,
    read_max_lines=None,
    self_identify_env="TMUX_PANE",
    supports_event_stream=False,
    native_worktrees=False,
)

# herdr-like: native agent status → only agent panes are topics.
HERDR_TARGET = "herdr-session-v1-" + "a" * 64

HERDR_CAPS = MultiplexerCapabilities(
    name="herdr",
    ids_stable_across_restart=False,
    exposes_pane_tty=False,
    native_agent_status=True,
    read_max_lines=1000,
    self_identify_env="HERDR_PANE_ID",
    supports_event_stream=True,
    native_worktrees=True,
)


def _win(window_id: str, command: str = "") -> WindowRef:
    return WindowRef(
        window_id=window_id,
        window_name="",
        cwd="/proj",
        pane_current_command=command,
    )


class TestIsAgentTopicWindow:
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("claude", True),  # tmux surfaces agent windows...
            ("", True),  # ...and bare/shell windows alike (unchanged behavior)
            ("zsh", True),
        ],
    )
    def test_tmux_surfaces_every_window(self, command: str, expected: bool) -> None:
        assert is_agent_topic_window(_win("@1", command), TMUX_CAPS) is expected

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("claude", True),  # an agent pane is a topic
            ("codex", True),
            ("", False),  # a bare shell pane is NOT a topic on herdr
            ("   ", False),  # whitespace-only label is not an agent
        ],
    )
    def test_herdr_only_agent_panes(self, command: str, expected: bool) -> None:
        assert (
            is_agent_topic_window(_win(HERDR_TARGET, command), HERDR_CAPS) is expected
        )

    def test_herdr_sessions_have_distinct_topic_targets(self) -> None:
        """Agent sessions, not their current panes, are independently bound."""
        session_a = _win(HERDR_TARGET, "claude")
        session_b = _win("herdr-session-v1-" + "b" * 64, "claude")
        assert is_agent_topic_window(session_a, HERDR_CAPS) is True
        assert is_agent_topic_window(session_b, HERDR_CAPS) is True
        assert session_a.window_id != session_b.window_id

    def test_herdr_rejects_malformed_prefixed_target(self) -> None:
        assert not is_agent_topic_window(
            _win("herdr-session-v1-not-a-sha256", "claude"), HERDR_CAPS
        )


class TestFormatAgentTopicPrefix:
    @pytest.mark.parametrize(
        ("workspace", "tab", "expected"),
        [
            # Lone tab → "<workspace> ▸ <tab>".
            ("ccgram", "herdr-support", "ccgram ▸ herdr-support"),
            ("ccgram", "ralphex", "ccgram ▸ ralphex"),
            # Same workspace, different tab labels → distinct titles (no collision).
            ("ccgram", "herdr-support", "ccgram ▸ herdr-support"),
            ("ccgram", "ralphex", "ccgram ▸ ralphex"),
            # Numeric / auto-generated tab labels still render usefully.
            ("myproject", "tab-1", "myproject ▸ tab-1"),
            ("myproject", "Tab 1", "myproject ▸ Tab 1"),
            # Shell tab (no agent) renders the same way — label is tab name.
            ("ccgram", "zsh", "ccgram ▸ zsh"),
            # Missing parts degrade without a stray separator.
            ("", "herdr-support", "herdr-support"),
            ("ccgram", "", "ccgram"),
            ("", "", ""),
            # Whitespace is trimmed off every part.
            ("  ccgram  ", "  herdr-support  ", "ccgram ▸ herdr-support"),
        ],
    )
    def test_renders_workspace_tab_label(
        self, workspace: str, tab: str, expected: str
    ) -> None:
        assert format_agent_topic_prefix(workspace, tab) == expected

    def test_same_agent_different_tabs_are_distinct(self) -> None:
        """Two tabs in the same workspace with different labels produce distinct titles.

        This is the core requirement: "ccgram ▸ herdr-support" and
        "ccgram ▸ ralphex" are distinct even when both run claude.
        """
        label_a = format_agent_topic_prefix("ccgram", "herdr-support")
        label_b = format_agent_topic_prefix("ccgram", "ralphex")
        assert label_a == "ccgram ▸ herdr-support"
        assert label_b == "ccgram ▸ ralphex"
        assert label_a != label_b

    def test_rename_changes_label_not_identity(self) -> None:
        """Renaming the workspace re-renders the label; the tab id is the key."""
        before = format_agent_topic_prefix("ccgram", "herdr-support")
        after = format_agent_topic_prefix("ccgram-v2", "herdr-support")
        assert before == "ccgram ▸ herdr-support"
        assert after == "ccgram-v2 ▸ herdr-support"
        assert before != after


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    thread_router.reset()
    window_store.window_states.clear()
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


class TestHerdrSessionRouting:
    """Each Herdr agent session routes by opaque target, never a pane locator."""

    def test_two_sessions_route_to_distinct_topics(self, mgr: SessionManager) -> None:
        target_a = "herdr-session-v1-a"
        target_b = "herdr-session-v1-b"
        thread_router.bind_thread(100, 11, target_a)
        thread_router.bind_thread(100, 12, target_b)
        window_store.window_states[target_a] = WindowState(
            session_id="sess-A", cwd="/proj"
        )
        window_store.window_states[target_b] = WindowState(
            session_id="sess-B", cwd="/proj"
        )

        assert session_resolver.find_users_for_session("sess-A") == [
            (100, target_a, 11, None)
        ]
        assert session_resolver.find_users_for_session("sess-B") == [
            (100, target_b, 12, None)
        ]

    def test_binding_is_keyed_per_session_target(self, mgr: SessionManager) -> None:
        target_a = "herdr-session-v1-a"
        target_b = "herdr-session-v1-b"
        thread_router.bind_thread(100, 11, target_a)
        thread_router.bind_thread(100, 12, target_b)

        assert thread_router.get_window_for_thread(100, 11) == target_a
        assert thread_router.get_window_for_thread(100, 12) == target_b
        assert thread_router.get_thread_for_window(100, target_a) == 11
        assert thread_router.get_thread_for_window(100, target_b) == 12

    def test_no_stream_crosstalk_between_session_targets(
        self, mgr: SessionManager
    ) -> None:
        target_a = "herdr-session-v1-a"
        target_b = "herdr-session-v1-b"
        thread_router.bind_thread(100, 11, target_a)
        thread_router.bind_thread(100, 12, target_b)
        window_store.window_states[target_a] = WindowState(
            session_id="sess-A", cwd="/proj"
        )
        window_store.window_states[target_b] = WindowState(
            session_id="sess-B", cwd="/proj"
        )

        threads_for_a = {
            t for _, _, t, _ in session_resolver.find_users_for_session("sess-A")
        }
        assert threads_for_a == {11}
