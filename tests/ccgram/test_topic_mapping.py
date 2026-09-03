"""Tests for the backend-neutral topic-mapping projection (Task 10).

Covers:
* ``is_agent_topic_window`` — the discovery filter that decides
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
    native_topic_targets=True,
)

# agterm: reports agent state natively, but addresses windows by their own
# session UUID rather than a guarded target.
AGTERM_CAPS = MultiplexerCapabilities(
    name="agterm",
    ids_stable_across_restart=True,
    exposes_pane_tty=False,
    native_agent_status=True,
    read_max_lines=None,
    self_identify_env="AGTERM_SESSION_ID",
    supports_event_stream=False,
    native_worktrees=False,
    supports_workspace_selection=True,
    native_topic_targets=False,
)

AGTERM_SESSION_UUID = "157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4"


def _win(window_id: str, command: str = "") -> WindowRef:
    return WindowRef(
        window_id=window_id,
        window_name="",
        cwd="/proj",
        pane_current_command=command,
    )


class TestIsAgentTopicWindow:
    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("claude", id="agent"),
            pytest.param("", id="bare"),
            pytest.param("zsh", id="shell"),
        ],
    )
    def test_tmux_surfaces_every_window(self, command: str) -> None:
        assert is_agent_topic_window(_win("@1", command), TMUX_CAPS) is True

    def test_herdr_agent_pane_is_a_topic(self) -> None:
        """The herdr rules themselves now live in the adapter's ``_live_ref``.

        Only the backend can apply them: whether a record carries a guarded
        target and an agent label is herdr's own knowledge, and asking that
        question here is what let it be keyed on the wrong capability twice.
        Verified against the adapter in ``test_herdr_backend.py``.
        """
        assert is_agent_topic_window(_win(HERDR_TARGET, "claude"), HERDR_CAPS)

    def test_a_window_the_backend_refused_is_not_a_topic(self) -> None:
        window = WindowRef(
            window_id=HERDR_TARGET,
            window_name="",
            cwd="/proj",
            pane_current_command="claude",
            topic_eligible=False,
        )
        assert is_agent_topic_window(window, HERDR_CAPS) is False

    def test_agterm_session_qualifies_without_a_herdr_target(self) -> None:
        """Keying the opaque-target check on ``native_agent_status`` made every
        agterm session permanently ineligible: a UUID never matches a target.

        Whether the session is running an agent is the backend's call, carried
        in ``topic_eligible``, because only it can read its own foreground.
        """
        assert is_agent_topic_window(_win(AGTERM_SESSION_UUID, "claude"), AGTERM_CAPS)

    def test_agterm_session_uuid_is_not_required_to_look_like_a_herdr_target(
        self,
    ) -> None:
        assert is_agent_topic_window(_win(AGTERM_SESSION_UUID, "claude"), AGTERM_CAPS)


class TestFormatAgentTopicPrefix:
    @pytest.mark.parametrize(
        ("workspace", "tab", "expected"),
        [
            # Two tabs in one workspace get distinct titles (no collision).
            ("ccgram", "herdr-support", "ccgram ▸ herdr-support"),
            ("ccgram", "ralphex", "ccgram ▸ ralphex"),
            # Renaming the workspace re-renders the label; the tab id is the key.
            ("ccgram-v2", "herdr-support", "ccgram-v2 ▸ herdr-support"),
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

    def test_provider_prefix_is_searchable(self) -> None:
        assert (
            format_agent_topic_prefix("ccgram", "1", "p3", provider="pi")
            == "Pi ▸ ccgram ▸ 1 ▸ p3"
        )


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


class TestTopicEligibleFlag:
    """Discovery consumes the reconciliation listing, not ``list_windows``.

    ``session_monitor`` calls ``list_windows_for_reconciliation`` and hands that
    same list to ``_emit_unbound_window_events``; it never calls
    ``list_windows``. So a backend's adoption filters cannot live only in
    ``list_windows`` — they have to travel on the window.
    """

    def test_an_ineligible_window_is_never_a_topic(self) -> None:
        window = WindowRef(
            window_id=AGTERM_SESSION_UUID,
            window_name="",
            cwd="/proj",
            pane_current_command="claude",
            topic_eligible=False,
        )
        assert is_agent_topic_window(window, AGTERM_CAPS) is False

    def test_the_flag_overrides_tmux_blanket_eligibility(self) -> None:
        window = WindowRef(
            window_id="@1",
            window_name="",
            cwd="/proj",
            pane_current_command="claude",
            topic_eligible=False,
        )
        assert is_agent_topic_window(window, TMUX_CAPS) is False

    def test_windows_default_to_eligible(self) -> None:
        """Backends with nothing to exclude say nothing and stay unaffected."""
        assert WindowRef(window_id="@1", window_name="", cwd="/p").topic_eligible
