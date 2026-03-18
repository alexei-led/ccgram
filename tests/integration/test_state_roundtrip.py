"""Integration tests for SessionManager state persistence round-trips.

Tests bind → save → reload → verify cycles using real file I/O,
ensuring state.json serialization is correct across restarts.
Pure in-memory behavior (notification cycling, one-topic-one-window)
is covered by unit tests in test_session.py.
"""

import json
from pathlib import Path

import pytest

from ccgram.session import SessionManager

pytestmark = pytest.mark.integration


@pytest.fixture
def make_session_manager(tmp_path, monkeypatch):
    """Factory: create a SessionManager with isolated state files."""

    def _make(state_file: Path | None = None) -> SessionManager:
        sf = state_file or (tmp_path / "state.json")
        monkeypatch.setattr("ccgram.config.config.state_file", sf)
        monkeypatch.setattr(
            "ccgram.config.config.session_map_file", tmp_path / "session_map.json"
        )
        return SessionManager()

    return _make


@pytest.mark.parametrize(
    "setup_fn, check_fn",
    [
        pytest.param(
            lambda sm: sm.bind_thread(
                user_id=1, thread_id=42, window_id="@0", window_name="test-proj"
            ),
            lambda sm: (
                sm.get_window_for_thread(user_id=1, thread_id=42) == "@0"
                and sm.get_display_name("@0") == "test-proj"
            ),
            id="bind-thread",
        ),
        pytest.param(
            lambda sm: (
                sm.bind_thread(user_id=1, thread_id=10, window_id="@0"),
                sm.bind_thread(user_id=1, thread_id=20, window_id="@1"),
                sm.unbind_thread(user_id=1, thread_id=10),
            ),
            lambda sm: (
                sm.get_window_for_thread(user_id=1, thread_id=10) is None
                and sm.get_window_for_thread(user_id=1, thread_id=20) == "@1"
            ),
            id="unbind-thread",
        ),
        pytest.param(
            lambda sm: (
                sm.bind_thread(
                    user_id=100, thread_id=1, window_id="@0", window_name="proj-a"
                ),
                sm.bind_thread(
                    user_id=200, thread_id=2, window_id="@1", window_name="proj-b"
                ),
            ),
            lambda sm: (
                sm.get_window_for_thread(100, 1) == "@0"
                and sm.get_window_for_thread(200, 2) == "@1"
                and sm.get_display_name("@0") == "proj-a"
                and sm.get_display_name("@1") == "proj-b"
            ),
            id="multiple-users",
        ),
        pytest.param(
            lambda sm: sm.set_group_chat_id(user_id=1, thread_id=42, chat_id=-100123),
            lambda sm: sm.resolve_chat_id(1, 42) == -100123,
            id="group-chat-ids",
        ),
        pytest.param(
            lambda sm: sm.update_user_window_offset(
                user_id=1, window_id="@0", offset=12345
            ),
            lambda sm: sm.get_user_window_offset(1, "@0") == 12345,
            id="user-offsets",
        ),
        pytest.param(
            lambda sm: (
                sm.toggle_user_star(user_id=1, path="/tmp/starred-proj"),
                sm.update_user_mru(user_id=1, path="/tmp/recent-proj"),
            ),
            lambda sm: (
                any("starred-proj" in s for s in sm.get_user_starred(1))
                and any("recent-proj" in s for s in sm.get_user_mru(1))
            ),
            id="directory-favorites",
        ),
    ],
)
async def test_persist_reload(make_session_manager, setup_fn, check_fn) -> None:
    sm1 = make_session_manager()
    setup_fn(sm1)
    sm1.flush_state()

    sm2 = make_session_manager()
    assert check_fn(sm2)


async def test_window_state_survives_reload(make_session_manager) -> None:
    sm1 = make_session_manager()
    state = sm1.get_window_state("@5")
    state.session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    state.cwd = "/tmp/myproject"
    sm1.set_window_provider("@5", "claude")
    sm1.set_notification_mode("@5", "errors_only")
    sm1.flush_state()

    sm2 = make_session_manager()
    reloaded = sm2.get_window_state("@5")
    assert reloaded.session_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert reloaded.cwd == "/tmp/myproject"
    assert reloaded.provider_name == "claude"
    assert reloaded.notification_mode == "errors_only"


async def test_duplicate_bindings_deduped_on_load(tmp_path, monkeypatch) -> None:
    """Old state with duplicate bindings — loader keeps highest thread_id."""
    state = {
        "window_states": {},
        "user_window_offsets": {},
        "thread_bindings": {"1": {"10": "@0", "20": "@0"}},
        "group_chat_ids": {},
        "window_display_names": {},
        "user_dir_favorites": {},
    }
    sf = tmp_path / "state.json"
    sf.write_text(json.dumps(state))
    monkeypatch.setattr("ccgram.config.config.state_file", sf)
    monkeypatch.setattr(
        "ccgram.config.config.session_map_file", tmp_path / "session_map.json"
    )

    sm = SessionManager()
    assert sm.get_window_for_thread(1, 10) is None
    assert sm.get_window_for_thread(1, 20) == "@0"
