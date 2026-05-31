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
from ccgram.thread_router import thread_router
from ccgram.user_preferences import user_preferences
from ccgram.window_state_store import window_store

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
            lambda sm: thread_router.bind_thread(
                user_id=1, thread_id=42, window_id="@0", window_name="test-proj"
            ),
            lambda sm: (
                thread_router.get_window_for_thread(user_id=1, thread_id=42) == "@0"
                and thread_router.get_display_name("@0") == "test-proj"
            ),
            id="bind-thread",
        ),
        pytest.param(
            lambda sm: (
                thread_router.bind_thread(user_id=1, thread_id=10, window_id="@0"),
                thread_router.bind_thread(user_id=1, thread_id=20, window_id="@1"),
                thread_router.unbind_thread(user_id=1, thread_id=10),
            ),
            lambda sm: (
                thread_router.get_window_for_thread(user_id=1, thread_id=10) is None
                and thread_router.get_window_for_thread(user_id=1, thread_id=20) == "@1"
            ),
            id="unbind-thread",
        ),
        pytest.param(
            lambda sm: (
                thread_router.bind_thread(
                    user_id=100, thread_id=1, window_id="@0", window_name="proj-a"
                ),
                thread_router.bind_thread(
                    user_id=200, thread_id=2, window_id="@1", window_name="proj-b"
                ),
            ),
            lambda sm: (
                thread_router.get_window_for_thread(100, 1) == "@0"
                and thread_router.get_window_for_thread(200, 2) == "@1"
                and thread_router.get_display_name("@0") == "proj-a"
                and thread_router.get_display_name("@1") == "proj-b"
            ),
            id="multiple-users",
        ),
        pytest.param(
            lambda sm: thread_router.set_group_chat_id(
                user_id=1, thread_id=42, chat_id=-100123
            ),
            lambda sm: thread_router.resolve_chat_id(1, 42) == -100123,
            id="group-chat-ids",
        ),
        pytest.param(
            lambda sm: user_preferences.update_user_window_offset(
                user_id=1, window_id="@0", offset=12345
            ),
            lambda sm: user_preferences.get_user_window_offset(1, "@0") == 12345,
            id="user-offsets",
        ),
        pytest.param(
            lambda sm: (
                user_preferences.toggle_user_star(user_id=1, path="/tmp/starred-proj"),
                user_preferences.update_user_mru(user_id=1, path="/tmp/recent-proj"),
            ),
            lambda sm: (
                any("starred-proj" in s for s in user_preferences.get_user_starred(1))
                and any("recent-proj" in s for s in user_preferences.get_user_mru(1))
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
    state = window_store.get_window_state("@5")
    state.session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    state.cwd = "/tmp/myproject"
    sm1.set_window_provider("@5", "claude")
    sm1.flush_state()

    _sm2 = make_session_manager()  # reload triggers __post_init__ -> _load_state
    reloaded = window_store.get_window_state("@5")
    assert reloaded.session_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert reloaded.cwd == "/tmp/myproject"
    assert reloaded.provider_name == "claude"


async def test_full_window_state_feature_groups_survive_reload(
    make_session_manager,
) -> None:
    """Every persisted WindowState feature group survives a SessionManager reload.

    Locks the integration-level persistence contract before the
    window-state feature-port refactor. If a future port forgets to
    serialize one of these fields, this test fails before any handler
    notices.
    """
    sm1 = make_session_manager()
    state = window_store.get_window_state("@7")
    state.session_id = "ffff-eeee-dddd-cccc"
    state.cwd = "/repo/x"
    state.window_name = "proj-x"
    state.transcript_path = "/tmp/transcripts/x.jsonl"
    sm1.set_window_provider("@7", "codex")
    sm1.set_window_approval_mode("@7", "yolo")
    sm1.set_batch_mode("@7", "verbose")
    sm1.set_tool_call_visibility("@7", "shown")
    sm1.set_window_origin("@7", "ccgram_created")
    sm1.set_window_worktree("@7", "/repo/x.worktrees/ccg-x", "ccg/x")
    window_store.set_pane_lifecycle_notify("@7", True)
    window_store.set_provider_manual_override("@7", value=True)
    window_store.upsert_pane(
        "@7",
        "%5",
        name="api",
        provider="codex",
        last_active_ts=1700000000.5,
        state="blocked",
        subscribed=True,
    )
    window_store.upsert_pane("@7", "%6", state="idle")
    sm1.flush_state()

    _sm2 = make_session_manager()
    reloaded = window_store.get_window_state("@7")
    assert reloaded.session_id == "ffff-eeee-dddd-cccc"
    assert reloaded.cwd == "/repo/x"
    assert reloaded.window_name == "proj-x"
    assert reloaded.transcript_path == "/tmp/transcripts/x.jsonl"
    assert reloaded.provider_name == "codex"
    assert reloaded.approval_mode == "yolo"
    assert reloaded.batch_mode == "verbose"
    assert reloaded.tool_call_visibility == "shown"
    assert reloaded.origin == "ccgram_created"
    assert reloaded.pane_lifecycle_notify is True
    assert reloaded.worktree_path == "/repo/x.worktrees/ccg-x"
    assert reloaded.worktree_branch == "ccg/x"
    assert reloaded.provider_manual_override is True
    assert set(reloaded.panes.keys()) == {"%5", "%6"}
    pane5 = reloaded.panes["%5"]
    assert pane5.name == "api"
    assert pane5.provider == "codex"
    assert pane5.last_active_ts == 1700000000.5
    assert pane5.state == "blocked"
    assert pane5.subscribed is True
    assert reloaded.panes["%6"].state == "idle"
    # Transient RC-probe fields must not have been resurrected.
    assert reloaded.rc_probe_state is None
    assert reloaded.rc_armed_at is None


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

    SessionManager()
    assert thread_router.get_window_for_thread(1, 10) is None
    assert thread_router.get_window_for_thread(1, 20) == "@0"
