from __future__ import annotations

from ccgram.window_state_ports.worktree_state import (
    WorktreeProjection,
    clear_worktree,
    get_worktree,
    set_worktree,
)
from ccgram.window_state_store import WindowState, WindowStateStore


class TestReads:
    def test_get_worktree_missing(self, store: WindowStateStore) -> None:
        assert get_worktree("@missing") is None

    def test_get_worktree_unset(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState()
        proj = get_worktree("@1")
        assert proj == WorktreeProjection(
            window_id="@1", worktree_path=None, worktree_branch=None
        )

    def test_get_worktree_populated(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(
            worktree_path="/tmp/wt", worktree_branch="ccg/foo"
        )
        proj = get_worktree("@1")
        assert proj == WorktreeProjection(
            window_id="@1",
            worktree_path="/tmp/wt",
            worktree_branch="ccg/foo",
        )


class TestWrites:
    def test_set_worktree_persists(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_worktree("@1", "/tmp/wt", "ccg/foo")
        state = store.window_states["@1"]
        assert state.worktree_path == "/tmp/wt"
        assert state.worktree_branch == "ccg/foo"
        assert len(save_calls) == 1

    def test_set_worktree_noop_when_unchanged(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_worktree("@1", "/tmp/wt", "ccg/foo")
        save_calls.clear()
        set_worktree("@1", "/tmp/wt", "ccg/foo")
        assert save_calls == []

    def test_clear_worktree(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_worktree("@1", "/tmp/wt", "ccg/foo")
        save_calls.clear()
        clear_worktree("@1")
        state = store.window_states["@1"]
        assert state.worktree_path is None
        assert state.worktree_branch is None
        assert len(save_calls) == 1

    def test_clear_worktree_noop_when_already_clear(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        store.window_states["@1"] = WindowState()
        clear_worktree("@1")
        assert save_calls == []

    def test_clear_worktree_missing_window_no_save(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        clear_worktree("@missing")
        assert save_calls == []
