"""Tests for window_resolver — ID format helpers and startup migration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ccgram.window_resolver import (
    LiveWindow,
    is_window_id,
    resolve_stale_ids,
)


class TestIsWindowId:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            pytest.param("@0", True, id="at_zero"),
            pytest.param("@12", True, id="at_multi_digit"),
            pytest.param("@", False, id="at_only"),
            pytest.param("0", False, id="no_at"),
            pytest.param("", False, id="empty"),
            pytest.param("mywindow", False, id="name"),
        ],
    )
    def test_is_window_id(self, key: str, expected: bool) -> None:
        assert is_window_id(key) == expected


def _ws(name: str) -> SimpleNamespace:
    """Minimal WindowState stand-in with mutable window_name."""
    return SimpleNamespace(window_name=name)


def _ws_sid(name: str, session_id: str) -> SimpleNamespace:
    """WindowState stand-in carrying a durable agent session id (herdr path)."""
    return SimpleNamespace(window_name=name, session_id=session_id)


class TestResolveStaleIds:
    def test_no_changes_when_ids_still_live(self) -> None:
        live = [LiveWindow("@0", "proj")]
        window_states = {"@0": _ws("proj")}
        thread_bindings: dict = {100: {42: "@0"}}
        offsets: dict = {100: {"@0": 10}}
        display_names = {"@0": "proj"}

        changed = resolve_stale_ids(
            live, window_states, thread_bindings, offsets, display_names
        )

        assert not changed
        assert "@0" in window_states
        assert thread_bindings[100][42] == "@0"

    def test_stale_id_remapped_via_display_name(self) -> None:
        # @0 is gone; tmux restarted and the same window is now @1. Every
        # persisted map resolves through one pre-mutation display-name snapshot.
        live = [LiveWindow("@1", "proj")]
        window_states = {"@0": _ws("proj")}
        thread_bindings: dict = {100: {42: "@0"}}
        offsets: dict = {}
        display_names = {"@0": "proj"}

        changed = resolve_stale_ids(
            live, window_states, thread_bindings, offsets, display_names
        )

        assert changed
        assert "@1" in window_states
        assert "@0" not in window_states
        assert display_names.get("@1") == "proj"
        assert "@0" not in display_names
        assert thread_bindings[100][42] == "@1"

    def test_dead_window_preserved_without_live_match(self) -> None:
        # Stale ID with no live window of that name — keep for /restore
        live: list[LiveWindow] = []
        window_states = {"@0": _ws("dead-proj")}
        thread_bindings: dict = {100: {42: "@0"}}
        offsets: dict = {}
        display_names: dict = {}

        changed = resolve_stale_ids(
            live, window_states, thread_bindings, offsets, display_names
        )

        assert not changed
        assert "@0" in window_states
        assert thread_bindings[100][42] == "@0"

    def test_old_format_name_key_migrated_to_window_id(self) -> None:
        # Pre-migration state: window_states keyed by name instead of @id
        live = [LiveWindow("@3", "myproject")]
        window_states = {"myproject": _ws("myproject")}
        thread_bindings: dict = {100: {7: "myproject"}}
        offsets: dict = {}
        display_names: dict = {}

        changed = resolve_stale_ids(
            live, window_states, thread_bindings, offsets, display_names
        )

        assert changed
        assert "@3" in window_states
        assert "myproject" not in window_states
        assert thread_bindings[100][7] == "@3"
        assert display_names.get("@3") == "myproject"

    def test_old_format_name_key_dropped_when_no_live_match(self) -> None:
        live: list[LiveWindow] = []
        window_states = {"oldname": _ws("oldname")}
        thread_bindings: dict = {}
        offsets: dict = {}
        display_names: dict = {}

        changed = resolve_stale_ids(
            live, window_states, thread_bindings, offsets, display_names
        )

        assert changed
        assert "oldname" not in window_states

    def test_empty_user_bindings_pruned(self) -> None:
        # After migration drops the only binding for a user, that user is removed
        live: list[LiveWindow] = []
        window_states: dict = {}
        thread_bindings: dict = {100: {42: "oldname"}}
        offsets: dict = {}
        display_names: dict = {}

        changed = resolve_stale_ids(
            live, window_states, thread_bindings, offsets, display_names
        )

        assert changed
        assert 100 not in thread_bindings

    def test_offsets_follow_stale_id_remap(self) -> None:
        # Read offsets use the same pre-mutation name mapping as window state
        # and thread bindings, rather than being dropped after display rewrite.
        live = [LiveWindow("@2", "proj")]
        window_states = {"@0": _ws("proj")}
        thread_bindings: dict = {}
        offsets: dict = {100: {"@0": 99}}
        display_names = {"@0": "proj"}

        changed = resolve_stale_ids(
            live, window_states, thread_bindings, offsets, display_names
        )

        assert changed
        assert "@2" in window_states
        assert offsets[100] == {"@2": 99}

    def test_returns_false_with_empty_state(self) -> None:
        changed = resolve_stale_ids([], {}, {}, {}, {})
        assert not changed


class TestGuardedTargetRecovery:
    """Non-stable backend targets are retained without display or locator recovery."""

    def test_opaque_target_missing_from_snapshot_is_retained(self) -> None:
        target = "herdr-session-v1-" + "a" * 64
        live = [LiveWindow("herdr-session-v1-" + "b" * 64, "claude")]
        window_states = {target: _ws_sid("ccgram", "T1")}
        thread_bindings: dict = {100: {42: target}}
        offsets: dict = {100: {target: 5}}
        display_names = {target: "claude"}

        changed = resolve_stale_ids(
            live,
            window_states,
            thread_bindings,
            offsets,
            display_names,
            ids_stable=False,
        )

        assert changed is False
        assert target in window_states
        assert thread_bindings[100][42] == target
        assert offsets[100] == {target: 5}

    def test_tmux_stable_path_keeps_display_recovery(self) -> None:
        live = [LiveWindow("@1", "proj")]
        window_states = {"@0": _ws("proj")}
        thread_bindings: dict = {}
        offsets: dict = {}
        display_names = {"@0": "proj"}
        assert (
            resolve_stale_ids(
                live, window_states, thread_bindings, offsets, display_names
            )
            is True
        )
        assert "@1" in window_states
