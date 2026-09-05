"""Tests for window_resolver — ID format helpers and startup migration."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from ccgram.window_state_store import (
    CCGRAM_CREATED_WINDOW_ORIGIN,
    PaneInfo,
    WindowState,
)
from ccgram.window_resolver import (
    LiveWindow,
    is_window_id,
    migrate_window_aliases,
    reset_alias_redirects,
    resolve_stale_ids as _resolve_stale_ids,
    resolve_window_alias,
)


def resolve_stale_ids(*args: Any, **kwargs: Any) -> bool:
    return _resolve_stale_ids(
        *args,
        window_id_predicate=is_window_id,
        recover_stale_ids_by_name=True,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _clean_alias_redirects() -> Iterator[None]:
    """Redirects are module state; one test must not answer the next."""
    reset_alias_redirects()
    yield
    reset_alias_redirects()


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


def _ws_full(
    name: str = "",
    session_id: str = "",
    cwd: str = "",
    transcript_path: str = "",
    provider_name: str = "",
) -> SimpleNamespace:
    """WindowState stand-in carrying every field the alias migration folds."""
    return SimpleNamespace(
        window_name=name,
        session_id=session_id,
        cwd=cwd,
        transcript_path=transcript_path,
        provider_name=provider_name,
    )


class TestMigrateWindowAliases:
    """The hook writes state under a provisional id; the topic binds the durable
    one. Unless the two are folded together, inbound routing — which matches on
    the *bound* window's session id — never matches and replies are dropped."""

    def test_folds_hook_state_onto_the_id_the_topic_bound(self) -> None:
        window_states = {
            "alias": _ws_full(
                session_id="sid-1", cwd="/repo", transcript_path="/t.jsonl"
            )
        }
        chat_bindings = {(7, -100, 41): "canonical"}

        migrations = migrate_window_aliases(
            {"alias": "canonical"},
            window_states,
            {},
            chat_bindings,
            {},
            {"alias": "proj ▸ 1"},
        )

        assert [(m.alias_id, m.canonical_id) for m in migrations] == [
            ("alias", "canonical")
        ]
        assert "alias" not in window_states
        # The bound window now carries the session id, which is the whole point.
        assert window_states["canonical"].session_id == "sid-1"
        assert window_states["canonical"].transcript_path == "/t.jsonl"
        assert chat_bindings[(7, -100, 41)] == "canonical"

    def test_repoints_a_topic_bound_to_the_superseded_id(self) -> None:
        window_states = {"canonical": _ws_full(session_id="sid-1")}
        thread_bindings = {7: {41: "alias"}}
        offsets = {7: {"alias": 12}}
        display_names = {"alias": "proj ▸ 1"}

        migrate_window_aliases(
            {"alias": "canonical"},
            window_states,
            thread_bindings,
            {},
            offsets,
            display_names,
        )

        assert thread_bindings[7][41] == "canonical"
        assert offsets[7] == {"canonical": 12}
        assert display_names == {"canonical": "proj ▸ 1"}

    def test_never_overwrites_what_the_live_id_already_resolved(self) -> None:
        window_states = {
            "alias": _ws_full(session_id="stale", cwd="/old", provider_name="claude"),
            "canonical": _ws_full(session_id="fresh"),
        }

        migrate_window_aliases({"alias": "canonical"}, window_states, {}, {}, {}, {})

        assert "alias" not in window_states
        assert window_states["canonical"].session_id == "fresh"
        # Gaps are still filled from the superseded entry.
        assert window_states["canonical"].cwd == "/old"
        assert window_states["canonical"].provider_name == "claude"

    def test_carries_creation_choices_onto_a_hook_built_state(self) -> None:
        # Real WindowState here: the carry-over reads each field's default off
        # the state class, which a SimpleNamespace stand-in cannot express.
        # The hook-built entry holds defaults for everything the creation flow
        # chose; letting them win drops the YOLO badge and puts a
        # ccgram-created window outside ccgram's lifecycle.
        alias = WindowState(
            session_id="sid-1",
            approval_mode="yolo",
            origin=CCGRAM_CREATED_WINDOW_ORIGIN,
        )
        window_states = {"alias": alias, "canonical": WindowState(session_id="fresh")}

        migrate_window_aliases({"alias": "canonical"}, window_states, {}, {}, {}, {})

        assert window_states["canonical"].approval_mode == "yolo"
        assert window_states["canonical"].origin == CCGRAM_CREATED_WINDOW_ORIGIN

    def test_never_overrides_a_choice_the_live_id_already_holds(self) -> None:
        canonical = WindowState(session_id="fresh", approval_mode="yolo")

        migrate_window_aliases(
            {"alias": "canonical"},
            {"alias": WindowState(session_id="sid-1"), "canonical": canonical},
            {},
            {},
            {},
            {},
        )

        assert canonical.approval_mode == "yolo"

    def test_preserves_all_non_default_state_during_a_collision(self) -> None:
        alias = WindowState(
            session_id="stale",
            batch_mode="batched",
            tool_call_visibility="hidden",
            panes={"%1": PaneInfo("%1", name="left")},
            pane_lifecycle_notify=False,
            rc_probe_state="armed",
            rc_armed_at=12.5,
            worktree_path="/repo/.worktrees/feature",
            worktree_branch="feature",
            provider_manual_override=True,
            legacy_herdr=True,
            legacy_herdr_archived=True,
            legacy_herdr_archive_user_id=7,
            legacy_herdr_archive_thread_id=42,
        )
        canonical = WindowState(
            session_id="fresh",
            panes={"%2": PaneInfo("%2", name="right")},
        )

        migrate_window_aliases(
            {"alias": "canonical"},
            {"alias": alias, "canonical": canonical},
            {},
            {},
            {},
            {},
        )

        assert canonical.session_id == "fresh"
        assert canonical.batch_mode == "batched"
        assert canonical.tool_call_visibility == "hidden"
        assert set(canonical.panes) == {"%1", "%2"}
        assert canonical.pane_lifecycle_notify is False
        assert canonical.rc_probe_state == "armed"
        assert canonical.rc_armed_at == 12.5
        assert canonical.worktree_path == "/repo/.worktrees/feature"
        assert canonical.worktree_branch == "feature"
        assert canonical.provider_manual_override is True
        assert canonical.legacy_herdr is True
        assert canonical.legacy_herdr_archived is True
        assert canonical.legacy_herdr_archive_user_id == 7
        assert canonical.legacy_herdr_archive_thread_id == 42

    def test_unreferenced_or_self_aliases_are_not_migrations(self) -> None:
        window_states = {"canonical": _ws_full(session_id="sid-1")}
        assert (
            migrate_window_aliases(
                {"unknown": "canonical", "canonical": "canonical"},
                window_states,
                {},
                {},
                {},
                {},
            )
            == []
        )
        assert window_states["canonical"].session_id == "sid-1"

    def test_a_topic_already_bound_to_the_canonical_id_is_unbound(self) -> None:
        """One window is one topic; a fold must not leave two answering for it.

        If the canonical id was discovered as an unbound window before ccgram
        learned it supersedes the alias, a second topic is already bound to it.
        The alias's topic carries the user's history, so it wins.
        """
        chat_bindings = {(7, -100, 41): "alias", (7, -100, 553): "canonical"}
        thread_bindings = {7: {41: "alias", 553: "canonical"}}

        migrate_window_aliases(
            {"alias": "canonical"},
            {"canonical": _ws_full(session_id="sid-1")},
            thread_bindings,
            chat_bindings,
            {},
            {},
        )

        assert chat_bindings == {(7, -100, 41): "canonical"}
        assert thread_bindings == {7: {41: "canonical"}}

    def test_the_same_window_bound_in_another_chat_is_not_a_duplicate(self) -> None:
        """``bind_thread`` scopes its own eviction per (user, chat); so does this."""
        chat_bindings = {(7, -100, 41): "alias", (7, -200, 9): "canonical"}

        migrate_window_aliases(
            {"alias": "canonical"},
            {"canonical": _ws_full(session_id="sid-1")},
            {},
            chat_bindings,
            {},
            {},
        )

        assert chat_bindings == {(7, -100, 41): "canonical", (7, -200, 9): "canonical"}


class TestResolveWindowAlias:
    """A flow holding an id minted before supersession — the topic-creation
    wait is the one that matters — must be able to ask what that window is
    called now, or it waits out its timeout on a key nothing will write."""

    def test_an_id_that_was_never_superseded_resolves_to_itself(self) -> None:
        assert resolve_window_alias("canonical") == "canonical"

    def test_resolves_a_superseded_id_to_the_current_one(self) -> None:
        migrate_window_aliases(
            {"alias": "canonical"},
            {"alias": _ws_full(session_id="sid-1")},
            {},
            {},
            {},
            {},
        )

        assert resolve_window_alias("alias") == "canonical"

    def test_follows_a_chain_of_supersessions(self) -> None:
        states = {"first": _ws_full(session_id="sid-1")}
        migrate_window_aliases({"first": "second"}, states, {}, {}, {}, {})
        migrate_window_aliases({"second": "third"}, states, {}, {}, {}, {})

        assert resolve_window_alias("first") == "third"

    def test_records_a_supersession_no_state_referenced_yet(self) -> None:
        # The window whose state was already swept, or bound after the rename:
        # nothing to migrate, but the identity moved and callers still need it.
        assert migrate_window_aliases({"alias": "canonical"}, {}, {}, {}, {}, {}) == []
        assert resolve_window_alias("alias") == "canonical"

    def test_keeps_every_alias_from_a_large_live_snapshot(self) -> None:
        aliases = {f"alias-{index}": f"canonical-{index}" for index in range(300)}

        migrate_window_aliases(aliases, {}, {}, {}, {}, {})

        assert all(
            resolve_window_alias(alias) == canonical
            for alias, canonical in aliases.items()
        )

    def test_stale_aliases_remain_bounded(self) -> None:
        for index in range(300):
            migrate_window_aliases(
                {f"alias-{index}": f"canonical-{index}"}, {}, {}, {}, {}, {}
            )

        assert resolve_window_alias("alias-0") == "alias-0"
        assert resolve_window_alias("alias-299") == "canonical-299"
