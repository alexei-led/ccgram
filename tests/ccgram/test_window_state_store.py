"""Tests for WindowStateStore — pane management, serialization, helpers."""

from __future__ import annotations

import pytest

from ccgram.session import SessionManager
from ccgram.window_state_store import (
    DEFAULT_PANE_STATE,
    DEFAULT_TOOL_CALL_VISIBILITY,
    TOOL_CALL_VISIBILITY_MODES,
    PaneInfo,
    WindowState,
    WindowStateStore,
    window_store,
)


@pytest.fixture
def store() -> WindowStateStore:
    save_calls: list[int] = []
    s = WindowStateStore(
        schedule_save=lambda: save_calls.append(1),
        on_hookless_provider_switch=lambda _wid: None,
    )
    s._save_calls = save_calls  # type: ignore[attr-defined]
    return s


class TestPaneInfoSerialization:
    def test_round_trip_full(self) -> None:
        pane = PaneInfo(
            pane_id="%5",
            name="api-gateway",
            provider="claude",
            last_active_ts=1700000000.5,
            state="blocked",
            subscribed=True,
        )
        loaded = PaneInfo.from_dict(pane.to_dict())
        assert loaded == pane

    def test_round_trip_defaults_omits_optional_keys(self) -> None:
        pane = PaneInfo(pane_id="%6")
        d = pane.to_dict()
        assert d == {"pane_id": "%6"}
        loaded = PaneInfo.from_dict(d)
        assert loaded == pane

    def test_invalid_state_falls_back_to_default(self) -> None:
        pane = PaneInfo.from_dict({"pane_id": "%7", "state": "garbage"})
        assert pane.state == DEFAULT_PANE_STATE

    def test_pane_id_filled_from_dict_key_when_missing(self) -> None:
        pane = PaneInfo.from_dict({"name": "build"})
        assert pane.pane_id == ""

    def test_last_active_ts_coerces_to_float(self) -> None:
        pane = PaneInfo.from_dict({"pane_id": "%9", "last_active_ts": 0})
        assert pane.last_active_ts == 0.0


class TestWindowStatePanes:
    def test_default_empty_dict(self) -> None:
        ws = WindowState()
        assert ws.panes == {}

    def test_to_dict_omits_panes_when_empty(self) -> None:
        ws = WindowState(cwd="/tmp/x")
        assert "panes" not in ws.to_dict()

    def test_to_dict_includes_panes_when_present(self) -> None:
        ws = WindowState(cwd="/tmp/x")
        ws.panes["%5"] = PaneInfo(pane_id="%5", state="active", subscribed=True)
        ws.panes["%6"] = PaneInfo(pane_id="%6")
        d = ws.to_dict()
        assert "panes" in d
        assert set(d["panes"].keys()) == {"%5", "%6"}
        assert d["panes"]["%5"]["state"] == "active"
        assert d["panes"]["%6"] == {"pane_id": "%6"}

    def test_from_dict_missing_panes_defaults_to_empty(self) -> None:
        ws = WindowState.from_dict({"session_id": "abc", "cwd": "/p"})
        assert ws.panes == {}

    def test_from_dict_round_trip(self) -> None:
        original = WindowState(
            session_id="sid",
            cwd="/p",
            window_name="proj",
            panes={
                "%5": PaneInfo(pane_id="%5", name="api", state="blocked"),
                "%6": PaneInfo(pane_id="%6", subscribed=True),
            },
        )
        loaded = WindowState.from_dict(original.to_dict())
        assert loaded == original

    def test_from_dict_skips_non_dict_pane_entries(self) -> None:
        ws = WindowState.from_dict(
            {
                "panes": {"%5": "garbage", "%6": {"pane_id": "%6"}},
            }
        )
        assert "%5" not in ws.panes
        assert ws.panes["%6"].pane_id == "%6"


class TestStoreCRUD:
    def test_get_pane_returns_none_for_missing_window(
        self, store: WindowStateStore
    ) -> None:
        assert store.get_pane("@1", "%5") is None

    def test_get_pane_returns_none_for_missing_pane(
        self, store: WindowStateStore
    ) -> None:
        store.get_window_state("@1")
        assert store.get_pane("@1", "%5") is None

    def test_upsert_pane_creates_entry(self, store: WindowStateStore) -> None:
        pane = store.upsert_pane("@1", "%5", provider="claude", state="active")
        assert pane.pane_id == "%5"
        assert pane.provider == "claude"
        assert pane.state == "active"
        assert store.get_pane("@1", "%5") is pane

    def test_upsert_pane_updates_only_provided_fields(
        self, store: WindowStateStore
    ) -> None:
        store.upsert_pane(
            "@1",
            "%5",
            name="orig",
            provider="claude",
            last_active_ts=10.0,
            state="active",
            subscribed=True,
        )
        store.upsert_pane("@1", "%5", state="idle")
        pane = store.get_pane("@1", "%5")
        assert pane is not None
        assert pane.name == "orig"
        assert pane.provider == "claude"
        assert pane.last_active_ts == 10.0
        assert pane.state == "idle"
        assert pane.subscribed is True

    def test_upsert_pane_clears_name_when_explicitly_none(
        self, store: WindowStateStore
    ) -> None:
        store.upsert_pane("@1", "%5", name="api")
        store.upsert_pane("@1", "%5", name=None)
        pane = store.get_pane("@1", "%5")
        assert pane is not None and pane.name is None

    def test_upsert_pane_rejects_invalid_state(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            store.upsert_pane("@1", "%5", state="garbage")  # type: ignore[arg-type]

    def test_upsert_pane_schedules_save(self, store: WindowStateStore) -> None:
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.upsert_pane("@1", "%5", state="active")
        assert len(store._save_calls) == 1  # type: ignore[attr-defined]

    def test_remove_pane_removes_entry(self, store: WindowStateStore) -> None:
        store.upsert_pane("@1", "%5")
        assert store.remove_pane("@1", "%5") is True
        assert store.get_pane("@1", "%5") is None

    def test_remove_pane_returns_false_when_missing(
        self, store: WindowStateStore
    ) -> None:
        assert store.remove_pane("@1", "%5") is False
        store.upsert_pane("@1", "%5")
        assert store.remove_pane("@1", "%99") is False

    def test_store_to_dict_round_trip_preserves_panes(
        self, store: WindowStateStore
    ) -> None:
        store.upsert_pane(
            "@1",
            "%5",
            name="api",
            provider="claude",
            state="blocked",
            subscribed=True,
        )
        store.upsert_pane("@1", "%6", state="idle")
        snapshot = store.to_dict()

        new_store = WindowStateStore(
            schedule_save=lambda: None,
            on_hookless_provider_switch=lambda _wid: None,
        )
        new_store.from_dict(snapshot)
        assert "@1" in new_store.window_states
        panes = new_store.window_states["@1"].panes
        assert panes["%5"].name == "api"
        assert panes["%5"].subscribed is True
        assert panes["%5"].state == "blocked"
        assert panes["%6"].state == "idle"

    def test_legacy_state_without_panes_loads_cleanly(
        self, store: WindowStateStore
    ) -> None:
        legacy = {
            "@1": {
                "session_id": "s",
                "cwd": "/p",
                "window_name": "proj",
            }
        }
        store.from_dict(legacy)
        assert store.window_states["@1"].panes == {}


class TestPaneLifecycleNotify:
    def test_window_state_default_is_none(self) -> None:
        ws = WindowState()
        assert ws.pane_lifecycle_notify is None

    def test_to_dict_omits_when_none(self) -> None:
        ws = WindowState(cwd="/p")
        assert "pane_lifecycle_notify" not in ws.to_dict()

    def test_to_dict_includes_when_set(self) -> None:
        ws = WindowState(cwd="/p", pane_lifecycle_notify=True)
        d = ws.to_dict()
        assert d["pane_lifecycle_notify"] is True

    def test_to_dict_includes_when_explicitly_false(self) -> None:
        ws = WindowState(cwd="/p", pane_lifecycle_notify=False)
        d = ws.to_dict()
        assert d["pane_lifecycle_notify"] is False

    def test_from_dict_round_trip(self) -> None:
        original = WindowState(cwd="/p", pane_lifecycle_notify=True)
        loaded = WindowState.from_dict(original.to_dict())
        assert loaded.pane_lifecycle_notify is True

    def test_from_dict_missing_field_defaults_to_none(self) -> None:
        ws = WindowState.from_dict({"session_id": "s", "cwd": "/p"})
        assert ws.pane_lifecycle_notify is None

    def test_get_returns_default_when_unknown_window(
        self, store: WindowStateStore
    ) -> None:
        assert store.get_pane_lifecycle_notify("@missing", default=False) is False
        assert store.get_pane_lifecycle_notify("@missing", default=True) is True

    def test_get_returns_default_when_override_unset(
        self, store: WindowStateStore
    ) -> None:
        store.get_window_state("@1")
        assert store.get_pane_lifecycle_notify("@1", default=False) is False
        assert store.get_pane_lifecycle_notify("@1", default=True) is True

    def test_get_returns_override_when_set(self, store: WindowStateStore) -> None:
        store.set_pane_lifecycle_notify("@1", True)
        assert store.get_pane_lifecycle_notify("@1", default=False) is True
        store.set_pane_lifecycle_notify("@1", False)
        assert store.get_pane_lifecycle_notify("@1", default=True) is False

    def test_set_schedules_save(self, store: WindowStateStore) -> None:
        save_calls = store._save_calls  # type: ignore[attr-defined]
        save_calls.clear()
        store.set_pane_lifecycle_notify("@1", True)
        assert len(save_calls) == 1

    def test_set_to_same_value_does_not_save(self, store: WindowStateStore) -> None:
        store.set_pane_lifecycle_notify("@1", True)
        save_calls = store._save_calls  # type: ignore[attr-defined]
        save_calls.clear()
        store.set_pane_lifecycle_notify("@1", True)
        assert save_calls == []

    def test_set_to_none_clears_override(self, store: WindowStateStore) -> None:
        store.set_pane_lifecycle_notify("@1", True)
        store.set_pane_lifecycle_notify("@1", None)
        assert store.get_pane_lifecycle_notify("@1", default=False) is False
        assert store.get_pane_lifecycle_notify("@1", default=True) is True


class TestWindowStateWorktree:
    def test_defaults_are_none(self) -> None:
        ws = WindowState()
        assert ws.worktree_path is None
        assert ws.worktree_branch is None

    def test_to_dict_omits_when_none(self) -> None:
        ws = WindowState(cwd="/p")
        d = ws.to_dict()
        assert "worktree_path" not in d
        assert "worktree_branch" not in d

    def test_to_dict_includes_when_set(self) -> None:
        ws = WindowState(
            cwd="/p",
            worktree_path="/p.worktrees/ccg-feature",
            worktree_branch="ccg/feature",
        )
        d = ws.to_dict()
        assert d["worktree_path"] == "/p.worktrees/ccg-feature"
        assert d["worktree_branch"] == "ccg/feature"

    def test_from_dict_round_trip(self) -> None:
        original = WindowState(
            cwd="/p",
            worktree_path="/p.worktrees/ccg-x",
            worktree_branch="ccg/x",
        )
        loaded = WindowState.from_dict(original.to_dict())
        assert loaded.worktree_path == "/p.worktrees/ccg-x"
        assert loaded.worktree_branch == "ccg/x"

    def test_from_dict_missing_fields_default_to_none(self) -> None:
        ws = WindowState.from_dict({"session_id": "s", "cwd": "/p"})
        assert ws.worktree_path is None
        assert ws.worktree_branch is None

    def test_set_window_worktree_persists(self, mgr: SessionManager) -> None:
        mgr.set_window_worktree("@3", "/repo.worktrees/ccg-y", "ccg/y")
        state = window_store.window_states["@3"]
        assert state.worktree_path == "/repo.worktrees/ccg-y"
        assert state.worktree_branch == "ccg/y"


class TestWindowStateGeminiExternalWarned:
    def test_default_is_false(self) -> None:
        assert WindowState().gemini_external_warned is False

    def test_to_dict_omits_when_false(self) -> None:
        assert "gemini_external_warned" not in WindowState(cwd="/p").to_dict()

    def test_round_trip_when_true(self) -> None:
        ws = WindowState(cwd="/p", gemini_external_warned=True)
        d = ws.to_dict()
        assert d["gemini_external_warned"] is True
        assert WindowState.from_dict(d).gemini_external_warned is True

    def test_from_dict_missing_defaults_false(self) -> None:
        ws = WindowState.from_dict({"session_id": "s", "cwd": "/p"})
        assert ws.gemini_external_warned is False

    def test_mark_is_idempotent_and_persists(self, mgr: SessionManager) -> None:
        assert window_store.was_gemini_external_warned("@4") is False
        window_store.mark_gemini_external_warned("@4")
        assert window_store.was_gemini_external_warned("@4") is True
        window_store.mark_gemini_external_warned("@4")
        assert window_store.window_states["@4"].gemini_external_warned is True


class TestProviderManualOverride:
    def test_default_is_false(self) -> None:
        assert WindowState().provider_manual_override is False

    def test_to_dict_omits_when_false(self) -> None:
        assert "provider_manual_override" not in WindowState(cwd="/p").to_dict()

    def test_to_dict_includes_when_true(self) -> None:
        ws = WindowState(cwd="/p", provider_manual_override=True)
        assert ws.to_dict()["provider_manual_override"] is True

    def test_from_dict_round_trip(self) -> None:
        ws = WindowState(cwd="/p", provider_manual_override=True)
        loaded = WindowState.from_dict(ws.to_dict())
        assert loaded.provider_manual_override is True

    def test_from_dict_missing_defaults_false(self) -> None:
        ws = WindowState.from_dict({"session_id": "s", "cwd": "/p"})
        assert ws.provider_manual_override is False

    def test_set_schedules_save_and_persists(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState()
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.set_provider_manual_override("@1", value=True)
        assert store.window_states["@1"].provider_manual_override is True
        assert len(store._save_calls) == 1  # type: ignore[attr-defined]

    def test_set_same_value_is_no_op(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState()
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.set_provider_manual_override("@1", value=False)
        assert store._save_calls == []  # type: ignore[attr-defined]

    def test_set_missing_window_is_no_op(self, store: WindowStateStore) -> None:
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.set_provider_manual_override("@missing", value=True)
        assert "@missing" not in store.window_states
        assert store._save_calls == []  # type: ignore[attr-defined]

    def test_clear_transcript_path_clears_and_saves(
        self, store: WindowStateStore
    ) -> None:
        store.window_states["@1"] = WindowState(transcript_path="/t.jsonl")
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.clear_transcript_path("@1")
        assert store.window_states["@1"].transcript_path == ""
        assert len(store._save_calls) == 1  # type: ignore[attr-defined]

    def test_clear_transcript_path_no_op_when_empty(
        self, store: WindowStateStore
    ) -> None:
        store.window_states["@1"] = WindowState()
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.clear_transcript_path("@1")
        assert store._save_calls == []  # type: ignore[attr-defined]

    def test_clear_transcript_path_no_op_missing_window(
        self, store: WindowStateStore
    ) -> None:
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.clear_transcript_path("@missing")
        assert store._save_calls == []  # type: ignore[attr-defined]


class TestApprovalMode:
    def test_default_is_normal(self, store: WindowStateStore) -> None:
        assert store.get_approval_mode("@1") == "normal"

    def test_unknown_window_returns_default(self, store: WindowStateStore) -> None:
        assert store.get_approval_mode("@missing") == "normal"

    def test_set_yolo(self, store: WindowStateStore) -> None:
        store.set_window_approval_mode("@1", "yolo")
        assert store.get_approval_mode("@1") == "yolo"

    def test_case_insensitive(self, store: WindowStateStore) -> None:
        store.set_window_approval_mode("@1", "YOLO")
        assert store.get_approval_mode("@1") == "yolo"

    def test_invalid_raises(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            store.set_window_approval_mode("@1", "turbo")

    def test_corrupt_stored_value_falls_back_to_default(
        self, store: WindowStateStore
    ) -> None:
        store.get_window_state("@1").approval_mode = "garbage"
        assert store.get_approval_mode("@1") == "normal"


class TestBatchMode:
    def test_default_is_batched(self, store: WindowStateStore) -> None:
        assert store.get_batch_mode("@1") == "ephemeral"

    def test_unknown_window_returns_default(self, store: WindowStateStore) -> None:
        assert store.get_batch_mode("@missing") == "ephemeral"

    def test_set_verbose(self, store: WindowStateStore) -> None:
        store.set_batch_mode("@1", "verbose")
        assert store.get_batch_mode("@1") == "verbose"

    def test_invalid_raises(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            store.set_batch_mode("@1", "stream")

    def test_set_same_value_skips_save(self, store: WindowStateStore) -> None:
        store.set_batch_mode("@1", "verbose")
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.set_batch_mode("@1", "verbose")
        assert store._save_calls == []  # type: ignore[attr-defined]

    def test_cycle_default_to_verbose(self, store: WindowStateStore) -> None:
        # Default is "ephemeral"; one cycle advances to "verbose".
        assert store.cycle_batch_mode("@1") == "verbose"

    def test_cycle_ephemeral_to_verbose(self, store: WindowStateStore) -> None:
        store.set_batch_mode("@1", "ephemeral")
        assert store.cycle_batch_mode("@1") == "verbose"

    def test_cycle_verbose_to_batched(self, store: WindowStateStore) -> None:
        store.set_batch_mode("@1", "verbose")
        assert store.cycle_batch_mode("@1") == "batched"

    def test_cycle_full_loop(self, store: WindowStateStore) -> None:
        # ephemeral -> verbose -> batched -> ephemeral.
        assert store.cycle_batch_mode("@1") == "verbose"
        assert store.cycle_batch_mode("@1") == "batched"
        assert store.cycle_batch_mode("@1") == "ephemeral"

    def test_corrupt_stored_value_falls_back_to_default(
        self, store: WindowStateStore
    ) -> None:
        store.get_window_state("@1").batch_mode = "garbage"
        assert store.get_batch_mode("@1") == "ephemeral"


class TestSetWindowOrigin:
    def test_set_ccgram_created(self, store: WindowStateStore) -> None:
        store.set_window_origin("@1", "ccgram_created")
        assert store.window_states["@1"].origin == "ccgram_created"

    def test_set_external_also_sets_external_flag(
        self, store: WindowStateStore
    ) -> None:
        store.set_window_origin("@1", "external")
        state = store.window_states["@1"]
        assert state.origin == "external"
        assert state.external is True

    def test_non_external_origin_does_not_set_external_flag(
        self, store: WindowStateStore
    ) -> None:
        store.set_window_origin("@1", "ccgram_created")
        assert store.window_states["@1"].external is False

    def test_invalid_origin_raises(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            store.set_window_origin("@1", "unknown_origin")

    def test_set_same_origin_skips_save(self, store: WindowStateStore) -> None:
        store.set_window_origin("@1", "ccgram_created")
        store._save_calls.clear()  # type: ignore[attr-defined]
        store.set_window_origin("@1", "ccgram_created")
        assert store._save_calls == []  # type: ignore[attr-defined]


class TestSetWindowProvider:
    def test_sets_provider(self, store: WindowStateStore) -> None:
        store.set_window_provider("@1", "codex", new_provider_supports_hook=True)
        assert store.window_states["@1"].provider_name == "codex"

    def test_sets_cwd_when_provided(self, store: WindowStateStore) -> None:
        store.set_window_provider(
            "@1", "claude", cwd="/tmp/proj", new_provider_supports_hook=True
        )
        assert store.window_states["@1"].cwd == "/tmp/proj"

    def test_hookless_switch_clears_session_fields(
        self, store: WindowStateStore
    ) -> None:
        state = store.get_window_state("@1")
        state.provider_name = "claude"
        state.session_id = "abc123"
        state.transcript_path = "/tmp/t.jsonl"
        store.set_window_provider("@1", "shell", new_provider_supports_hook=False)
        assert store.window_states["@1"].session_id == ""
        assert store.window_states["@1"].transcript_path == ""

    def test_hookless_switch_invokes_callback(self, store: WindowStateStore) -> None:
        called: list[str] = []
        store._on_hookless_provider_switch = called.append
        state = store.get_window_state("@1")
        state.provider_name = "claude"
        store.set_window_provider("@1", "shell", new_provider_supports_hook=False)
        assert called == ["@1"]

    def test_hook_provider_does_not_clear_session(
        self, store: WindowStateStore
    ) -> None:
        state = store.get_window_state("@1")
        state.provider_name = "claude"
        state.session_id = "keep-me"
        store.set_window_provider("@1", "codex", new_provider_supports_hook=True)
        assert store.window_states["@1"].session_id == "keep-me"

    def test_empty_provider_name_reset_does_not_trigger_hookless_callback(
        self, store: WindowStateStore
    ) -> None:
        called: list[str] = []
        store._on_hookless_provider_switch = called.append
        state = store.get_window_state("@1")
        state.provider_name = "claude"
        store.set_window_provider("@1", "", new_provider_supports_hook=False)
        assert called == []

    def test_same_provider_hookless_does_not_trigger_callback(
        self, store: WindowStateStore
    ) -> None:
        called: list[str] = []
        store._on_hookless_provider_switch = called.append
        state = store.get_window_state("@1")
        state.provider_name = "shell"
        store.set_window_provider("@1", "shell", new_provider_supports_hook=False)
        assert called == []


class TestPruneStaleWindowStates:
    def test_removes_stale_windows(self, store: WindowStateStore) -> None:
        store.get_window_state("@1")
        store.get_window_state("@2")
        store.get_window_state("@3")
        changed = store.prune_stale_window_states(
            live_window_ids=set(),
            session_map_wids=set(),
            bound_window_ids=set(),
        )
        assert changed is True
        assert store.window_states == {}

    def test_keeps_live_windows(self, store: WindowStateStore) -> None:
        store.get_window_state("@1")
        store.get_window_state("@2")
        changed = store.prune_stale_window_states(
            live_window_ids={"@1"},
            session_map_wids=set(),
            bound_window_ids=set(),
        )
        assert changed is True
        assert "@1" in store.window_states
        assert "@2" not in store.window_states

    def test_keeps_session_map_windows(self, store: WindowStateStore) -> None:
        store.get_window_state("@1")
        changed = store.prune_stale_window_states(
            live_window_ids=set(),
            session_map_wids={"@1"},
            bound_window_ids=set(),
        )
        assert changed is False
        assert "@1" in store.window_states

    def test_keeps_bound_windows(self, store: WindowStateStore) -> None:
        store.get_window_state("@1")
        changed = store.prune_stale_window_states(
            live_window_ids=set(),
            session_map_wids=set(),
            bound_window_ids={"@1"},
        )
        assert changed is False
        assert "@1" in store.window_states

    def test_no_stale_returns_false(self, store: WindowStateStore) -> None:
        changed = store.prune_stale_window_states(
            live_window_ids=set(),
            session_map_wids=set(),
            bound_window_ids=set(),
        )
        assert changed is False


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    window_store.window_states.clear()
    return SessionManager()


class TestToolCallVisibilityConstants:
    def test_modes_tuple(self):
        assert TOOL_CALL_VISIBILITY_MODES == ("default", "shown", "hidden")

    def test_default_value(self):
        assert DEFAULT_TOOL_CALL_VISIBILITY == "default"


class TestToolCallVisibilityStore:
    def test_get_default(self, mgr: SessionManager) -> None:
        assert mgr.get_tool_call_visibility("@0") == "default"

    def test_get_nonexistent_window(self, mgr: SessionManager) -> None:
        assert mgr.get_tool_call_visibility("@999") == "default"

    def test_set_valid(self, mgr: SessionManager) -> None:
        mgr.set_tool_call_visibility("@0", "hidden")
        assert mgr.get_tool_call_visibility("@0") == "hidden"
        mgr.set_tool_call_visibility("@0", "shown")
        assert mgr.get_tool_call_visibility("@0") == "shown"
        mgr.set_tool_call_visibility("@0", "default")
        assert mgr.get_tool_call_visibility("@0") == "default"

    def test_set_invalid_raises(self, mgr: SessionManager) -> None:
        with pytest.raises(ValueError, match="Invalid tool_call_visibility"):
            mgr.set_tool_call_visibility("@0", "bogus")

    @pytest.mark.parametrize(
        ("start", "expected"),
        [
            ("default", "shown"),
            ("shown", "hidden"),
            ("hidden", "default"),
        ],
    )
    def test_cycle(self, mgr: SessionManager, start: str, expected: str) -> None:
        mgr.set_tool_call_visibility("@0", start)
        assert mgr.cycle_tool_call_visibility("@0") == expected
        assert mgr.get_tool_call_visibility("@0") == expected

    def test_cycle_full_circle(self, mgr: SessionManager) -> None:
        assert mgr.cycle_tool_call_visibility("@1") == "shown"
        assert mgr.cycle_tool_call_visibility("@1") == "hidden"
        assert mgr.cycle_tool_call_visibility("@1") == "default"


class TestToolCallVisibilitySerialization:
    @pytest.mark.parametrize(
        ("mode", "expect_key"),
        [("default", False), ("shown", True), ("hidden", True)],
    )
    def test_to_dict(self, mode: str, expect_key: bool) -> None:
        ws = WindowState(session_id="s1", cwd="/tmp", tool_call_visibility=mode)
        d = ws.to_dict()
        if expect_key:
            assert d["tool_call_visibility"] == mode
        else:
            assert "tool_call_visibility" not in d

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"session_id": "s1", "cwd": "/tmp"}, "default"),
            (
                {"session_id": "s1", "cwd": "/tmp", "tool_call_visibility": "shown"},
                "shown",
            ),
            (
                {"session_id": "s1", "cwd": "/tmp", "tool_call_visibility": "hidden"},
                "hidden",
            ),
        ],
    )
    def test_from_dict(self, data: dict[str, str], expected: str) -> None:
        assert WindowState.from_dict(data).tool_call_visibility == expected

    @pytest.mark.parametrize("mode", list(TOOL_CALL_VISIBILITY_MODES))
    def test_roundtrip(self, mode: str) -> None:
        ws = WindowState(session_id="s1", cwd="/tmp", tool_call_visibility=mode)
        assert WindowState.from_dict(ws.to_dict()).tool_call_visibility == mode


class TestWindowStateFullPersistenceCharacterization:
    """Pin every persisted feature group on WindowState.to_dict/from_dict.

    Locks the persisted shape before the window-state feature-port migration
    so the per-port refactors cannot silently drop or rename a persisted
    field. Each block covers one feature group exposed to handlers, the
    Mini App, or providers.
    """

    def test_full_round_trip_preserves_all_feature_groups(self) -> None:
        original = WindowState(
            session_id="sid-abc",
            cwd="/repo/path",
            window_name="proj",
            transcript_path="/tmp/t.jsonl",
            provider_name="claude",
            approval_mode="yolo",
            batch_mode="verbose",
            tool_call_visibility="shown",
            external=True,
            origin="external",
            panes={
                "%5": PaneInfo(
                    pane_id="%5",
                    name="api",
                    provider="claude",
                    last_active_ts=1700000000.5,
                    state="blocked",
                    subscribed=True,
                ),
                "%6": PaneInfo(pane_id="%6", state="idle"),
            },
            pane_lifecycle_notify=True,
            worktree_path="/repo.worktrees/ccg-x",
            worktree_branch="ccg/x",
            gemini_external_warned=True,
            provider_manual_override=True,
        )
        loaded = WindowState.from_dict(original.to_dict())
        assert loaded == original

    def test_provider_and_session_identity_round_trip(self) -> None:
        ws = WindowState(
            session_id="aaaa-bbbb",
            cwd="/p",
            window_name="proj",
            transcript_path="/tmp/t.jsonl",
            provider_name="codex",
        )
        d = ws.to_dict()
        assert d["session_id"] == "aaaa-bbbb"
        assert d["cwd"] == "/p"
        assert d["window_name"] == "proj"
        assert d["transcript_path"] == "/tmp/t.jsonl"
        assert d["provider_name"] == "codex"
        assert WindowState.from_dict(d) == ws

    def test_origin_external_round_trip(self) -> None:
        ws = WindowState(cwd="/p", external=True, origin="external")
        d = ws.to_dict()
        assert d["external"] is True
        assert d["origin"] == "external"
        loaded = WindowState.from_dict(d)
        assert loaded.external is True
        assert loaded.origin == "external"

    def test_origin_ccgram_created_round_trip(self) -> None:
        ws = WindowState(cwd="/p", origin="ccgram_created")
        d = ws.to_dict()
        assert d["origin"] == "ccgram_created"
        assert WindowState.from_dict(d).origin == "ccgram_created"

    def test_invalid_origin_falls_back_to_default(self) -> None:
        ws = WindowState.from_dict({"session_id": "s", "cwd": "/p", "origin": "wat"})
        assert ws.origin == "manual_discovered"

    def test_approval_mode_round_trip(self) -> None:
        for mode in ("normal", "yolo"):
            ws = WindowState(cwd="/p", approval_mode=mode)
            assert WindowState.from_dict(ws.to_dict()).approval_mode == mode

    def test_batch_mode_round_trip(self) -> None:
        for mode in ("batched", "ephemeral", "verbose"):
            ws = WindowState(cwd="/p", batch_mode=mode)
            assert WindowState.from_dict(ws.to_dict()).batch_mode == mode

    def test_pane_lifecycle_override_round_trip(self) -> None:
        for value in (True, False):
            ws = WindowState(cwd="/p", pane_lifecycle_notify=value)
            assert WindowState.from_dict(ws.to_dict()).pane_lifecycle_notify is value

    def test_worktree_metadata_round_trip(self) -> None:
        ws = WindowState(
            cwd="/repo",
            worktree_path="/repo.worktrees/ccg-y",
            worktree_branch="ccg/y",
        )
        loaded = WindowState.from_dict(ws.to_dict())
        assert loaded.worktree_path == "/repo.worktrees/ccg-y"
        assert loaded.worktree_branch == "ccg/y"

    def test_gemini_external_warned_round_trip(self) -> None:
        ws = WindowState(cwd="/p", gemini_external_warned=True)
        assert WindowState.from_dict(ws.to_dict()).gemini_external_warned is True

    def test_provider_manual_override_omitted_when_false(self) -> None:
        assert "provider_manual_override" not in WindowState(cwd="/p").to_dict()

    def test_provider_manual_override_round_trip(self) -> None:
        ws = WindowState(cwd="/p", provider_manual_override=True)
        d = ws.to_dict()
        assert d["provider_manual_override"] is True
        assert WindowState.from_dict(d).provider_manual_override is True

    def test_provider_manual_override_defaults_false_on_load(self) -> None:
        ws = WindowState.from_dict({"session_id": "s", "cwd": "/p"})
        assert ws.provider_manual_override is False


class TestWindowStateTransientRcProbeFieldsNotSerialized:
    """RC probe lifecycle fields are in-memory only — must never survive a round trip."""

    def test_rc_probe_state_absent_from_to_dict(self) -> None:
        ws = WindowState(cwd="/p")
        ws.rc_probe_state = "armed"
        ws.rc_armed_at = 12345.0
        d = ws.to_dict()
        assert "rc_probe_state" not in d
        assert "rc_armed_at" not in d

    def test_rc_probe_state_defaults_after_from_dict(self) -> None:
        ws = WindowState(cwd="/p")
        ws.rc_probe_state = "classified"
        ws.rc_armed_at = 999.0
        loaded = WindowState.from_dict(ws.to_dict())
        assert loaded.rc_probe_state is None
        assert loaded.rc_armed_at is None

    def test_rc_probe_state_ignored_if_present_in_input_dict(self) -> None:
        # Even if a corrupt state.json carries the transient field, the
        # dataclass constructor must drop it on the floor.
        loaded = WindowState.from_dict(
            {
                "session_id": "s",
                "cwd": "/p",
                "rc_probe_state": "armed",
                "rc_armed_at": 42.0,
            }
        )
        assert loaded.rc_probe_state is None
        assert loaded.rc_armed_at is None
