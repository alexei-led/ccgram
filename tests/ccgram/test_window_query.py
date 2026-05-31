"""Tests for window_query — read-only free functions over window_store."""

from __future__ import annotations

import pytest

from ccgram.window_query import (
    get_approval_mode,
    get_batch_mode,
    get_session_id_for_window,
    get_tool_call_visibility,
    get_window_provider,
    is_ephemeral_tools,
    is_tool_calls_hidden,
    iter_window_ids,
    view_window,
    window_count,
)
from ccgram.window_state_store import WindowState, WindowStateStore


@pytest.fixture(autouse=True)
def _store(monkeypatch) -> WindowStateStore:
    store = WindowStateStore(
        schedule_save=lambda: None,
        on_hookless_provider_switch=lambda _wid: None,
    )
    monkeypatch.setattr("ccgram.window_query.window_store", store)
    monkeypatch.setattr("ccgram.window_state_store.window_store", store)
    monkeypatch.setattr("ccgram.window_state_ports.identity_state.window_store", store)
    monkeypatch.setattr("ccgram.window_state_ports.lifecycle_state.window_store", store)
    monkeypatch.setattr("ccgram.window_state_ports.tool_state.window_store", store)
    monkeypatch.setattr("ccgram.window_state_ports.worktree_state.window_store", store)
    return store


@pytest.fixture
def populated(_store: WindowStateStore) -> WindowStateStore:
    _store.window_states["@1"] = WindowState(
        session_id="sid1",
        cwd="/proj",
        provider_name="claude",
        approval_mode="yolo",
        batch_mode="verbose",
        window_name="myproj",
        transcript_path="/tmp/t.jsonl",
    )
    return _store


class TestViewWindow:
    def test_returns_none_for_unknown(self) -> None:
        assert view_window("@missing") is None

    def test_returns_snapshot(self, populated) -> None:
        v = view_window("@1")
        assert v is not None
        assert v.window_id == "@1"
        assert v.cwd == "/proj"
        assert v.provider_name == "claude"
        assert v.session_id == "sid1"
        assert v.window_name == "myproj"
        assert v.origin == "manual_discovered"

    def test_transcript_path_as_path_object(self, populated) -> None:
        v = view_window("@1")
        assert v is not None
        assert v.transcript_path is not None
        assert str(v.transcript_path) == "/tmp/t.jsonl"

    def test_no_transcript_path_is_none(self, _store: WindowStateStore) -> None:
        _store.window_states["@2"] = WindowState(cwd="/p")
        v = view_window("@2")
        assert v is not None
        assert v.transcript_path is None


class TestGetWindowProvider:
    def test_returns_none_for_unknown(self) -> None:
        assert get_window_provider("@missing") is None

    def test_returns_provider_name(self, populated) -> None:
        assert get_window_provider("@1") == "claude"

    def test_empty_provider_returns_empty_string(
        self, _store: WindowStateStore
    ) -> None:
        _store.window_states["@2"] = WindowState()
        assert get_window_provider("@2") == ""


class TestGetApprovalMode:
    def test_unknown_window_returns_default(self) -> None:
        assert get_approval_mode("@missing") == "normal"

    def test_returns_stored_mode(self, populated) -> None:
        assert get_approval_mode("@1") == "yolo"

    def test_corrupt_value_falls_back_to_default(
        self, _store: WindowStateStore
    ) -> None:
        _store.window_states["@2"] = WindowState(approval_mode="garbage")
        assert get_approval_mode("@2") == "normal"


class TestGetBatchMode:
    def test_unknown_window_returns_batched(self) -> None:
        assert get_batch_mode("@missing") == "ephemeral"

    def test_returns_stored_mode(self, populated) -> None:
        assert get_batch_mode("@1") == "verbose"

    def test_corrupt_value_falls_back_to_default(
        self, _store: WindowStateStore
    ) -> None:
        _store.window_states["@2"] = WindowState(batch_mode="garbage")
        assert get_batch_mode("@2") == "ephemeral"


class TestGetSessionId:
    def test_returns_none_for_unknown(self) -> None:
        assert get_session_id_for_window("@missing") is None

    def test_returns_session_id(self, populated) -> None:
        assert get_session_id_for_window("@1") == "sid1"

    def test_empty_session_id_returns_none(self, _store: WindowStateStore) -> None:
        _store.window_states["@2"] = WindowState()
        assert get_session_id_for_window("@2") is None


class TestWindowCount:
    def test_empty_store(self) -> None:
        assert window_count() == 0

    def test_counts_windows(self, populated) -> None:
        assert window_count() == 1


class TestIterWindowIds:
    def test_empty(self) -> None:
        assert iter_window_ids() == []

    def test_returns_ids(self, populated) -> None:
        assert iter_window_ids() == ["@1"]


class TestToolCallDelegations:
    def test_is_ephemeral_tools_unknown_window(self) -> None:
        assert is_ephemeral_tools("@missing") is True

    def test_is_ephemeral_tools_returns_false_for_non_ephemeral(
        self, _store: WindowStateStore
    ) -> None:
        _store.window_states["@1"] = WindowState(batch_mode="batched")
        assert is_ephemeral_tools("@1") is False

    def test_get_tool_call_visibility_unknown_window(self) -> None:
        assert get_tool_call_visibility("@missing") == "default"

    def test_get_tool_call_visibility_returns_stored_value(
        self, _store: WindowStateStore
    ) -> None:
        _store.window_states["@1"] = WindowState(tool_call_visibility="hidden")
        assert get_tool_call_visibility("@1") == "hidden"

    def test_is_tool_calls_hidden_unknown_window_uses_global_config(
        self, _store: WindowStateStore, monkeypatch
    ) -> None:
        from ccgram.window_state_ports import tool_state as _ts

        monkeypatch.setattr(_ts, "_resolve_tool_calls_hidden", lambda _v: True)
        assert is_tool_calls_hidden("@missing") is True

    def test_is_tool_calls_hidden_shown_overrides_global(
        self, _store: WindowStateStore
    ) -> None:
        _store.window_states["@1"] = WindowState(tool_call_visibility="shown")
        assert is_tool_calls_hidden("@1") is False

    def test_is_tool_calls_hidden_hidden_overrides_global(
        self, _store: WindowStateStore
    ) -> None:
        _store.window_states["@1"] = WindowState(tool_call_visibility="hidden")
        assert is_tool_calls_hidden("@1") is True
