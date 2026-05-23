from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ccgram.window_state_ports.identity_state import (
    IdentityProjection,
    clear_transcript_path,
    get_approval_mode,
    get_cwd,
    get_identity,
    get_provider_name,
    get_session_id,
    get_transcript_path,
    get_window_name,
    is_provider_manually_overridden,
    set_provider_manual_override,
    set_window_approval_mode,
)
from ccgram.window_state_store import WindowState, WindowStateStore


class TestReads:
    def test_get_identity_missing(self, store: WindowStateStore) -> None:
        assert get_identity("@missing") is None

    def test_get_identity_full(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(
            session_id="sid",
            cwd="/proj",
            window_name="ccgram",
            transcript_path="/tmp/t.jsonl",
            provider_name="claude",
            approval_mode="yolo",
        )
        ident = get_identity("@1")
        assert ident == IdentityProjection(
            window_id="@1",
            provider_name="claude",
            session_id="sid",
            cwd="/proj",
            transcript_path=Path("/tmp/t.jsonl"),
            window_name="ccgram",
            approval_mode="yolo",
        )

    def test_identity_no_transcript_path(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(cwd="/p")
        ident = get_identity("@1")
        assert ident is not None
        assert ident.transcript_path is None

    def test_identity_invalid_approval_falls_back(
        self, store: WindowStateStore
    ) -> None:
        store.window_states["@1"] = WindowState(approval_mode="garbage")
        ident = get_identity("@1")
        assert ident is not None
        assert ident.approval_mode == "normal"

    def test_individual_field_reads(self, store: WindowStateStore) -> None:
        store.window_states["@1"] = WindowState(
            session_id="sid",
            cwd="/proj",
            window_name="ccgram",
            transcript_path="/tmp/t.jsonl",
            provider_name="claude",
        )
        assert get_provider_name("@1") == "claude"
        assert get_session_id("@1") == "sid"
        assert get_cwd("@1") == "/proj"
        assert get_transcript_path("@1") == "/tmp/t.jsonl"
        assert get_window_name("@1") == "ccgram"

    def test_field_reads_default_on_missing(self, store: WindowStateStore) -> None:
        assert get_provider_name("@missing") is None
        assert get_session_id("@missing") is None
        assert get_cwd("@missing") == ""
        assert get_transcript_path("@missing") == ""
        assert get_window_name("@missing") == ""

    def test_get_approval_mode(self, store: WindowStateStore) -> None:
        assert get_approval_mode("@missing") == "normal"
        store.window_states["@1"] = WindowState(approval_mode="yolo")
        assert get_approval_mode("@1") == "yolo"

    def test_get_approval_mode_invalid_value_falls_back(
        self, store: WindowStateStore
    ) -> None:
        store.window_states["@1"] = WindowState(approval_mode="garbage")
        assert get_approval_mode("@1") == "normal"

    @pytest.mark.parametrize(
        "field_value, expected",
        [
            (True, True),
            (False, False),
            # MagicMock is truthy but not `is True` — verifies the production
            # guard that prevents a mock attribute from short-circuiting detection.
            (MagicMock(), False),
        ],
    )
    def test_is_provider_manually_overridden_is_true_guard(
        self,
        store: WindowStateStore,
        field_value: object,
        expected: bool,
    ) -> None:
        store.window_states["@1"] = WindowState()
        store.window_states["@1"].provider_manual_override = field_value  # type: ignore[assignment]
        assert is_provider_manually_overridden("@1") is expected

    def test_is_provider_manually_overridden_missing_window(
        self, store: WindowStateStore
    ) -> None:
        assert is_provider_manually_overridden("@missing") is False


class TestWrites:
    def test_set_approval_mode_persists(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_window_approval_mode("@1", "yolo")
        assert store.window_states["@1"].approval_mode == "yolo"
        assert len(save_calls) == 1

    def test_set_approval_mode_case_insensitive(self, store: WindowStateStore) -> None:
        set_window_approval_mode("@1", "YOLO")
        assert store.window_states["@1"].approval_mode == "yolo"

    def test_set_approval_mode_rejects_invalid(self, store: WindowStateStore) -> None:
        with pytest.raises(ValueError):
            set_window_approval_mode("@1", "garbage")

    def test_set_provider_manual_override_sets_and_saves(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        store.window_states["@1"] = WindowState()
        set_provider_manual_override("@1", value=True)
        assert store.window_states["@1"].provider_manual_override is True
        assert len(save_calls) == 1

    def test_set_provider_manual_override_clears_and_saves(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        store.window_states["@1"] = WindowState(provider_manual_override=True)
        set_provider_manual_override("@1", value=False)
        assert store.window_states["@1"].provider_manual_override is False
        assert len(save_calls) == 1

    def test_set_provider_manual_override_no_op_same_value(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        store.window_states["@1"] = WindowState()
        set_provider_manual_override("@1", value=False)
        assert save_calls == []

    def test_set_provider_manual_override_no_op_missing_window(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        set_provider_manual_override("@missing", value=True)
        assert "@missing" not in store.window_states
        assert save_calls == []

    def test_clear_transcript_path_clears_and_saves(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        store.window_states["@1"] = WindowState(transcript_path="/tmp/t.jsonl")
        clear_transcript_path("@1")
        assert store.window_states["@1"].transcript_path == ""
        assert len(save_calls) == 1

    def test_clear_transcript_path_no_op_when_already_empty(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        store.window_states["@1"] = WindowState()
        clear_transcript_path("@1")
        assert save_calls == []

    def test_clear_transcript_path_no_op_missing_window(
        self, store: WindowStateStore, save_calls: list[int]
    ) -> None:
        clear_transcript_path("@missing")
        assert save_calls == []
