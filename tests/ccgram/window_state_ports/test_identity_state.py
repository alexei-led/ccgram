from __future__ import annotations

from pathlib import Path

import pytest

from ccgram.window_state_ports.identity_state import (
    IdentityProjection,
    get_approval_mode,
    get_cwd,
    get_identity,
    get_provider_name,
    get_session_id,
    get_transcript_path,
    get_window_name,
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
