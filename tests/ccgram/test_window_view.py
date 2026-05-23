"""Tests for WindowView projection and SessionManager.view_window."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from ccgram.session import SessionManager
from ccgram.window_state_store import WindowState, window_store
from ccgram.window_view import WindowView


def _wire_store() -> SessionManager:
    """Instantiate a fresh SessionManager so window_store has a real save callback."""
    with patch.object(SessionManager, "_load_state"):
        return SessionManager()


class TestWindowViewProjection:
    def test_view_window_returns_none_for_missing(self) -> None:
        sm = _wire_store()
        window_store.window_states.clear()
        assert sm.view_window("@404") is None

    def test_view_window_projects_existing_state(self) -> None:
        sm = _wire_store()
        window_store.window_states["@1"] = WindowState(
            cwd="/tmp/proj",
            provider_name="claude",
            approval_mode="normal",
            transcript_path="/tmp/log.jsonl",
        )
        view = sm.view_window("@1")
        assert view == WindowView(
            window_id="@1",
            cwd="/tmp/proj",
            provider_name="claude",
            approval_mode="normal",
            batch_mode="ephemeral",
            tool_call_visibility="default",
            transcript_path=Path("/tmp/log.jsonl"),
            window_name="",
            session_id="",
            external=False,
            origin="manual_discovered",
        )
        # cleanup
        window_store.window_states.pop("@1", None)

    def test_view_window_normalizes_empty_strings(self) -> None:
        sm = _wire_store()
        window_store.window_states["@2"] = WindowState(
            cwd="",
            provider_name="codex",
            approval_mode="normal",
            transcript_path="",
        )
        view = sm.view_window("@2")
        assert view is not None
        assert view.cwd == ""
        assert view.transcript_path is None
        window_store.window_states.pop("@2", None)

    def test_view_window_is_frozen(self) -> None:
        sm = _wire_store()
        window_store.window_states["@3"] = WindowState(cwd="/x", provider_name="claude")
        view = sm.view_window("@3")
        assert view is not None
        with pytest.raises(FrozenInstanceError):
            view.cwd = "/y"  # type: ignore[misc]
        window_store.window_states.pop("@3", None)
