"""Tests for topic_creation_draft.py — module helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from ccgram.handlers.topics.directory_browser import BROWSE_PATH_KEY
from ccgram.handlers.topics.topic_creation_draft import (
    _browser_flow_stale,
    _required_selected_path,
)
from ccgram.handlers.user_state import PENDING_THREAD_ID


def _make_update(thread_id: int) -> MagicMock:
    update = MagicMock()
    update.message = None
    update.callback_query = MagicMock()
    update.callback_query.message = MagicMock()
    update.callback_query.message.message_thread_id = thread_id
    return update


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data
    return ctx


# ── _browser_flow_stale ──────────────────────────────────────────────────────


class TestBrowserFlowStale:
    def test_stale_when_no_pending_thread(self) -> None:
        context = _make_context({})
        assert _browser_flow_stale(_make_update(42), context) is True

    def test_stale_when_thread_id_mismatches(self) -> None:
        context = _make_context({PENDING_THREAD_ID: 99})
        assert _browser_flow_stale(_make_update(42), context) is True

    def test_not_stale_when_thread_id_matches(self) -> None:
        context = _make_context({PENDING_THREAD_ID: 42})
        assert _browser_flow_stale(_make_update(42), context) is False

    def test_stale_when_user_data_is_none(self) -> None:
        context = _make_context(None)
        assert _browser_flow_stale(_make_update(42), context) is True


# ── _required_selected_path ──────────────────────────────────────────────────


class TestRequiredSelectedPath:
    def test_returns_none_when_user_data_is_none(self) -> None:
        context = _make_context(None)
        assert _required_selected_path(context) is None

    def test_returns_none_when_browse_path_missing(self) -> None:
        context = _make_context({})
        assert _required_selected_path(context) is None

    def test_returns_none_for_empty_string(self) -> None:
        context = _make_context({BROWSE_PATH_KEY: ""})
        assert _required_selected_path(context) is None

    def test_returns_path_when_present(self) -> None:
        context = _make_context({BROWSE_PATH_KEY: "/my/project"})
        assert _required_selected_path(context) == "/my/project"
