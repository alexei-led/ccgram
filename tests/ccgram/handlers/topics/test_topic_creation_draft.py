"""Tests for topic_creation_draft.py — module helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

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


class TestBrowserFlowStale:
    def test_not_stale_when_thread_id_matches(self) -> None:
        assert (
            _browser_flow_stale(
                _make_update(42), _make_context({PENDING_THREAD_ID: 42})
            )
            is False
        )

    @pytest.mark.parametrize(
        "user_data",
        [None, {}, {PENDING_THREAD_ID: 99}],
        ids=["no_user_data", "flow_reset", "cross_topic_tap"],
    )
    def test_stale(self, user_data: dict | None) -> None:
        assert _browser_flow_stale(_make_update(42), _make_context(user_data)) is True


class TestRequiredSelectedPath:
    def test_returns_path_when_present(self) -> None:
        context = _make_context({BROWSE_PATH_KEY: "/my/project"})
        assert _required_selected_path(context) == "/my/project"

    @pytest.mark.parametrize(
        "user_data",
        [None, {}, {BROWSE_PATH_KEY: ""}, {BROWSE_PATH_KEY: 42}],
        ids=["no_user_data", "key_missing", "empty_string", "non_string"],
    )
    def test_returns_none_rather_than_falling_back_to_cwd(
        self, user_data: dict[str, Any] | None
    ) -> None:
        assert _required_selected_path(_make_context(user_data)) is None
