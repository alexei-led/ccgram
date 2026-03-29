"""Tests for topic_orchestration module."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.topic_orchestration import (
    _collect_target_chats,
    _is_window_already_bound,
    _topic_create_retry_until,
    adopt_unbound_windows,
    handle_new_window,
)
from ccgram.session_monitor import NewWindowEvent


@pytest.fixture(autouse=True)
def _clear_retry_state():
    _topic_create_retry_until.clear()
    yield
    _topic_create_retry_until.clear()


class TestIsWindowAlreadyBound:
    def test_bound_window(self):
        with patch("ccgram.handlers.topic_orchestration.thread_router") as mock_router:
            mock_router.iter_thread_bindings.return_value = [
                (1, 100, "@0"),
                (1, 200, "@5"),
            ]
            assert _is_window_already_bound("@5") is True

    def test_unbound_window(self):
        with patch("ccgram.handlers.topic_orchestration.thread_router") as mock_router:
            mock_router.iter_thread_bindings.return_value = [
                (1, 100, "@0"),
            ]
            assert _is_window_already_bound("@5") is False

    def test_no_bindings(self):
        with patch("ccgram.handlers.topic_orchestration.thread_router") as mock_router:
            mock_router.iter_thread_bindings.return_value = []
            assert _is_window_already_bound("@0") is False


class TestCollectTargetChats:
    def test_from_bindings(self):
        with patch("ccgram.handlers.topic_orchestration.thread_router") as mock_router:
            mock_router.iter_thread_bindings.return_value = [
                (1, 100, "@0"),
            ]
            mock_router.resolve_chat_id.return_value = -1001
            result = _collect_target_chats("@5")
            assert result == {-1001}

    def test_fallback_to_group_chat_ids(self):
        with patch("ccgram.handlers.topic_orchestration.thread_router") as mock_router:
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {1: -2002}
            result = _collect_target_chats("@5")
            assert result == {-2002}

    def test_fallback_to_config_group_id(self):
        with (
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_router,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {}
            mock_config.group_id = -3003
            result = _collect_target_chats("@5")
            assert result == {-3003}

    def test_no_chats_available(self):
        with (
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_router,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {}
            mock_config.group_id = None
            result = _collect_target_chats("@5")
            assert result == set()


class TestHandleNewWindow:
    @pytest.mark.asyncio
    async def test_skips_already_bound(self):
        event = NewWindowEvent(
            window_id="@0", session_id="s1", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()
        with patch(
            "ccgram.handlers.topic_orchestration._is_window_already_bound",
            return_value=True,
        ):
            await handle_new_window(event, bot)
        bot.create_forum_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_topic(self):
        event = NewWindowEvent(
            window_id="@5", session_id="s2", window_name="myproject", cwd="/tmp"
        )
        bot = AsyncMock()
        topic = MagicMock()
        topic.message_thread_id = 999
        bot.create_forum_topic.return_value = topic

        with (
            patch(
                "ccgram.handlers.topic_orchestration._is_window_already_bound",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.topic_orchestration._auto_detect_provider",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topic_orchestration._collect_target_chats",
                return_value={-1001},
            ),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_router,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = [(1, 100, "@0")]
            mock_router.resolve_chat_id.return_value = -1001
            mock_config.allowed_users = set()
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(chat_id=-1001, name="myproject")

    @pytest.mark.asyncio
    async def test_skips_when_no_chats(self):
        event = NewWindowEvent(
            window_id="@5", session_id="s2", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()

        with (
            patch(
                "ccgram.handlers.topic_orchestration._is_window_already_bound",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.topic_orchestration._auto_detect_provider",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topic_orchestration._collect_target_chats",
                return_value=set(),
            ),
        ):
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_backoff(self):
        event = NewWindowEvent(
            window_id="@5", session_id="s2", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()
        _topic_create_retry_until[-1001] = time.monotonic() + 60

        with (
            patch(
                "ccgram.handlers.topic_orchestration._is_window_already_bound",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.topic_orchestration._auto_detect_provider",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topic_orchestration._collect_target_chats",
                return_value={-1001},
            ),
        ):
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()


class TestAdoptUnboundWindows:
    @pytest.mark.asyncio
    async def test_adopts_orphaned_windows(self):
        bot = AsyncMock()
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.window_name = "test"

        mock_audit = MagicMock()
        mock_issue = MagicMock()
        mock_issue.category = "orphaned_window"
        mock_audit.issues = [mock_issue]

        with (
            patch("ccgram.handlers.topic_orchestration.tmux_manager") as mock_tmux,
            patch("ccgram.handlers.topic_orchestration.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.topic_orchestration._adopt_orphaned_windows",
                new_callable=AsyncMock,
                create=True,
            ),
        ):
            mock_tmux.list_windows = AsyncMock(return_value=[mock_window])
            mock_sm.audit_state.return_value = mock_audit

            with patch(
                "ccgram.handlers.sync_command._adopt_orphaned_windows",
                new_callable=AsyncMock,
            ) as mock_adopt:
                await adopt_unbound_windows(bot)
                mock_adopt.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_orphans_skips(self):
        bot = AsyncMock()
        mock_audit = MagicMock()
        mock_audit.issues = []

        with (
            patch("ccgram.handlers.topic_orchestration.tmux_manager") as mock_tmux,
            patch("ccgram.handlers.topic_orchestration.session_manager") as mock_sm,
        ):
            mock_tmux.list_windows = AsyncMock(return_value=[])
            mock_sm.audit_state.return_value = mock_audit
            await adopt_unbound_windows(bot)
