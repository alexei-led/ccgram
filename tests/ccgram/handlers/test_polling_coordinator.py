"""Tests for the polling coordinator loop and orchestration functions."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot
from telegram.error import BadRequest

from ccgram.handlers.polling_coordinator import (
    _BACKOFF_MAX,
    _BACKOFF_MIN,
    _check_autoclose_timers,
    _check_unbound_window_ttl,
    _handle_dead_window_notification,
    _probe_topic_existence,
    _prune_stale_state,
)
from ccgram.handlers.polling_strategies import (
    lifecycle_strategy,
    terminal_strategy,
)


@pytest.fixture(autouse=True)
def _clean_strategy_state():
    """Reset all strategy state between tests."""
    terminal_strategy._states.clear()
    lifecycle_strategy._states.clear()
    lifecycle_strategy._dead_notified.clear()
    yield
    terminal_strategy._states.clear()
    lifecycle_strategy._states.clear()
    lifecycle_strategy._dead_notified.clear()


class TestCheckAutocloseTimers:
    @pytest.mark.asyncio
    async def test_no_topics_is_noop(self):
        bot = AsyncMock(spec=Bot)
        await _check_autoclose_timers(bot)
        bot.delete_forum_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_expired_done_topic_gets_closed(self):
        bot = AsyncMock(spec=Bot)
        bot.delete_forum_topic = AsyncMock()
        user_id, thread_id = 1, 100
        lifecycle_strategy.start_autoclose_timer(
            user_id, thread_id, "done", time.monotonic() - 99999
        )
        with (
            patch("ccgram.handlers.polling_coordinator.config") as mock_config,
            patch("ccgram.handlers.polling_coordinator.thread_router") as mock_router,
            patch(
                "ccgram.handlers.polling_coordinator.clear_topic_state",
                new_callable=AsyncMock,
            ),
        ):
            mock_config.autoclose_done_minutes = 1
            mock_router.resolve_chat_id.return_value = 42
            mock_router.get_window_for_thread.return_value = "@0"
            await _check_autoclose_timers(bot)
        bot.delete_forum_topic.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_yet_expired_topic_stays(self):
        bot = AsyncMock(spec=Bot)
        user_id, thread_id = 1, 100
        lifecycle_strategy.start_autoclose_timer(
            user_id, thread_id, "done", time.monotonic()
        )
        with patch("ccgram.handlers.polling_coordinator.config") as mock_config:
            mock_config.autoclose_done_minutes = 60
            await _check_autoclose_timers(bot)
        bot.delete_forum_topic.assert_not_called()


class TestCheckUnboundWindowTtl:
    @pytest.mark.asyncio
    async def test_no_timeout_is_noop(self):
        with patch("ccgram.handlers.polling_coordinator.config") as mock_config:
            mock_config.autoclose_done_minutes = 0
            await _check_unbound_window_ttl([])

    @pytest.mark.asyncio
    async def test_bound_window_timer_cleared(self):
        ws = terminal_strategy.get_state("@0")
        ws.unbound_timer = time.monotonic() - 100
        mock_window = MagicMock(window_id="@0", window_name="test")
        with (
            patch("ccgram.handlers.polling_coordinator.config") as mock_config,
            patch("ccgram.handlers.polling_coordinator.thread_router") as mock_router,
        ):
            mock_config.autoclose_done_minutes = 1
            mock_router.iter_thread_bindings.return_value = [(1, 100, "@0")]
            await _check_unbound_window_ttl([mock_window])
        assert ws.unbound_timer is None


class TestHandleDeadWindowNotification:
    @pytest.mark.asyncio
    async def test_sends_notification_once(self):
        bot = AsyncMock(spec=Bot)
        bot.send_message = AsyncMock(return_value=MagicMock())
        with (
            patch("ccgram.handlers.polling_coordinator.thread_router") as mock_router,
            patch("ccgram.handlers.polling_coordinator.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.polling_coordinator.update_topic_emoji",
                new_callable=AsyncMock,
            ),
            patch("ccgram.handlers.polling_coordinator.clear_tool_msg_ids_for_topic"),
            patch(
                "ccgram.handlers.polling_coordinator.rate_limit_send_message",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_router.resolve_chat_id.return_value = 42
            mock_router.get_display_name.return_value = "test"
            mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
            mock_send.return_value = MagicMock()

            await _handle_dead_window_notification(bot, 1, 100, "@0")
            assert (1, 100, "@0") in lifecycle_strategy._dead_notified

            mock_send.reset_mock()
            await _handle_dead_window_notification(bot, 1, 100, "@0")
            mock_send.assert_not_called()


class TestPruneStaleState:
    @pytest.mark.asyncio
    async def test_syncs_display_names(self):
        mock_window = MagicMock(window_id="@0", window_name="test")
        with patch("ccgram.handlers.polling_coordinator.session_manager") as mock_sm:
            await _prune_stale_state([mock_window])
            mock_sm.sync_display_names.assert_called_once_with([("@0", "test")])
            mock_sm.prune_stale_state.assert_called_once_with({"@0"})


class TestProbeTopicExistence:
    @pytest.mark.asyncio
    async def test_deleted_topic_unbinds(self):
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages = AsyncMock(
            side_effect=BadRequest("Topic_id_invalid")
        )
        with (
            patch("ccgram.handlers.polling_coordinator.thread_router") as mock_router,
            patch("ccgram.handlers.polling_coordinator.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.polling_coordinator.clear_topic_state",
                new_callable=AsyncMock,
            ),
        ):
            mock_router.iter_thread_bindings.return_value = [(1, 100, "@0")]
            mock_router.resolve_chat_id.return_value = 42
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.kill_window = AsyncMock()
            await _probe_topic_existence(bot)
            mock_router.unbind_thread.assert_called_once_with(1, 100)

    @pytest.mark.asyncio
    async def test_suspended_probe_skipped(self):
        bot = AsyncMock(spec=Bot)
        ws = terminal_strategy.get_state("@0")
        ws.probe_failures = 999
        with patch("ccgram.handlers.polling_coordinator.thread_router") as mock_router:
            mock_router.iter_thread_bindings.return_value = [(1, 100, "@0")]
            await _probe_topic_existence(bot)
        bot.unpin_all_forum_topic_messages.assert_not_called()


class TestBackoffConstants:
    def test_backoff_bounds(self):
        assert _BACKOFF_MIN == 2.0
        assert _BACKOFF_MAX == 30.0
