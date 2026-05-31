"""Tests for vim mode detection and auto-INSERT recovery in tmux_manager."""

from unittest.mock import AsyncMock, patch

import pytest

from ccgram.tmux_manager import (
    TmuxManager,
    has_insert_indicator,
    _vim_locks,
    _vim_state,
    clear_vim_state,
    notify_vim_insert_seen,
    reset_vim_state,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_vim_state()
    yield
    reset_vim_state()


# ── has_insert_indicator ──────────────────────────────────────────────


class TestHasInsertIndicator:
    def test_insert_on_last_line(self):
        pane = "some output\nprompt> hello\n-- INSERT --"
        assert has_insert_indicator(pane) is True

    def test_insert_second_to_last(self):
        pane = "line1\n-- INSERT --\nlast line"
        assert has_insert_indicator(pane) is True

    def test_insert_third_to_last(self):
        pane = "-- INSERT --\nsecond\nthird"
        assert has_insert_indicator(pane) is True

    def test_no_insert_indicator(self):
        pane = "some output\nprompt> hello\n"
        assert has_insert_indicator(pane) is False

    def test_insert_too_far_up(self):
        pane = "-- INSERT --\nline2\nline3\nline4"
        assert has_insert_indicator(pane) is False

    def test_empty_pane(self):
        assert has_insert_indicator("") is False

    def test_insert_with_surrounding_text(self):
        pane = "output\nstatus: -- INSERT -- (paste)\ndone"
        assert has_insert_indicator(pane) is False

    def test_insert_with_whitespace(self):
        pane = "output\n  -- INSERT --  \ndone"
        assert has_insert_indicator(pane) is True

    def test_claude_status_bar_insert(self):
        pane = "output\n-- INSERT -- ⏸ plan mode on (shift+tab to cycle)\ndone"
        assert has_insert_indicator(pane) is False


# ── notify / clear / reset ─────────────────────────────────────────────


class TestVimStateCache:
    def test_notify_sets_true(self):
        notify_vim_insert_seen("@1")
        assert _vim_state["@1"] is True

    def test_clear_removes_entry(self):
        _vim_state["@1"] = True
        clear_vim_state("@1")
        assert "@1" not in _vim_state

    def test_clear_also_removes_lock(self):
        import asyncio

        _vim_locks["@1"] = asyncio.Lock()
        clear_vim_state("@1")
        assert "@1" not in _vim_locks

    def test_clear_missing_key_is_noop(self):
        clear_vim_state("@999")

    def test_reset_clears_all(self):
        import asyncio

        _vim_state["@1"] = True
        _vim_state["@2"] = False
        _vim_locks["@1"] = asyncio.Lock()
        reset_vim_state()
        assert _vim_state == {}
        assert _vim_locks == {}


# ── _ensure_vim_insert_mode ────────────────────────────────────────────


def _make_manager() -> TmuxManager:
    m = TmuxManager.__new__(TmuxManager)
    m.session_name = "test"
    m._server = None
    return m


class TestEnsureVimInsertMode:
    @pytest.fixture()
    def manager(self):
        return _make_manager()

    async def test_cache_false_skips_entirely(self, manager):
        _vim_state["@1"] = False
        with (
            patch.object(manager, "capture_pane", new_callable=AsyncMock) as cap,
            patch.object(manager, "_pane_send") as send,
        ):
            await manager._ensure_vim_insert_mode("@1")
        cap.assert_not_called()
        send.assert_not_called()

    async def test_unknown_state_skips_entirely(self, manager):
        """Unknown vim state never speculatively probes — no 'i' can leak."""
        assert "@1" not in _vim_state
        with (
            patch.object(manager, "capture_pane", new_callable=AsyncMock) as cap,
            patch.object(manager, "_pane_send") as send,
        ):
            await manager._ensure_vim_insert_mode("@1")
        cap.assert_not_called()
        send.assert_not_called()
        assert "@1" not in _vim_state

    async def test_cache_true_already_insert_sends_nothing(self, manager):
        _vim_state["@1"] = True
        with (
            patch.object(
                manager,
                "capture_pane",
                new_callable=AsyncMock,
                return_value="prompt\n-- INSERT --",
            ),
            patch.object(manager, "_pane_send") as send,
        ):
            await manager._ensure_vim_insert_mode("@1")
        send.assert_not_called()
        assert _vim_state["@1"] is True

    async def test_cache_true_normal_mode_enters_insert(self, manager):
        _vim_state["@1"] = True
        with (
            patch.object(
                manager,
                "capture_pane",
                new_callable=AsyncMock,
                return_value="prompt>",
            ),
            patch.object(manager, "_pane_send", return_value=True) as send,
        ):
            await manager._ensure_vim_insert_mode("@1")
        send.assert_called_once_with("@1", "i", enter=False, literal=True)
        assert _vim_state["@1"] is True

    async def test_cache_true_capture_failure_sends_nothing(self, manager):
        """Capture failure on a vim window sends nothing — resilient, no leak."""
        _vim_state["@1"] = True
        with (
            patch.object(
                manager, "capture_pane", new_callable=AsyncMock, return_value=None
            ),
            patch.object(manager, "_pane_send") as send,
        ):
            await manager._ensure_vim_insert_mode("@1")
        send.assert_not_called()
        assert _vim_state["@1"] is True


# ── Self-correction scenarios ──────────────────────────────────────────


class TestSelfCorrection:
    @pytest.fixture()
    def manager(self):
        return _make_manager()

    async def test_vim_enabled_mid_session(self, manager):
        _vim_state["@1"] = False
        with patch.object(manager, "capture_pane", new_callable=AsyncMock) as cap:
            await manager._ensure_vim_insert_mode("@1")
            cap.assert_not_called()

        notify_vim_insert_seen("@1")
        assert _vim_state["@1"] is True


# ── _send_literal_then_enter integration ───────────────────────────────


class TestSendLiteralVimIntegration:
    @pytest.fixture()
    def manager(self):
        return _make_manager()

    async def test_vim_check_runs_before_text_send(self, manager):
        """_send_literal_then_enter calls _ensure_vim_insert_mode first."""
        _vim_state["@1"] = False  # fast path — skip vim check
        with (
            patch.object(
                manager, "_ensure_vim_insert_mode", new_callable=AsyncMock
            ) as vim_check,
            patch.object(manager, "_pane_send", return_value=True),
            patch("ccgram.tmux_manager.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await manager._send_literal_then_enter("@1", "hello")
        assert result is True
        vim_check.assert_awaited_once_with("@1")

    async def test_per_window_lock_serializes_sends(self, manager):
        """Concurrent sends to the same window are serialized by lock."""
        import asyncio

        order = []

        async def slow_vim_check(_wid):
            order.append("vim_start")
            await asyncio.sleep(0)
            order.append("vim_end")

        with (
            patch.object(
                manager, "_ensure_vim_insert_mode", side_effect=slow_vim_check
            ),
            patch.object(manager, "_pane_send", return_value=True),
            patch("ccgram.tmux_manager.asyncio.sleep", new_callable=AsyncMock),
        ):
            await asyncio.gather(
                manager._send_literal_then_enter("@1", "a"),
                manager._send_literal_then_enter("@1", "b"),
            )
        # Both calls should complete; lock ensures no interleaving of probe+send
        assert order.count("vim_start") == 2
        assert order.count("vim_end") == 2


# ── Polling + cleanup integration ──────────────────────────────────────


class TestPollingAndCleanupIntegration:
    def test_notify_from_polling_warms_cache(self):
        assert "@5" not in _vim_state
        notify_vim_insert_seen("@5")
        assert _vim_state["@5"] is True

    def test_cleanup_removes_state(self):
        _vim_state["@5"] = True
        clear_vim_state("@5")
        assert "@5" not in _vim_state

    async def test_cleanup_clears_vim_state_with_window_id(self):
        """clear_topic_state calls clear_vim_state when window_id is provided."""
        from ccgram.handlers.cleanup import clear_topic_state

        _vim_state["@7"] = True
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update"),
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.thread_router.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, client=AsyncMock(), window_id="@7")
        assert "@7" not in _vim_state

    async def test_cleanup_skips_vim_state_without_window_id(self):
        """clear_topic_state does NOT call clear_vim_state when window_id is None."""
        from ccgram.handlers.cleanup import clear_topic_state

        _vim_state["@7"] = True
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update"),
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.thread_router.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, client=AsyncMock(), window_id=None)
        # Vim state for @7 should remain untouched
        assert _vim_state["@7"] is True

    async def test_update_status_calls_notify_when_insert_in_tail(self):
        """Polling calls notify_vim_insert_seen when INSERT is in last 3 lines."""
        with (
            patch(
                "ccgram.handlers.polling.window_tick.apply.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
            ) as mock_find,
            patch(
                "ccgram.handlers.polling.window_tick.apply.tmux_manager.capture_pane",
                new_callable=AsyncMock,
                return_value="output\nprompt\n-- INSERT --",
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe.notify_vim_insert_seen",
                wraps=notify_vim_insert_seen,
            ) as mock_notify,
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window"
            ) as mock_gpw,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply._apply_tick_decision",
                new_callable=AsyncMock,
            ),
        ):
            from unittest.mock import MagicMock

            mock_win = MagicMock()
            mock_win.window_id = "@9"
            mock_win.pane_current_command = "claude"
            mock_find.return_value = mock_win
            mock_provider = MagicMock()
            mock_provider.parse_terminal_status.return_value = None
            mock_provider.capabilities.uses_pane_title = False
            mock_gpw.return_value = mock_provider

            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )

            await update_status_message(AsyncMock(), 1, "@9", thread_id=42)
        mock_notify.assert_called_once_with("@9")

    async def test_update_status_skips_notify_when_insert_not_in_tail(self):
        """Polling does NOT call notify when INSERT is only in historical output."""
        with (
            patch(
                "ccgram.handlers.polling.window_tick.apply.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
            ) as mock_find,
            patch(
                "ccgram.handlers.polling.window_tick.apply.tmux_manager.capture_pane",
                new_callable=AsyncMock,
                return_value="-- INSERT --\nline2\nline3\nline4",
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe.notify_vim_insert_seen",
                wraps=notify_vim_insert_seen,
            ) as mock_notify,
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window"
            ) as mock_gpw,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply._apply_tick_decision",
                new_callable=AsyncMock,
            ),
        ):
            from unittest.mock import MagicMock

            mock_win = MagicMock()
            mock_win.window_id = "@9"
            mock_win.pane_current_command = "claude"
            mock_find.return_value = mock_win
            mock_provider = MagicMock()
            mock_provider.parse_terminal_status.return_value = None
            mock_provider.capabilities.uses_pane_title = False
            mock_gpw.return_value = mock_provider

            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )

            await update_status_message(AsyncMock(), 1, "@9", thread_id=42)
        mock_notify.assert_not_called()
