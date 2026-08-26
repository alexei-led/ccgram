"""Tests for vim mode detection and auto-INSERT recovery in tmux_manager."""

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.multiplexer.tmux import TmuxManager
from ccgram.multiplexer.vim_state import (
    _vim_locks,
    _vim_state,
    clear_vim_state,
    has_insert_indicator,
    notify_vim_insert_seen,
    reset_vim_state,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_vim_state()
    yield
    reset_vim_state()


@pytest.fixture
def manager() -> TmuxManager:
    m = TmuxManager.__new__(TmuxManager)
    m.session_name = "test"
    m._server = None
    return m


# ── has_insert_indicator ──────────────────────────────────────────────


class TestHasInsertIndicator:
    @pytest.mark.parametrize(
        ("pane", "expected"),
        [
            pytest.param(
                "some output\nprompt> hello\n-- INSERT --", True, id="last-line"
            ),
            pytest.param("line1\n-- INSERT --\nlast line", True, id="second-to-last"),
            pytest.param("-- INSERT --\nsecond\nthird", True, id="third-to-last"),
            pytest.param("output\n  -- INSERT --  \ndone", True, id="padded"),
            pytest.param("some output\nprompt> hello\n", False, id="absent"),
            pytest.param("", False, id="empty-pane"),
            pytest.param(
                "-- INSERT --\nline2\nline3\nline4", False, id="outside-tail-window"
            ),
            pytest.param(
                "output\nstatus: -- INSERT -- (paste)\ndone",
                False,
                id="not-the-whole-line",
            ),
            pytest.param(
                "output\n-- INSERT -- ⏸ plan mode on (shift+tab to cycle)\ndone",
                False,
                id="claude-status-bar",
            ),
        ],
    )
    def test_detects_only_a_bare_insert_line_in_the_tail(
        self, pane: str, expected: bool
    ) -> None:
        assert has_insert_indicator(pane) is expected


# ── notify / clear / reset ─────────────────────────────────────────────


class TestVimStateCache:
    def test_notify_sets_true(self):
        notify_vim_insert_seen("@1")
        assert _vim_state["@1"] is True

    def test_clear_removes_entry_and_lock(self):
        _vim_state["@1"] = True
        _vim_locks["@1"] = asyncio.Lock()
        clear_vim_state("@1")
        assert "@1" not in _vim_state
        assert "@1" not in _vim_locks

    def test_clear_missing_key_is_noop(self):
        clear_vim_state("@999")

    def test_reset_clears_all(self):
        _vim_state["@1"] = True
        _vim_state["@2"] = False
        _vim_locks["@1"] = asyncio.Lock()
        reset_vim_state()
        assert _vim_state == {}
        assert _vim_locks == {}


# ── _ensure_vim_insert_mode ────────────────────────────────────────────


class TestEnsureVimInsertMode:
    @pytest.mark.parametrize(
        "cached",
        [pytest.param(False, id="known-not-vim"), pytest.param(None, id="unknown")],
    )
    async def test_never_probes_a_window_not_known_to_run_vim(
        self, manager, cached: bool | None
    ):
        """An unknown window is never speculatively probed — no 'i' can leak."""
        if cached is not None:
            _vim_state["@1"] = cached
        with (
            patch.object(manager, "capture_pane", new_callable=AsyncMock) as cap,
            patch.object(manager, "_pane_send") as send,
        ):
            await manager._ensure_vim_insert_mode("@1")
        cap.assert_not_called()
        send.assert_not_called()
        assert _vim_state.get("@1") is cached

    async def test_normal_mode_enters_insert(self, manager):
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

    @pytest.mark.parametrize(
        "pane_text",
        [
            pytest.param("prompt\n-- INSERT --", id="already-insert"),
            pytest.param(None, id="capture-failed"),
        ],
    )
    async def test_sends_nothing_when_insert_is_not_needed_or_unknown(
        self, manager, pane_text: str | None
    ):
        _vim_state["@1"] = True
        with (
            patch.object(
                manager,
                "capture_pane",
                new_callable=AsyncMock,
                return_value=pane_text,
            ),
            patch.object(manager, "_pane_send") as send,
        ):
            await manager._ensure_vim_insert_mode("@1")
        send.assert_not_called()
        assert _vim_state["@1"] is True


# ── mid-session self-correction ────────────────────────────────────────


class TestSelfCorrection:
    async def test_vim_enabled_mid_session(self, manager):
        """A window cached as not-vim must be upgraded once polling sees INSERT.

        Without the overwrite the window would stay `False` forever and its
        first keystrokes after opening vim would be swallowed.
        """
        _vim_state["@1"] = False
        with patch.object(manager, "capture_pane", new_callable=AsyncMock) as cap:
            await manager._ensure_vim_insert_mode("@1")
            cap.assert_not_called()

        notify_vim_insert_seen("@1")
        assert _vim_state["@1"] is True


# ── _send_literal_then_enter integration ───────────────────────────────


class TestSendLiteralVimIntegration:
    async def test_vim_check_runs_before_text_send(self, manager):
        _vim_state["@1"] = False  # fast path — skip vim check
        with (
            patch.object(
                manager, "_ensure_vim_insert_mode", new_callable=AsyncMock
            ) as vim_check,
            patch.object(manager, "_pane_send", return_value=True),
            patch("ccgram.multiplexer.tmux.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await manager._send_literal_then_enter("@1", "hello")
        assert result is True
        vim_check.assert_awaited_once_with("@1")

    async def test_per_window_lock_serializes_sends(self, manager):
        """Concurrent sends to the same window are serialized by lock."""
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
            patch("ccgram.multiplexer.tmux.asyncio.sleep", new_callable=AsyncMock),
        ):
            await asyncio.gather(
                manager._send_literal_then_enter("@1", "a"),
                manager._send_literal_then_enter("@1", "b"),
            )
        assert order == ["vim_start", "vim_end", "vim_start", "vim_end"]


# ── Polling + cleanup integration ──────────────────────────────────────


async def _run_update_status(pane_text: str) -> MagicMock:
    """Drive one status poll over *pane_text*; return the notify spy."""
    window = MagicMock(window_id="@9", pane_current_command="claude")
    provider = MagicMock()
    provider.parse_terminal_status.return_value = None
    provider.capabilities.uses_pane_title = False

    with ExitStack() as stack:
        patches = {
            "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id": AsyncMock(
                return_value=window
            ),
            "ccgram.multiplexer.tmux.tmux_manager.capture_pane": AsyncMock(
                return_value=pane_text
            ),
            "ccgram.handlers.polling.window_tick.observe._parse_with_pyte": MagicMock(
                return_value=None
            ),
            "ccgram.handlers.polling.window_tick.apply.get_provider_for_window": (
                MagicMock(return_value=provider)
            ),
            "ccgram.handlers.polling.window_tick.apply.get_interactive_window": (
                MagicMock(return_value=None)
            ),
            "ccgram.handlers.polling.window_tick.apply._apply_tick_decision": (
                AsyncMock()
            ),
        }
        for target, replacement in patches.items():
            stack.enter_context(patch(target, replacement))
        notify = stack.enter_context(
            patch(
                "ccgram.handlers.polling.window_tick.observe.notify_vim_insert_seen",
                wraps=notify_vim_insert_seen,
            )
        )

        from ccgram.handlers.polling.window_tick import _update_status

        await _update_status(AsyncMock(), 1, "@9", thread_id=42)
    return notify


class TestPollingAndCleanupIntegration:
    @pytest.mark.parametrize(
        ("pane_text", "notified"),
        [
            pytest.param("output\nprompt\n-- INSERT --", True, id="insert-in-tail"),
            pytest.param(
                "-- INSERT --\nline2\nline3\nline4", False, id="insert-only-in-history"
            ),
        ],
    )
    async def test_polling_warms_the_cache_only_from_the_pane_tail(
        self, pane_text: str, notified: bool
    ):
        notify = await _run_update_status(pane_text)
        assert notify.called is notified
        assert _vim_state.get("@9") is (True if notified else None)

    @pytest.mark.parametrize(
        ("window_id", "cleared"),
        [
            pytest.param("@7", True, id="with-window-id"),
            pytest.param(None, False, id="without-window-id"),
        ],
    )
    async def test_clear_topic_state_clears_vim_state_only_with_a_window_id(
        self, window_id: str | None, cleared: bool
    ):
        from ccgram.handlers.cleanup import clear_topic_state

        _vim_state["@7"] = True
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update"),
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.thread_router.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, client=AsyncMock(), window_id=window_id)
        assert ("@7" not in _vim_state) is cleared
