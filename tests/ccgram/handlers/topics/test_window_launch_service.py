"""Tests for window_launch_service.py — _cwd_within, _persist_worktree_state, launch_window."""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from ccgram.multiplexer.base import TopicTargetResult
from ccgram.handlers.topics.window_launch_service import (
    WindowLaunchRequest,
    _cwd_within,
    _persist_worktree_state,
    launch_window,
)
from ccgram.handlers.user_state import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_PATH,
)


# ── _cwd_within ──────────────────────────────────────────────────────────────


class TestCwdWithin:
    def test_exact_match_returns_true(self, tmp_path):
        assert _cwd_within(str(tmp_path), str(tmp_path)) is True

    def test_subdir_returns_true(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        assert _cwd_within(str(sub), str(tmp_path)) is True

    def test_sibling_returns_false(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert _cwd_within(str(a), str(b)) is False

    def test_parent_returns_false(self, tmp_path):
        sub = tmp_path / "child"
        sub.mkdir()
        assert _cwd_within(str(tmp_path), str(sub)) is False

    def test_nonexistent_path_returns_false(self):
        # Path.resolve() is non-strict by default — does not raise on nonexistent paths.
        # Both paths resolve to themselves, so a != b → False.
        assert _cwd_within("/nonexistent/a", "/nonexistent/b") is False


# ── _persist_worktree_state ──────────────────────────────────────────────────


class TestPersistWorktreeState:
    @patch("ccgram.handlers.topics.window_launch_service.session_manager")
    def test_writes_worktree_state_when_cwd_matches(
        self, mock_sm: MagicMock, tmp_path
    ) -> None:
        wt_path = str(tmp_path)
        user_data = {
            PENDING_WORKTREE_PATH: wt_path,
            PENDING_WORKTREE_BRANCH: "ccg/feat",
        }
        context = MagicMock()
        context.user_data = user_data

        _persist_worktree_state("@1", wt_path, context)

        mock_sm.set_window_worktree.assert_called_once_with("@1", wt_path, "ccg/feat")

    @patch("ccgram.handlers.topics.window_launch_service.session_manager")
    def test_clears_worktree_state_after_call(
        self, mock_sm: MagicMock, tmp_path
    ) -> None:
        wt_path = str(tmp_path)
        user_data = {
            PENDING_WORKTREE_PATH: wt_path,
            PENDING_WORKTREE_BRANCH: "ccg/feat",
        }
        context = MagicMock()
        context.user_data = user_data

        _persist_worktree_state("@1", wt_path, context)

        # clear_worktree_state removes the pending worktree keys
        assert PENDING_WORKTREE_PATH not in user_data
        assert PENDING_WORKTREE_BRANCH not in user_data

    @patch("ccgram.handlers.topics.window_launch_service.session_manager")
    def test_skips_write_when_cwd_outside_worktree(
        self, mock_sm: MagicMock, tmp_path
    ) -> None:
        wt_path = str(tmp_path / "worktrees" / "feat")
        cwd = str(tmp_path / "other")
        user_data = {
            PENDING_WORKTREE_PATH: wt_path,
            PENDING_WORKTREE_BRANCH: "ccg/feat",
        }
        context = MagicMock()
        context.user_data = user_data

        _persist_worktree_state("@1", cwd, context)

        mock_sm.set_window_worktree.assert_not_called()
        # Keys are still cleared
        assert PENDING_WORKTREE_PATH not in user_data

    @patch("ccgram.handlers.topics.window_launch_service.session_manager")
    def test_no_op_when_worktree_path_missing(
        self, mock_sm: MagicMock, tmp_path
    ) -> None:
        context = MagicMock()
        context.user_data = {}
        _persist_worktree_state("@1", str(tmp_path), context)
        mock_sm.set_window_worktree.assert_not_called()


# ── launch_window ────────────────────────────────────────────────────────────


def _make_query() -> AsyncMock:
    query = AsyncMock()
    query.answer = AsyncMock()
    query.message = MagicMock()
    query.message.chat.type = "supergroup"
    query.message.chat.id = -100999
    return query


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot = AsyncMock()
    ctx.bot.edit_forum_topic = AsyncMock()
    return ctx


def _base_patches():
    """Return a dict of patch targets → return values for a successful launch."""
    return {
        "ccgram.handlers.topics.window_launch_service.tmux_manager": None,
        "ccgram.handlers.topics.window_launch_service.session_manager": None,
        "ccgram.handlers.topics.window_launch_service.thread_router": None,
        "ccgram.handlers.topics.window_launch_service.topic_orchestration": None,
        "ccgram.handlers.topics.window_launch_service.user_preferences": None,
        "ccgram.handlers.topics.window_launch_service.session_map_sync": None,
        "ccgram.handlers.topics.window_launch_service.safe_edit": None,
        "ccgram.handlers.topics.window_launch_service.provider_registry": None,
    }


@pytest.mark.asyncio
class TestLaunchWindowSuccess:
    async def test_creates_window_and_binds_thread(self, tmp_path) -> None:
        """Happy path: window created, thread bound, success message sent."""
        user_data = {PENDING_THREAD_ID: 42}
        query = _make_query()
        context = _make_context(user_data)

        with (
            patch(
                "ccgram.handlers.topics.window_launch_service.tmux_manager"
            ) as mock_mux,
            patch(
                "ccgram.handlers.topics.window_launch_service.session_manager"
            ) as mock_sm,
            patch(
                "ccgram.handlers.topics.window_launch_service.thread_router"
            ) as mock_tr,
            patch(
                "ccgram.handlers.topics.window_launch_service.topic_orchestration"
            ) as mock_orch,
            patch("ccgram.handlers.topics.window_launch_service.user_preferences"),
            patch(
                "ccgram.handlers.topics.window_launch_service.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.topics.window_launch_service.safe_edit",
                new_callable=AsyncMock,
            ) as mock_edit,
            patch(
                "ccgram.handlers.topics.window_launch_service.provider_registry"
            ) as mock_reg,
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        ):
            mock_mux.create_topic_target = AsyncMock(
                return_value=TopicTargetResult("@5", "my-win", "@5")
            )
            mock_mux.stamp_pane_title = AsyncMock()
            mock_mux.capabilities.native_worktrees = False
            mock_mux.capabilities.native_agent_status = True
            mock_tr.get_window_for_thread.return_value = None
            mock_tr.resolve_chat_id.return_value = -100999
            mock_sm.set_window_provider = MagicMock()
            mock_sm.set_window_origin = MagicMock()
            mock_sm.set_window_cwd = MagicMock()
            mock_sm.set_window_approval_mode = MagicMock()
            mock_sms.wait_for_session_map_entry = AsyncMock()

            caps = MagicMock()
            caps.chat_first_command_path = False
            caps.has_yolo_confirmation = False
            caps.supports_hook = False
            mock_reg.get.return_value.capabilities = caps

            await launch_window(
                query,
                context,
                WindowLaunchRequest(
                    user_id=100,
                    thread_id=42,
                    provider_name="claude",
                    cwd=str(tmp_path),
                    mode="normal",
                    pending_text=None,
                ),
            )

        mock_mux.create_topic_target.assert_awaited_once()
        mock_tr.bind_thread.assert_called_once()
        mock_orch.register_pending_creation.assert_called_once_with("@5")
        mock_orch.clear_pending_creation.assert_called_once_with("@5")
        mock_edit.assert_awaited_once()
        assert "✅" in mock_edit.call_args[0][1]

    async def test_post_create_stamp_error_closes_target_before_reraising(
        self, tmp_path
    ) -> None:
        query = _make_query()
        context = _make_context({PENDING_THREAD_ID: 42})
        with (
            patch("ccgram.handlers.topics.window_launch_service.tmux_manager") as mux,
            patch("ccgram.handlers.topics.window_launch_service.session_manager"),
            patch(
                "ccgram.handlers.topics.window_launch_service.thread_router"
            ) as router,
            patch(
                "ccgram.handlers.topics.window_launch_service.topic_orchestration"
            ) as orchestration,
            patch("ccgram.handlers.topics.window_launch_service.user_preferences"),
            patch("ccgram.handlers.topics.window_launch_service.session_map_sync"),
            patch("ccgram.handlers.topics.window_launch_service.provider_registry"),
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        ):
            mux.create_topic_target = AsyncMock(
                return_value=TopicTargetResult("@5", "new", "@5")
            )
            mux.stamp_pane_title = AsyncMock(side_effect=RuntimeError("stamp failed"))
            mux.kill_window = AsyncMock(return_value=True)
            mux.capabilities.native_worktrees = False
            mux.capabilities.native_agent_status = True
            with pytest.raises(RuntimeError, match="stamp failed"):
                await launch_window(
                    query,
                    context,
                    WindowLaunchRequest(
                        100, 42, "claude", str(tmp_path), "normal", None
                    ),
                )
        mux.kill_window.assert_awaited_once_with("@5")
        orchestration.clear_pending_creation.assert_called_once_with("@5")
        router.unbind_thread.assert_called_once_with(100, 42)

    async def test_session_map_timeout_closes_target_before_unbinding_late_hook(
        self, tmp_path
    ) -> None:
        """A late hook cannot orphan the just-created target after timeout."""
        query = _make_query()
        context = _make_context({PENDING_THREAD_ID: 42})
        cleanup_order: list[str] = []

        with (
            patch(
                "ccgram.handlers.topics.window_launch_service.tmux_manager"
            ) as mock_mux,
            patch(
                "ccgram.handlers.topics.window_launch_service.session_manager"
            ) as mock_sm,
            patch(
                "ccgram.handlers.topics.window_launch_service.thread_router"
            ) as mock_tr,
            patch(
                "ccgram.handlers.topics.window_launch_service.topic_orchestration"
            ) as mock_orch,
            patch("ccgram.handlers.topics.window_launch_service.user_preferences"),
            patch(
                "ccgram.handlers.topics.window_launch_service.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.topics.window_launch_service.safe_edit",
                new_callable=AsyncMock,
            ) as mock_edit,
            patch(
                "ccgram.handlers.topics.window_launch_service.provider_registry"
            ) as mock_reg,
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        ):

            async def close_created_target(target_id: str) -> bool:
                cleanup_order.append(f"close:{target_id}")
                return True

            mock_mux.create_topic_target = AsyncMock(
                return_value=TopicTargetResult("@5", "my-win", "@5")
            )
            mock_mux.kill_window = AsyncMock(side_effect=close_created_target)
            mock_mux.stamp_pane_title = AsyncMock()
            mock_mux.capabilities.native_worktrees = False
            mock_mux.capabilities.native_agent_status = True
            mock_tr.resolve_chat_id.return_value = -100999
            mock_tr.unbind_thread.side_effect = lambda *_: cleanup_order.append(
                "unbind"
            )
            mock_orch.clear_pending_creation.side_effect = lambda *_: (
                cleanup_order.append("clear-pending")
            )
            mock_sm.set_window_provider = MagicMock()
            mock_sm.set_window_origin = MagicMock()
            mock_sm.set_window_cwd = MagicMock()
            mock_sm.set_window_approval_mode = MagicMock()
            mock_sms.wait_for_session_map_entry = AsyncMock(return_value=False)
            caps = MagicMock(
                chat_first_command_path=False,
                has_yolo_confirmation=False,
                supports_hook=True,
            )
            mock_reg.get.return_value.capabilities = caps

            result = await launch_window(
                query,
                context,
                WindowLaunchRequest(100, 42, "claude", str(tmp_path), "normal", None),
            )

        assert result.success is False
        mock_mux.kill_window.assert_awaited_once_with("@5")
        mock_tr.unbind_thread.assert_called_once_with(100, 42)
        # The target is closed before the pending guard and binding are removed,
        # so a late hook cannot adopt it into an orphan topic.
        assert cleanup_order == ["close:@5", "clear-pending", "unbind"]
        assert "❌" in mock_edit.call_args.args[1]

    async def test_session_map_timeout_keeps_guard_and_binding_when_close_fails(
        self, tmp_path
    ) -> None:
        query = _make_query()
        context = _make_context({PENDING_THREAD_ID: 42})
        with (
            patch("ccgram.handlers.topics.window_launch_service.tmux_manager") as mux,
            patch("ccgram.handlers.topics.window_launch_service.session_manager"),
            patch(
                "ccgram.handlers.topics.window_launch_service.thread_router"
            ) as router,
            patch(
                "ccgram.handlers.topics.window_launch_service.topic_orchestration"
            ) as orchestration,
            patch("ccgram.handlers.topics.window_launch_service.user_preferences"),
            patch(
                "ccgram.handlers.topics.window_launch_service.session_map_sync"
            ) as maps,
            patch(
                "ccgram.handlers.topics.window_launch_service.safe_edit",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.window_launch_service.provider_registry"
            ) as registry,
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        ):
            mux.create_topic_target = AsyncMock(
                return_value=TopicTargetResult("@5", "my-win", "@5")
            )
            mux.stamp_pane_title = AsyncMock()
            mux.kill_window = AsyncMock(return_value=False)
            mux.capabilities.native_worktrees = False
            mux.capabilities.native_agent_status = True
            maps.wait_for_session_map_entry = AsyncMock(return_value=False)
            registry.get.return_value.capabilities = MagicMock(
                chat_first_command_path=False,
                has_yolo_confirmation=False,
                supports_hook=True,
            )

            result = await launch_window(
                query,
                context,
                WindowLaunchRequest(100, 42, "claude", str(tmp_path), "normal", None),
            )

        assert not result.success and "cleanup failed" in (result.error_message or "")
        orchestration.clear_pending_creation.assert_not_called()
        router.unbind_thread.assert_not_called()

    async def test_no_thread_success_releases_pending_creation_guard(
        self, tmp_path
    ) -> None:
        query = _make_query()
        context = _make_context()
        with (
            patch("ccgram.handlers.topics.window_launch_service.tmux_manager") as mux,
            patch("ccgram.handlers.topics.window_launch_service.session_manager"),
            patch("ccgram.handlers.topics.window_launch_service.thread_router"),
            patch(
                "ccgram.handlers.topics.window_launch_service.topic_orchestration"
            ) as orchestration,
            patch("ccgram.handlers.topics.window_launch_service.user_preferences"),
            patch("ccgram.handlers.topics.window_launch_service.session_map_sync"),
            patch(
                "ccgram.handlers.topics.window_launch_service.safe_edit",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.window_launch_service.provider_registry"
            ) as registry,
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        ):
            mux.create_topic_target = AsyncMock(
                return_value=TopicTargetResult("@5", "my-win", "@5")
            )
            mux.stamp_pane_title = AsyncMock()
            mux.capabilities.native_worktrees = False
            mux.capabilities.native_agent_status = True
            registry.get.return_value.capabilities = MagicMock(
                chat_first_command_path=False,
                has_yolo_confirmation=False,
                supports_hook=False,
            )

            result = await launch_window(
                query,
                context,
                WindowLaunchRequest(100, None, "claude", str(tmp_path), "normal", None),
            )

        assert result.success
        orchestration.clear_pending_creation.assert_called_once_with("@5")

    async def test_create_window_failure_calls_abort(self, tmp_path) -> None:
        """When create_window returns success=False, abort is called, no bind."""
        user_data = {PENDING_THREAD_ID: 42}
        query = _make_query()
        context = _make_context(user_data)

        with (
            patch(
                "ccgram.handlers.topics.window_launch_service.tmux_manager"
            ) as mock_mux,
            patch("ccgram.handlers.topics.window_launch_service.session_manager"),
            patch(
                "ccgram.handlers.topics.window_launch_service.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.window_launch_service.topic_orchestration"),
            patch("ccgram.handlers.topics.window_launch_service.user_preferences"),
            patch("ccgram.handlers.topics.window_launch_service.session_map_sync"),
            patch(
                "ccgram.handlers.topics.window_launch_service.safe_edit",
                new_callable=AsyncMock,
            ) as mock_edit,
            patch(
                "ccgram.handlers.topics.window_launch_service.provider_registry"
            ) as mock_reg,
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        ):
            mock_mux.create_topic_target = AsyncMock(
                side_effect=RuntimeError("tmux error")
            )
            mock_mux.capabilities.native_worktrees = False
            mock_mux.capabilities.native_agent_status = True

            caps = MagicMock()
            caps.chat_first_command_path = False
            caps.has_yolo_confirmation = False
            caps.supports_hook = False
            mock_reg.get.return_value.capabilities = caps

            await launch_window(
                query,
                context,
                WindowLaunchRequest(
                    user_id=100,
                    thread_id=42,
                    provider_name="claude",
                    cwd=str(tmp_path),
                    mode="normal",
                    pending_text=None,
                ),
            )

        # Thread must not be bound on failure
        mock_tr.bind_thread.assert_not_called()
        # Error message sent via safe_edit
        mock_edit.assert_awaited_once()
        assert "❌" in mock_edit.call_args[0][1]

    async def test_pending_text_forwarded_via_send_to_window(self, tmp_path) -> None:
        """PENDING_THREAD_TEXT is forwarded to the new window after bind."""
        user_data = {
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "hello agent",
        }
        query = _make_query()
        context = _make_context(user_data)

        with (
            patch(
                "ccgram.handlers.topics.window_launch_service.tmux_manager"
            ) as mock_mux,
            patch(
                "ccgram.handlers.topics.window_launch_service.session_manager"
            ) as mock_sm,
            patch(
                "ccgram.handlers.topics.window_launch_service.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.window_launch_service.topic_orchestration"),
            patch("ccgram.handlers.topics.window_launch_service.user_preferences"),
            patch(
                "ccgram.handlers.topics.window_launch_service.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.topics.window_launch_service.safe_edit",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.window_launch_service.provider_registry"
            ) as mock_reg,
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
            patch(
                "ccgram.handlers.topics.window_launch_service.send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
        ):
            mock_mux.create_topic_target = AsyncMock(
                return_value=TopicTargetResult("@5", "my-win", "@5")
            )
            mock_mux.stamp_pane_title = AsyncMock()
            mock_mux.capabilities.native_worktrees = False
            mock_mux.capabilities.native_agent_status = True
            mock_tr.get_window_for_thread.return_value = None
            mock_tr.resolve_chat_id.return_value = -100999
            mock_sm.set_window_provider = MagicMock()
            mock_sm.set_window_origin = MagicMock()
            mock_sm.set_window_cwd = MagicMock()
            mock_sm.set_window_approval_mode = MagicMock()
            mock_sms.wait_for_session_map_entry = AsyncMock()

            caps = MagicMock()
            caps.chat_first_command_path = False
            caps.has_yolo_confirmation = False
            caps.supports_hook = False
            mock_reg.get.return_value.capabilities = caps

            await launch_window(
                query,
                context,
                WindowLaunchRequest(
                    user_id=100,
                    thread_id=42,
                    provider_name="claude",
                    cwd=str(tmp_path),
                    mode="normal",
                    pending_text="hello agent",
                ),
            )

        mock_send.assert_awaited_once_with(100, "@5", 42, "hello agent", ANY)
        # Keys consumed after forwarding
        assert PENDING_THREAD_TEXT not in user_data
