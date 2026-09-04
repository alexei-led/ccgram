"""Tests for window_launch_service.py — pure helpers plus the launch_window flow."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from ccgram.multiplexer.base import TopicTargetResult
from ccgram.handlers.topics.window_launch_service import (
    WindowLaunchRequest,
    _create_topic_window,
    _cwd_within,
    _follow_supersession,
    _persist_worktree_state,
    launch_window,
)
from ccgram.handlers.user_state import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_PATH,
    PENDING_WORKTREE_REPO,
)

_MODULE = "ccgram.handlers.topics.window_launch_service."


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
    @staticmethod
    def _context(user_data: dict) -> MagicMock:
        context = MagicMock()
        context.user_data = user_data
        return context

    def test_writes_worktree_state_and_clears_keys_when_cwd_matches(
        self, tmp_path
    ) -> None:
        wt_path = str(tmp_path)
        user_data = {
            PENDING_WORKTREE_PATH: wt_path,
            PENDING_WORKTREE_BRANCH: "ccg/feat",
        }

        with patch(f"{_MODULE}session_manager") as mock_sm:
            _persist_worktree_state("@1", wt_path, self._context(user_data))

        mock_sm.set_window_worktree.assert_called_once_with("@1", wt_path, "ccg/feat")
        assert PENDING_WORKTREE_PATH not in user_data
        assert PENDING_WORKTREE_BRANCH not in user_data

    def test_skips_write_when_cwd_outside_worktree(self, tmp_path) -> None:
        """A stale path from an aborted attempt must not attach to a new window."""
        user_data = {
            PENDING_WORKTREE_PATH: str(tmp_path / "worktrees" / "feat"),
            PENDING_WORKTREE_BRANCH: "ccg/feat",
        }

        with patch(f"{_MODULE}session_manager") as mock_sm:
            _persist_worktree_state(
                "@1", str(tmp_path / "other"), self._context(user_data)
            )

        mock_sm.set_window_worktree.assert_not_called()
        assert PENDING_WORKTREE_PATH not in user_data

    def test_no_op_when_worktree_path_missing(self, tmp_path) -> None:
        with patch(f"{_MODULE}session_manager") as mock_sm:
            _persist_worktree_state("@1", str(tmp_path), self._context({}))
        mock_sm.set_window_worktree.assert_not_called()


# ── _follow_supersession ─────────────────────────────────────────────────────


class TestFollowSupersession:
    def test_stable_id_leaves_the_creation_guard_alone(self) -> None:
        with (
            patch(f"{_MODULE}window_query") as mock_wq,
            patch(f"{_MODULE}topic_orchestration") as mock_orch,
        ):
            mock_wq.resolve_window_alias.return_value = "@5"
            assert _follow_supersession("@5") == "@5"

        mock_orch.register_pending_creation.assert_not_called()
        mock_orch.clear_pending_creation.assert_not_called()

    def test_superseded_id_carries_the_creation_guard_forward(self) -> None:
        """Herdr re-keys a target mid-creation; the guard must move with it."""
        with (
            patch(f"{_MODULE}window_query") as mock_wq,
            patch(f"{_MODULE}topic_orchestration") as mock_orch,
        ):
            mock_wq.resolve_window_alias.return_value = "w1:t9"
            assert _follow_supersession("w1:t1") == "w1:t9"

        mock_orch.register_pending_creation.assert_called_once_with("w1:t9")
        mock_orch.clear_pending_creation.assert_called_once_with("w1:t1")


# ── _create_topic_window ─────────────────────────────────────────────────────


class TestCreateTopicWindow:
    @staticmethod
    def _worktree_context() -> MagicMock:
        context = MagicMock()
        context.user_data = {
            PENDING_WORKTREE_REPO: "/repo",
            PENDING_WORKTREE_BRANCH: "ccg/feat",
            PENDING_WORKTREE_PATH: "/repo.worktrees/ccg-feat",
        }
        return context

    async def test_native_worktree_delegates_to_one_call(self) -> None:
        with patch(f"{_MODULE}tmux_manager") as mux:
            mux.capabilities.native_worktrees = True
            mux.create_worktree_window = AsyncMock(
                return_value=(True, "ok", "ccg-feat", "w1:t1")
            )
            result = await _create_topic_window(
                "/repo.worktrees/ccg-feat", "claude", None, self._worktree_context()
            )

        assert result == (True, "ok", "ccg-feat", "w1:t1")
        mux.create_worktree_window.assert_awaited_once_with(
            "/repo",
            "/repo.worktrees/ccg-feat",
            "ccg/feat",
            window_name="ccg-feat",
            launch_command="claude",
        )

    async def test_preselected_workspace_blocks_native_worktree_creation(self) -> None:
        """The native worktree API cannot pin a workspace — refuse, don't override."""
        with patch(f"{_MODULE}tmux_manager") as mux:
            mux.capabilities.native_worktrees = True
            mux.create_worktree_window = AsyncMock()
            success, message, name, window_id = await _create_topic_window(
                "/repo.worktrees/ccg-feat",
                "claude",
                "ws-chosen",
                self._worktree_context(),
            )

        assert success is False
        assert message == "Selected workspace cannot create a native worktree"
        assert (name, window_id) == ("", "")
        mux.create_worktree_window.assert_not_awaited()

    async def test_tmux_backend_uses_create_window(self) -> None:
        context = MagicMock()
        context.user_data = {}
        with patch(f"{_MODULE}tmux_manager") as mux:
            mux.capabilities.native_worktrees = False
            mux.capabilities.native_agent_status = False
            mux.create_window = AsyncMock(return_value=(True, "ok", "proj", "@7"))
            result = await _create_topic_window("/proj", "claude", "ws1", context)

        assert result == (True, "ok", "proj", "@7")
        mux.create_window.assert_awaited_once_with(
            "/proj", launch_command="claude", workspace_id="ws1"
        )

    async def test_guarded_target_failure_is_reported_not_raised(self) -> None:
        context = MagicMock()
        context.user_data = {}
        with patch(f"{_MODULE}tmux_manager") as mux:
            mux.capabilities.native_worktrees = False
            mux.capabilities.native_agent_status = True
            mux.capabilities.native_topic_targets = True
            mux.create_topic_target = AsyncMock(side_effect=RuntimeError("no socket"))
            success, message, name, window_id = await _create_topic_window(
                "/proj", "claude", None, context
            )

        assert success is False
        assert message == "no socket"
        assert (name, window_id) == ("", "")


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


@dataclass
class _LaunchMocks:
    """Every collaborator ``launch_window`` reaches for, so tests can steer them."""

    mux: MagicMock
    session: MagicMock
    router: MagicMock
    orchestration: MagicMock
    preferences: MagicMock
    session_map: MagicMock
    edit: AsyncMock
    registry: MagicMock


@contextlib.contextmanager
def _launch_env(
    *,
    supports_hook: bool = False,
    has_yolo_confirmation: bool = False,
    chat_first_command_path: bool = False,
    launch_command: str = "claude",
) -> Iterator[_LaunchMocks]:
    """Patch launch_window's collaborators for a successful hookless launch.

    Defaults describe a guarded-target backend creating ``@5`` with no hook to
    wait for; each test overrides only the piece it is about.
    """
    with (
        patch(f"{_MODULE}tmux_manager") as mux,
        patch(f"{_MODULE}session_manager") as session,
        patch(f"{_MODULE}thread_router") as router,
        patch(f"{_MODULE}topic_orchestration") as orchestration,
        patch(f"{_MODULE}user_preferences") as preferences,
        patch(f"{_MODULE}session_map_sync") as session_map,
        patch(f"{_MODULE}safe_edit", new_callable=AsyncMock) as edit,
        patch(f"{_MODULE}provider_registry") as registry,
        patch("ccgram.providers.resolve_launch_command", return_value=launch_command),
    ):
        mux.create_topic_target = AsyncMock(
            return_value=TopicTargetResult("@5", "my-win", "@5")
        )
        mux.stamp_pane_title = AsyncMock()
        mux.kill_window = AsyncMock(return_value=True)
        mux.capabilities.native_worktrees = False
        mux.capabilities.native_agent_status = True
        mux.capabilities.native_topic_targets = True
        router.get_window_for_thread.return_value = None
        router.resolve_chat_id.return_value = -100999
        session_map.wait_for_session_map_entry = AsyncMock(return_value=True)
        registry.get.return_value.capabilities = MagicMock(
            supports_hook=supports_hook,
            has_yolo_confirmation=has_yolo_confirmation,
            chat_first_command_path=chat_first_command_path,
        )
        yield _LaunchMocks(
            mux=mux,
            session=session,
            router=router,
            orchestration=orchestration,
            preferences=preferences,
            session_map=session_map,
            edit=edit,
            registry=registry,
        )


def _request(**overrides) -> WindowLaunchRequest:
    fields = {
        "user_id": 100,
        "thread_id": 42,
        "provider_name": "claude",
        "cwd": "/tmp/proj",
        "mode": "normal",
        "pending_text": None,
    }
    fields.update(overrides)
    return WindowLaunchRequest(**fields)


class TestLaunchWindowSuccess:
    async def test_creates_window_and_binds_thread(self, tmp_path) -> None:
        query = _make_query()
        context = _make_context({PENDING_THREAD_ID: 42})

        with _launch_env() as m:
            await launch_window(query, context, _request(cwd=str(tmp_path)))

        m.mux.create_topic_target.assert_awaited_once()
        m.router.bind_thread.assert_called_once()
        m.orchestration.pending_creation_transaction.assert_called_once_with()
        m.orchestration.register_pending_creation.assert_called_once_with("@5")
        m.orchestration.clear_pending_creation.assert_called_once_with("@5")
        m.edit.assert_awaited_once()
        assert "✅" in m.edit.call_args[0][1]

    @patch(f"{_MODULE}_accept_yolo_confirmation", new_callable=AsyncMock)
    async def test_yolo_creation_guard_outlives_configured_confirmation(
        self, mock_accept: AsyncMock, tmp_path, monkeypatch
    ) -> None:
        mock_accept.return_value = True
        monkeypatch.setattr(
            "ccgram.handlers.topics.window_launch_service.config.yolo_confirmation_timeout",
            45.0,
        )

        with _launch_env(has_yolo_confirmation=True) as m:
            await launch_window(
                _make_query(),
                _make_context({PENDING_THREAD_ID: 42}),
                _request(cwd=str(tmp_path), mode="yolo"),
            )

        m.orchestration.register_pending_creation.assert_called_once_with(
            "@5", ttl_s=55.0
        )

    async def test_no_thread_success_releases_pending_creation_guard(
        self, tmp_path
    ) -> None:
        with _launch_env() as m:
            result = await launch_window(
                _make_query(),
                _make_context(),
                _request(thread_id=None, cwd=str(tmp_path)),
            )

        assert result.success
        m.orchestration.clear_pending_creation.assert_called_once_with("@5")

    async def test_pending_text_is_forwarded_after_binding(self, tmp_path) -> None:
        """A non-chat-first provider's first prompt uses the bound-window send path."""
        user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hello agent"}
        context = _make_context(user_data)

        with (
            _launch_env(launch_command="agy") as m,
            patch(
                f"{_MODULE}send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
        ):
            await launch_window(
                _make_query(),
                context,
                _request(
                    provider_name="antigravity",
                    cwd=str(tmp_path),
                    pending_text="hello agent",
                ),
            )

        m.mux.create_topic_target.assert_awaited_once_with(
            str(tmp_path), launch_command="agy", workspace_id=None
        )
        mock_send.assert_awaited_once_with(100, "@5", 42, "hello agent", ANY)
        assert PENDING_THREAD_TEXT not in user_data


class TestLaunchWindowFailure:
    async def test_create_failure_aborts_without_binding_and_clears_flow_state(
        self, tmp_path
    ) -> None:
        user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hi"}
        context = _make_context(user_data)

        with _launch_env() as m:
            m.mux.create_topic_target = AsyncMock(
                side_effect=RuntimeError("tmux error")
            )
            result = await launch_window(
                _make_query(), context, _request(cwd=str(tmp_path))
            )

        assert result.success is False
        m.router.bind_thread.assert_not_called()
        m.edit.assert_awaited_once()
        assert "❌" in m.edit.call_args[0][1]
        assert PENDING_THREAD_ID not in user_data
        assert PENDING_THREAD_TEXT not in user_data

    async def test_post_create_stamp_error_closes_target_before_reraising(
        self, tmp_path
    ) -> None:
        with _launch_env() as m:
            m.mux.stamp_pane_title = AsyncMock(side_effect=RuntimeError("stamp failed"))
            with pytest.raises(RuntimeError, match="stamp failed"):
                await launch_window(
                    _make_query(),
                    _make_context({PENDING_THREAD_ID: 42}),
                    _request(cwd=str(tmp_path)),
                )

        m.mux.kill_window.assert_awaited_once_with("@5")
        m.orchestration.clear_pending_creation.assert_called_once_with("@5")
        m.router.unbind_thread.assert_called_once_with(
            100,
            42,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )

    async def test_session_map_timeout_closes_target_before_unbinding_late_hook(
        self, tmp_path
    ) -> None:
        """A late hook cannot orphan the just-created target after timeout."""
        cleanup_order: list[str] = []

        with _launch_env(supports_hook=True) as m:
            m.mux.kill_window = AsyncMock(
                side_effect=lambda target_id: (
                    cleanup_order.append(f"close:{target_id}") or True
                )
            )
            m.router.unbind_thread.side_effect = lambda *_args, **_kwargs: (
                cleanup_order.append("unbind")
            )
            m.orchestration.clear_pending_creation.side_effect = lambda *_: (
                cleanup_order.append("clear-pending")
            )
            m.session_map.wait_for_session_map_entry = AsyncMock(return_value=False)

            result = await launch_window(
                _make_query(),
                _make_context({PENDING_THREAD_ID: 42}),
                _request(cwd=str(tmp_path)),
            )

        assert result.success is False
        m.mux.kill_window.assert_awaited_once_with("@5")
        m.router.unbind_thread.assert_called_once_with(
            100,
            42,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )
        # The target is closed before the pending guard and binding are removed,
        # so a late hook cannot adopt it into an orphan topic.
        assert cleanup_order == ["close:@5", "clear-pending", "unbind"]
        assert "❌" in m.edit.call_args.args[1]

    async def test_session_map_timeout_keeps_guard_and_binding_when_close_fails(
        self, tmp_path
    ) -> None:
        with _launch_env(supports_hook=True) as m:
            m.mux.kill_window = AsyncMock(return_value=False)
            m.session_map.wait_for_session_map_entry = AsyncMock(return_value=False)

            result = await launch_window(
                _make_query(),
                _make_context({PENDING_THREAD_ID: 42}),
                _request(cwd=str(tmp_path)),
            )

        assert not result.success and "cleanup failed" in (result.error_message or "")
        m.orchestration.clear_pending_creation.assert_not_called()
        m.router.unbind_thread.assert_not_called()

    async def test_pending_text_send_failure_reports_back_to_the_topic(
        self, tmp_path
    ) -> None:
        with (
            _launch_env(),
            patch(
                f"{_MODULE}send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(False, "pane is gone"),
            ),
            patch(f"{_MODULE}safe_send", new_callable=AsyncMock) as mock_safe_send,
        ):
            result = await launch_window(
                _make_query(),
                _make_context({PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hi"}),
                _request(cwd=str(tmp_path), pending_text="hi"),
            )

        assert result.success is True
        mock_safe_send.assert_awaited_once()
        assert "pane is gone" in mock_safe_send.call_args.args[2]
        assert mock_safe_send.call_args.kwargs == {"message_thread_id": 42}
