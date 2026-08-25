from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.recovery.restore_command import restore_command


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch("ccgram.handlers.recovery.restore_command.thread_router") as mock_tr,
        patch("ccgram.handlers.recovery.restore_command.tmux_manager") as mock_tm,
        patch("ccgram.handlers.recovery.restore_command.config") as mock_cfg,
        patch("ccgram.handlers.recovery.restore_command.window_query") as mock_wq,
        patch("ccgram.handlers.recovery.restore_command.render_banner") as mock_render,
    ):
        mock_cfg.is_user_allowed.return_value = True
        mock_tr.resolve_window_for_thread.return_value = None
        mock_tr.get_display_name.return_value = "my-project"
        mock_tm.find_window_by_id = AsyncMock(return_value=None)
        mock_wq.get_window_provider.return_value = "claude"
        mock_render.return_value = ("⚠ Banner text", MagicMock())

        yield mock_tr, mock_tm, mock_cfg, mock_wq, mock_render


def _make_update(*, user_id: int = 100, thread_id: int | None = 42):
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.message = AsyncMock()
    update.message.message_thread_id = thread_id
    update.message.chat.type = "supergroup"
    update.message.chat.id = -100999
    return update


def _make_context():
    context = MagicMock()
    context.user_data = {}
    context.bot = AsyncMock()
    return context


class TestRestoreCommand:
    async def test_no_user_returns_early(self, _patch_deps) -> None:
        update = MagicMock()
        update.effective_user = None
        update.message = AsyncMock()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            mock_reply.assert_not_called()

    async def test_no_message_returns_early(self, _patch_deps) -> None:
        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = None

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            mock_reply.assert_not_called()

    async def test_unauthorized_user_rejected(self, _patch_deps) -> None:
        _, _, mock_cfg, _, _ = _patch_deps
        mock_cfg.is_user_allowed.return_value = False
        update = _make_update()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, _make_context())
            mock_reply.assert_called_once()
            assert "not authorized" in mock_reply.call_args[0][1]

    async def test_no_thread_id(self, _patch_deps) -> None:
        update = _make_update(thread_id=None)

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, _make_context())
            assert "inside a topic" in mock_reply.call_args[0][1]

    async def test_unbound_topic(self, _patch_deps) -> None:
        mock_tr, _, _, _, _ = _patch_deps
        mock_tr.resolve_window_for_thread.return_value = None
        update = _make_update()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, _make_context())
            assert "No session bound" in mock_reply.call_args[0][1]

    async def test_alive_window(self, _patch_deps) -> None:
        mock_tr, mock_tm, _, _, _ = _patch_deps
        mock_tr.resolve_window_for_thread.return_value = "@5"
        mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock())
        update = _make_update()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, _make_context())
            assert "still running" in mock_reply.call_args[0][1]

    async def test_dead_window_no_cwd_reports_missing_state_not_missing_dir(
        self, _patch_deps
    ) -> None:
        """No window state means unknown folder, not a deleted one (#176)."""
        mock_tr, _, _, mock_wq, _ = _patch_deps
        mock_tr.resolve_window_for_thread.return_value = "@5"
        mock_wq.view_window.return_value = MagicMock(cwd="")
        update = _make_update()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, _make_context())
            text = mock_reply.call_args[0][1]
            assert "Directory no longer exists" not in text
            assert "session state is gone" in text
            assert "/resume" in text

    async def test_dead_window_nonexistent_cwd(self, _patch_deps) -> None:
        mock_tr, _, _, mock_wq, _ = _patch_deps
        mock_tr.resolve_window_for_thread.return_value = "@5"
        mock_wq.view_window.return_value = MagicMock(cwd="/nonexistent/path")
        update = _make_update()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, _make_context())
            assert "Directory no longer exists" in mock_reply.call_args[0][1]

    async def test_dead_window_renders_banner(self, _patch_deps, tmp_path) -> None:
        mock_tr, _, _, mock_wq, mock_render = _patch_deps
        mock_tr.resolve_window_for_thread.return_value = "@5"
        mock_wq.view_window.return_value = MagicMock(cwd=str(tmp_path))
        update = _make_update()
        ctx = _make_context()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply") as mock_reply:
            await restore_command(update, ctx)

            mock_render.assert_called_once()
            banner = mock_render.call_args.args[0]
            assert banner.window_id == "@5"
            assert banner.mode == "restore"
            assert banner.cwd == str(tmp_path)
            assert banner.display == "my-project"
            assert banner.thread_id == 42
            mock_reply.assert_called_once()
            args = mock_reply.call_args
            assert args[0][1] == "⚠ Banner text"
            assert args[1]["reply_markup"] is not None

    async def test_dead_window_pending_state_recorded(
        self, _patch_deps, tmp_path
    ) -> None:
        from ccgram.handlers.user_state import (
            PENDING_THREAD_ID,
            RECOVERY_WINDOW_ID,
        )

        mock_tr, _, _, mock_wq, _ = _patch_deps
        mock_tr.resolve_window_for_thread.return_value = "@5"
        mock_wq.view_window.return_value = MagicMock(cwd=str(tmp_path))
        update = _make_update()
        ctx = _make_context()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply"):
            await restore_command(update, ctx)

        assert ctx.user_data[PENDING_THREAD_ID] == 42
        assert ctx.user_data[RECOVERY_WINDOW_ID] == "@5"

    async def test_dead_window_does_not_create_window(
        self, _patch_deps, tmp_path
    ) -> None:
        mock_tr, mock_tm, _, mock_wq, _ = _patch_deps
        mock_tr.resolve_window_for_thread.return_value = "@5"
        mock_wq.view_window.return_value = MagicMock(cwd=str(tmp_path))
        update = _make_update()
        ctx = _make_context()

        with patch("ccgram.handlers.recovery.restore_command.safe_reply"):
            await restore_command(update, ctx)

        mock_tm.create_window.assert_not_called()
        mock_tr.bind_thread.assert_not_called()
        mock_tr.unbind_thread.assert_not_called()
