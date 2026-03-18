"""Tests for /restore command — dead topic auto-recovery."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.restore_command import restore_command
from ccgram.session import WindowState


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch("ccgram.handlers.restore_command.session_manager") as mock_sm,
        patch("ccgram.handlers.restore_command.tmux_manager") as mock_tm,
        patch("ccgram.handlers.restore_command.config") as mock_cfg,
        patch("ccgram.handlers.restore_command.clear_dead_notification") as mock_cdn,
        patch("ccgram.handlers.restore_command.get_provider_for_window") as mock_gpw,
        patch("ccgram.handlers.restore_command.resolve_launch_command") as mock_rlc,
    ):
        mock_cfg.is_user_allowed.return_value = True
        mock_sm.resolve_window_for_thread.return_value = None
        mock_sm.wait_for_session_map_entry = AsyncMock()
        mock_tm.find_window_by_id = AsyncMock(return_value=None)
        mock_tm.create_window = AsyncMock(
            return_value=(True, "Created window", "my-project", "@10")
        )

        # Provider mock
        caps = MagicMock()
        caps.name = "claude"
        caps.supports_hook = True
        provider = MagicMock()
        provider.capabilities = caps
        provider.make_launch_args.return_value = "--continue"
        mock_gpw.return_value = provider
        mock_rlc.return_value = "claude"

        yield mock_sm, mock_tm, mock_cfg, mock_cdn, mock_gpw, mock_rlc


def _make_update(*, user_id=100, thread_id=42):  # noqa: ANN001
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.message = AsyncMock()
    update.message.message_thread_id = thread_id
    return update


def _make_context():  # noqa: ANN001
    context = MagicMock()
    context.bot = AsyncMock()
    return context


class TestRestoreCommand:
    async def test_no_user_returns_early(self, _patch_deps) -> None:
        update = MagicMock()
        update.effective_user = None
        update.message = AsyncMock()

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            mock_reply.assert_not_called()

    async def test_no_message_returns_early(self, _patch_deps) -> None:
        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = None

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            mock_reply.assert_not_called()

    async def test_unauthorized_user_rejected(self, _patch_deps) -> None:
        _, _, mock_cfg, _, _, _ = _patch_deps
        mock_cfg.is_user_allowed.return_value = False
        update = _make_update()

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            mock_reply.assert_called_once()
            assert "not authorized" in mock_reply.call_args[0][1]

    async def test_no_thread_id(self, _patch_deps) -> None:
        update = _make_update(thread_id=None)

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            assert "inside a topic" in mock_reply.call_args[0][1]

    async def test_unbound_topic(self, _patch_deps) -> None:
        mock_sm, _, _, _, _, _ = _patch_deps
        mock_sm.resolve_window_for_thread.return_value = None
        update = _make_update()

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            assert "No session bound" in mock_reply.call_args[0][1]

    async def test_alive_window(self, _patch_deps) -> None:
        mock_sm, mock_tm, _, _, _, _ = _patch_deps
        mock_sm.resolve_window_for_thread.return_value = "@5"
        mock_tm.find_window_by_id.return_value = MagicMock()
        update = _make_update()

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            assert "still running" in mock_reply.call_args[0][1]

    async def test_dead_window_no_cwd(self, _patch_deps) -> None:
        mock_sm, mock_tm, _, _, _, _ = _patch_deps
        mock_sm.resolve_window_for_thread.return_value = "@5"
        mock_tm.find_window_by_id.return_value = None
        mock_sm.get_window_state.return_value = WindowState()
        update = _make_update()

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            assert "Directory no longer exists" in mock_reply.call_args[0][1]

    async def test_dead_window_nonexistent_cwd(self, _patch_deps) -> None:
        mock_sm, mock_tm, _, _, _, _ = _patch_deps
        mock_sm.resolve_window_for_thread.return_value = "@5"
        mock_tm.find_window_by_id.return_value = None
        mock_sm.get_window_state.return_value = WindowState(cwd="/nonexistent/path")
        update = _make_update()

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, MagicMock())
            assert "Directory no longer exists" in mock_reply.call_args[0][1]

    async def test_dead_window_auto_continues(self, _patch_deps, tmp_path) -> None:
        mock_sm, mock_tm, _, mock_cdn, mock_gpw, _ = _patch_deps
        mock_sm.resolve_window_for_thread.return_value = "@5"
        mock_tm.find_window_by_id.return_value = None
        mock_sm.get_window_state.return_value = WindowState(cwd=str(tmp_path))
        mock_sm.get_approval_mode.return_value = "normal"
        update = _make_update()
        context = _make_context()

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, context)

            # Unbinds old, clears dead notification
            mock_sm.unbind_thread.assert_called_once_with(100, 42)
            mock_cdn.assert_called_once_with(100, 42)

            # Creates window with --continue
            mock_tm.create_window.assert_called_once()
            call_kwargs = mock_tm.create_window.call_args
            assert call_kwargs[1]["agent_args"] == "--continue"

            # Binds new window
            mock_sm.bind_thread.assert_called_once()

            # Success message
            mock_reply.assert_called_once()
            text = mock_reply.call_args[0][1]
            assert "\u2705" in text
            assert "Continuing previous session" in text

    async def test_dead_window_create_fails(self, _patch_deps, tmp_path) -> None:
        mock_sm, mock_tm, _, _, _, _ = _patch_deps
        mock_sm.resolve_window_for_thread.return_value = "@5"
        mock_tm.find_window_by_id.return_value = None
        mock_sm.get_window_state.return_value = WindowState(cwd=str(tmp_path))
        mock_sm.get_approval_mode.return_value = "normal"
        mock_tm.create_window.return_value = (False, "tmux error", "", "")
        update = _make_update()
        context = _make_context()

        with patch("ccgram.handlers.restore_command.safe_reply") as mock_reply:
            await restore_command(update, context)
            mock_reply.assert_called_once()
            text = mock_reply.call_args[0][1]
            assert "\u274c" in text
            assert "tmux error" in text
            # Should not bind on failure
            mock_sm.bind_thread.assert_not_called()
