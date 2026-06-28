import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup

from ccgram.handlers.callback_data import (
    CB_DIR_CANCEL,
    CB_DIR_STAR,
    CB_WT_CONFIRM,
    CB_WT_EDIT_NAME,
    CB_WT_NEW,
    CB_WT_USE_CURRENT,
)
from ccgram.handlers.topics.directory_browser import BROWSE_PATH_KEY
from ccgram.handlers.topics.directory_callbacks import (
    _handle_confirm,
    _handle_star,
    _handle_up,
    _handle_worktree_callback,
    _handle_wt_confirm,
    _handle_wt_edit_name,
    _handle_wt_new,
    _handle_wt_use_current,
)
from ccgram.handlers.user_state import (
    AWAITING_WORKTREE_BRANCH_NAME,
    PENDING_THREAD_ID,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_DIRTY,
    PENDING_WORKTREE_PATH,
    PENDING_WORKTREE_REPO,
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "file.txt").write_text("hello")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "branch", "-M", "main")
    return repo


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot = AsyncMock()
    return ctx


def _make_query() -> AsyncMock:
    query = AsyncMock()
    query.answer = AsyncMock()
    query.message = MagicMock()
    query.message.chat.type = "supergroup"
    query.message.chat.id = -100999
    return query


def _make_update(thread_id: int = 42) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 100
    update.message = None
    update.callback_query = MagicMock()
    update.callback_query.message = MagicMock()
    update.callback_query.message.message_thread_id = thread_id
    return update


class TestConfirmWorktreeGating:
    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.thread_router")
    async def test_eligible_repo_shows_worktree_picker(
        self, mock_tr: MagicMock, mock_edit: AsyncMock, git_repo: Path
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        user_data = {BROWSE_PATH_KEY: str(git_repo), PENDING_THREAD_ID: 42}
        context = _make_context(user_data)

        await _handle_confirm(_make_query(), 100, _make_update(42), context)

        text = mock_edit.call_args[0][1]
        assert "Git Worktree" in text
        assert user_data[PENDING_WORKTREE_REPO] == str(git_repo.resolve())
        assert user_data[PENDING_WORKTREE_DIRTY] is False

    @patch(
        "ccgram.handlers.topics.workspace_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.thread_router")
    async def test_non_git_dir_shows_provider_picker(
        self, mock_tr: MagicMock, mock_edit: AsyncMock, tmp_path: Path
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        plain = tmp_path / "plain"
        plain.mkdir()
        user_data = {BROWSE_PATH_KEY: str(plain), PENDING_THREAD_ID: 42}
        context = _make_context(user_data)

        await _handle_confirm(_make_query(), 100, _make_update(42), context)

        text = mock_edit.call_args[0][1]
        assert "Select Provider" in text
        assert PENDING_WORKTREE_REPO not in user_data

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.thread_router")
    async def test_confirm_after_new_command_reset_fails_closed(
        self, mock_tr: MagicMock, mock_edit: AsyncMock, git_repo: Path
    ) -> None:
        from ccgram.handlers.topics.new_command import new_command

        mock_tr.get_window_for_thread.return_value = None
        user_data = {BROWSE_PATH_KEY: str(git_repo), PENDING_THREAD_ID: 42}
        context = _make_context(user_data)

        nc_update = MagicMock()
        nc_update.effective_user = MagicMock(id=100)
        nc_update.message = AsyncMock()
        with patch(
            "ccgram.handlers.topics.new_command.config.is_user_allowed",
            return_value=True,
        ):
            await new_command(nc_update, context)

        query = _make_query()
        await _handle_confirm(query, 100, _make_update(42), context)

        query.answer.assert_called_once_with(
            "Stale browser (flow reset)", show_alert=True
        )
        mock_edit.assert_not_called()
        assert PENDING_WORKTREE_REPO not in user_data

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.thread_router")
    async def test_confirm_with_repopulated_path_but_no_pending_thread(
        self, mock_tr: MagicMock, mock_edit: AsyncMock, git_repo: Path
    ) -> None:
        mock_tr.get_window_for_thread.return_value = None
        user_data = {BROWSE_PATH_KEY: str(git_repo)}
        context = _make_context(user_data)

        query = _make_query()
        await _handle_confirm(query, 100, _make_update(42), context)

        query.answer.assert_called_once_with(
            "Stale browser (flow reset)", show_alert=True
        )
        mock_edit.assert_not_called()
        assert PENDING_WORKTREE_REPO not in user_data


class TestHandleWtUseCurrent:
    @patch(
        "ccgram.handlers.topics.workspace_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_clears_state_and_shows_provider_picker(
        self, mock_edit: AsyncMock
    ) -> None:
        user_data = {
            BROWSE_PATH_KEY: "/tmp/proj",
            PENDING_WORKTREE_REPO: "/tmp/proj",
            PENDING_WORKTREE_DIRTY: True,
        }
        context = _make_context(user_data)

        await _handle_wt_use_current(_make_query(), context)

        text = mock_edit.call_args[0][1]
        assert "Select Provider" in text
        assert PENDING_WORKTREE_REPO not in user_data
        assert PENDING_WORKTREE_DIRTY not in user_data

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_state_lost_fails_closed(self, mock_edit: AsyncMock) -> None:
        context = _make_context({})
        await _handle_wt_use_current(_make_query(), context)
        assert "state lost" in mock_edit.call_args[0][1].lower()
        assert "Select Provider" not in mock_edit.call_args[0][1]

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_browse_path_gone_fails_closed(self, mock_edit: AsyncMock) -> None:
        context = _make_context({PENDING_WORKTREE_REPO: "/tmp/proj"})
        await _handle_wt_use_current(_make_query(), context)
        assert "state lost" in mock_edit.call_args[0][1].lower()

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_stale_button_after_new_command_reset_fails_closed(
        self, mock_edit: AsyncMock
    ) -> None:
        from ccgram.handlers.topics.new_command import new_command

        user_data = {
            BROWSE_PATH_KEY: "/tmp/proj",
            PENDING_WORKTREE_REPO: "/tmp/proj",
            PENDING_THREAD_ID: 42,
        }
        context = _make_context(user_data)

        nc_update = MagicMock()
        nc_update.effective_user = MagicMock(id=100)
        nc_update.message = AsyncMock()
        with patch(
            "ccgram.handlers.topics.new_command.config.is_user_allowed",
            return_value=True,
        ):
            await new_command(nc_update, context)

        query = _make_query()
        await _handle_worktree_callback(
            query, CB_WT_USE_CURRENT, _make_update(42), context
        )

        query.answer.assert_awaited_once_with(
            "Stale browser (flow reset)", show_alert=True
        )
        mock_edit.assert_not_called()


class TestHandleWtNew:
    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_suggests_branch_and_shows_confirm(
        self, mock_edit: AsyncMock, git_repo: Path
    ) -> None:
        user_data = {
            PENDING_WORKTREE_REPO: str(git_repo),
            PENDING_WORKTREE_DIRTY: False,
        }
        context = _make_context(user_data)

        await _handle_wt_new(_make_query(), context)

        branch = user_data[PENDING_WORKTREE_BRANCH]
        assert branch.startswith("ccg/")
        assert user_data[PENDING_WORKTREE_PATH].endswith(
            f"repo.worktrees/{branch.replace('/', '-')}"
        )
        text = mock_edit.call_args[0][1]
        assert "New Worktree" in text
        assert branch in text

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_missing_repo_state_errors(self, mock_edit: AsyncMock) -> None:
        context = _make_context({})
        await _handle_wt_new(_make_query(), context)
        assert "state lost" in mock_edit.call_args[0][1].lower()


class TestHandleWtConfirm:
    @patch(
        "ccgram.handlers.topics.workspace_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_creates_worktree_and_shows_provider_picker(
        self, mock_edit: AsyncMock, git_repo: Path
    ) -> None:
        wt_path = git_repo.parent / "repo.worktrees" / "ccg-feat"
        user_data = {
            PENDING_WORKTREE_REPO: str(git_repo),
            PENDING_WORKTREE_BRANCH: "ccg/feat",
            PENDING_WORKTREE_PATH: str(wt_path),
        }
        context = _make_context(user_data)

        await _handle_wt_confirm(_make_query(), context)

        assert wt_path.is_dir()
        assert (wt_path / "file.txt").exists()
        assert user_data[BROWSE_PATH_KEY] == str(wt_path)
        assert "Select Provider" in mock_edit.call_args[0][1]

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_branch_conflict_shows_error_with_cancel(
        self, mock_edit: AsyncMock, git_repo: Path
    ) -> None:
        _git(git_repo, "branch", "ccg/dup")
        wt_path = git_repo.parent / "repo.worktrees" / "ccg-dup"
        user_data = {
            PENDING_WORKTREE_REPO: str(git_repo),
            PENDING_WORKTREE_BRANCH: "ccg/dup",
            PENDING_WORKTREE_PATH: str(wt_path),
        }
        context = _make_context(user_data)

        await _handle_wt_confirm(_make_query(), context)

        text = mock_edit.call_args[0][1]
        assert "Could not create worktree" in text
        keyboard = mock_edit.call_args.kwargs["reply_markup"]
        assert isinstance(keyboard, InlineKeyboardMarkup)
        callbacks = [b.callback_data for row in keyboard.inline_keyboard for b in row]
        assert CB_DIR_CANCEL in callbacks
        assert not wt_path.exists()


class TestHandleWtConfirmNativeWorktrees:
    """native_worktrees (herdr) path must skip the workspace picker and go
    straight to the provider picker (MAJOR-2 regression)."""

    @patch("ccgram.handlers.topics.directory_callbacks.tmux_manager")
    @patch(
        "ccgram.handlers.topics.workspace_callbacks._show_workspace_picker_or_provider",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.topics.workspace_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_native_worktrees_skips_workspace_picker(
        self,
        mock_edit: AsyncMock,
        mock_workspace_picker: AsyncMock,
        mock_mux: MagicMock,
    ) -> None:
        mock_mux.capabilities.native_worktrees = True
        user_data = {
            PENDING_WORKTREE_REPO: "/repo",
            PENDING_WORKTREE_BRANCH: "ccg/feat",
            PENDING_WORKTREE_PATH: "/wt/path",
        }
        context = _make_context(user_data)

        await _handle_wt_confirm(_make_query(), context)

        # workspace picker must NOT be called on the native_worktrees path
        mock_workspace_picker.assert_not_called()
        # provider picker edit must have been called (shows "Select Provider")
        assert mock_edit.called
        assert "Select Provider" in mock_edit.call_args[0][1]


class TestHandleWtEditName:
    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_sets_awaiting_flag(self, mock_edit: AsyncMock) -> None:
        context = _make_context({PENDING_WORKTREE_REPO: "/tmp/proj"})
        await _handle_wt_edit_name(_make_query(), context)
        assert context.user_data[AWAITING_WORKTREE_BRANCH_NAME] is True
        assert "branch name" in mock_edit.call_args[0][1].lower()

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_missing_repo_state_does_not_arm_flag(
        self, mock_edit: AsyncMock
    ) -> None:
        context = _make_context({})
        await _handle_wt_edit_name(_make_query(), context)
        assert AWAITING_WORKTREE_BRANCH_NAME not in context.user_data
        assert "state lost" in mock_edit.call_args[0][1].lower()

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_stale_edit_name_after_new_command_reset_fails_closed(
        self, mock_edit: AsyncMock
    ) -> None:
        from ccgram.handlers.topics.new_command import new_command

        user_data = {
            BROWSE_PATH_KEY: "/tmp/proj",
            PENDING_WORKTREE_REPO: "/tmp/proj",
            PENDING_THREAD_ID: 42,
        }
        context = _make_context(user_data)

        nc_update = MagicMock()
        nc_update.effective_user = MagicMock(id=100)
        nc_update.message = AsyncMock()
        with patch(
            "ccgram.handlers.topics.new_command.config.is_user_allowed",
            return_value=True,
        ):
            await new_command(nc_update, context)

        query = _make_query()
        await _handle_worktree_callback(
            query, CB_WT_EDIT_NAME, _make_update(42), context
        )

        assert AWAITING_WORKTREE_BRANCH_NAME not in user_data
        query.answer.assert_awaited_once_with(
            "Stale browser (flow reset)", show_alert=True
        )
        mock_edit.assert_not_called()


class TestWorktreeDispatchStaleGuard:
    async def test_thread_mismatch_is_rejected(self) -> None:
        query = _make_query()
        context = _make_context({PENDING_THREAD_ID: 99})
        await _handle_worktree_callback(
            query, CB_WT_USE_CURRENT, _make_update(42), context
        )
        query.answer.assert_awaited_once()
        assert query.answer.await_args.kwargs.get("show_alert") is True

    @patch(
        "ccgram.handlers.topics.directory_callbacks._handle_wt_use_current",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.topics.directory_callbacks._handle_wt_new",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.topics.directory_callbacks._handle_wt_confirm",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.topics.directory_callbacks._handle_wt_edit_name",
        new_callable=AsyncMock,
    )
    async def test_routes_each_callback(
        self,
        mock_edit_name: AsyncMock,
        mock_confirm: AsyncMock,
        mock_new: AsyncMock,
        mock_use_current: AsyncMock,
    ) -> None:
        context = _make_context({PENDING_THREAD_ID: 42})
        update = _make_update(42)
        for data, mock in (
            (CB_WT_USE_CURRENT, mock_use_current),
            (CB_WT_NEW, mock_new),
            (CB_WT_CONFIRM, mock_confirm),
            (CB_WT_EDIT_NAME, mock_edit_name),
        ):
            await _handle_worktree_callback(_make_query(), data, update, context)
            mock.assert_awaited_once()

    @patch(
        "ccgram.handlers.topics.directory_callbacks._handle_wt_use_current",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.topics.directory_callbacks._handle_wt_new",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.topics.directory_callbacks._handle_wt_confirm",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.topics.directory_callbacks._handle_wt_edit_name",
        new_callable=AsyncMock,
    )
    async def test_no_pending_thread_fails_closed(
        self,
        mock_edit_name: AsyncMock,
        mock_confirm: AsyncMock,
        mock_new: AsyncMock,
        mock_use_current: AsyncMock,
    ) -> None:
        update = _make_update(42)
        for data, mock in (
            (CB_WT_USE_CURRENT, mock_use_current),
            (CB_WT_NEW, mock_new),
            (CB_WT_CONFIRM, mock_confirm),
            (CB_WT_EDIT_NAME, mock_edit_name),
        ):
            context = _make_context({PENDING_WORKTREE_REPO: "/stale/repo"})
            query = _make_query()
            await _handle_worktree_callback(query, data, update, context)
            query.answer.assert_awaited_once_with(
                "Stale browser (flow reset)", show_alert=True
            )
            mock.assert_not_awaited()


class TestNavigationStaleGuard:
    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_stale_up_after_new_command_reset_fails_closed(
        self, mock_edit: AsyncMock
    ) -> None:
        from ccgram.handlers.topics.new_command import new_command

        user_data = {BROWSE_PATH_KEY: "/some/old/path", PENDING_THREAD_ID: 42}
        context = _make_context(user_data)
        nc_update = MagicMock()
        nc_update.effective_user = MagicMock(id=100)
        nc_update.message = AsyncMock()
        with patch(
            "ccgram.handlers.topics.new_command.config.is_user_allowed",
            return_value=True,
        ):
            await new_command(nc_update, context)

        query = _make_query()
        await _handle_up(query, 100, _make_update(42), context)

        query.answer.assert_called_once_with(
            "Stale browser (flow reset)", show_alert=True
        )
        mock_edit.assert_not_called()
        assert BROWSE_PATH_KEY not in user_data

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.user_preferences")
    async def test_stale_star_after_new_command_reset_does_not_toggle(
        self, mock_prefs: MagicMock, mock_edit: AsyncMock
    ) -> None:
        from ccgram.handlers.topics.new_command import new_command

        user_data = {BROWSE_PATH_KEY: "/some/old/path", PENDING_THREAD_ID: 42}
        context = _make_context(user_data)
        nc_update = MagicMock()
        nc_update.effective_user = MagicMock(id=100)
        nc_update.message = AsyncMock()
        with patch(
            "ccgram.handlers.topics.new_command.config.is_user_allowed",
            return_value=True,
        ):
            await new_command(nc_update, context)

        query = _make_query()
        await _handle_star(query, 100, f"{CB_DIR_STAR}0", _make_update(42), context)

        query.answer.assert_called_once_with(
            "Stale browser (flow reset)", show_alert=True
        )
        mock_prefs.toggle_user_star.assert_not_called()
        mock_edit.assert_not_called()
        assert BROWSE_PATH_KEY not in user_data

    async def test_cross_topic_up_is_rejected(self) -> None:
        context = _make_context({PENDING_THREAD_ID: 99})
        query = _make_query()
        await _handle_up(query, 100, _make_update(42), context)
        query.answer.assert_called_once_with(
            "Stale browser (flow reset)", show_alert=True
        )

    async def test_stale_up_with_no_pending_thread_fails_closed(self) -> None:
        context = _make_context({BROWSE_PATH_KEY: "/some/old/path"})
        query = _make_query()
        await _handle_up(query, 100, _make_update(42), context)
        query.answer.assert_called_once_with(
            "Stale browser (flow reset)", show_alert=True
        )
        assert context.user_data.get(BROWSE_PATH_KEY) == "/some/old/path"
