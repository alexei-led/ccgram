import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.topics.directory_browser import BROWSE_PATH_KEY
from ccgram.handlers.topics.directory_callbacks import (
    _create_window_and_bind,
    _handle_confirm,
    _handle_wt_confirm,
    _handle_wt_new,
    _handle_wt_use_current,
)
from ccgram.handlers.text.text_handler import _handle_worktree_name_reply
from ccgram.handlers.user_state import (
    AWAITING_WORKTREE_BRANCH_NAME,
    PENDING_THREAD_ID,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_CREATING,
    PENDING_WORKTREE_DIRTY,
    PENDING_WORKTREE_PATH,
    PENDING_WORKTREE_REPO,
)
from ccgram.session import SessionManager
from ccgram.window_state_store import window_store

pytestmark = pytest.mark.integration


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


@pytest.fixture
def session_manager(tmp_path, monkeypatch) -> SessionManager:
    monkeypatch.setattr("ccgram.config.config.state_file", tmp_path / "state.json")
    monkeypatch.setattr(
        "ccgram.config.config.session_map_file", tmp_path / "session_map.json"
    )
    return SessionManager()


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


def _make_context(user_data: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data
    ctx.bot = AsyncMock()
    return ctx


@patch("ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock)
@patch("ccgram.handlers.topics.directory_callbacks.thread_router")
async def test_use_current_branch_skips_to_provider_picker(
    mock_tr: MagicMock, mock_edit: AsyncMock, git_repo: Path
) -> None:
    mock_tr.get_window_for_thread.return_value = None
    user_data = {BROWSE_PATH_KEY: str(git_repo), PENDING_THREAD_ID: 42}
    context = _make_context(user_data)

    await _handle_confirm(_make_query(), 100, _make_update(42), context)
    assert "Git Worktree" in mock_edit.call_args[0][1]

    await _handle_wt_use_current(_make_query(), context)
    assert "Select Provider" in mock_edit.call_args[0][1]
    assert PENDING_WORKTREE_REPO not in user_data


async def test_new_worktree_creates_and_persists_to_window_state(
    session_manager: SessionManager, git_repo: Path
) -> None:
    user_data = {
        PENDING_WORKTREE_REPO: str(git_repo),
        PENDING_WORKTREE_DIRTY: False,
    }
    context = _make_context(user_data)

    with patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit",
        new_callable=AsyncMock,
    ):
        await _handle_wt_new(_make_query(), context)
        branch = user_data[PENDING_WORKTREE_BRANCH]
        await _handle_wt_confirm(_make_query(), context)

    worktree_path = Path(user_data[PENDING_WORKTREE_PATH])
    assert worktree_path.is_dir()
    assert (worktree_path / "file.txt").exists()
    assert user_data[BROWSE_PATH_KEY] == str(worktree_path)

    mock_provider = MagicMock()
    mock_provider.capabilities.supports_hook = False
    mock_provider.capabilities.chat_first_command_path = False
    mock_provider.capabilities.has_yolo_confirmation = False

    with (
        patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        patch(
            "ccgram.handlers.topics.directory_callbacks.safe_edit",
            new_callable=AsyncMock,
        ),
        patch("ccgram.handlers.topics.directory_callbacks.tmux_manager") as mock_tmux,
        patch(
            "ccgram.handlers.topics.directory_callbacks.provider_registry"
        ) as mock_registry,
    ):
        mock_tmux.create_window = AsyncMock(
            return_value=(True, "Created window 'repo'", "repo", "@7")
        )
        mock_tmux.stamp_pane_title = AsyncMock()
        mock_registry.is_valid.return_value = True
        mock_registry.get.return_value = mock_provider

        await _create_window_and_bind(
            _make_query(), 100, str(worktree_path), "claude", "normal", context
        )

    state = window_store.window_states["@7"]
    assert state.worktree_path == str(worktree_path)
    assert state.worktree_branch == branch
    assert PENDING_WORKTREE_PATH not in user_data


async def test_create_window_failure_clears_worktree_state(
    session_manager: SessionManager, git_repo: Path
) -> None:
    user_data = {
        PENDING_WORKTREE_REPO: str(git_repo),
        PENDING_WORKTREE_DIRTY: False,
    }
    context = _make_context(user_data)

    with patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit",
        new_callable=AsyncMock,
    ):
        await _handle_wt_new(_make_query(), context)
        await _handle_wt_confirm(_make_query(), context)

    assert user_data[PENDING_WORKTREE_CREATING] is True

    with (
        patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        patch(
            "ccgram.handlers.topics.directory_callbacks.safe_edit",
            new_callable=AsyncMock,
        ),
        patch("ccgram.handlers.topics.directory_callbacks.tmux_manager") as mock_tmux,
    ):
        mock_tmux.create_window = AsyncMock(
            return_value=(False, "tmux refused", None, None)
        )
        await _create_window_and_bind(
            _make_query(),
            100,
            user_data[BROWSE_PATH_KEY],
            "claude",
            "normal",
            context,
        )

    assert PENDING_WORKTREE_CREATING not in user_data
    assert PENDING_WORKTREE_REPO not in user_data
    assert PENDING_WORKTREE_PATH not in user_data

    with patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit",
        new_callable=AsyncMock,
    ):
        q = _make_query()
        await _handle_wt_confirm(q, context)
    assert ("Creating worktree…",) not in [c.args for c in q.answer.await_args_list]


@patch("ccgram.handlers.topics.directory_callbacks.thread_router")
async def test_new_worktree_from_subdir_roots_topic_in_subdir(
    mock_tr: MagicMock, git_repo: Path
) -> None:
    mock_tr.get_window_for_thread.return_value = None
    (git_repo / "frontend").mkdir()
    (git_repo / "frontend" / "app.txt").write_text("x")
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-m", "add frontend")
    subdir = git_repo / "frontend"

    user_data = {BROWSE_PATH_KEY: str(subdir), PENDING_THREAD_ID: 42}
    context = _make_context(user_data)

    with patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit",
        new_callable=AsyncMock,
    ):
        await _handle_confirm(_make_query(), 100, _make_update(42), context)
        await _handle_wt_new(_make_query(), context)
        worktree_root = Path(user_data[PENDING_WORKTREE_PATH])
        await _handle_wt_confirm(_make_query(), context)

    expected = worktree_root / "frontend"
    assert expected.is_dir()
    assert user_data[BROWSE_PATH_KEY] == str(expected)


@patch("ccgram.handlers.topics.directory_callbacks.thread_router")
async def test_new_worktree_untracked_subdir_falls_back_to_root(
    mock_tr: MagicMock, git_repo: Path
) -> None:
    mock_tr.get_window_for_thread.return_value = None
    # Subdir exists on disk but is NOT committed → absent in fresh HEAD checkout.
    (git_repo / "scratch").mkdir()
    subdir = git_repo / "scratch"

    user_data = {BROWSE_PATH_KEY: str(subdir), PENDING_THREAD_ID: 42}
    context = _make_context(user_data)

    with patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit",
        new_callable=AsyncMock,
    ):
        await _handle_confirm(_make_query(), 100, _make_update(42), context)
        await _handle_wt_new(_make_query(), context)
        worktree_root = Path(user_data[PENDING_WORKTREE_PATH])
        await _handle_wt_confirm(_make_query(), context)

    assert not (worktree_root / "scratch").exists()
    assert user_data[BROWSE_PATH_KEY] == str(worktree_root)


def test_persist_worktree_state_accepts_subdir_cwd(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    from ccgram.handlers.topics.directory_callbacks import _persist_worktree_state

    worktree_root = tmp_path / "repo.worktrees" / "ccg-x"
    (worktree_root / "frontend").mkdir(parents=True)
    user_data = {
        PENDING_WORKTREE_PATH: str(worktree_root),
        PENDING_WORKTREE_BRANCH: "ccg/x",
    }
    context = _make_context(user_data)

    _persist_worktree_state("@9", str(worktree_root / "frontend"), context)

    state = window_store.window_states["@9"]
    assert state.worktree_path == str(worktree_root)
    assert state.worktree_branch == "ccg/x"
    assert PENDING_WORKTREE_PATH not in user_data


def test_persist_worktree_state_rejects_unrelated_cwd(
    session_manager: SessionManager, tmp_path: Path
) -> None:
    from ccgram.handlers.topics.directory_callbacks import _persist_worktree_state

    user_data = {
        PENDING_WORKTREE_PATH: str(tmp_path / "repo.worktrees" / "ccg-x"),
        PENDING_WORKTREE_BRANCH: "ccg/x",
    }
    context = _make_context(user_data)

    _persist_worktree_state("@11", str(tmp_path / "somewhere-else"), context)

    assert "@11" not in window_store.window_states


async def test_superseded_worktree_flow_cleared_by_ui_guard(git_repo: Path) -> None:
    from ccgram.handlers.text.text_handler import _check_ui_guards
    from ccgram.handlers.topics.directory_browser import (
        STATE_BROWSING_DIRECTORY,
        STATE_KEY,
    )
    from ccgram.handlers.user_state import PENDING_THREAD_TEXT

    user_data = {
        STATE_KEY: STATE_BROWSING_DIRECTORY,
        PENDING_THREAD_ID: 42,
        PENDING_THREAD_TEXT: "topic A message",
        PENDING_WORKTREE_REPO: str(git_repo),
        PENDING_WORKTREE_BRANCH: "ccg/agent-1",
        PENDING_WORKTREE_PATH: str(git_repo) + ".worktrees/ccg-agent-1",
        PENDING_WORKTREE_CREATING: True,
    }

    handled = await _check_ui_guards(user_data, 99, MagicMock())

    assert handled is False
    assert PENDING_WORKTREE_CREATING not in user_data
    assert PENDING_WORKTREE_REPO not in user_data
    assert PENDING_WORKTREE_BRANCH not in user_data
    assert PENDING_WORKTREE_PATH not in user_data
    assert STATE_KEY not in user_data
    assert PENDING_THREAD_ID not in user_data


async def test_provider_select_thread_mismatch_clears_worktree_state(
    git_repo: Path,
) -> None:
    from ccgram.handlers.topics.directory_callbacks import _validate_provider_select

    user_data = {
        PENDING_THREAD_ID: 42,
        PENDING_WORKTREE_REPO: str(git_repo),
        PENDING_WORKTREE_BRANCH: "ccg/agent-1",
        PENDING_WORKTREE_PATH: str(git_repo) + ".worktrees/ccg-agent-1",
        PENDING_WORKTREE_CREATING: True,
    }
    context = _make_context(user_data)
    query = _make_query()

    ok = await _validate_provider_select(query, 100, _make_update(99), context, 42)

    assert ok is False
    assert PENDING_WORKTREE_CREATING not in user_data
    assert PENDING_WORKTREE_REPO not in user_data
    assert PENDING_WORKTREE_BRANCH not in user_data
    assert PENDING_WORKTREE_PATH not in user_data


async def test_wt_confirm_double_tap_creates_worktree_once(git_repo: Path) -> None:
    worktree_path = git_repo.parent / "repo.worktrees" / "ccg-x"
    user_data = {
        PENDING_WORKTREE_REPO: str(git_repo),
        PENDING_WORKTREE_BRANCH: "ccg/x",
        PENDING_WORKTREE_PATH: str(worktree_path),
    }
    context = _make_context(user_data)
    create_mock = MagicMock()
    q1, q2 = _make_query(), _make_query()

    with (
        patch(
            "ccgram.handlers.topics.directory_callbacks.create_worktree",
            create_mock,
        ),
        patch(
            "ccgram.handlers.topics.directory_callbacks.safe_edit",
            new_callable=AsyncMock,
        ),
    ):
        await _handle_wt_confirm(q1, context)
        await _handle_wt_confirm(q2, context)

    assert create_mock.call_count == 1
    q2.answer.assert_awaited_with("Creating worktree…")


async def test_edit_name_text_reply_revalidates_and_reconfirms(
    git_repo: Path,
) -> None:
    user_data = {
        PENDING_THREAD_ID: 42,
        PENDING_WORKTREE_REPO: str(git_repo),
        PENDING_WORKTREE_DIRTY: False,
        AWAITING_WORKTREE_BRANCH_NAME: True,
    }
    message = MagicMock()

    with patch(
        "ccgram.handlers.text.text_handler.safe_reply", new_callable=AsyncMock
    ) as mock_reply:
        handled = await _handle_worktree_name_reply(
            user_data, 42, "feature/login", message
        )

    assert handled is True
    assert user_data[PENDING_WORKTREE_BRANCH] == "feature/login"
    assert user_data[PENDING_WORKTREE_PATH].endswith("repo.worktrees/feature-login")
    assert AWAITING_WORKTREE_BRANCH_NAME not in user_data
    assert "New Worktree" in mock_reply.call_args[0][1]


async def test_edit_name_invalid_branch_reprompts(git_repo: Path) -> None:
    user_data = {
        PENDING_THREAD_ID: 42,
        PENDING_WORKTREE_REPO: str(git_repo),
        AWAITING_WORKTREE_BRANCH_NAME: True,
    }
    message = MagicMock()

    with patch(
        "ccgram.handlers.text.text_handler.safe_reply", new_callable=AsyncMock
    ) as mock_reply:
        handled = await _handle_worktree_name_reply(
            user_data, 42, "bad branch..name", message
        )

    assert handled is True
    assert user_data[AWAITING_WORKTREE_BRANCH_NAME] is True
    assert "Invalid branch name" in mock_reply.call_args[0][1]


async def test_edit_name_inactive_when_flag_unset() -> None:
    handled = await _handle_worktree_name_reply({}, 42, "x", MagicMock())
    assert handled is False


@patch("ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock)
@patch("ccgram.handlers.topics.directory_callbacks.thread_router")
async def test_non_git_directory_skips_worktree_picker(
    mock_tr: MagicMock, mock_edit: AsyncMock, tmp_path: Path
) -> None:
    mock_tr.get_window_for_thread.return_value = None
    plain = tmp_path / "plain"
    plain.mkdir()
    user_data = {BROWSE_PATH_KEY: str(plain), PENDING_THREAD_ID: 42}
    context = _make_context(user_data)

    await _handle_confirm(_make_query(), 100, _make_update(42), context)

    assert "Select Provider" in mock_edit.call_args[0][1]
    assert PENDING_WORKTREE_REPO not in user_data
