import subprocess
from pathlib import Path
from types import SimpleNamespace
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
from ccgram.handlers.topics.window_launch_service import WindowLaunchRequest
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

_MOD_DC = "ccgram.handlers.topics.directory_callbacks"
_MOD_WC = "ccgram.handlers.topics.workspace_callbacks"
_MOD_WL = "ccgram.handlers.topics.window_launch_service"


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


def _agent_provider() -> MagicMock:
    """A provider with every optional launch step switched off."""
    provider = MagicMock()
    provider.capabilities.supports_hook = False
    provider.capabilities.chat_first_command_path = False
    provider.capabilities.has_yolo_confirmation = False
    return provider


@pytest.fixture(autouse=True)
def safe_edit():
    """The flow's two ``safe_edit`` call sites, patched.

    ``.dc`` renders the directory/worktree screens, ``.wc`` the workspace and
    provider pickers; a test asserts on whichever screen it drove the flow to.
    """
    with (
        patch(f"{_MOD_DC}.safe_edit", new_callable=AsyncMock) as dc,
        patch(f"{_MOD_WC}.safe_edit", new_callable=AsyncMock) as wc,
    ):
        yield SimpleNamespace(dc=dc, wc=wc)


@pytest.fixture
def unbound_thread():
    """The target topic has no window yet, so the creation flow proceeds."""
    with patch(f"{_MOD_DC}.thread_router") as mock_tr:
        mock_tr.get_window_for_thread.return_value = None
        yield mock_tr


@patch(f"{_MOD_WC}.tmux_manager")
async def test_use_current_branch_skips_to_provider_picker(
    mock_mux: MagicMock,
    unbound_thread: MagicMock,
    safe_edit: SimpleNamespace,
    git_repo: Path,
) -> None:
    mock_mux.capabilities.native_agent_status = False
    user_data = {BROWSE_PATH_KEY: str(git_repo), PENDING_THREAD_ID: 42}
    context = _make_context(user_data)

    await _handle_confirm(_make_query(), 100, _make_update(42), context)
    assert "Git Worktree" in safe_edit.dc.call_args[0][1]

    await _handle_wt_use_current(_make_query(), context)
    assert "Select Provider" in safe_edit.wc.call_args[0][1]
    assert PENDING_WORKTREE_REPO not in user_data


async def test_new_worktree_creates_and_persists_to_window_state(
    session_manager: SessionManager, git_repo: Path
) -> None:
    user_data = {
        PENDING_WORKTREE_REPO: str(git_repo),
        PENDING_WORKTREE_DIRTY: False,
    }
    context = _make_context(user_data)

    await _handle_wt_new(_make_query(), context)
    branch = user_data[PENDING_WORKTREE_BRANCH]
    await _handle_wt_confirm(_make_query(), context)

    worktree_path = Path(user_data[PENDING_WORKTREE_PATH])
    assert worktree_path.is_dir()
    assert (worktree_path / "file.txt").exists()
    assert user_data[BROWSE_PATH_KEY] == str(worktree_path)

    with (
        patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        patch(f"{_MOD_WL}.tmux_manager") as mock_tmux,
        patch(f"{_MOD_WL}.provider_registry") as mock_registry,
    ):
        mock_tmux.capabilities.native_worktrees = False
        mock_tmux.create_window = AsyncMock(
            return_value=(True, "Created window 'repo'", "repo", "@7")
        )
        mock_tmux.stamp_pane_title = AsyncMock()
        mock_registry.is_valid.return_value = True
        mock_registry.get.return_value = _agent_provider()

        await _create_window_and_bind(
            _make_query(),
            context,
            WindowLaunchRequest(
                user_id=100,
                thread_id=user_data.get(PENDING_THREAD_ID),
                provider_name="claude",
                cwd=str(worktree_path),
                mode="normal",
                pending_text=None,
            ),
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

    await _handle_wt_new(_make_query(), context)
    await _handle_wt_confirm(_make_query(), context)

    assert user_data[PENDING_WORKTREE_CREATING] is True

    with (
        patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        patch(f"{_MOD_WL}.tmux_manager") as mock_tmux,
    ):
        mock_tmux.capabilities.native_worktrees = False
        mock_tmux.create_window = AsyncMock(
            return_value=(False, "tmux refused", None, None)
        )
        await _create_window_and_bind(
            _make_query(),
            context,
            WindowLaunchRequest(
                user_id=100,
                thread_id=user_data.get(PENDING_THREAD_ID),
                provider_name="claude",
                cwd=user_data[BROWSE_PATH_KEY],
                mode="normal",
                pending_text=None,
            ),
        )

    assert PENDING_WORKTREE_CREATING not in user_data
    assert PENDING_WORKTREE_REPO not in user_data
    assert PENDING_WORKTREE_PATH not in user_data

    q = _make_query()
    await _handle_wt_confirm(q, context)
    assert ("Creating worktree…",) not in [c.args for c in q.answer.await_args_list]


async def test_herdr_delegates_worktree_creation(
    session_manager: SessionManager, git_repo: Path
) -> None:
    # native_worktrees backend: _create_window_and_bind routes through
    # create_worktree_window (one herdr `worktree create`), never create_window.
    worktree_path = str(git_repo.parent / "repo.worktrees" / "ccg-feature")
    user_data = {
        PENDING_WORKTREE_REPO: str(git_repo),
        PENDING_WORKTREE_BRANCH: "ccg/feature",
        PENDING_WORKTREE_PATH: worktree_path,
        PENDING_THREAD_ID: 42,
    }
    context = _make_context(user_data)

    with (
        patch("ccgram.providers.resolve_launch_command", return_value="claude"),
        patch(f"{_MOD_WL}.thread_router"),
        patch(f"{_MOD_WL}.tmux_manager") as mock_tmux,
        patch(f"{_MOD_WL}.provider_registry") as mock_registry,
    ):
        mock_tmux.capabilities.native_worktrees = True
        mock_tmux.create_worktree_window = AsyncMock(
            return_value=(True, "Created herdr worktree", "ccg-feature", "w5:t1")
        )
        mock_tmux.stamp_pane_title = AsyncMock()
        mock_registry.is_valid.return_value = True
        mock_registry.get.return_value = _agent_provider()

        await _create_window_and_bind(
            _make_query(),
            context,
            WindowLaunchRequest(
                user_id=100,
                thread_id=user_data.get(PENDING_THREAD_ID),
                provider_name="claude",
                cwd=worktree_path,
                mode="normal",
                pending_text=None,
            ),
        )

    # Delegated to herdr; the git create_window path is never used.
    mock_tmux.create_worktree_window.assert_awaited_once()
    call = mock_tmux.create_worktree_window.await_args
    assert call is not None
    assert call.args == (str(git_repo), worktree_path, "ccg/feature")
    assert call.kwargs["window_name"] == "ccg-feature"
    assert call.kwargs["launch_command"] == "claude"
    mock_tmux.create_window.assert_not_called()

    state = window_store.window_states["w5:t1"]
    assert state.worktree_path == worktree_path
    assert state.worktree_branch == "ccg/feature"
    assert PENDING_WORKTREE_PATH not in user_data


async def test_new_worktree_from_subdir_roots_topic_in_subdir(
    unbound_thread: MagicMock, git_repo: Path
) -> None:
    (git_repo / "frontend").mkdir()
    (git_repo / "frontend" / "app.txt").write_text("x")
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-m", "add frontend")
    subdir = git_repo / "frontend"

    user_data = {BROWSE_PATH_KEY: str(subdir), PENDING_THREAD_ID: 42}
    context = _make_context(user_data)

    await _handle_confirm(_make_query(), 100, _make_update(42), context)
    await _handle_wt_new(_make_query(), context)
    worktree_root = Path(user_data[PENDING_WORKTREE_PATH])
    await _handle_wt_confirm(_make_query(), context)

    expected = worktree_root / "frontend"
    assert expected.is_dir()
    assert user_data[BROWSE_PATH_KEY] == str(expected)


async def test_new_worktree_untracked_subdir_falls_back_to_root(
    unbound_thread: MagicMock, git_repo: Path
) -> None:
    # Subdir exists on disk but is NOT committed → absent in fresh HEAD checkout.
    (git_repo / "scratch").mkdir()
    subdir = git_repo / "scratch"

    user_data = {BROWSE_PATH_KEY: str(subdir), PENDING_THREAD_ID: 42}
    context = _make_context(user_data)

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

    with patch(f"{_MOD_DC}.create_worktree", create_mock):
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


@patch(f"{_MOD_WC}.tmux_manager")
async def test_non_git_directory_skips_worktree_picker(
    mock_mux: MagicMock,
    unbound_thread: MagicMock,
    safe_edit: SimpleNamespace,
    tmp_path: Path,
) -> None:
    mock_mux.capabilities.native_agent_status = False
    plain = tmp_path / "plain"
    plain.mkdir()
    user_data = {BROWSE_PATH_KEY: str(plain), PENDING_THREAD_ID: 42}
    context = _make_context(user_data)

    await _handle_confirm(_make_query(), 100, _make_update(42), context)

    assert "Select Provider" in safe_edit.wc.call_args[0][1]
    assert PENDING_WORKTREE_REPO not in user_data
