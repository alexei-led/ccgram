import shutil
import subprocess
from pathlib import Path

import pytest

from ccgram.handlers.topics.worktree import (
    WorktreeError,
    check_worktree_eligibility,
    create_worktree,
    slug_for_path,
    suggest_branch_name,
    validate_branch_name,
    worktree_path_for,
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def _git_repo_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One-time git repo creation per session (or per xdist worker).

    Seven git subprocess calls run once instead of once per test.
    Each test gets an isolated copy via shutil.copytree (see git_repo).
    """
    base = tmp_path_factory.mktemp("git_template")
    repo = base / "repo"
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
def git_repo(tmp_path: Path, _git_repo_template: Path) -> Path:
    """Per-test copy of the template repo — filesystem copy, no git subprocess."""
    repo = tmp_path / "repo"
    shutil.copytree(_git_repo_template, repo)
    return repo


class TestCheckWorktreeEligibility:
    def test_clean_repo_is_eligible(self, git_repo: Path) -> None:
        result = check_worktree_eligibility(git_repo)
        assert result.eligible is True
        assert result.current_branch == "main"
        assert result.dirty is False
        assert result.repo_path == git_repo.resolve()
        assert result.reason is None

    @pytest.mark.parametrize(
        "filename", ["file.txt", "new.txt"], ids=["modified_tracked", "untracked"]
    )
    def test_dirty_repo_is_eligible_with_dirty_flag(
        self, git_repo: Path, filename: str
    ) -> None:
        (git_repo / filename).write_text("changed")
        result = check_worktree_eligibility(git_repo)
        assert result.eligible is True
        assert result.dirty is True

    def test_bare_repo_is_ineligible(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare.git"
        bare.mkdir()
        _git(bare, "init", "--bare")
        result = check_worktree_eligibility(bare)
        assert result.eligible is False
        assert result.reason is not None

    def test_detached_head_is_ineligible(self, git_repo: Path) -> None:
        sha = _git(git_repo, "rev-parse", "HEAD").stdout.strip()
        _git(git_repo, "checkout", sha)
        result = check_worktree_eligibility(git_repo)
        assert result.eligible is False
        assert result.reason == "detached HEAD"

    def test_mid_rebase_is_ineligible(self, git_repo: Path) -> None:
        git_dir = Path(_git(git_repo, "rev-parse", "--git-dir").stdout.strip())
        if not git_dir.is_absolute():
            git_dir = git_repo / git_dir
        (git_dir / "rebase-merge").mkdir()
        result = check_worktree_eligibility(git_repo)
        assert result.eligible is False
        assert result.reason == "merge or rebase in progress"

    def test_merge_head_is_ineligible(self, git_repo: Path) -> None:
        git_dir = Path(_git(git_repo, "rev-parse", "--git-dir").stdout.strip())
        if not git_dir.is_absolute():
            git_dir = git_repo / git_dir
        (git_dir / "MERGE_HEAD").write_text("deadbeef\n")
        result = check_worktree_eligibility(git_repo)
        assert result.eligible is False
        assert result.reason == "merge or rebase in progress"

    def test_non_git_dir_is_ineligible(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        result = check_worktree_eligibility(plain)
        assert result.eligible is False
        assert result.reason == "not a git work tree"
        assert result.repo_path is None


class TestSuggestBranchName:
    def test_kebab_case_from_title(self, git_repo: Path) -> None:
        assert suggest_branch_name("Fix the Bug!", git_repo) == "ccg/fix-the-bug"

    @pytest.mark.parametrize("title", [None, "   "], ids=["none", "blank"])
    def test_missing_title_falls_back_to_agent(
        self, git_repo: Path, title: str | None
    ) -> None:
        assert suggest_branch_name(title, git_repo) == "ccg/agent-1"

    def test_collision_with_existing_branch(self, git_repo: Path) -> None:
        _git(git_repo, "branch", "ccg/feature")
        assert suggest_branch_name("feature", git_repo) == "ccg/feature-2"

    def test_double_collision_increments(self, git_repo: Path) -> None:
        _git(git_repo, "branch", "ccg/feature")
        _git(git_repo, "branch", "ccg/feature-2")
        assert suggest_branch_name("feature", git_repo) == "ccg/feature-3"

    def test_collision_with_existing_worktree(self, git_repo: Path) -> None:
        wt = git_repo.parent / "wt-agent-1"
        _git(git_repo, "worktree", "add", str(wt), "-b", "ccg/agent-1", "HEAD")
        assert suggest_branch_name(None, git_repo) == "ccg/agent-2"


class TestValidateBranchName:
    @pytest.mark.parametrize(
        "name",
        ["ccg/feature", "feature", "ccg/fix-the-bug", "a" * 200],
    )
    def test_accepts_valid_name(self, name: str) -> None:
        assert validate_branch_name(name) is True

    @pytest.mark.parametrize(
        ("name", "reason"),
        [
            ("", "empty"),
            ("-leading", "leading dash reads as a git option"),
            ("a" * 201, "one over the length cap"),
            ("has space", "space"),
            ("bad..name", "double dot"),
            ("feature.lock", "reserved .lock suffix"),
            ("foo~1", "revision operator"),
            ("a:b", "colon"),
            ("foo^", "caret"),
            ("foo//bar", "empty path component"),
            ("foo.", "trailing dot"),
        ],
    )
    def test_rejects_invalid_name(self, name: str, reason: str) -> None:
        assert validate_branch_name(name) is False, reason


class TestPathHelpers:
    @pytest.mark.parametrize(
        ("branch", "expected"),
        [("ccg/foo/bar", "ccg-foo-bar"), ("ccg-x", "ccg-x")],
    )
    def test_slug_for_path(self, branch: str, expected: str) -> None:
        assert slug_for_path(branch) == expected

    def test_worktree_path_for(self) -> None:
        repo = Path("/a/b/myrepo")
        assert worktree_path_for(repo, "ccg-x") == Path("/a/b/myrepo.worktrees/ccg-x")


class TestCreateWorktree:
    def test_success_creates_dir_and_branch(self, git_repo: Path) -> None:
        slug = slug_for_path("ccg/new")
        target = worktree_path_for(git_repo, slug)
        create_worktree(git_repo, "ccg/new", target)
        assert target.is_dir()
        assert (target / "file.txt").read_text() == "hello"
        branches = _git(
            git_repo, "branch", "--list", "--format=%(refname:short)"
        ).stdout.split()
        assert "ccg/new" in branches

    def test_duplicate_branch_raises_worktree_error(self, git_repo: Path) -> None:
        first = worktree_path_for(git_repo, slug_for_path("ccg/dup"))
        create_worktree(git_repo, "ccg/dup", first)
        second = worktree_path_for(git_repo, "other-dup")
        with pytest.raises(WorktreeError):
            create_worktree(git_repo, "ccg/dup", second)

    def test_occupied_target_path_raises_worktree_error(self, git_repo: Path) -> None:
        target = worktree_path_for(git_repo, "occupied")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir()
        (target / "stray.txt").write_text("x")
        with pytest.raises(WorktreeError):
            create_worktree(git_repo, "ccg/occupied", target)

    def test_parent_dir_mkdir_failure_raises_worktree_error(
        self, git_repo: Path
    ) -> None:
        target = worktree_path_for(git_repo, "blocked")
        target.parent.parent.mkdir(parents=True, exist_ok=True)
        target.parent.write_text("not a directory")
        with pytest.raises(WorktreeError):
            create_worktree(git_repo, "ccg/blocked", target)
