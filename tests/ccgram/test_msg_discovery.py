import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ccgram.msg_discovery import (
    WindowInfo,
    clear_declared,
    list_peers,
    register_declared,
)


@pytest.fixture()
def declared_path(tmp_path: Path) -> Path:
    return tmp_path / "declared.json"


class TestListPeers:
    def test_returns_peers_from_window_states(
        self, tmp_path: Path, declared_path: Path
    ) -> None:
        window_states = {
            "@0": WindowInfo(
                cwd="/home/user/payment-svc",
                window_name="payment-svc",
                provider_name="claude",
            ),
            "@5": WindowInfo(
                cwd="/home/user/api-gw",
                window_name="api-gw",
                provider_name="codex",
            ),
        }
        peers = list_peers(
            window_states=window_states,
            tmux_session="ccgram",
            declared_path=declared_path,
        )
        assert len(peers) == 2
        ids = {p.window_id for p in peers}
        assert ids == {"ccgram:@0", "ccgram:@5"}

    def test_includes_provider_and_cwd(
        self, tmp_path: Path, declared_path: Path
    ) -> None:
        window_states = {
            "@0": WindowInfo(
                cwd="/proj",
                window_name="proj",
                provider_name="claude",
            ),
        }
        peers = list_peers(
            window_states=window_states,
            tmux_session="ccgram",
            declared_path=declared_path,
        )
        assert peers[0].provider == "claude"
        assert peers[0].cwd == "/proj"
        assert peers[0].name == "proj"

    def test_merges_declared_overlay(self, tmp_path: Path, declared_path: Path) -> None:
        window_states = {
            "@0": WindowInfo(
                cwd="/proj",
                window_name="proj",
                provider_name="claude",
            ),
        }
        register_declared(
            "ccgram:@0", task="Implementing refund", team="backend", path=declared_path
        )
        peers = list_peers(
            window_states=window_states,
            tmux_session="ccgram",
            declared_path=declared_path,
        )
        assert peers[0].task == "Implementing refund"
        assert peers[0].team == "backend"

    def test_empty_window_states(self, declared_path: Path) -> None:
        peers = list_peers(
            window_states={},
            tmux_session="ccgram",
            declared_path=declared_path,
        )
        assert peers == []


class TestFiltering:
    @pytest.fixture()
    def window_states(self) -> dict[str, WindowInfo]:
        return {
            "@0": WindowInfo(
                cwd="/home/user/payment-svc",
                window_name="payment-svc",
                provider_name="claude",
            ),
            "@5": WindowInfo(
                cwd="/home/user/api-gw",
                window_name="api-gw",
                provider_name="codex",
            ),
            "@8": WindowInfo(
                cwd="/home/user/dashboard",
                window_name="dashboard",
                provider_name="gemini",
            ),
        }

    def test_filter_by_provider(
        self, window_states: dict[str, WindowInfo], declared_path: Path
    ) -> None:
        peers = list_peers(
            window_states=window_states,
            tmux_session="ccgram",
            declared_path=declared_path,
            filter_provider="claude",
        )
        assert len(peers) == 1
        assert peers[0].provider == "claude"

    def test_filter_by_team(
        self, window_states: dict[str, WindowInfo], declared_path: Path
    ) -> None:
        register_declared("ccgram:@0", team="backend", path=declared_path)
        register_declared("ccgram:@5", team="backend", path=declared_path)
        register_declared("ccgram:@8", team="frontend", path=declared_path)

        peers = list_peers(
            window_states=window_states,
            tmux_session="ccgram",
            declared_path=declared_path,
            filter_team="backend",
        )
        assert len(peers) == 2
        ids = {p.window_id for p in peers}
        assert ids == {"ccgram:@0", "ccgram:@5"}

    def test_filter_by_cwd_glob(
        self, window_states: dict[str, WindowInfo], declared_path: Path
    ) -> None:
        peers = list_peers(
            window_states=window_states,
            tmux_session="ccgram",
            declared_path=declared_path,
            filter_cwd="*/api-*",
        )
        assert len(peers) == 1
        assert peers[0].name == "api-gw"

    def test_combined_filters(
        self, window_states: dict[str, WindowInfo], declared_path: Path
    ) -> None:
        register_declared("ccgram:@0", team="backend", path=declared_path)
        register_declared("ccgram:@5", team="backend", path=declared_path)

        peers = list_peers(
            window_states=window_states,
            tmux_session="ccgram",
            declared_path=declared_path,
            filter_provider="codex",
            filter_team="backend",
        )
        assert len(peers) == 1
        assert peers[0].window_id == "ccgram:@5"

    def test_no_match_returns_empty(
        self, window_states: dict[str, WindowInfo], declared_path: Path
    ) -> None:
        peers = list_peers(
            window_states=window_states,
            tmux_session="ccgram",
            declared_path=declared_path,
            filter_provider="shell",
        )
        assert peers == []


class TestDeclaredOverlay:
    def test_register_task_and_team(self, declared_path: Path) -> None:
        register_declared(
            "ccgram:@0", task="Implementing refund", team="backend", path=declared_path
        )
        info = _load_declared(declared_path)
        assert info["ccgram:@0"]["task"] == "Implementing refund"
        assert info["ccgram:@0"]["team"] == "backend"

    def test_update_existing(self, declared_path: Path) -> None:
        register_declared("ccgram:@0", task="old task", path=declared_path)
        register_declared("ccgram:@0", task="new task", path=declared_path)
        info = _load_declared(declared_path)
        assert info["ccgram:@0"]["task"] == "new task"

    def test_partial_update_preserves_fields(self, declared_path: Path) -> None:
        register_declared(
            "ccgram:@0", task="my task", team="backend", path=declared_path
        )
        register_declared("ccgram:@0", task="updated task", path=declared_path)
        info = _load_declared(declared_path)
        assert info["ccgram:@0"]["task"] == "updated task"
        assert info["ccgram:@0"]["team"] == "backend"

    def test_clear_on_window_death(self, declared_path: Path) -> None:
        register_declared("ccgram:@0", task="my task", path=declared_path)
        clear_declared("ccgram:@0", path=declared_path)
        info = _load_declared(declared_path)
        assert "ccgram:@0" not in info

    def test_clear_nonexistent_is_noop(self, declared_path: Path) -> None:
        clear_declared("ccgram:@99", path=declared_path)

    def test_register_without_path_uses_default(self, tmp_path: Path) -> None:
        with patch(
            "ccgram.msg_discovery._default_declared_path",
            return_value=tmp_path / "declared.json",
        ):
            register_declared("ccgram:@0", task="test")
            info = _load_declared(tmp_path / "declared.json")
            assert info["ccgram:@0"]["task"] == "test"


class TestBranchDetection:
    def test_branch_detected_from_cwd(
        self, tmp_path: Path, declared_path: Path
    ) -> None:
        window_states = {
            "@0": WindowInfo(
                cwd=str(tmp_path),
                window_name="proj",
                provider_name="claude",
            ),
        }
        with patch(
            "ccgram.msg_discovery.detect_branch",
            return_value="feat/refund",
        ):
            peers = list_peers(
                window_states=window_states,
                tmux_session="ccgram",
                declared_path=declared_path,
            )
        assert peers[0].branch == "feat/refund"

    def test_branch_empty_on_git_failure(
        self, tmp_path: Path, declared_path: Path
    ) -> None:
        window_states = {
            "@0": WindowInfo(
                cwd=str(tmp_path),
                window_name="proj",
                provider_name="claude",
            ),
        }
        with patch(
            "ccgram.msg_discovery.detect_branch",
            return_value="",
        ):
            peers = list_peers(
                window_states=window_states,
                tmux_session="ccgram",
                declared_path=declared_path,
            )
        assert peers[0].branch == ""

    def test_detect_branch_real_git_repo(self, tmp_path: Path) -> None:
        from ccgram.msg_discovery import detect_branch

        git_env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(tmp_path),
        }
        subprocess.run(
            ["git", "init", "-b", "feat/test", str(tmp_path)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "--allow-empty", "-m", "init"],
            check=True,
            capture_output=True,
            env=git_env,
        )
        assert detect_branch(str(tmp_path)) == "feat/test"

    def test_detect_branch_nonexistent_dir(self) -> None:
        from ccgram.msg_discovery import detect_branch

        assert detect_branch("/nonexistent/path") == ""


def _load_declared(path: Path) -> dict:
    import json

    if not path.exists():
        return {}
    return json.loads(path.read_text())
