"""Tests for TmuxManager.discover_external_sessions and _scan_session_windows."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.tmux_manager import TmuxManager, TmuxWindow


def _make_proc(stdout: str = "", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout.encode(), b"")
    proc.returncode = returncode
    return proc


class TestDiscoverExternalSessions:
    @pytest.fixture
    def manager(self) -> TmuxManager:
        tm = TmuxManager.__new__(TmuxManager)
        tm.session_name = "ccgram"
        tm._external_cache = []
        tm._external_cache_expires = 0.0
        return tm

    @pytest.mark.asyncio
    async def test_returns_windows_with_ai_processes(self, manager):
        sessions_proc = _make_proc("my-project\nccgram\n")
        windows_proc = _make_proc(
            "@0\tproject\t/home/user/project\tclaude\n@1\tbash\t/home/user\tbash\n"
        )

        with (
            patch("ccgram.tmux_manager.asyncio.create_subprocess_exec") as mock_exec,
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = ""
            mock_exec.side_effect = [sessions_proc, windows_proc]
            result = await manager.discover_external_sessions()

        assert len(result) == 1
        assert result[0].window_id == "my-project:@0"
        assert result[0].window_name == "project"
        assert result[0].cwd == "/home/user/project"
        assert result[0].pane_current_command == "claude"

    @pytest.mark.asyncio
    async def test_skips_own_session(self, manager):
        sessions_proc = _make_proc("ccgram\n")

        with (
            patch("ccgram.tmux_manager.asyncio.create_subprocess_exec") as mock_exec,
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = ""
            mock_exec.return_value = sessions_proc
            result = await manager.discover_external_sessions()

        assert result == []
        assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_pattern_filtering(self, manager):
        sessions_proc = _make_proc("omc-abc\nrandom-session\n")
        omc_windows = _make_proc("@0\tagent\t/tmp\tclaude\n")

        with (
            patch("ccgram.tmux_manager.asyncio.create_subprocess_exec") as mock_exec,
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = "omc-*"
            mock_exec.side_effect = [sessions_proc, omc_windows]
            result = await manager.discover_external_sessions()

        assert len(result) == 1
        assert result[0].window_id == "omc-abc:@0"

    @pytest.mark.asyncio
    async def test_multiple_patterns(self, manager):
        sessions_proc = _make_proc("omc-abc\nomx-xyz\nother\n")
        omc_windows = _make_proc("@0\tagent\t/tmp\tclaude\n")
        omx_windows = _make_proc("@0\tgemini\t/tmp\tgemini\n")

        with (
            patch("ccgram.tmux_manager.asyncio.create_subprocess_exec") as mock_exec,
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = "omc-*, omx-*"
            mock_exec.side_effect = [sessions_proc, omc_windows, omx_windows]
            result = await manager.discover_external_sessions()

        assert len(result) == 2
        ids = {w.window_id for w in result}
        assert ids == {"omc-abc:@0", "omx-xyz:@0"}

    @pytest.mark.asyncio
    async def test_no_patterns_scans_all_sessions(self, manager):
        sessions_proc = _make_proc("sess-a\nsess-b\n")
        win_a = _make_proc("@0\twin\t/tmp\tclaude\n")
        win_b = _make_proc("@0\twin\t/tmp\tcodex\n")

        with (
            patch("ccgram.tmux_manager.asyncio.create_subprocess_exec") as mock_exec,
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = ""
            mock_exec.side_effect = [sessions_proc, win_a, win_b]
            result = await manager.discover_external_sessions()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_emdash_sessions_included(self, manager):
        sessions_proc = _make_proc("emdash-claude-main-abc123\n")
        windows_proc = _make_proc("@0\temdash\t/home/user\tclaude\n")

        with (
            patch("ccgram.tmux_manager.asyncio.create_subprocess_exec") as mock_exec,
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = ""
            mock_exec.side_effect = [sessions_proc, windows_proc]
            result = await manager.discover_external_sessions()

        assert len(result) == 1
        assert result[0].window_id == "emdash-claude-main-abc123:@0"

    @pytest.mark.asyncio
    async def test_cache_returns_cached_results(self, manager):
        manager._external_cache = [
            TmuxWindow(
                window_id="cached:@0",
                window_name="cached",
                cwd="/tmp",
                pane_current_command="claude",
            )
        ]
        manager._external_cache_expires = asyncio.get_event_loop().time() + 100

        with patch("ccgram.tmux_manager.asyncio.create_subprocess_exec") as mock_exec:
            result = await manager.discover_external_sessions()

        assert len(result) == 1
        assert result[0].window_id == "cached:@0"
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_is_a_copy(self, manager):
        original = TmuxWindow(
            window_id="x:@0", window_name="x", cwd="/tmp", pane_current_command="claude"
        )
        manager._external_cache = [original]
        manager._external_cache_expires = asyncio.get_event_loop().time() + 100

        result = await manager.discover_external_sessions()
        result.append(
            TmuxWindow(
                window_id="extra:@0",
                window_name="extra",
                cwd="/tmp",
                pane_current_command="claude",
            )
        )
        assert len(manager._external_cache) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_timeout_returns_empty(self, manager):
        with (
            patch(
                "ccgram.tmux_manager.asyncio.create_subprocess_exec",
                side_effect=TimeoutError,
            ),
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = ""
            result = await manager.discover_external_sessions()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_sessions_oserror_returns_empty(self, manager):
        with (
            patch(
                "ccgram.tmux_manager.asyncio.create_subprocess_exec",
                side_effect=OSError("no tmux"),
            ),
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = ""
            result = await manager.discover_external_sessions()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_sessions_nonzero_exit_returns_empty(self, manager):
        sessions_proc = _make_proc("", returncode=1)

        with (
            patch(
                "ccgram.tmux_manager.asyncio.create_subprocess_exec",
                return_value=sessions_proc,
            ),
            patch("ccgram.tmux_manager.config") as mock_config,
        ):
            mock_config.tmux_external_patterns = ""
            result = await manager.discover_external_sessions()

        assert result == []

    @pytest.mark.asyncio
    async def test_deprecated_alias_delegates(self, manager):
        manager.discover_external_sessions = AsyncMock(return_value=[])
        result = await manager.discover_emdash_sessions()
        assert result == []
        manager.discover_external_sessions.assert_awaited_once()


class TestScanSessionWindows:
    @pytest.fixture
    def manager(self) -> TmuxManager:
        tm = TmuxManager.__new__(TmuxManager)
        tm.session_name = "ccgram"
        return tm

    @pytest.mark.asyncio
    async def test_filters_non_ai_windows(self, manager):
        proc = _make_proc(
            "@0\tproject\t/home/user\tclaude\n"
            "@1\tbash\t/home/user\tbash\n"
            "@2\tvim\t/home/user\tvim\n"
        )

        with patch(
            "ccgram.tmux_manager.asyncio.create_subprocess_exec", return_value=proc
        ):
            result = await manager._scan_session_windows("my-session")

        assert len(result) == 1
        assert result[0].window_id == "my-session:@0"

    @pytest.mark.asyncio
    async def test_multiple_ai_windows(self, manager):
        proc = _make_proc(
            "@0\tproject-a\t/home/a\tclaude\n"
            "@1\tproject-b\t/home/b\tcodex\n"
            "@2\tproject-c\t/home/c\tgemini\n"
        )

        with patch(
            "ccgram.tmux_manager.asyncio.create_subprocess_exec", return_value=proc
        ):
            result = await manager._scan_session_windows("ext")

        assert len(result) == 3
        assert {w.pane_current_command for w in result} == {"claude", "codex", "gemini"}

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self, manager):
        with patch(
            "ccgram.tmux_manager.asyncio.create_subprocess_exec",
            side_effect=TimeoutError,
        ):
            result = await manager._scan_session_windows("my-session")
        assert result == []

    @pytest.mark.asyncio
    async def test_oserror_returns_empty(self, manager):
        with patch(
            "ccgram.tmux_manager.asyncio.create_subprocess_exec",
            side_effect=OSError("fail"),
        ):
            result = await manager._scan_session_windows("my-session")
        assert result == []

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_empty(self, manager):
        proc = _make_proc("", returncode=1)
        with patch(
            "ccgram.tmux_manager.asyncio.create_subprocess_exec", return_value=proc
        ):
            result = await manager._scan_session_windows("my-session")
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_lines_skipped(self, manager):
        proc = _make_proc(
            "@0\tproject\t/home\tclaude\n\nincomplete\tdata\n@1\tok\t/tmp\tgemini\n"
        )

        with patch(
            "ccgram.tmux_manager.asyncio.create_subprocess_exec", return_value=proc
        ):
            result = await manager._scan_session_windows("ext")

        assert len(result) == 2
        assert result[0].window_id == "ext:@0"
        assert result[1].window_id == "ext:@1"

    @pytest.mark.asyncio
    async def test_emdash_fallback_name(self, manager):
        proc = _make_proc("@0\t\t/home\tclaude\n")

        with patch(
            "ccgram.tmux_manager.asyncio.create_subprocess_exec", return_value=proc
        ):
            result = await manager._scan_session_windows("emdash-claude-main-abc")

        assert len(result) == 1
        assert result[0].window_name == "claude-main-abc"
