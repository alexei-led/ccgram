"""Tests for provider auto-detection from tmux pane commands."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.providers import (
    _reset_provider,
    detect_provider_from_command,
    detect_provider_from_runtime,
    should_probe_pane_title_for_provider_detection,
)
from ccgram.session_monitor import SessionMonitor


class TestDetectProviderFromCommand:
    @pytest.fixture(autouse=True)
    def _reset(self):
        _reset_provider()
        yield
        _reset_provider()

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            pytest.param("claude", "claude", id="bare-claude"),
            pytest.param("codex", "codex", id="bare-codex"),
            pytest.param("gemini", "gemini", id="bare-gemini"),
            pytest.param("/usr/local/bin/claude", "claude", id="full-path-claude"),
            pytest.param("/opt/bin/codex --resume", "codex", id="codex-with-args"),
            pytest.param("gemini-cli", "gemini", id="gemini-cli-variant"),
            pytest.param("Claude", "claude", id="case-insensitive-claude"),
            pytest.param("CODEX", "codex", id="uppercase-codex"),
            pytest.param("  claude  ", "claude", id="whitespace-padded"),
        ],
    )
    def test_known_commands(self, command: str, expected: str) -> None:
        assert detect_provider_from_command(command) == expected

    def test_unknown_command_returns_empty(self) -> None:
        assert detect_provider_from_command("vim") == ""

    def test_shell_command_detected(self) -> None:
        assert detect_provider_from_command("bash") == "shell"
        assert detect_provider_from_command("zsh") == "shell"
        assert detect_provider_from_command("fish") == "shell"
        assert detect_provider_from_command("-bash") == "shell"

    def test_empty_command_returns_empty(self) -> None:
        assert detect_provider_from_command("") == ""

    def test_priority_order_first_match(self) -> None:
        # "claude" is checked first, so "claude-codex" matches "claude"
        assert detect_provider_from_command("claude-codex") == "claude"


class TestDetectProviderFromRuntime:
    @pytest.fixture(autouse=True)
    def _reset(self):
        _reset_provider()
        yield
        _reset_provider()

    def test_probe_hint_for_gemini_wrappers(self) -> None:
        assert should_probe_pane_title_for_provider_detection("bun") is True
        assert should_probe_pane_title_for_provider_detection("node") is True
        assert should_probe_pane_title_for_provider_detection("bash") is False

    def test_detects_gemini_from_wrapper_and_title_marker(self) -> None:
        assert (
            detect_provider_from_runtime("bun", pane_title="◇ Ready (ccbot)")
            == "gemini"
        )

    def test_does_not_detect_gemini_from_generic_title_text(self) -> None:
        assert (
            detect_provider_from_runtime("bun", pane_title="Working on build...") == ""
        )

    def test_prefers_command_detection_when_available(self) -> None:
        assert detect_provider_from_runtime("codex", pane_title="◇ Ready") == "codex"

    def test_detects_provider_from_ccgram_title_stamp(self) -> None:
        assert detect_provider_from_runtime("bun", pane_title="ccgram:codex") == "codex"
        assert (
            detect_provider_from_runtime("node", pane_title="ccgram:claude") == "claude"
        )
        assert (
            detect_provider_from_runtime("bun", pane_title="ccgram:gemini") == "gemini"
        )
        assert detect_provider_from_runtime("bun", pane_title="ccgram:shell") == "shell"

    def test_ignores_invalid_ccgram_stamp(self) -> None:
        assert detect_provider_from_runtime("bun", pane_title="ccgram:unknown") == ""


class TestHandleNewWindowAutoDetection:
    @patch("ccgram.handlers.topic_orchestration.tmux_manager")
    @patch("ccgram.handlers.topic_orchestration.session_manager")
    @patch("ccgram.handlers.topic_orchestration.config")
    @patch(
        "ccgram.handlers.topic_orchestration.detect_provider_from_pane",
        new_callable=AsyncMock,
        return_value="codex",
    )
    async def test_sets_detected_provider(
        self,
        mock_detect: MagicMock,
        mock_config: MagicMock,
        mock_sm: MagicMock,
        mock_tmux: MagicMock,
    ) -> None:
        from ccgram.handlers.topic_orchestration import (
            handle_new_window as _handle_new_window,
        )
        from ccgram.session_monitor import NewWindowEvent

        mock_config.group_id = None
        mock_sm.iter_thread_bindings.return_value = []
        mock_sm.get_window_state.return_value = MagicMock(provider_name="")

        mock_window = MagicMock()
        mock_window.pane_current_command = "codex"
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)

        event = NewWindowEvent(
            window_id="@5", session_id="uuid-1", window_name="proj", cwd="/tmp/proj"
        )
        bot = AsyncMock()

        await _handle_new_window(event, bot)

        mock_detect.assert_awaited_once()
        mock_sm.set_window_provider.assert_called_once_with("@5", "codex")

    @patch("ccgram.handlers.topic_orchestration.tmux_manager")
    @patch("ccgram.handlers.topic_orchestration.session_manager")
    @patch("ccgram.handlers.topic_orchestration.config")
    @patch(
        "ccgram.handlers.topic_orchestration.detect_provider_from_pane",
        new_callable=AsyncMock,
    )
    async def test_skips_detection_when_no_pane_command(
        self,
        mock_detect: MagicMock,
        mock_config: MagicMock,
        mock_sm: MagicMock,
        mock_tmux: MagicMock,
    ) -> None:
        from ccgram.handlers.topic_orchestration import (
            handle_new_window as _handle_new_window,
        )
        from ccgram.session_monitor import NewWindowEvent

        mock_config.group_id = None
        mock_sm.iter_thread_bindings.return_value = []

        mock_window = MagicMock()
        mock_window.pane_current_command = ""
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)

        event = NewWindowEvent(
            window_id="@6", session_id="uuid-2", window_name="proj", cwd="/tmp"
        )
        bot = AsyncMock()

        await _handle_new_window(event, bot)

        mock_detect.assert_not_called()
        mock_sm.set_window_provider.assert_not_called()

    @patch("ccgram.handlers.topic_orchestration.tmux_manager")
    @patch("ccgram.handlers.topic_orchestration.session_manager")
    @patch("ccgram.handlers.topic_orchestration.config")
    @patch(
        "ccgram.handlers.topic_orchestration.detect_provider_from_pane",
        new_callable=AsyncMock,
    )
    async def test_skips_detection_when_window_not_found(
        self,
        mock_detect: MagicMock,
        mock_config: MagicMock,
        mock_sm: MagicMock,
        mock_tmux: MagicMock,
    ) -> None:
        from ccgram.handlers.topic_orchestration import (
            handle_new_window as _handle_new_window,
        )
        from ccgram.session_monitor import NewWindowEvent

        mock_config.group_id = None
        mock_sm.iter_thread_bindings.return_value = []

        mock_tmux.find_window_by_id = AsyncMock(return_value=None)

        event = NewWindowEvent(
            window_id="@7", session_id="uuid-3", window_name="proj", cwd="/tmp"
        )
        bot = AsyncMock()

        await _handle_new_window(event, bot)

        mock_detect.assert_not_called()
        mock_sm.set_window_provider.assert_not_called()

    @patch("ccgram.handlers.topic_orchestration.tmux_manager")
    @patch("ccgram.handlers.topic_orchestration.session_manager")
    @patch("ccgram.handlers.topic_orchestration.config")
    @patch(
        "ccgram.handlers.topic_orchestration.detect_provider_from_pane",
        new_callable=AsyncMock,
        return_value="",
    )
    async def test_detects_gemini_from_pane_title_when_command_is_bun(
        self,
        mock_detect: MagicMock,
        mock_config: MagicMock,
        mock_sm: MagicMock,
        mock_tmux: MagicMock,
    ) -> None:
        from ccgram.handlers.topic_orchestration import (
            handle_new_window as _handle_new_window,
        )
        from ccgram.session_monitor import NewWindowEvent

        mock_config.group_id = None
        mock_sm.iter_thread_bindings.return_value = []
        mock_sm.get_window_state.return_value = MagicMock(provider_name="")

        mock_window = MagicMock()
        mock_window.pane_current_command = "bun"
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
        mock_tmux.get_pane_title = AsyncMock(return_value="◇  Ready (ccbot)")

        event = NewWindowEvent(
            window_id="@8", session_id="uuid-4", window_name="proj", cwd="/tmp"
        )
        bot = AsyncMock()

        await _handle_new_window(event, bot)

        mock_detect.assert_awaited_once()
        mock_tmux.get_pane_title.assert_awaited_once_with("@8")
        mock_sm.set_window_provider.assert_called_once_with("@8", "gemini")

    @patch("ccgram.handlers.topic_orchestration.tmux_manager")
    @patch("ccgram.handlers.topic_orchestration.session_manager")
    @patch("ccgram.handlers.topic_orchestration.config")
    @patch(
        "ccgram.handlers.topic_orchestration.detect_provider_from_pane",
        new_callable=AsyncMock,
        return_value="",
    )
    async def test_does_not_detect_gemini_from_generic_working_text(
        self,
        mock_detect: MagicMock,
        mock_config: MagicMock,
        mock_sm: MagicMock,
        mock_tmux: MagicMock,
    ) -> None:
        from ccgram.handlers.topic_orchestration import (
            handle_new_window as _handle_new_window,
        )
        from ccgram.session_monitor import NewWindowEvent

        mock_config.group_id = None
        mock_sm.iter_thread_bindings.return_value = []
        mock_sm.get_window_state.return_value = MagicMock(provider_name="")

        mock_window = MagicMock()
        mock_window.pane_current_command = "bun"
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
        mock_tmux.get_pane_title = AsyncMock(return_value="Working on build...")

        event = NewWindowEvent(
            window_id="@10", session_id="uuid-6", window_name="proj", cwd="/tmp"
        )
        bot = AsyncMock()

        await _handle_new_window(event, bot)

        mock_detect.assert_awaited_once()
        mock_tmux.get_pane_title.assert_awaited_once_with("@10")
        mock_sm.set_window_provider.assert_not_called()

    @patch("ccgram.handlers.topic_orchestration.tmux_manager")
    @patch("ccgram.handlers.topic_orchestration.session_manager")
    @patch("ccgram.handlers.topic_orchestration.config")
    @patch(
        "ccgram.handlers.topic_orchestration.detect_provider_from_pane",
        new_callable=AsyncMock,
        return_value="",
    )
    async def test_skips_provider_set_for_unrecognized_command(
        self,
        mock_detect: MagicMock,
        mock_config: MagicMock,
        mock_sm: MagicMock,
        mock_tmux: MagicMock,
    ) -> None:
        from ccgram.handlers.topic_orchestration import (
            handle_new_window as _handle_new_window,
        )
        from ccgram.session_monitor import NewWindowEvent

        mock_config.group_id = None
        mock_sm.iter_thread_bindings.return_value = []
        mock_sm.get_window_state.return_value = MagicMock(provider_name="")

        mock_window = MagicMock()
        mock_window.pane_current_command = "bash"
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)

        event = NewWindowEvent(
            window_id="@9", session_id="uuid-5", window_name="proj", cwd="/tmp"
        )
        bot = AsyncMock()

        await _handle_new_window(event, bot)

        mock_detect.assert_awaited_once()
        mock_sm.set_window_provider.assert_not_called()


class TestSessionMonitorProviderFromMap:
    async def test_sets_provider_from_session_map(self, tmp_path) -> None:
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            poll_interval=0.1,
            state_file=tmp_path / "monitor_state.json",
        )
        monitor._last_session_map = {}

        new_map = {
            "@5": {
                "session_id": "uuid-1",
                "cwd": "/tmp",
                "window_name": "proj",
                "provider_name": "codex",
            }
        }

        with (
            patch.object(
                monitor,
                "_load_current_session_map",
                new_callable=AsyncMock,
                return_value=new_map,
            ),
            patch("ccgram.session.session_manager") as mock_sm,
        ):
            await monitor._detect_and_cleanup_changes()
            mock_sm.set_window_provider.assert_called_once_with("@5", "codex")

    async def test_skips_provider_when_not_in_map(self, tmp_path) -> None:
        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            poll_interval=0.1,
            state_file=tmp_path / "monitor_state.json",
        )
        monitor._last_session_map = {}

        new_map = {
            "@6": {
                "session_id": "uuid-2",
                "cwd": "/tmp",
                "window_name": "proj",
            }
        }

        with (
            patch.object(
                monitor,
                "_load_current_session_map",
                new_callable=AsyncMock,
                return_value=new_map,
            ),
            patch("ccgram.session.session_manager") as mock_sm,
        ):
            await monitor._detect_and_cleanup_changes()
            mock_sm.set_window_provider.assert_not_called()
