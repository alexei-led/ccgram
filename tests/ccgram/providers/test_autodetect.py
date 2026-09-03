from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.multiplexer.base import ForegroundInfo
from ccgram.providers import (
    _reset_provider,
    detect_provider_from_command,
    detect_provider_from_pane,
    detect_provider_from_runtime,
    should_probe_pane_title_for_provider_detection,
)
from ccgram.providers.process_detection import _pgid_cache
from ccgram.session_monitor import SessionMonitor


class TestDetectProviderFromPane:
    """``detect_provider_from_pane`` resolves the foreground process through
    the multiplexer seam (``Multiplexer.foreground``) — never a tty/ps path."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _pgid_cache.clear()
        yield
        _pgid_cache.clear()

    async def test_fast_path_skips_foreground(self) -> None:
        mock_mux = MagicMock()
        mock_mux.foreground = AsyncMock()
        with patch("ccgram.multiplexer.multiplexer", mock_mux):
            result = await detect_provider_from_pane("claude", window_id="@0")
        assert result == "claude"
        mock_mux.foreground.assert_not_called()

    async def test_js_runtime_routes_through_foreground(self) -> None:
        mock_mux = MagicMock()
        mock_mux.foreground = AsyncMock(
            return_value=ForegroundInfo(
                pid=8668,
                pgid=8668,
                argv=["bun", "/Users/x/.bun/bin/codex", "--full-auto"],
                cwd="/tmp",
            )
        )
        with patch("ccgram.multiplexer.multiplexer", mock_mux):
            result = await detect_provider_from_pane("bun", window_id="@0")
        assert result == "codex"
        mock_mux.foreground.assert_awaited_once_with("@0")

    async def test_non_runtime_command_skips_foreground(self) -> None:
        mock_mux = MagicMock()
        mock_mux.foreground = AsyncMock()
        with patch("ccgram.multiplexer.multiplexer", mock_mux):
            result = await detect_provider_from_pane("vim", window_id="@0")
        assert result == ""
        mock_mux.foreground.assert_not_called()

    async def test_missing_window_id_returns_empty(self) -> None:
        mock_mux = MagicMock()
        mock_mux.foreground = AsyncMock()
        with patch("ccgram.multiplexer.multiplexer", mock_mux):
            result = await detect_provider_from_pane("bun", window_id="")
        assert result == ""
        mock_mux.foreground.assert_not_called()

    async def test_no_foreground_returns_empty(self) -> None:
        mock_mux = MagicMock()
        mock_mux.foreground = AsyncMock(return_value=None)
        with patch("ccgram.multiplexer.multiplexer", mock_mux):
            result = await detect_provider_from_pane("node", window_id="@0")
        assert result == ""
        mock_mux.foreground.assert_awaited_once_with("@0")

    async def test_empty_command_classifies_shell_via_foreground(self) -> None:
        # herdr reports no pane_current_command for a bare shell pane; the
        # foreground argv must still classify it as a shell so binding/adopting
        # the pane sets the shell provider.
        mock_mux = MagicMock()
        mock_mux.foreground = AsyncMock(
            return_value=ForegroundInfo(pid=4242, pgid=4242, argv=["-fish"], cwd="/tmp")
        )
        with patch("ccgram.multiplexer.multiplexer", mock_mux):
            result = await detect_provider_from_pane("", window_id="@7")
        assert result == "shell"
        mock_mux.foreground.assert_awaited_once_with("@7")

    async def test_empty_command_without_window_skips_foreground(self) -> None:
        mock_mux = MagicMock()
        mock_mux.foreground = AsyncMock()
        with patch("ccgram.multiplexer.multiplexer", mock_mux):
            result = await detect_provider_from_pane("", window_id="")
        assert result == ""
        mock_mux.foreground.assert_not_called()


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
            pytest.param("pi", "pi", id="bare-pi"),
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
            detect_provider_from_runtime("bun", pane_title="◇ Ready (ccgram)")
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


class _NewWindowHarness:
    """Drives ``topic_orchestration.handle_new_window`` with every collaborator
    stubbed, so the tests read as pane-state → provider-set decisions."""

    def __init__(self, detect, session_manager, mux) -> None:
        self.detect = detect
        self.session_manager = session_manager
        self.mux = mux

    async def run(
        self,
        window_id: str,
        *,
        pane_command: str | None,
        pane_title: str = "",
    ) -> None:
        # Lazy: importing at module scope would pull the handler package in
        # before the patches above are installed.
        from ccgram.handlers.topics.topic_orchestration import handle_new_window
        from ccgram.session_monitor import NewWindowEvent

        window = None
        if pane_command is not None:
            window = MagicMock()
            window.pane_current_command = pane_command
        self.mux.find_window_by_id = AsyncMock(return_value=window)
        self.mux.get_pane_title = AsyncMock(return_value=pane_title)

        event = NewWindowEvent(
            window_id=window_id,
            session_id=f"uuid-{window_id}",
            window_name="proj",
            cwd="/tmp/proj",
        )
        await handle_new_window(event, AsyncMock())


@pytest.fixture
def new_window():
    module = "ccgram.handlers.topics.topic_orchestration"
    with (
        patch(f"{module}.tmux_manager") as mux,
        patch(f"{module}.session_manager") as session_manager,
        patch(f"{module}.config") as config,
        patch(
            f"{module}.detect_provider_from_pane",
            new_callable=AsyncMock,
            return_value="",
        ) as detect,
    ):
        config.group_id = None
        session_manager.iter_thread_bindings.return_value = []
        session_manager.view_window.return_value = MagicMock(provider_name="")
        yield _NewWindowHarness(detect, session_manager, mux)


class TestHandleNewWindowAutoDetection:
    async def test_sets_detected_provider(self, new_window) -> None:
        new_window.detect.return_value = "codex"

        await new_window.run("@5", pane_command="codex")

        new_window.detect.assert_awaited_once()
        new_window.session_manager.set_window_provider.assert_called_once_with(
            "@5", "codex"
        )

    @pytest.mark.parametrize(
        "pane_command",
        [pytest.param("", id="empty_command"), pytest.param(None, id="window_gone")],
    )
    async def test_skips_detection_without_a_pane_command(
        self, new_window, pane_command: str | None
    ) -> None:
        await new_window.run("@6", pane_command=pane_command)

        new_window.detect.assert_not_called()
        new_window.session_manager.set_window_provider.assert_not_called()

    async def test_detects_gemini_from_pane_title_when_command_is_a_js_runtime(
        self, new_window
    ) -> None:
        await new_window.run("@8", pane_command="bun", pane_title="◇  Ready (ccgram)")

        new_window.detect.assert_awaited_once()
        new_window.mux.get_pane_title.assert_awaited_once_with("@8")
        new_window.session_manager.set_window_provider.assert_called_once_with(
            "@8", "gemini"
        )

    async def test_generic_pane_title_does_not_imply_gemini(self, new_window) -> None:
        await new_window.run(
            "@10", pane_command="bun", pane_title="Working on build..."
        )

        new_window.detect.assert_awaited_once()
        new_window.mux.get_pane_title.assert_awaited_once_with("@10")
        new_window.session_manager.set_window_provider.assert_not_called()

    async def test_unrecognized_command_leaves_provider_unset(self, new_window) -> None:
        await new_window.run("@9", pane_command="bash")

        new_window.detect.assert_awaited_once()
        new_window.session_manager.set_window_provider.assert_not_called()


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
            await monitor._detect_and_cleanup_changes(adoptable_window_ids=None)
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
            await monitor._detect_and_cleanup_changes(adoptable_window_ids=None)
            mock_sm.set_window_provider.assert_not_called()
