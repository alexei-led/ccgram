"""Tests for ShellProvider — shell-specific behavior beyond contract tests.

Contract tests (test_provider_contracts.py) skip transcript-related branches
for ShellProvider via ``pytest.skip("No transcript support")``. These tests
verify the override behavior directly and assert exact capability values.
"""

from unittest.mock import AsyncMock, patch

import pytest

from ccgram.providers.shell import ShellProvider, detect_pane_shell
from ccgram.tmux_manager import TmuxWindow


class TestShellCapabilities:
    """Verify exact capability values — contract tests only check name is truthy."""

    @pytest.fixture
    def caps(self):
        return ShellProvider().capabilities

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("name", "shell"),
            ("launch_command", ""),
            ("supports_hook", False),
            ("supports_hook_events", False),
            ("supports_resume", False),
            ("supports_continue", False),
            ("supports_structured_transcript", False),
            ("supports_incremental_read", False),
            ("transcript_format", "plain"),
            ("builtin_commands", ()),
            ("uses_pane_title", False),
            ("supports_user_command_discovery", False),
        ],
    )
    def test_capability_value(self, caps, field: str, expected: object) -> None:
        assert getattr(caps, field) == expected


class TestShellOverrides:
    """ShellProvider overrides that differ from JsonlProvider base class.

    The base class (JsonlProvider) has different behavior for each of these
    methods. ShellProvider overrides them to return no-op/empty values because
    shell sessions have no structured transcripts, no hooks, and no commands.
    """

    @pytest.fixture
    def provider(self) -> ShellProvider:
        return ShellProvider()

    @pytest.mark.parametrize(
        ("resume_id", "use_continue"),
        [
            (None, False),
            ("abc123", False),
            (None, True),
            ("abc123", True),
        ],
        ids=["fresh", "resume", "continue", "resume+continue"],
    )
    def test_make_launch_args_always_empty(
        self, provider: ShellProvider, resume_id: str | None, use_continue: bool
    ) -> None:
        assert (
            provider.make_launch_args(resume_id=resume_id, use_continue=use_continue)
            == ""
        )

    def test_parse_transcript_line_returns_none_for_valid_json(
        self, provider: ShellProvider
    ) -> None:
        assert (
            provider.parse_transcript_line(
                '{"type": "assistant", "message": {"content": "hi"}}'
            )
            is None
        )

    def test_read_transcript_file_returns_empty(self, provider: ShellProvider) -> None:
        entries, offset = provider.read_transcript_file("/any/path.jsonl", 0)
        assert entries == []
        assert offset == 0

    def test_extract_bash_output_returns_none_even_with_match(
        self, provider: ShellProvider
    ) -> None:
        pane = "some text\n! ls -la\ntotal 42\n"
        assert provider.extract_bash_output(pane, "ls") is None

    def test_discover_commands_returns_empty(self, provider: ShellProvider) -> None:
        assert provider.discover_commands("/any/dir") == []

    def test_parse_hook_payload_returns_none(self, provider: ShellProvider) -> None:
        payload = {
            "session_id": "test-sid",
            "cwd": "/tmp",
            "transcript_path": "/tmp/t.jsonl",
            "window_key": "ccgram:@0",
        }
        assert provider.parse_hook_payload(payload) is None

    def test_parse_terminal_status_returns_none_for_spinner(
        self, provider: ShellProvider
    ) -> None:
        sep = "─" * 30
        pane = f"output\n✻ Reading files\n{sep}\n❯ \n{sep}\n"
        assert provider.parse_terminal_status(pane) is None


class TestDetectPaneShell:
    @pytest.fixture
    def mock_tmux(self):
        with patch("ccgram.tmux_manager.tmux_manager") as mock_tm:
            yield mock_tm

    @pytest.mark.parametrize(
        ("pane_cmd", "expected"),
        [
            ("bash", "bash"),
            ("zsh", "zsh"),
            ("fish", "fish"),
            ("-bash", "bash"),
            ("-zsh", "zsh"),
            ("dash", "dash"),
            ("ksh", "ksh"),
            ("/opt/homebrew/bin/fish", "fish"),
            ("/bin/bash", "bash"),
        ],
        ids=[
            "bash",
            "zsh",
            "fish",
            "login-bash",
            "login-zsh",
            "dash",
            "ksh",
            "full-path-fish",
            "full-path-bash",
        ],
    )
    async def test_detects_shell_from_pane_command(
        self, mock_tmux, pane_cmd: str, expected: str
    ) -> None:
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command=pane_cmd,
            )
        )
        assert await detect_pane_shell("@0") == expected

    async def test_falls_back_to_env_when_pane_not_found(self, mock_tmux) -> None:
        mock_tmux.find_window_by_id = AsyncMock(return_value=None)
        with patch("ccgram.providers.shell.os.environ.get", return_value="/bin/zsh"):
            assert await detect_pane_shell("@0") == "zsh"

    async def test_falls_back_to_env_when_command_not_a_shell(self, mock_tmux) -> None:
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command="python",
            )
        )
        with patch("ccgram.providers.shell.os.environ.get", return_value="/bin/fish"):
            assert await detect_pane_shell("@0") == "fish"

    async def test_falls_back_to_env_when_command_empty(self, mock_tmux) -> None:
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command="",
            )
        )
        with patch("ccgram.providers.shell.os.environ.get", return_value="/bin/bash"):
            assert await detect_pane_shell("@0") == "bash"
