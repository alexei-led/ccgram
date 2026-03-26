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

    async def test_whitespace_only_command_falls_back(self, mock_tmux) -> None:
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command="   ",
            )
        )
        with patch("ccgram.providers.shell.os.environ.get", return_value="/bin/zsh"):
            assert await detect_pane_shell("@0") == "zsh"


class TestSetupShellPrompt:
    @pytest.fixture
    def mock_tmux(self):
        with patch("ccgram.tmux_manager.tmux_manager") as mock_tm:
            mock_tm.capture_pane = AsyncMock(return_value=None)
            yield mock_tm

    @pytest.mark.parametrize(
        ("shell", "expected_substring"),
        [
            ("fish", "fish_prompt"),
            ("bash", "PS1="),
            ("zsh", "PROMPT="),
            ("tcsh", "set prompt"),
            ("ksh", "PS1="),
        ],
        ids=["fish", "bash", "zsh", "tcsh", "ksh-fallback"],
    )
    async def test_sends_correct_prompt_command(
        self, mock_tmux, shell: str, expected_substring: str
    ) -> None:
        from ccgram.providers.shell import setup_shell_prompt

        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command=shell,
            )
        )
        mock_tmux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        calls = mock_tmux.send_keys.call_args_list
        # calls[0] is the C-c to clear partial input, calls[1] is the prompt command
        prompt_call = calls[1]
        assert expected_substring in prompt_call[0][1]

    async def test_sends_clear_after_prompt(self, mock_tmux) -> None:
        from ccgram.providers.shell import setup_shell_prompt

        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command="bash",
            )
        )
        mock_tmux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        calls = mock_tmux.send_keys.call_args_list
        # calls: [0]=C-c, [1]=prompt command, [2]=clear
        assert len(calls) == 3
        assert calls[2][0][1] == "clear"

    async def test_send_keys_uses_raw_true(self, mock_tmux) -> None:
        from ccgram.providers.shell import setup_shell_prompt

        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command="bash",
            )
        )
        mock_tmux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        calls = mock_tmux.send_keys.call_args_list
        # calls[1] = prompt command, calls[2] = clear — both should use raw=True
        assert calls[1][1].get("raw") is True
        assert calls[2][1].get("raw") is True

    @pytest.mark.usefixtures("_wrap_mode")
    async def test_skips_setup_when_marker_present(self, mock_tmux) -> None:
        from ccgram.providers.shell import setup_shell_prompt

        mock_tmux.capture_pane = AsyncMock(return_value="~/code main ❯ ⌘0⌘ ")
        mock_tmux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        mock_tmux.send_keys.assert_not_called()


class TestGetShellName:
    def test_returns_basename_of_shell_env(self) -> None:
        from ccgram.providers.shell import get_shell_name

        with patch("ccgram.providers.shell.os.environ.get", return_value="/bin/zsh"):
            assert get_shell_name() == "zsh"

    def test_returns_empty_when_shell_unset(self) -> None:
        from ccgram.providers.shell import get_shell_name

        with patch.dict("os.environ", {}, clear=True):
            assert get_shell_name() == ""

    def test_returns_basename_from_full_path(self) -> None:
        from ccgram.providers.shell import get_shell_name

        with patch(
            "ccgram.providers.shell.os.environ.get",
            return_value="/opt/homebrew/bin/fish",
        ):
            assert get_shell_name() == "fish"


# ── Wrap-mode tests ──────────────────────────────────────────────────────


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeRegex:
    def test_match_prompt_finds_wrap_marker(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code main ❯ ⌘0⌘ ls -la")
        assert m is not None
        assert m.group(1) == "0"
        assert m.group(2) == "ls -la"

    def test_match_prompt_bare_prompt_idle(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code main ❯ ⌘0⌘ ")
        assert m is not None
        assert m.group(1) == "0"
        assert m.group(2).strip() == ""

    def test_match_prompt_nonzero_exit(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code main ❯ ⌘127⌘ bad-cmd")
        assert m is not None
        assert m.group(1) == "127"
        assert m.group(2) == "bad-cmd"

    def test_match_prompt_no_marker(self) -> None:
        from ccgram.providers.shell import match_prompt

        assert match_prompt("$ ls -la") is None

    def test_match_prompt_marker_only_line(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("⌘0⌘")
        assert m is not None
        assert m.group(1) == "0"


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeSetup:
    @pytest.fixture
    def mock_tmux(self):
        with patch("ccgram.tmux_manager.tmux_manager") as mock_tm:
            mock_tm.capture_pane = AsyncMock(return_value=None)
            yield mock_tm

    @pytest.mark.parametrize(
        ("shell", "expected_substring"),
        [
            ("fish", "__ccgram_orig_prompt"),
            ("bash", "PROMPT_COMMAND"),
            ("zsh", "PROMPT="),
            ("tcsh", "set prompt"),
        ],
        ids=["fish-wrap", "bash-wrap", "zsh-wrap", "tcsh-wrap"],
    )
    async def test_wrap_sends_correct_prompt_command(
        self, mock_tmux, shell: str, expected_substring: str
    ) -> None:
        from ccgram.providers.shell import setup_shell_prompt

        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command=shell,
            )
        )
        mock_tmux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        calls = mock_tmux.send_keys.call_args_list
        # calls[0] is C-c, calls[1] is the prompt command
        prompt_call = calls[1]
        assert expected_substring in prompt_call[0][1]

    async def test_wrap_fish_preserves_original_prompt(self, mock_tmux) -> None:
        from ccgram.providers.shell import setup_shell_prompt

        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command="fish",
            )
        )
        mock_tmux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        cmd = mock_tmux.send_keys.call_args_list[1][0][1]
        assert "functions --copy fish_prompt __ccgram_orig_prompt" in cmd
        assert "functions -q __ccgram_orig_prompt" in cmd
        assert "__ccgram_orig_prompt" in cmd
        assert "⌘%d⌘" in cmd
        assert "set_color brblack" in cmd

    async def test_wrap_has_prompt_marker_detects_wrap_marker(self, mock_tmux) -> None:
        from ccgram.providers.shell import has_prompt_marker

        mock_tmux.capture_pane = AsyncMock(return_value="~/code main ❯ ⌘0⌘ ")
        assert await has_prompt_marker("@0") is True

    async def test_wrap_has_prompt_marker_rejects_no_marker(self, mock_tmux) -> None:
        from ccgram.providers.shell import has_prompt_marker

        mock_tmux.capture_pane = AsyncMock(return_value="~/code main ❯ ")
        assert await has_prompt_marker("@0") is False


# ── Prompt mode resolution tests ─────────────────────────────────────────


class TestGetPromptMode:
    def test_defaults_to_wrap(self) -> None:
        from ccgram.config import config
        from ccgram.providers.shell import _get_prompt_mode

        config.prompt_mode = "wrap"
        assert _get_prompt_mode() == "wrap"

    def test_returns_replace(self) -> None:
        from ccgram.config import config
        from ccgram.providers.shell import _get_prompt_mode

        config.prompt_mode = "replace"
        assert _get_prompt_mode() == "replace"

    def test_invalid_mode_defaults_to_wrap(self) -> None:
        import ccgram.providers.shell as shell_mod
        from ccgram.config import config
        from ccgram.providers.shell import _get_prompt_mode

        original_warned = shell_mod._WARNED_INVALID_MODE
        shell_mod._WARNED_INVALID_MODE = False
        try:
            config.prompt_mode = "bogus"
            assert _get_prompt_mode() == "wrap"
        finally:
            shell_mod._WARNED_INVALID_MODE = original_warned

    def test_empty_string_defaults_to_wrap(self) -> None:
        from ccgram.config import config
        from ccgram.providers.shell import _get_prompt_mode

        config.prompt_mode = ""
        assert _get_prompt_mode() == "wrap"


class TestMatchPromptModeSwitching:
    """Verify match_prompt uses re.match in replace mode and re.search in wrap."""

    def test_replace_mode_anchors_at_start(self) -> None:
        from ccgram.config import config
        from ccgram.providers.shell import match_prompt

        config.prompt_mode = "replace"
        assert match_prompt("ccgram:0❯ ls") is not None
        assert match_prompt("some prefix ccgram:0❯ ls") is None

    @pytest.mark.usefixtures("_wrap_mode")
    def test_wrap_mode_searches_anywhere(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code main ❯ ⌘0⌘ ls")
        assert m is not None
        assert m.group(1) == "0"
        assert m.group(2) == "ls"

    @pytest.mark.usefixtures("_wrap_mode")
    def test_wrap_mode_matches_marker_at_start(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("⌘0⌘ ls")
        assert m is not None
        assert m.group(2) == "ls"


# ── Setup command content tests ──────────────────────────────────────────


class TestWrapSetupCommands:
    @pytest.mark.parametrize(
        ("shell", "expected"),
        [
            ("fish", "__ccgram_orig_prompt"),
            ("fish", "set_color brblack"),
            ("fish", "or function __ccgram_orig_prompt"),
            ("fish", "functions -q __ccgram_orig_prompt"),
            ("bash", "PROMPT_COMMAND"),
            ("bash", "⌘\\${__ccgram_x}⌘"),
            ("bash", "type __ccgram_sc"),
            ("zsh", "⌘%?⌘"),
            ("zsh", "⌘%\\?⌘"),
            ("sh", "⌘0⌘"),
            ("sh", "⌘*⌘"),
            ("dash", "⌘0⌘"),
            ("ksh", "⌘0⌘"),
            ("tcsh", "⌘$status⌘"),
            ("csh", "⌘$status⌘"),
        ],
        ids=[
            "fish-wraps-original",
            "fish-uses-set_color",
            "fish-has-fallback",
            "fish-guard",
            "bash-prompt-command",
            "bash-marker-format",
            "bash-guard",
            "zsh-marker-format",
            "zsh-guard",
            "sh-marker",
            "sh-guard",
            "dash-marker",
            "ksh-marker",
            "tcsh-marker-format",
            "csh-marker-format",
        ],
    )
    def test_wrap_command_contains(self, shell: str, expected: str) -> None:
        from ccgram.providers.shell import _wrap_setup_commands

        assert expected in _wrap_setup_commands(shell)

    def test_unknown_shell_falls_back_to_posix(self) -> None:
        from ccgram.providers.shell import _wrap_setup_commands

        cmd = _wrap_setup_commands("unknown_shell")
        assert "⌘0⌘" in cmd


class TestReplaceSetupCommands:
    @pytest.mark.parametrize(
        ("shell", "expected"),
        [
            ("fish", 'printf "ccgram:$status❯ "'),
            ("bash", "PS1='ccgram:$?❯ '"),
            ("zsh", "PROMPT='ccgram:%?❯ '"),
            ("tcsh", 'set prompt = "ccgram:$status❯ "'),
        ],
        ids=["fish", "bash", "zsh", "tcsh"],
    )
    def test_replace_command_contains(self, shell: str, expected: str) -> None:
        from ccgram.providers.shell import _replace_setup_commands

        assert expected in _replace_setup_commands(shell, "ccgram")

    def test_custom_prefix(self) -> None:
        from ccgram.providers.shell import _replace_setup_commands

        cmd = _replace_setup_commands("bash", "mybot")
        assert "mybot:$?❯" in cmd

    def test_unknown_shell_falls_back_to_bash(self) -> None:
        from ccgram.providers.shell import _replace_setup_commands

        cmd = _replace_setup_commands("unknown_shell", "ccgram")
        assert "PS1=" in cmd


class TestSetupShellPromptClearsBefore:
    """Verify setup_shell_prompt sends C-c to cancel partial input."""

    @pytest.fixture
    def mock_tmux(self):
        with patch("ccgram.tmux_manager.tmux_manager") as mock_tm:
            mock_tm.capture_pane = AsyncMock(return_value=None)
            yield mock_tm

    async def test_sends_ctrl_c_before_prompt_command(self, mock_tmux) -> None:
        from ccgram.providers.shell import setup_shell_prompt

        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command="bash",
            )
        )
        mock_tmux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        first_call = mock_tmux.send_keys.call_args_list[0]
        assert first_call[0][1] == "C-c"
        assert first_call[1].get("enter") is False
        assert first_call[1].get("literal") is False

    async def test_no_clear_when_clear_false(self, mock_tmux) -> None:
        from ccgram.providers.shell import setup_shell_prompt

        mock_tmux.find_window_by_id = AsyncMock(
            return_value=TmuxWindow(
                window_id="@0",
                window_name="test",
                cwd="/tmp",
                pane_current_command="bash",
            )
        )
        mock_tmux.send_keys = AsyncMock()

        await setup_shell_prompt("@0", clear=False)

        calls = mock_tmux.send_keys.call_args_list
        # calls: [0]=C-c, [1]=prompt command (no clear)
        assert len(calls) == 2
        assert all(c[0][1] != "clear" for c in calls)
