from unittest.mock import AsyncMock, patch

import pytest

from ccgram.providers.shell import (
    PromptMatch,
    ShellProvider,
    detect_pane_shell,
    has_prompt_marker,
    setup_shell_prompt,
)
from ccgram.multiplexer.base import ForegroundInfo, WindowRef as TmuxWindow


def _window(pane_cmd: str = "") -> TmuxWindow:
    return TmuxWindow(
        window_id="@0",
        window_name="test",
        cwd="/tmp",
        pane_current_command=pane_cmd,
    )


def _env_shell(path: str):
    return patch("ccgram.providers.shell_infra.os.environ.get", return_value=path)


@pytest.fixture
def prompt_setup_mux():
    """multiplexer proxy for setup_shell_prompt: interactive shell, bare pane."""
    with (
        patch("ccgram.multiplexer.multiplexer") as mock_tm,
        patch(
            "ccgram.providers.shell_infra._is_interactive_shell",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        mock_tm.capture_pane = AsyncMock(return_value=None)
        mock_tm.send_keys = AsyncMock()
        yield mock_tm


class TestShellCapabilities:
    @pytest.fixture
    def caps(self):
        return ShellProvider().capabilities

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("name", "shell"),
            ("launch_command", ""),
            ("supports_hook", False),
            ("supports_resume", False),
            ("supports_continue", False),
            ("supports_structured_transcript", False),
            ("supports_incremental_read", False),
            ("builtin_commands", ()),
            ("uses_pane_title", False),
            ("supports_user_command_discovery", False),
        ],
    )
    def test_capability_value(self, caps, field: str, expected: object) -> None:
        assert getattr(caps, field) == expected


class TestShellOverrides:
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

    def test_parse_terminal_status_returns_none_for_spinner(
        self, provider: ShellProvider
    ) -> None:
        sep = "─" * 30
        pane = f"output\n✻ Reading files\n{sep}\n❯ \n{sep}\n"
        assert provider.parse_terminal_status(pane) is None


class TestDetectPaneShell:
    @pytest.fixture
    def mock_tmux(self):
        with patch("ccgram.multiplexer.multiplexer") as mock_tm:
            # Default: no foreground process (backends that don't surface the
            # shell in pane_current_command fall through to the env). Tests that
            # exercise the foreground path override this.
            mock_tm.foreground = AsyncMock(return_value=None)
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
        mock_tmux.find_window_by_id = AsyncMock(return_value=_window(pane_cmd))
        assert await detect_pane_shell("@0") == expected

    @pytest.mark.parametrize(
        "window",
        [
            pytest.param(None, id="pane_not_found"),
            pytest.param(_window("python"), id="foreground_is_not_a_shell"),
            pytest.param(_window(""), id="empty_command"),
            pytest.param(_window("   "), id="whitespace_only_command"),
        ],
    )
    async def test_falls_back_to_env_shell(self, mock_tmux, window) -> None:
        mock_tmux.find_window_by_id = AsyncMock(return_value=window)
        with _env_shell("/bin/zsh"):
            assert await detect_pane_shell("@0") == "zsh"

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["-fish"], "fish"),
            (["/bin/zsh"], "zsh"),
            (["bash"], "bash"),
        ],
    )
    async def test_detects_shell_from_foreground_when_command_empty(
        self, mock_tmux, argv: list[str], expected: str
    ) -> None:
        # herdr leaves pane_current_command empty for a bare shell; the shell is
        # identified from the foreground process via the seam, not the bot $SHELL.
        mock_tmux.find_window_by_id = AsyncMock(return_value=_window(""))
        mock_tmux.foreground = AsyncMock(
            return_value=ForegroundInfo(pid=1, pgid=1, argv=argv, cwd="/tmp", tty="")
        )
        with _env_shell("/bin/dash"):
            assert await detect_pane_shell("@0") == expected

    async def test_foreground_non_shell_falls_back_to_env(self, mock_tmux) -> None:
        # A non-shell foreground (e.g. a JS agent runtime) is not a shell match;
        # detection still falls back to the env shell name.
        mock_tmux.find_window_by_id = AsyncMock(return_value=_window(""))
        mock_tmux.foreground = AsyncMock(
            return_value=ForegroundInfo(
                pid=1, pgid=1, argv=["node", "cli.js"], cwd="/tmp", tty=""
            )
        )
        with _env_shell("/bin/bash"):
            assert await detect_pane_shell("@0") == "bash"


class TestSetupShellPrompt:
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
        self, prompt_setup_mux, shell: str, expected_substring: str
    ) -> None:
        prompt_setup_mux.find_window_by_id = AsyncMock(return_value=_window(shell))
        prompt_setup_mux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        calls = prompt_setup_mux.send_keys.call_args_list
        prompt_call = calls[1]
        assert expected_substring in prompt_call[0][1]

    async def test_sends_clear_after_prompt(self, prompt_setup_mux) -> None:
        prompt_setup_mux.find_window_by_id = AsyncMock(return_value=_window("bash"))
        prompt_setup_mux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        calls = prompt_setup_mux.send_keys.call_args_list
        assert len(calls) == 3
        assert calls[2][0][1] == "clear"

    async def test_send_keys_uses_raw_true(self, prompt_setup_mux) -> None:
        prompt_setup_mux.find_window_by_id = AsyncMock(return_value=_window("bash"))
        prompt_setup_mux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        calls = prompt_setup_mux.send_keys.call_args_list
        assert calls[1][1].get("raw") is True
        assert calls[2][1].get("raw") is True

    @pytest.mark.usefixtures("_wrap_mode")
    async def test_skips_setup_when_marker_present(self, prompt_setup_mux) -> None:
        prompt_setup_mux.capture_pane = AsyncMock(return_value="~/code main ❯ ⌘0⌘ ")
        prompt_setup_mux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        prompt_setup_mux.send_keys.assert_not_called()


class TestGetShellName:
    @pytest.mark.parametrize(
        ("shell_env", "expected"),
        [
            pytest.param("/bin/zsh", "zsh", id="absolute_path"),
            pytest.param("/opt/homebrew/bin/fish", "fish", id="nested_path"),
            pytest.param("bash", "bash", id="bare_name"),
        ],
    )
    def test_returns_basename_of_shell_env(self, shell_env: str, expected: str) -> None:
        from ccgram.providers.shell import get_shell_name

        with _env_shell(shell_env):
            assert get_shell_name() == expected

    def test_returns_empty_when_shell_unset(self) -> None:
        from ccgram.providers.shell import get_shell_name

        with patch.dict("os.environ", {}, clear=True):
            assert get_shell_name() == ""


class TestPromptMatch:
    def test_prompt_match_frozen(self) -> None:
        pm = PromptMatch(
            sequence_number=0, trailing_text="ls", exit_code=0, raw_line="⌘0⌘ ls"
        )
        with pytest.raises(AttributeError):
            pm.exit_code = 1  # type: ignore[misc]

    @pytest.mark.usefixtures("_wrap_mode")
    def test_wrap_mode_bare_prompt(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code ❯ ⌘0⌘ ")
        assert m is not None
        assert m.sequence_number == 0
        assert m.exit_code == 0
        assert m.trailing_text.strip() == ""
        assert m.raw_line == "~/code ❯ ⌘0⌘ "

    @pytest.mark.usefixtures("_wrap_mode")
    def test_wrap_mode_with_trailing(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code ❯ ⌘0⌘ git status")
        assert m is not None
        assert m.sequence_number == 0
        assert m.exit_code == 0
        assert m.trailing_text == "git status"
        assert m.raw_line == "~/code ❯ ⌘0⌘ git status"

    def test_replace_mode_bare_prompt(self) -> None:
        from ccgram.config import config
        from ccgram.providers.shell import match_prompt

        original = config.prompt_mode
        config.prompt_mode = "replace"
        try:
            m = match_prompt("ccgram:0❯ ")
            assert m is not None
            assert m.sequence_number == 0
            assert m.exit_code == 0
            assert m.trailing_text.strip() == ""
            assert m.raw_line == "ccgram:0❯ "
        finally:
            config.prompt_mode = original


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeRegex:
    def test_match_prompt_finds_wrap_marker(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code main ❯ ⌘0⌘ ls -la")
        assert m is not None
        assert isinstance(m, PromptMatch)
        assert m.exit_code == 0
        assert m.trailing_text == "ls -la"
        assert m.raw_line == "~/code main ❯ ⌘0⌘ ls -la"

    def test_match_prompt_bare_prompt_idle(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code main ❯ ⌘0⌘ ")
        assert m is not None
        assert m.exit_code == 0
        assert m.trailing_text.strip() == ""

    def test_match_prompt_nonzero_exit(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("~/code main ❯ ⌘127⌘ bad-cmd")
        assert m is not None
        assert m.exit_code == 127
        assert m.trailing_text == "bad-cmd"

    def test_match_prompt_no_marker(self) -> None:
        from ccgram.providers.shell import match_prompt

        assert match_prompt("$ ls -la") is None

    def test_match_prompt_marker_only_line(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("⌘0⌘")
        assert m is not None
        assert m.exit_code == 0


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeSetup:
    @pytest.mark.parametrize(
        ("shell", "expected_substring"),
        [
            ("fish", "__ccgram_orig_prompt"),
            ("bash", "PROMPT_COMMAND"),
            ("zsh", "PROMPT+="),
            ("tcsh", "set prompt"),
        ],
        ids=["fish-wrap", "bash-wrap", "zsh-wrap", "tcsh-wrap"],
    )
    async def test_wrap_sends_correct_prompt_command(
        self, prompt_setup_mux, shell: str, expected_substring: str
    ) -> None:
        prompt_setup_mux.find_window_by_id = AsyncMock(return_value=_window(shell))
        prompt_setup_mux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        calls = prompt_setup_mux.send_keys.call_args_list
        prompt_call = calls[1]
        assert expected_substring in prompt_call[0][1]

    async def test_wrap_fish_preserves_original_prompt(self, prompt_setup_mux) -> None:
        prompt_setup_mux.find_window_by_id = AsyncMock(return_value=_window("fish"))
        prompt_setup_mux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        cmd = prompt_setup_mux.send_keys.call_args_list[1][0][1]
        assert "builtin functions --copy fish_prompt __ccgram_orig_prompt" in cmd
        assert "builtin functions --query __ccgram_orig_prompt" in cmd
        assert "__ccgram_orig_prompt" in cmd
        assert "⌘%d⌘" in cmd
        assert "set_color brblack" in cmd

    async def test_wrap_has_prompt_marker_detects_wrap_marker(
        self, prompt_setup_mux
    ) -> None:
        prompt_setup_mux.capture_pane = AsyncMock(return_value="~/code main ❯ ⌘0⌘ ")
        assert await has_prompt_marker("@0") is True

    async def test_wrap_has_prompt_marker_rejects_no_marker(
        self, prompt_setup_mux
    ) -> None:
        prompt_setup_mux.capture_pane = AsyncMock(return_value="~/code main ❯ ")
        assert await has_prompt_marker("@0") is False


class TestGetPromptMode:
    def test_defaults_to_wrap(self) -> None:
        from ccgram.config import config
        from ccgram.providers.shell_infra import _get_prompt_mode

        config.prompt_mode = "wrap"
        assert _get_prompt_mode() == "wrap"

    def test_returns_replace(self) -> None:
        from ccgram.config import config
        from ccgram.providers.shell_infra import _get_prompt_mode

        config.prompt_mode = "replace"
        assert _get_prompt_mode() == "replace"

    def test_invalid_mode_defaults_to_wrap(self) -> None:
        import ccgram.providers.shell_infra as shell_mod
        from ccgram.config import config
        from ccgram.providers.shell_infra import _get_prompt_mode

        original_warned = shell_mod._WARNED_INVALID_MODE
        shell_mod._WARNED_INVALID_MODE = False
        try:
            config.prompt_mode = "bogus"
            assert _get_prompt_mode() == "wrap"
        finally:
            shell_mod._WARNED_INVALID_MODE = original_warned

    def test_empty_string_defaults_to_wrap(self) -> None:
        from ccgram.config import config
        from ccgram.providers.shell_infra import _get_prompt_mode

        config.prompt_mode = ""
        assert _get_prompt_mode() == "wrap"


class TestMatchPromptModeSwitching:
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
        assert m.exit_code == 0
        assert m.trailing_text == "ls"

    @pytest.mark.usefixtures("_wrap_mode")
    def test_wrap_mode_matches_marker_at_start(self) -> None:
        from ccgram.providers.shell import match_prompt

        m = match_prompt("⌘0⌘ ls")
        assert m is not None
        assert m.trailing_text == "ls"


class TestWrapSetupCommands:
    @pytest.mark.parametrize(
        ("shell", "expected"),
        [
            ("fish", "__ccgram_orig_prompt"),
            ("fish", "set_color brblack"),
            ("fish", "or function __ccgram_orig_prompt"),
            ("fish", "builtin functions --query __ccgram_orig_prompt"),
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
        from ccgram.providers.shell_infra import _wrap_setup_commands

        assert expected in _wrap_setup_commands(shell)

    def test_zsh_wrap_command_uses_real_escape_sequence(self) -> None:
        from ccgram.providers.shell_infra import _wrap_setup_commands

        cmd = _wrap_setup_commands("zsh")
        assert "$'%{\\e[2m%}⌘%?⌘%{\\e[0m%} '" in cmd
        assert "\\033[2m" not in cmd
        assert "\\033[0m" not in cmd

    def test_unknown_shell_falls_back_to_posix(self) -> None:
        from ccgram.providers.shell_infra import _wrap_setup_commands

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
        from ccgram.providers.shell_infra import _replace_setup_commands

        assert expected in _replace_setup_commands(shell, "ccgram")

    def test_custom_prefix(self) -> None:
        from ccgram.providers.shell_infra import _replace_setup_commands

        cmd = _replace_setup_commands("bash", "mybot")
        assert "mybot:$?❯" in cmd

    def test_unknown_shell_falls_back_to_bash(self) -> None:
        from ccgram.providers.shell_infra import _replace_setup_commands

        cmd = _replace_setup_commands("unknown_shell", "ccgram")
        assert "PS1=" in cmd


class TestSetupShellPromptClearsBefore:
    async def test_sends_ctrl_c_before_prompt_command(self, prompt_setup_mux) -> None:
        prompt_setup_mux.find_window_by_id = AsyncMock(return_value=_window("bash"))
        prompt_setup_mux.send_keys = AsyncMock()

        await setup_shell_prompt("@0")

        first_call = prompt_setup_mux.send_keys.call_args_list[0]
        assert first_call[0][1] == "C-c"
        assert first_call[1].get("enter") is False
        assert first_call[1].get("literal") is False

    async def test_no_clear_when_clear_false(self, prompt_setup_mux) -> None:
        prompt_setup_mux.find_window_by_id = AsyncMock(return_value=_window("bash"))
        prompt_setup_mux.send_keys = AsyncMock()

        await setup_shell_prompt("@0", clear=False)

        calls = prompt_setup_mux.send_keys.call_args_list
        assert len(calls) == 2
        assert all(c[0][1] != "clear" for c in calls)
