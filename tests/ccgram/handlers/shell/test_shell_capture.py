from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot

from ccgram.handlers.shell.shell_capture import (
    _extract_command_output,
    strip_terminal_glyphs,
)
from ccgram.multiplexer.base import CaptureResult

_MOD = "ccgram.handlers.shell.shell_capture"


class TestStripTerminalGlyphs:
    def test_strips_nerd_font_glyphs(self) -> None:
        assert strip_terminal_glyphs("\ue0b0 hello") == " hello"

    def test_strips_pua_supplement(self) -> None:
        assert strip_terminal_glyphs("\U000f0001 icon") == " icon"

    def test_preserves_normal_text(self) -> None:
        assert strip_terminal_glyphs("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert strip_terminal_glyphs("") == ""


class TestExtractCommandOutput:
    @pytest.mark.parametrize(
        ("pane", "expected_text", "expected_code"),
        [
            (
                "ccgram:0❯ ls\nfile1.txt\nfile2.txt\nccgram:0❯",
                "file1.txt\nfile2.txt",
                0,
            ),
            (
                "ccgram:0❯ bad-cmd\nerror: not found\nccgram:127❯",
                "error: not found",
                127,
            ),
            ("ccgram:0❯ true\nccgram:0❯", "", 0),
            ("ccgram:0❯", "", 0),
        ],
        ids=["success", "failure-127", "no-output", "bare-prompt"],
    )
    def test_marker_extraction(
        self, pane: str, expected_text: str, expected_code: int
    ) -> None:
        result = _extract_command_output(pane)
        assert result.text == expected_text
        assert result.exit_code == expected_code

    def test_no_markers_returns_empty(self) -> None:
        current = "$ ls\nfile1.txt"
        result = _extract_command_output(current)
        assert result.text == ""
        assert result.exit_code is None

    def test_empty_current(self) -> None:
        result = _extract_command_output("")
        assert result.text == ""
        assert result.exit_code is None

    def test_multiline_output_with_markers(self) -> None:
        current = (
            "ccgram:0❯ find . -name '*.py'\n"
            "./src/main.py\n"
            "./src/utils.py\n"
            "./tests/test_main.py\n"
            "ccgram:0❯"
        )
        result = _extract_command_output(current)
        assert result.exit_code == 0
        assert "./src/main.py" in result.text
        assert "./tests/test_main.py" in result.text

    def test_command_still_running_no_end_marker(self) -> None:
        current = "ccgram:0❯ long-cmd\npartial output line 1\npartial output line 2"
        result = _extract_command_output(current)
        assert result.exit_code is None


class TestUpdateErrorMessage:
    async def test_formats_with_code_fence(self) -> None:
        from ccgram.handlers.shell.shell_capture import _update_error_message

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit:
            await _update_error_message(bot, -100, 99, 1, "some error output")

        formatted = mock_edit.call_args[0][3]
        assert formatted.startswith("\u274c exit 1\n```\n")
        assert formatted.endswith("\n```")
        assert "some error output" in formatted

    async def test_escapes_backticks_in_output(self) -> None:
        from ccgram.handlers.shell.shell_capture import _update_error_message

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit:
            await _update_error_message(bot, -100, 99, 1, "has ``` backticks")

        formatted = mock_edit.call_args[0][3]
        body = formatted.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert "```" not in body


class TestRelayOutputBackticks:
    async def test_triple_backticks_escaped_in_relay(self) -> None:
        from ccgram.handlers.shell.shell_capture import _relay_output

        bot = AsyncMock(spec=Bot)

        mock_sent = MagicMock()
        mock_sent.message_id = 42

        with patch(
            f"{_MOD}.rate_limit_send_message",
            new_callable=AsyncMock,
            return_value=mock_sent,
        ) as mock_send:
            await _relay_output(bot, -100, 42, "output has ``` backticks")

        formatted = mock_send.call_args[0][2]
        inner = formatted.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert "```" not in inner
        assert "` ` `" in inner

    async def test_relay_skips_whitespace_only_output(self) -> None:
        from ccgram.handlers.shell.shell_capture import _relay_output

        bot = AsyncMock(spec=Bot)

        with patch(
            f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock
        ) as mock_send:
            await _relay_output(bot, -100, 42, "   \n  \n  ")

        mock_send.assert_not_called()


class TestFindCommandEcho:
    def test_finds_echo_above_bare_prompt(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_command_echo

        lines = ["ccgram:0❯ ls", "file1.txt", "ccgram:0❯"]
        assert _find_command_echo(lines) == ("ccgram:0❯ ls", 0)

    def test_returns_none_for_idle(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_command_echo

        lines = ["ccgram:0❯"]
        assert _find_command_echo(lines) is None

    def test_returns_none_for_no_markers(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_command_echo

        lines = ["$ ls", "file.txt"]
        assert _find_command_echo(lines) is None

    def test_finds_last_command(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_command_echo

        lines = [
            "ccgram:0❯ ls",
            "file1.txt",
            "ccgram:0❯ pwd",
            "/home",
            "ccgram:0❯",
        ]
        assert _find_command_echo(lines) == ("ccgram:0❯ pwd", 2)


class TestFindInProgress:
    def test_finds_running_command(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_in_progress

        lines = ["ccgram:0❯ tail -f log", "line1", "line2"]
        result = _find_in_progress(lines)
        assert result is not None
        assert result.command_echo == "ccgram:0❯ tail -f log"
        assert result.echo_index == 0
        assert result.text == "line1\nline2"
        assert result.exit_code is None

    def test_returns_none_for_bare_prompt(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_in_progress

        lines = ["ccgram:0❯"]
        assert _find_in_progress(lines) is None

    def test_empty_output_in_progress(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_in_progress

        lines = ["ccgram:0❯ slow-cmd"]
        result = _find_in_progress(lines)
        assert result is not None
        assert result.text == ""


class TestExtractPassiveOutput:
    @pytest.mark.parametrize(
        ("pane", "echo", "expected_text", "expected_code"),
        [
            (
                "ccgram:0❯ ls\nfile1.txt\nfile2.txt\nccgram:0❯",
                "ccgram:0❯ ls",
                "file1.txt\nfile2.txt",
                0,
            ),
            (
                "ccgram:0❯ bad-cmd\nerror: not found\nccgram:127❯",
                "ccgram:0❯ bad-cmd",
                "error: not found",
                127,
            ),
            (
                "ccgram:0❯ true\nccgram:0❯",
                "ccgram:0❯ true",
                "",
                0,
            ),
        ],
        ids=["success", "failure-127", "no-output"],
    )
    def test_completed_commands(
        self,
        pane: str,
        echo: str,
        expected_text: str,
        expected_code: int,
    ) -> None:
        from ccgram.handlers.shell.shell_capture import _extract_passive_output

        result = _extract_passive_output(pane)
        assert result is not None
        assert result.command_echo == echo
        assert result.text == expected_text
        assert result.exit_code == expected_code

    def test_idle_returns_none(self) -> None:
        from ccgram.handlers.shell.shell_capture import _extract_passive_output

        assert _extract_passive_output("ccgram:0❯") is None

    def test_no_markers_returns_none(self) -> None:
        from ccgram.handlers.shell.shell_capture import _extract_passive_output

        assert _extract_passive_output("$ ls\nfile.txt") is None

    def test_empty_returns_none(self) -> None:
        from ccgram.handlers.shell.shell_capture import _extract_passive_output

        assert _extract_passive_output("") is None

    def test_in_progress_command(self) -> None:
        from ccgram.handlers.shell.shell_capture import _extract_passive_output

        pane = "ccgram:0❯ tail -f log\nline1\nline2"
        result = _extract_passive_output(pane)
        assert result is not None
        assert result.command_echo == "ccgram:0❯ tail -f log"
        assert result.text == "line1\nline2"
        assert result.exit_code is None


@pytest.fixture()
def _clean_monitor_state():
    from ccgram.handlers.shell.shell_capture import reset_shell_monitor_state

    reset_shell_monitor_state()
    yield
    reset_shell_monitor_state()


@pytest.mark.usefixtures("_clean_monitor_state")
class TestCheckPassiveShellOutput:
    @pytest.mark.asyncio()
    async def test_skips_when_no_markers(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock) as m:
            await check_passive_shell_output(bot, 1, 42, "@0", "$ ls\nfile.txt")
        m.assert_not_called()
        assert (
            "@0" not in _shell_monitor_state
            or _shell_monitor_state["@0"].msg_id is None
        )

    @pytest.mark.asyncio()
    async def test_relays_completed_command(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 99

        pane = "ccgram:0❯ ls\nfile1.txt\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        mock_send.assert_called_once()
        state = _shell_monitor_state["@0"]
        assert state.msg_id == 99
        assert state.last_command_echo == "ccgram:0❯ ls"

    @pytest.mark.asyncio()
    async def test_skips_unchanged_content(self) -> None:
        from ccgram.handlers.shell.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 99
        pane = "ccgram:0❯ ls\nfile1.txt\nccgram:0❯"

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        with (
            patch(
                f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock
            ) as mock_send2,
            patch(f"{_MOD}.thread_router") as mock_sm2,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
        ):
            mock_sm2.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        mock_send2.assert_not_called()

    @pytest.mark.asyncio()
    async def test_error_indicator_for_nonzero_exit(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 77

        pane = "ccgram:0❯ bad-cmd\nerror: not found\nccgram:127❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        assert mock_edit.called
        state = _shell_monitor_state["@0"]
        assert state.exit_code_sent is True

    @pytest.mark.asyncio()
    async def test_new_command_resets_state(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 50

        pane1 = "ccgram:0❯ ls\nfile.txt\nccgram:0❯"
        pane2 = "ccgram:0❯ pwd\n/home\nccgram:0❯"

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane1),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane1)
            assert _shell_monitor_state["@0"].last_command_echo == "ccgram:0❯ ls"

        mock_sent2 = MagicMock()
        mock_sent2.message_id = 51

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent2,
            ),
            patch(f"{_MOD}.thread_router") as mock_sm2,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane2),
            ),
        ):
            mock_sm2.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane2)

        state = _shell_monitor_state["@0"]
        assert state.last_command_echo == "ccgram:0❯ pwd"
        assert state.msg_id == mock_sent2.message_id

    @pytest.mark.asyncio()
    async def test_long_output_with_scrollback(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 88

        visible = "\n".join([f"file{i}.txt" for i in range(20)] + ["ccgram:0❯"])
        scrollback = "ccgram:0❯ ls -al\n" + visible

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=scrollback),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", visible)

        mock_send.assert_called_once()
        state = _shell_monitor_state["@0"]
        assert state.msg_id == 88
        assert state.last_command_echo == "ccgram:0❯ ls -al"


@pytest.mark.usefixtures("_clean_monitor_state")
class TestScrollbackTruncation:
    """The herdr read cap (1000 lines) clips scrollback; a clipped capture must
    surface a truncation notice rather than read as the full command output."""

    @pytest.mark.asyncio()
    async def test_truncated_capture_prepends_notice(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _TRUNCATION_NOTICE,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 400

        pane = "ccgram:0❯ dump\nlots of output\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane, truncated=True),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        sent_text = mock_send.call_args[0][2]
        assert _TRUNCATION_NOTICE in sent_text
        assert "lots of output" in sent_text

    @pytest.mark.asyncio()
    async def test_untruncated_capture_has_no_notice(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _TRUNCATION_NOTICE,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 401

        pane = "ccgram:0❯ dump\nshort output\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane, truncated=False),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        sent_text = mock_send.call_args[0][2]
        assert _TRUNCATION_NOTICE not in sent_text
        assert "short output" in sent_text


@pytest.mark.usefixtures("_clean_monitor_state")
@pytest.mark.usefixtures("_clean_monitor_state")
class TestPassiveTruncatedTail:
    """A command longer than the backend's readable history scrolls its echo
    out of even a full-cap capture. On capped backends (herdr) the captured
    tail is still relayed, flagged truncated, instead of dropping all output;
    uncapped backends (tmux) keep the echo-anchored behavior."""

    @staticmethod
    def _mux(capture: CaptureResult, cap: int | None) -> MagicMock:
        mux = MagicMock()
        mux.capabilities.read_max_lines = cap
        mux.capture_scrollback = AsyncMock(return_value=capture)
        return mux

    def test_passive_scrollback_lines_uses_cap(self) -> None:
        from ccgram.handlers.shell.shell_capture import _passive_scrollback_lines

        mux = self._mux(CaptureResult(text="", truncated=False), 1000)
        with patch(f"{_MOD}.tmux_manager", mux):
            assert _passive_scrollback_lines() == 1000

    def test_passive_scrollback_lines_default_when_uncapped(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _SCROLLBACK_LINES,
            _passive_scrollback_lines,
        )

        mux = self._mux(CaptureResult(text="", truncated=False), None)
        with patch(f"{_MOD}.tmux_manager", mux):
            assert _passive_scrollback_lines() == _SCROLLBACK_LINES

    def test_truncated_tail_recovers_output_without_echo(self) -> None:
        from ccgram.handlers.shell.shell_capture import _truncated_tail_output

        result = _truncated_tail_output("999\n1000\nccgram:0❯")
        assert result is not None
        assert result.text == "999\n1000"
        assert result.exit_code == 0
        assert result.command_echo == ""

    def test_truncated_tail_none_when_idle(self) -> None:
        from ccgram.handlers.shell.shell_capture import _truncated_tail_output

        assert _truncated_tail_output("ccgram:0❯") is None

    def test_truncated_tail_none_without_completed_prompt(self) -> None:
        from ccgram.handlers.shell.shell_capture import _truncated_tail_output

        assert _truncated_tail_output("still running\nmore output") is None

    @pytest.mark.asyncio()
    async def test_capped_backend_relays_truncated_tail(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _TRUNCATION_NOTICE,
            check_passive_shell_output,
            mark_telegram_command,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 700

        # A Telegram-issued command (evidence it actually ran) whose echo fell
        # outside the readable window: output lines + a completed bare prompt,
        # no prompt-with-command above it. The visible tail is still relayed.
        mark_telegram_command("@0", "yes | head -100000", 1, 42)
        pane = "997\n998\n999\n1000\nccgram:0❯"
        mux = self._mux(CaptureResult(text=pane, truncated=False), 1000)
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager", mux),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        mux.capture_scrollback.assert_awaited_once_with("@0", lines=1000)
        sent_text = mock_send.call_args[0][2]
        assert _TRUNCATION_NOTICE in sent_text
        assert "1000" in sent_text

    @pytest.mark.asyncio()
    async def test_stale_scrollback_above_fresh_prompt_not_relayed(self) -> None:
        """Regression: an existing herdr shell bound with ``clear=False`` keeps
        old scrollback above the new ``ccgram:0❯`` prompt. With no command run
        in this prompt session (fresh monitor state), the echoless-tail recovery
        must treat it as idle and relay nothing — not dump stale scrollback."""
        from ccgram.handlers.shell.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        # Old output preserved by clear=False, then the freshly set-up prompt.
        pane = "old output\nccgram:0❯"
        mux = self._mux(CaptureResult(text=pane, truncated=False), 1000)
        with (
            patch(
                f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager", mux),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        mock_send.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_uncapped_backend_drops_echoless_capture(self) -> None:
        from ccgram.handlers.shell.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        pane = "997\n998\n999\n1000\nccgram:0❯"
        mux = self._mux(CaptureResult(text=pane, truncated=False), None)
        with (
            patch(
                f"{_MOD}.rate_limit_send_message", new_callable=AsyncMock
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager", mux),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        # Uncapped backend: no echo found → nothing relayed (no ambiguous tail).
        mock_send.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_in_flight_tail_edits_existing_message(self) -> None:
        """A long command relayed while its echo was visible must keep editing
        the same message once the echo scrolls out and only an echoless tail is
        left — not spawn a second message."""
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 700

        in_progress = "ccgram:0❯ long-cmd\nline1\nline2"
        completed_tail = "line998\nline999\nline1000\nccgram:0❯"
        mux = MagicMock()
        mux.capabilities.read_max_lines = 1000
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager", mux),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            # Poll 1: echo visible, command in progress → first relay (new msg).
            mux.capture_scrollback = AsyncMock(
                return_value=CaptureResult(text=in_progress, truncated=False)
            )
            await check_passive_shell_output(bot, 1, 42, "@0", in_progress)
            # Poll 2: echo scrolled out, command completed → echoless tail.
            mux.capture_scrollback = AsyncMock(
                return_value=CaptureResult(text=completed_tail, truncated=False)
            )
            await check_passive_shell_output(bot, 1, 42, "@0", completed_tail)

        # One send (poll 1), then an edit of that same message (poll 2) — not a
        # second send.
        mock_send.assert_awaited_once()
        mock_edit.assert_awaited()
        assert _shell_monitor_state["@0"].msg_id == 700

    @pytest.mark.asyncio()
    async def test_consecutive_echoless_commands_get_separate_messages(self) -> None:
        """Two Telegram commands that each complete within a single poll with
        their echo already scrolled out must each get their own message — the
        second must not overwrite the first."""
        from ccgram.handlers.shell.shell_capture import (
            check_passive_shell_output,
            mark_telegram_command,
        )

        bot = AsyncMock(spec=Bot)
        sent_a = MagicMock(message_id=801)
        sent_b = MagicMock(message_id=802)

        pane_a = "a-out-1\na-out-2\nccgram:0❯"
        pane_b = "b-out-1\nb-out-2\nccgram:0❯"
        mux = MagicMock()
        mux.capabilities.read_max_lines = 1000
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                side_effect=[sent_a, sent_b],
            ) as mock_send,
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.tmux_manager", mux),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mark_telegram_command("@0", "cmd-a", 1, 42)
            mux.capture_scrollback = AsyncMock(
                return_value=CaptureResult(text=pane_a, truncated=False)
            )
            await check_passive_shell_output(bot, 1, 42, "@0", pane_a)

            mark_telegram_command("@0", "cmd-b", 1, 42)
            mux.capture_scrollback = AsyncMock(
                return_value=CaptureResult(text=pane_b, truncated=False)
            )
            await check_passive_shell_output(bot, 1, 42, "@0", pane_b)

        # Two distinct sends, no edit of the first command's message.
        assert mock_send.await_count == 2
        mock_edit.assert_not_awaited()


class TestClearShellMonitorState:
    def test_clear_removes_state(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _ShellMonitorState,
            _shell_monitor_state,
            clear_shell_monitor_state,
        )

        _shell_monitor_state["@5"] = _ShellMonitorState(last_command_echo="test")
        clear_shell_monitor_state("@5")
        assert "@5" not in _shell_monitor_state

    def test_clear_nonexistent_is_noop(self) -> None:
        from ccgram.handlers.shell.shell_capture import clear_shell_monitor_state

        clear_shell_monitor_state("@99")

    def test_reset_clears_all(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _ShellMonitorState,
            _shell_monitor_state,
            reset_shell_monitor_state,
        )

        _shell_monitor_state["@1"] = _ShellMonitorState()
        _shell_monitor_state["@2"] = _ShellMonitorState()
        reset_shell_monitor_state()
        assert len(_shell_monitor_state) == 0


@pytest.mark.usefixtures("_clean_monitor_state")
class TestPassiveEdgeCases:
    @pytest.mark.asyncio()
    async def test_same_command_rerun_creates_new_message(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent1 = MagicMock()
        mock_sent1.message_id = 60
        mock_sent2 = MagicMock()
        mock_sent2.message_id = 61

        pane1 = "ccgram:0❯ ls\nfile.txt\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent1,
            ),
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane1),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane1)

        assert _shell_monitor_state["@0"].msg_id == 60

        pane2 = "ccgram:0❯ ls\nfile.txt\nccgram:0❯ ls\nfile.txt\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent2,
            ),
            patch(f"{_MOD}.thread_router") as mock_sm2,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane2),
            ),
        ):
            mock_sm2.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane2)

        assert _shell_monitor_state["@0"].msg_id == 61

    @pytest.mark.asyncio()
    async def test_scroll_out_preserves_in_progress(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _ShellMonitorState,
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)

        _shell_monitor_state["@0"] = _ShellMonitorState(
            last_command_echo="ccgram:0❯ long-cmd",
            last_echo_index=0,
            msg_id=70,
            last_output="partial",
        )

        no_marker_pane = "\n".join([f"output line {i}" for i in range(20)])
        with patch(f"{_MOD}.thread_router") as mock_sm:
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", no_marker_pane)

        state = _shell_monitor_state["@0"]
        assert state.last_command_echo == "ccgram:0❯ long-cmd"
        assert state.msg_id == 70


class TestCommandFromEcho:
    def test_extracts_command_text(self) -> None:
        from ccgram.handlers.shell.shell_capture import _command_from_echo

        assert _command_from_echo("ccgram:0❯ ls -al") == "ls -al"

    def test_strips_whitespace(self) -> None:
        from ccgram.handlers.shell.shell_capture import _command_from_echo

        assert _command_from_echo("ccgram:0❯ echo hi   ") == "echo hi"

    def test_error_exit_code(self) -> None:
        from ccgram.handlers.shell.shell_capture import _command_from_echo

        assert _command_from_echo("ccgram:127❯ bad-cmd") == "bad-cmd"

    def test_non_matching_returns_input(self) -> None:
        from ccgram.handlers.shell.shell_capture import _command_from_echo

        assert _command_from_echo("$ ls") == "$ ls"


class TestHasMarkersInTail:
    def test_marker_at_end(self) -> None:
        from ccgram.handlers.shell.shell_capture import _has_markers_in_tail

        text = "file1.txt\nfile2.txt\nccgram:0❯"
        assert _has_markers_in_tail(text) is True

    def test_no_markers(self) -> None:
        from ccgram.handlers.shell.shell_capture import _has_markers_in_tail

        text = "file1.txt\nfile2.txt\n$ "
        assert _has_markers_in_tail(text) is False

    def test_marker_with_leading_whitespace(self) -> None:
        from ccgram.handlers.shell.shell_capture import _has_markers_in_tail

        text = "line1\n                    ccgram:0❯"
        assert _has_markers_in_tail(text) is True

    def test_marker_with_command(self) -> None:
        from ccgram.handlers.shell.shell_capture import _has_markers_in_tail

        text = "output\nccgram:0❯ ls -al"
        assert _has_markers_in_tail(text) is True


@pytest.mark.usefixtures("_clean_monitor_state")
class TestPassiveRelayFormatting:
    @pytest.mark.asyncio()
    async def test_output_includes_command_header(self) -> None:
        from ccgram.handlers.shell.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 200

        pane = "ccgram:0❯ echo hi\nhello\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        sent_text = mock_send.call_args[0][2]
        assert "❯ echo hi" in sent_text
        assert "hello" in sent_text
        assert sent_text.startswith("```\n")

    @pytest.mark.asyncio()
    async def test_multiline_output_formatted(self) -> None:
        from ccgram.handlers.shell.shell_capture import check_passive_shell_output

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 201

        pane = "ccgram:0❯ seq 1 3\n1\n2\n3\nccgram:0❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ) as mock_send,
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        sent_text = mock_send.call_args[0][2]
        assert "❯ seq 1 3" in sent_text
        assert "1\n2\n3" in sent_text

    @pytest.mark.asyncio()
    async def test_error_command_shows_exit_indicator(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
        )

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 202

        pane = "ccgram:0❯ bad-cmd\nbad-cmd: not found\nccgram:127❯"
        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock) as mock_edit,
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        assert _shell_monitor_state["@0"].exit_code_sent is True
        assert mock_edit.called
        edit_text = mock_edit.call_args[0][3]  # (bot, chat_id, msg_id, text)
        assert "exit 127" in edit_text

    @pytest.mark.asyncio()
    async def test_telegram_command_reacts_done_on_success(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            check_passive_shell_output,
            mark_telegram_command,
            reset_shell_monitor_state,
        )

        reset_shell_monitor_state()
        mark_telegram_command("@0", "ls", 1, 42, message_id=600)

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 1
        pane = "ccgram:0❯ ls\nfile.txt\nccgram:0❯"

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
            patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        from ccgram.handlers.reactions import REACT_DONE

        assert mock_react.call_args.args[3] == REACT_DONE
        assert mock_react.call_args.args[2] == 600
        # Once consumed on success, message_id is cleared so re-runs of the
        # same poll cycle don't double-react.
        assert _shell_monitor_state["@0"].telegram_message_id == 0
        reset_shell_monitor_state()

    @pytest.mark.asyncio()
    async def test_telegram_command_reacts_fail_on_nonzero_exit(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            check_passive_shell_output,
            mark_telegram_command,
            reset_shell_monitor_state,
        )

        reset_shell_monitor_state()
        mark_telegram_command("@0", "fail", 1, 42, message_id=601)

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 1
        pane = "ccgram:0❯ fail\nerror\nccgram:1❯"

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
            patch(
                f"{_MOD}._maybe_suggest_fix", new_callable=AsyncMock
            ),  # silence LLM path
            patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        from ccgram.handlers.reactions import REACT_FAIL

        assert mock_react.call_args.args[3] == REACT_FAIL
        assert mock_react.call_args.args[2] == 601
        reset_shell_monitor_state()

    @pytest.mark.asyncio()
    async def test_no_message_id_skips_reaction(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            check_passive_shell_output,
            mark_telegram_command,
            reset_shell_monitor_state,
        )

        reset_shell_monitor_state()
        # No message_id passed — typical for legacy callers.
        mark_telegram_command("@0", "ls", 1, 42)

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 1
        pane = "ccgram:0❯ ls\nfile\nccgram:0❯"

        with (
            patch(
                f"{_MOD}.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=mock_sent,
            ),
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(f"{_MOD}.thread_router") as mock_sm,
            patch(
                f"{_MOD}._capture_with_scrollback",
                new_callable=AsyncMock,
                return_value=CaptureResult(text=pane),
            ),
            patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react,
        ):
            mock_sm.resolve_chat_id.return_value = -100
            await check_passive_shell_output(bot, 1, 42, "@0", pane)

        mock_react.assert_not_awaited()
        reset_shell_monitor_state()


class TestCaptureWithScrollback:
    @pytest.mark.asyncio()
    async def test_returns_text_on_success(self) -> None:
        from ccgram.handlers.shell.shell_capture import _capture_with_scrollback

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"line1\nline2\n", b"")
            mock_exec.return_value = mock_proc
            result = await _capture_with_scrollback("@4")

        assert result == CaptureResult(text="line1\nline2", truncated=False)

    @pytest.mark.asyncio()
    async def test_returns_none_on_empty(self) -> None:
        from ccgram.handlers.shell.shell_capture import _capture_with_scrollback

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"  \n  \n", b"")
            mock_exec.return_value = mock_proc
            result = await _capture_with_scrollback("@4")

        assert result is None

    @pytest.mark.asyncio()
    async def test_uses_correct_tmux_flags(self) -> None:
        from ccgram.handlers.shell.shell_capture import _capture_with_scrollback

        with patch(
            "asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"output", b"")
            mock_exec.return_value = mock_proc
            await _capture_with_scrollback("@4", history=100)

        args = mock_exec.call_args[0]
        assert "tmux" in args
        assert "capture-pane" in args
        assert "-J" in args
        assert "-S" in args
        assert "-100" in args
        assert "@4" in args

    @pytest.mark.asyncio()
    async def test_delegates_clamp_and_surfaces_truncation(self) -> None:
        # The line cap lives in the backend: shell passes its requested depth
        # straight through and surfaces the backend's truncated flag.
        from ccgram.handlers.shell.shell_capture import _capture_with_scrollback

        clamped = CaptureResult(text="tail only", truncated=True)
        fake_mux = MagicMock()
        fake_mux.capture_scrollback = AsyncMock(return_value=clamped)
        with patch(f"{_MOD}.tmux_manager", fake_mux):
            result = await _capture_with_scrollback("@4", history=5000)

        assert result == clamped
        fake_mux.capture_scrollback.assert_awaited_once_with("@4", lines=5000)


class TestMarkTelegramCommand:
    def test_marks_command_in_state(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            mark_telegram_command,
            reset_shell_monitor_state,
        )

        reset_shell_monitor_state()
        mark_telegram_command("@0", "ls -la", 1, 42, 7777)
        state = _shell_monitor_state["@0"]
        assert state.telegram_command == "ls -la"
        assert state.telegram_user_id == 1
        assert state.telegram_thread_id == 42
        assert state.telegram_message_id == 7777
        reset_shell_monitor_state()

    def test_default_message_id_is_zero(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            mark_telegram_command,
            reset_shell_monitor_state,
        )

        reset_shell_monitor_state()
        mark_telegram_command("@0", "ls", 1, 42)
        assert _shell_monitor_state["@0"].telegram_message_id == 0
        reset_shell_monitor_state()

    def test_overwrites_previous(self) -> None:
        from ccgram.handlers.shell.shell_capture import (
            _shell_monitor_state,
            mark_telegram_command,
            reset_shell_monitor_state,
        )

        reset_shell_monitor_state()
        mark_telegram_command("@0", "ls", 1, 42, 100)
        mark_telegram_command("@0", "pwd", 2, 99, 200)
        state = _shell_monitor_state["@0"]
        assert state.telegram_command == "pwd"
        assert state.telegram_user_id == 2
        assert state.telegram_thread_id == 99
        assert state.telegram_message_id == 200
        reset_shell_monitor_state()


class TestExitReaction:
    async def test_react_exit_zero_uses_done(self) -> None:
        from ccgram.handlers.shell.shell_capture import _react_exit

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react:
            await _react_exit(bot, -100, 42, 0)
        mock_react.assert_awaited_once()
        from ccgram.handlers.reactions import REACT_DONE

        assert mock_react.call_args.args[3] == REACT_DONE

    async def test_react_exit_nonzero_uses_fail(self) -> None:
        from ccgram.handlers.shell.shell_capture import _react_exit

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react:
            await _react_exit(bot, -100, 42, 1)
        mock_react.assert_awaited_once()
        from ccgram.handlers.reactions import REACT_FAIL

        assert mock_react.call_args.args[3] == REACT_FAIL

    async def test_react_exit_zero_message_id_skips(self) -> None:
        from ccgram.handlers.shell.shell_capture import _react_exit

        bot = AsyncMock(spec=Bot)
        with patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react:
            await _react_exit(bot, -100, 0, 1)
        mock_react.assert_not_awaited()


class TestRelayOutputTruncation:
    async def test_long_output_gets_truncated_with_ellipsis(self) -> None:
        from ccgram.handlers.shell.shell_capture import _relay_output

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 300

        long_output = "x" * 5000

        with patch(
            f"{_MOD}.rate_limit_send_message",
            new_callable=AsyncMock,
            return_value=mock_sent,
        ) as mock_send:
            await _relay_output(bot, -100, 42, long_output)

        sent_text = mock_send.call_args[0][2]
        assert sent_text.startswith("```\n\u2026 ")
        assert len(sent_text) < 5000

    async def test_short_output_not_truncated(self) -> None:
        from ccgram.handlers.shell.shell_capture import _relay_output

        bot = AsyncMock(spec=Bot)
        mock_sent = MagicMock()
        mock_sent.message_id = 301

        with patch(
            f"{_MOD}.rate_limit_send_message",
            new_callable=AsyncMock,
            return_value=mock_sent,
        ) as mock_send:
            await _relay_output(bot, -100, 42, "short output")

        sent_text = mock_send.call_args[0][2]
        assert "\u2026" not in sent_text
        assert "short output" in sent_text


@pytest.mark.usefixtures("_clean_monitor_state")
class TestMaybeSuggestFix:
    async def test_calls_llm_and_shows_approval_on_error(self) -> None:
        from ccgram.handlers.shell.shell_capture import _maybe_suggest_fix
        from ccgram.handlers.shell.shell_commands import show_command_approval

        bot = AsyncMock(spec=Bot)
        mock_completer = AsyncMock()
        from ccgram.llm.base import CommandResult

        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="ls -la", explanation="Fixed", is_dangerous=False
            )
        )

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(
                "ccgram.llm.get_completer",
                return_value=mock_completer,
            ),
            patch(
                "ccgram.handlers.shell.shell_context.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell.shell_commands.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
            patch(f"{_MOD}._approval_callback", new=show_command_approval),
        ):
            await _maybe_suggest_fix(
                bot,
                1,
                -100,
                42,
                "@0",
                command="lss",
                exit_code=127,
                msg_id=50,
                output="lss: not found",
            )

        mock_completer.generate_command.assert_called_once()
        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][2]
        assert "ls -la" in sent_text

    async def test_skips_when_no_llm(self) -> None:
        from ccgram.handlers.shell.shell_capture import _maybe_suggest_fix

        bot = AsyncMock(spec=Bot)

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(
                "ccgram.llm.get_completer",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.shell.shell_commands.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            await _maybe_suggest_fix(
                bot,
                1,
                -100,
                42,
                "@0",
                command="bad",
                exit_code=1,
                msg_id=50,
                output="error",
            )

        mock_send.assert_not_called()

    async def test_skips_when_fix_equals_original(self) -> None:
        from ccgram.handlers.shell.shell_capture import _maybe_suggest_fix

        bot = AsyncMock(spec=Bot)
        mock_completer = AsyncMock()
        from ccgram.llm.base import CommandResult

        mock_completer.generate_command = AsyncMock(
            return_value=CommandResult(
                command="bad-cmd", explanation="Same", is_dangerous=False
            )
        )

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(
                "ccgram.llm.get_completer",
                return_value=mock_completer,
            ),
            patch(
                "ccgram.handlers.shell.shell_commands.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell.shell_commands.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            await _maybe_suggest_fix(
                bot,
                1,
                -100,
                42,
                "@0",
                command="bad-cmd",
                exit_code=1,
                msg_id=50,
                output="error",
            )

        mock_send.assert_not_called()

    async def test_skips_when_llm_errors(self) -> None:
        from ccgram.handlers.shell.shell_capture import _maybe_suggest_fix

        bot = AsyncMock(spec=Bot)
        mock_completer = AsyncMock()
        mock_completer.generate_command = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with (
            patch(f"{_MOD}.edit_with_fallback", new_callable=AsyncMock),
            patch(
                "ccgram.llm.get_completer",
                return_value=mock_completer,
            ),
            patch(
                "ccgram.handlers.shell.shell_commands.gather_llm_context",
                new_callable=AsyncMock,
                return_value={"cwd": "/tmp", "shell": "bash", "shell_tools": ""},
            ),
            patch(
                "ccgram.handlers.shell.shell_commands.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            await _maybe_suggest_fix(
                bot,
                1,
                -100,
                42,
                "@0",
                command="bad",
                exit_code=1,
                msg_id=50,
                output="error",
            )

        mock_send.assert_not_called()


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeExtraction:
    def test_extract_completed_command(self) -> None:
        pane = "~/code main ❯ ⌘0⌘ ls\nfile1.txt\nfile2.txt\n~/code main ❯ ⌘0⌘"
        result = _extract_command_output(pane)
        assert result.text == "file1.txt\nfile2.txt"
        assert result.exit_code == 0

    def test_extract_failed_command(self) -> None:
        pane = "~/code main ❯ ⌘0⌘ bad-cmd\nerror: not found\n~/code main ❯ ⌘127⌘"
        result = _extract_command_output(pane)
        assert result.text == "error: not found"
        assert result.exit_code == 127

    def test_idle_returns_exit_code_only(self) -> None:
        pane = "~/code main ❯ ⌘0⌘"
        result = _extract_command_output(pane)
        assert result.exit_code == 0
        assert result.text == ""

    def test_no_markers_returns_empty(self) -> None:
        pane = "~/code main ❯ ls"
        result = _extract_command_output(pane)
        assert result.exit_code is None
        assert result.text == ""

    def test_still_running_no_bare_prompt(self) -> None:
        pane = "~/code main ❯ ⌘0⌘ long-cmd\npartial output"
        result = _extract_command_output(pane)
        assert result.exit_code is None


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeFindCommandEcho:
    def test_finds_echo_above_bare_prompt(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_command_echo

        lines = [
            "~/code main ❯ ⌘0⌘ ls",
            "file1.txt",
            "~/code main ❯ ⌘0⌘",
        ]
        assert _find_command_echo(lines) == ("~/code main ❯ ⌘0⌘ ls", 0)

    def test_returns_none_for_idle(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_command_echo

        lines = ["~/code main ❯ ⌘0⌘"]
        assert _find_command_echo(lines) is None


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeFindInProgress:
    def test_finds_running_command(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_in_progress

        lines = ["~/code main ❯ ⌘0⌘ tail -f log", "line1", "line2"]
        result = _find_in_progress(lines)
        assert result is not None
        assert result.echo_index == 0
        assert result.text == "line1\nline2"
        assert result.exit_code is None

    def test_returns_none_for_bare_prompt(self) -> None:
        from ccgram.handlers.shell.shell_capture import _find_in_progress

        lines = ["~/code main ❯ ⌘0⌘"]
        assert _find_in_progress(lines) is None


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModePassiveOutput:
    def test_completed_command(self) -> None:
        from ccgram.handlers.shell.shell_capture import _extract_passive_output

        pane = "~/code main ❯ ⌘0⌘ ls\nfile1.txt\n~/code main ❯ ⌘0⌘"
        result = _extract_passive_output(pane)
        assert result is not None
        assert result.text == "file1.txt"
        assert result.exit_code == 0

    def test_idle_returns_none(self) -> None:
        from ccgram.handlers.shell.shell_capture import _extract_passive_output

        assert _extract_passive_output("~/code main ❯ ⌘0⌘") is None

    def test_in_progress_command(self) -> None:
        from ccgram.handlers.shell.shell_capture import _extract_passive_output

        pane = "~/code main ❯ ⌘0⌘ tail -f log\nline1\nline2"
        result = _extract_passive_output(pane)
        assert result is not None
        assert result.text == "line1\nline2"
        assert result.exit_code is None


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeCommandFromEcho:
    def test_extracts_command_text(self) -> None:
        from ccgram.handlers.shell.shell_capture import _command_from_echo

        assert _command_from_echo("~/code main ❯ ⌘0⌘ ls -al") == "ls -al"

    def test_non_matching_returns_input(self) -> None:
        from ccgram.handlers.shell.shell_capture import _command_from_echo

        assert _command_from_echo("$ ls") == "$ ls"


@pytest.mark.usefixtures("_wrap_mode")
class TestWrapModeHasMarkersInTail:
    def test_marker_at_end(self) -> None:
        from ccgram.handlers.shell.shell_capture import _has_markers_in_tail

        text = "file1.txt\nfile2.txt\n~/code main ❯ ⌘0⌘"
        assert _has_markers_in_tail(text) is True

    def test_no_markers(self) -> None:
        from ccgram.handlers.shell.shell_capture import _has_markers_in_tail

        text = "file1.txt\nfile2.txt\n~/code main ❯ "
        assert _has_markers_in_tail(text) is False


class TestRegisterApprovalCallback:
    def test_double_registration_raises(self) -> None:
        from ccgram.handlers.shell import shell_capture

        shell_capture._reset_approval_callback_for_testing()
        shell_capture.register_approval_callback(AsyncMock())
        with pytest.raises(RuntimeError, match="already registered"):
            shell_capture.register_approval_callback(AsyncMock())

    async def test_default_raises_when_not_wired(self) -> None:
        from ccgram.handlers.shell import shell_capture

        shell_capture._reset_approval_callback_for_testing()
        with pytest.raises(RuntimeError, match="not wired"):
            await shell_capture._approval_callback(
                AsyncMock(spec=Bot), 1, 1, "@0", MagicMock(), 1
            )
