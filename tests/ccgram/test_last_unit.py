from unittest.mock import AsyncMock, patch

import pytest

from ccgram.last_unit import capture_for_screenshot, extract_last_shell_block

# Wrap-mode markers: ⌘N⌘ with optional ANSI around them.
# Bare prompt: marker followed only by ANSI reset codes (strip → empty after strip)
# Command echo: marker followed by ANSI reset + command text (non-empty after strip)
# The conftest sets replace mode by default; tests that use wrap markers must
# request the _wrap_mode fixture to override it for the duration of the test.

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"

BARE = f"user@host $ {_DIM}⌘0⌘{_RESET}"
ECHO = f"user@host $ {_DIM}⌘0⌘{_RESET} ls -la"
OUTPUT = "total 8\ndrwxr-xr-x  2 user group 64 Jan  1 00:00 ."


def _scrollback(*lines: str) -> str:
    return "\n".join(lines)


def test_extract_happy_path(_wrap_mode: None) -> None:
    scrollback = _scrollback(
        "some earlier output",
        ECHO,
        OUTPUT,
        BARE,
    )
    result = extract_last_shell_block(scrollback)
    assert result is not None
    assert result == _scrollback(ECHO, OUTPUT, BARE)


def test_extract_no_markers_returns_none() -> None:
    scrollback = _scrollback("line one", "line two", "line three")
    assert extract_last_shell_block(scrollback) is None


def test_extract_only_bare_prompt_no_echo_returns_none(_wrap_mode: None) -> None:
    scrollback = _scrollback("some output", "more output", BARE)
    assert extract_last_shell_block(scrollback) is None


def test_extract_command_running_returns_none(_wrap_mode: None) -> None:
    scrollback = _scrollback("earlier output", ECHO, "partial output")
    assert extract_last_shell_block(scrollback) is None


@pytest.mark.asyncio
async def test_capture_shell_with_markers(_wrap_mode: None) -> None:
    scrollback = _scrollback("earlier", ECHO, OUTPUT, BARE)
    with patch(
        "ccgram.last_unit.tmux_manager.capture_pane_scrollback",
        new=AsyncMock(return_value=scrollback),
    ):
        result = await capture_for_screenshot("@0", "shell")
    assert result == _scrollback(ECHO, OUTPUT, BARE)


@pytest.mark.asyncio
async def test_capture_shell_without_markers_returns_full_scrollback() -> None:
    scrollback = "no markers here\njust plain text"
    with patch(
        "ccgram.last_unit.tmux_manager.capture_pane_scrollback",
        new=AsyncMock(return_value=scrollback),
    ):
        result = await capture_for_screenshot("@0", "shell")
    assert result == scrollback


@pytest.mark.asyncio
async def test_capture_claude_returns_full_scrollback(_wrap_mode: None) -> None:
    scrollback = _scrollback("earlier", ECHO, OUTPUT, BARE)
    with patch(
        "ccgram.last_unit.tmux_manager.capture_pane_scrollback",
        new=AsyncMock(return_value=scrollback),
    ):
        result = await capture_for_screenshot("@0", "claude")
    assert result == scrollback


@pytest.mark.asyncio
async def test_capture_returns_none_when_scrollback_fails() -> None:
    with patch(
        "ccgram.last_unit.tmux_manager.capture_pane_scrollback",
        new=AsyncMock(return_value=None),
    ):
        result = await capture_for_screenshot("@0", "shell")
    assert result is None
