from unittest.mock import AsyncMock, patch

import pytest

from ccgram.providers.claude import (
    ClaudeProvider,
    _find_mode_line,
    _mode_short_label,
)
from ccgram.providers.codex import CodexProvider
from ccgram.providers.gemini import GeminiProvider
from ccgram.providers.shell import ShellProvider


class TestHasYoloConfirmation:
    def test_claude_has_yolo(self):
        assert ClaudeProvider().capabilities.has_yolo_confirmation is True

    @pytest.mark.parametrize("cls", [CodexProvider, GeminiProvider, ShellProvider])
    def test_others_no_yolo(self, cls):
        assert cls().capabilities.has_yolo_confirmation is False


class TestClaudePickerCommands:
    def test_exact_set(self):
        assert ClaudeProvider().capabilities.tui_picker_commands == frozenset(
            {
                "agents",
                "copy",
                "diff",
                "effort",
                "model",
                "permissions",
                "release-notes",
                "rewind",
                "settings",
                "skills",
                "theme",
                "tui",
            }
        )

    def test_status_excluded(self):
        assert "status" not in ClaudeProvider().capabilities.tui_picker_commands


class TestCodexPickerCommands:
    def test_exact_set(self):
        assert CodexProvider().capabilities.tui_picker_commands == frozenset(
            {"model", "permissions", "skills", "statusline", "personality"}
        )


class TestGeminiPickerCommands:
    def test_exact_set(self):
        assert GeminiProvider().capabilities.tui_picker_commands == frozenset(
            {
                "agents",
                "auth",
                "chat",
                "editor",
                "extensions",
                "ide",
                "model",
                "privacy",
                "rewind",
                "settings",
                "terminal-setup",
                "theme",
            }
        )


class TestScrapeCurrentModeEdit:
    async def test_edit_mode(self):
        provider = ClaudeProvider()
        mock_capture = AsyncMock(return_value="some output\n⏵⏵ auto-accept edits on  >")
        with patch("ccgram.multiplexer.multiplexer", capture_pane=mock_capture):
            result = await provider.scrape_current_mode("@0")
        assert result == "Edit"


class TestScrapeCurrentModePlan:
    async def test_plan_mode(self):
        provider = ClaudeProvider()
        mock_capture = AsyncMock(return_value="some output\n⏸ plan mode  >")
        with patch("ccgram.multiplexer.multiplexer", capture_pane=mock_capture):
            result = await provider.scrape_current_mode("@0")
        assert result == "Plan"


class TestScrapeCurrentModeFull:
    async def test_yolo_mode(self):
        provider = ClaudeProvider()
        mock_capture = AsyncMock(return_value="some output\n⏵⏵ bypass permissions  >")
        with patch("ccgram.multiplexer.multiplexer", capture_pane=mock_capture):
            result = await provider.scrape_current_mode("@0")
        assert result == "YOLO"


class TestScrapeCurrentModeNone:
    async def test_no_mode_line(self):
        provider = ClaudeProvider()
        mock_capture = AsyncMock(return_value="just regular output\nno mode here")
        with patch("ccgram.multiplexer.multiplexer", capture_pane=mock_capture):
            result = await provider.scrape_current_mode("@0")
        assert result is None

    async def test_empty_capture(self):
        provider = ClaudeProvider()
        mock_capture = AsyncMock(return_value="")
        with patch("ccgram.multiplexer.multiplexer", capture_pane=mock_capture):
            result = await provider.scrape_current_mode("@0")
        assert result is None

    async def test_capture_failure(self):
        provider = ClaudeProvider()
        mock_capture = AsyncMock(side_effect=OSError("tmux gone"))
        with patch("ccgram.multiplexer.multiplexer", capture_pane=mock_capture):
            result = await provider.scrape_current_mode("@0")
        assert result is None


class TestShellScrapeCurrentModeDefault:
    async def test_returns_none(self):
        provider = ShellProvider()
        result = await provider.scrape_current_mode("@0")
        assert result is None


class TestFindModeLine:
    def test_finds_chrome_marker(self):
        pane = "output\n─────\n⏵⏵ auto-accept edits on  >"
        result = _find_mode_line(pane)
        assert result is not None
        assert "auto-accept" in result

    def test_returns_none_for_no_mode(self):
        assert _find_mode_line("just some text\nno markers") is None

    def test_hint_fallback(self):
        pane = "line1\nline2\nauto mode enabled\nlast"
        result = _find_mode_line(pane)
        assert result is not None
        assert "auto mode" in result


class TestModeShortLabel:
    @pytest.mark.parametrize(
        ("mode_line", "expected"),
        [
            ("⏵⏵ auto-accept edits on  >", "Edit"),
            ("⏸ plan mode  >", "Plan"),
            ("⏵⏵ bypass permissions  >", "YOLO"),
            ("⏵⏵ auto mode  >", "Auto"),
        ],
    )
    def test_known_labels(self, mode_line, expected):
        assert _mode_short_label(mode_line) == expected

    def test_unknown_returns_none(self):
        assert _mode_short_label("something weird") is None
