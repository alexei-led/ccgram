"""Tests for screen_buffer — pyte-based VT100 screen rendering."""

from ccgram.screen_buffer import ScreenBuffer


class TestScreenBufferInit:
    def test_default_dimensions(self):
        buf = ScreenBuffer()
        assert buf.columns == 200
        assert buf.rows == 50

    def test_custom_dimensions(self):
        buf = ScreenBuffer(columns=80, rows=24)
        assert buf.columns == 80
        assert buf.rows == 24


class TestFeedAndDisplay:
    def test_plain_text(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("Hello, world!")
        lines = buf.display
        assert lines[0] == "Hello, world!"

    def test_ansi_colors_stripped(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("\x1b[31mred text\x1b[0m normal")
        lines = buf.display
        assert lines[0] == "red text normal"

    def test_ansi_bold_stripped(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("\x1b[1mbold\x1b[0m plain")
        assert buf.display[0] == "bold plain"

    def test_multiline_feed(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("line one\r\nline two\r\nline three")
        assert buf.display[0] == "line one"
        assert buf.display[1] == "line two"
        assert buf.display[2] == "line three"

    def test_empty_lines_are_empty_strings(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("first")
        for line in buf.display[1:]:
            assert line == ""

    def test_trailing_whitespace_stripped(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("text")
        raw = buf._screen.display[0]
        assert len(raw) == 40
        assert buf.display[0] == "text"


class TestCursorPosition:
    def test_cursor_after_text(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("hello")
        assert buf.cursor_row == 0

    def test_cursor_after_newline(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("hello\r\n")
        assert buf.cursor_row == 1


class TestReset:
    def test_reset_clears_content(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("some content")
        buf.reset()
        assert all(line == "" for line in buf.display)

    def test_reset_resets_cursor(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("hello\r\nworld")
        buf.reset()
        assert buf.cursor_row == 0


class TestSequentialFeeds:
    def test_incremental_feed_accumulates(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("hello")
        buf.feed(" world")
        assert buf.display[0] == "hello world"

    def test_reset_then_feed(self):
        buf = ScreenBuffer(columns=40, rows=5)
        buf.feed("old content")
        buf.reset()
        buf.feed("new content")
        assert buf.display[0] == "new content"


class TestRealWorldCapture:
    """Test with content resembling real Claude Code terminal output."""

    def test_claude_status_with_ansi(self):
        sep = "─" * 30
        raw = (
            "some output\r\n"
            "\x1b[36m✻ Reading file\x1b[0m\r\n"
            f"{sep}\r\n"
            "❯ \r\n"
            f"{sep}\r\n"
            "  \x1b[90m[Opus 4.6]\x1b[0m Context: 34%"
        )
        buf = ScreenBuffer(columns=80, rows=10)
        buf.feed(raw)
        lines = buf.display

        assert "✻ Reading file" in lines[1]
        assert "\x1b" not in lines[1]
        assert lines[2] == sep
        assert lines[4] == sep

    def test_interactive_ui_checkboxes(self):
        raw = (
            "  ☐ Option A\r\n"
            "  \x1b[1m✔ Option B\x1b[0m\r\n"
            "  ☐ Option C\r\n"
            "  Enter to select"
        )
        buf = ScreenBuffer(columns=80, rows=10)
        buf.feed(raw)
        lines = buf.display

        assert "☐ Option A" in lines[0]
        assert "✔ Option B" in lines[1]
        assert "Enter to select" in lines[3]
