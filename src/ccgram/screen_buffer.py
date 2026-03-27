"""VT100 screen buffer — wraps pyte for clean terminal text rendering.

Feeds raw tmux pane captures (with ANSI escape sequences) into a pyte
virtual terminal, producing clean rendered lines stripped of control codes.
Used by terminal_parser.py for robust status and interactive UI detection.

Key class: ScreenBuffer — create, feed raw text, read rendered lines.
"""

import structlog
import pyte

logger = structlog.get_logger()


class ScreenBuffer:
    """Virtual terminal screen backed by pyte.

    Wraps a pyte Screen + Stream to accept raw terminal text and
    expose clean rendered output lines, cursor position, and separator
    detection.
    """

    def __init__(self, columns: int = 200, rows: int = 50) -> None:
        self._screen = pyte.Screen(columns, rows)
        self._stream = pyte.Stream(self._screen)

    @property
    def columns(self) -> int:
        return self._screen.columns

    @property
    def rows(self) -> int:
        return self._screen.lines

    def feed(self, raw_text: str) -> None:
        """Feed raw terminal text (with ANSI escapes) into the screen."""
        try:
            self._stream.feed(raw_text)
        except TypeError, ValueError, KeyError, IndexError, UnicodeDecodeError:
            logger.debug("pyte feed error, resetting screen", exc_info=True)
            self._screen.reset()

    @property
    def display(self) -> list[str]:
        """Rendered lines with trailing whitespace stripped."""
        return [line.rstrip() for line in self._screen.display]

    @property
    def rendered_text(self) -> str:
        """Full rendered text with trailing blank lines trimmed."""
        lines = self.display
        last = len(lines) - 1
        while last >= 0 and not lines[last].strip():
            last -= 1
        return "\n".join(lines[: last + 1]) if last >= 0 else ""

    @property
    def cursor_row(self) -> int:
        return self._screen.cursor.y

    def resize(self, columns: int, rows: int) -> None:
        """Resize the screen and clear content."""
        if columns < 1 or rows < 1:
            return
        self._screen.resize(rows, columns)
        self._screen.reset()

    def reset(self) -> None:
        """Clear all screen state for reuse."""
        self._screen.reset()
