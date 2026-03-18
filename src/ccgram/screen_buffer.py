"""VT100 screen buffer — wraps pyte for clean terminal text rendering.

Feeds raw tmux pane captures (with ANSI escape sequences) into a pyte
virtual terminal, producing clean rendered lines stripped of control codes.
Used by terminal_parser.py for robust status and interactive UI detection.

Key class: ScreenBuffer — create, feed raw text, read rendered lines.
"""

import pyte


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
        self._stream.feed(raw_text)

    @property
    def display(self) -> list[str]:
        """Rendered lines with trailing whitespace stripped."""
        return [line.rstrip() for line in self._screen.display]

    @property
    def cursor_row(self) -> int:
        return self._screen.cursor.y

    def reset(self) -> None:
        """Clear all screen state for reuse."""
        self._screen.reset()
