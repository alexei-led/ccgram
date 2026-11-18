from unittest.mock import MagicMock

import pytest

from ccgram.multiplexer.tmux import TmuxManager


@pytest.mark.parametrize(
    ("text", "literal", "expected", "wire_literal"),
    [
        ("S-Enter", False, "\x1b[13;2u", True),
        ("Enter", False, "Enter", False),
        ("S-Enter", True, "S-Enter", True),
    ],
)
def test_modified_enter_preserves_editor_newlines(
    text, literal, expected, wire_literal
):
    pane = MagicMock()
    TmuxManager._send_pane_input(pane, text, enter=False, literal=literal)
    pane.send_keys.assert_called_once_with(expected, enter=False, literal=wire_literal)
