"""The Pi keyboard contract must remain a leaf, not join handler import cycles."""

import ast
from pathlib import Path

from ccgram.handlers.interactive import pi_controls


def test_pi_keyboard_has_no_stateful_ccgram_dependencies():
    tree = ast.parse(Path(pi_controls.__file__).read_text())
    imports = [
        (node.level, node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.level or (node.module or "").startswith("ccgram"))
    ]
    assert imports == [(2, "callback_data")]
    assert not any(
        alias.name.startswith("ccgram")
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
