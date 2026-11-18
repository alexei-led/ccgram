"""The keyboard leaf may depend on pure models, never stateful handlers."""

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
    assert imports == [(2, "callback_data"), (3, "providers.base")]
    assert not any(
        alias.name.startswith("ccgram")
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
