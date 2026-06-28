"""Fitness gate (T1): event/session-map I/O paths route through hooks.state_files.

Verifies that:
- ccgram.event_reader imports parse_event_record from ccgram.hooks.state_files
- ccgram.session_map imports parse_session_map_entry from ccgram.hooks.state_files
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent.parent.parent / "src" / "ccgram"


def _collect_imports_from(tree: ast.AST, module_suffix: str) -> list[str]:
    """Return names imported from any module ending with module_suffix."""
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module and node.module.endswith(module_suffix):
            names.extend(alias.name for alias in node.names)
    return names


def test_event_reader_imports_parse_event_record_from_state_files() -> None:
    """event_reader.py must import parse_event_record from hooks.state_files."""
    source = (SRC_ROOT / "event_reader.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="event_reader.py")
    imported = _collect_imports_from(tree, "hooks.state_files")
    assert "parse_event_record" in imported, (
        "event_reader.py must import parse_event_record from hooks.state_files; "
        f"currently imports: {imported}"
    )


def test_session_map_imports_parse_session_map_entry_from_state_files() -> None:
    """session_map.py must import parse_session_map_entry from hooks.state_files."""
    source = (SRC_ROOT / "session_map.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="session_map.py")
    imported = _collect_imports_from(tree, "hooks.state_files")
    assert "parse_session_map_entry" in imported, (
        "session_map.py must import parse_session_map_entry from hooks.state_files; "
        f"currently imports: {imported}"
    )
