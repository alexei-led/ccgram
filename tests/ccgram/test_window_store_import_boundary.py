"""Enforce that handlers and Mini App do not import ``window_store`` for reads.

Task 6 hardens the F3 window-state boundary. Handler and Mini App modules
must reach window state through the read projections in ``window_query`` /
``session_query`` or through the feature ports in ``window_state_ports``.
Importing ``window_state_store.window_store`` (or the ``get_window_store``
proxy accessor) directly from a handler re-couples the handler to the full
``WindowStateStore`` API and reverses the strength reduction the feature
ports achieved.

Two narrow write/coordination seams are still allowed to import the store
directly:

- ``handlers/status/rc_probe.py`` — transient in-memory RC-probe state
  that intentionally lives in ``WindowState`` to avoid bringing in a new
  store, but is never persisted.
- ``handlers/commands/forward.py`` — calls
  ``window_store.clear_window_session(window_id)`` to coordinate a
  post-``/clear`` reset that also touches polling state and status
  bubbles. Lifting this into a port would require duplicating the
  coordination logic.

Constants re-exported from ``window_state_store`` (``CCGRAM_CREATED_WINDOW_ORIGIN``)
are not part of the store API and are allowed to be imported anywhere.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "ccgram"
HANDLERS_ROOT = SRC_ROOT / "handlers"
MINIAPP_ROOT = SRC_ROOT / "miniapp"

# Symbols on ``window_state_store`` that expose the store object itself.
# Anything else (constants, dataclasses, enums) is fine to import.
STORE_SYMBOLS: frozenset[str] = frozenset(
    {
        "window_store",
        "get_window_store",
        "install_window_store",
        "is_window_store_wired",
        "WindowStateStore",
    }
)

# Files allowed to import the raw store object. Each one is an
# explicitly named coordination seam — see module docstring.
ALLOWED_STORE_IMPORTERS: frozenset[Path] = frozenset(
    {
        HANDLERS_ROOT / "status" / "rc_probe.py",
        HANDLERS_ROOT / "commands" / "forward.py",
    }
)


def _iter_target_files() -> list[Path]:
    files: list[Path] = []
    for root in (HANDLERS_ROOT, MINIAPP_ROOT):
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.py")))
    return files


def _imports_store(tree: ast.AST) -> list[tuple[str, int]]:
    """Return (symbol, lineno) for each ``window_state_store`` import of the store."""
    hits: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not mod.endswith("window_state_store"):
                continue
            for alias in node.names:
                if alias.name in STORE_SYMBOLS:
                    hits.append((alias.name, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("window_state_store"):
                    hits.append((alias.name, node.lineno))
    return hits


@pytest.mark.parametrize("path", _iter_target_files(), ids=lambda p: p.name)
def test_handler_or_miniapp_does_not_import_window_store(path: Path) -> None:
    """Handlers and Mini App go through window_query / window_state_ports."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits = _imports_store(tree)
    if not hits:
        return
    assert path in ALLOWED_STORE_IMPORTERS, (
        f"{path.relative_to(SRC_ROOT.parent)} imports the raw window_store "
        f"({hits}). Reads must go through ccgram.window_query or "
        "ccgram.window_state_ports; writes must go through SessionManager "
        "or a window_state_ports write function. Add this path to "
        "ALLOWED_STORE_IMPORTERS only when the access is a documented "
        "coordination seam."
    )


def test_allowed_importers_exist() -> None:
    """Guard against rename drift in ALLOWED_STORE_IMPORTERS."""
    missing = sorted(str(p) for p in ALLOWED_STORE_IMPORTERS if not p.is_file())
    assert not missing, (
        f"ALLOWED_STORE_IMPORTERS contains paths that no longer exist: {missing}"
    )


def test_at_least_one_target_file_exists() -> None:
    """Sanity check: the walker covers handlers and miniapp."""
    assert _iter_target_files(), "expected handlers/miniapp files to scan"
