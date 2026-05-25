"""Audit: only the typed backend config may read ``CCGRAM_TERMINAL_BACKEND``
or ``CCGRAM_CMUX_*`` env vars.

Mirrors :mod:`tests.ccgram.test_window_state_access_audit`. Reading
these env names anywhere outside :mod:`ccgram.terminal_backends.config`
(and the future bootstrap glue) would let cmux configuration sprawl
through handlers, undoing the typed-projection boundary established in
Task 3 of the cmux sidecar plan.

The audit walks every ``.py`` under ``src/ccgram`` and looks for two
shapes:

1. ``os.getenv("CCGRAM_TERMINAL_BACKEND…")`` / ``os.environ.get(...)``
   calls whose key string matches the audited prefixes.
2. Any string literal that exactly matches one of the audited env var
   names — covers ``os.environ[...]`` lookups and direct compares.

Tests in ``tests/`` and the audited config module itself are skipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "ccgram"

EXCLUDED_FILES: frozenset[Path] = frozenset(
    {
        SRC_ROOT / "terminal_backends" / "config.py",
        SRC_ROOT / "terminal_backends" / "__init__.py",
    }
)

AUDITED_PREFIXES: tuple[str, ...] = (
    "CCGRAM_TERMINAL_BACKEND",
    "CCGRAM_CMUX_",
)


def _is_audited_name(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in AUDITED_PREFIXES)


def _audit_file(path: Path) -> list[tuple[str, int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    hits: list[tuple[str, int, str]] = []
    rel = str(path.relative_to(SRC_ROOT.parent))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _is_audited_name(node.value)
        ):
            hits.append((rel, node.lineno, node.value))
    return hits


def _iter_audited_files() -> list[Path]:
    return [p for p in sorted(SRC_ROOT.rglob("*.py")) if p not in EXCLUDED_FILES]


def test_no_cmux_env_reads_outside_config_module() -> None:
    """No source file outside the typed config module references the
    ``CCGRAM_TERMINAL_BACKEND`` / ``CCGRAM_CMUX_*`` env vars."""
    hits: list[tuple[str, int, str]] = []
    for path in _iter_audited_files():
        hits.extend(_audit_file(path))

    assert not hits, (
        "Direct cmux/terminal-backend env var access detected outside "
        "ccgram/terminal_backends/config.py. Route the read through "
        "ccgram.terminal_backends.config.load_terminal_backend_config() "
        "and consume the typed TerminalBackendConfig instead. Hits:\n"
        + "\n".join(f"  {rel}:{lineno}  {name!r}" for rel, lineno, name in hits)
    )


def test_excluded_files_exist() -> None:
    """Guard against rename drift in the excluded set."""
    for path in EXCLUDED_FILES:
        assert path.is_file(), f"audited-config excluded path missing: {path}"


def test_audited_prefixes_are_documented() -> None:
    """Catch typos by asserting prefix shape."""
    for prefix in AUDITED_PREFIXES:
        assert prefix.startswith("CCGRAM_"), prefix
