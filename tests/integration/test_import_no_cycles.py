"""Verify ``import ccgram`` and submodules succeed in a clean interpreter.

Regression guard for round-4 modularity decouple (F6.2): hoisting in-function
imports to module level can introduce import cycles that don't surface in the
in-process test suite (other tests warm caches first). A fresh subprocess
catches a cycle that the in-process suite would miss.

Round 5 expands the parametrize list from a hand-maintained set of 29 modules
to a programmatic walk that yields every top-level ``src/ccgram/*.py`` file
plus every package under ``src/ccgram/`` (one entry per ``__init__``).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "ccgram"


def _walk_package(pkg_path: Path, dotted: str) -> Iterator[str]:
    yield dotted
    for entry in sorted(pkg_path.iterdir()):
        if entry.name.startswith("_"):
            continue
        if entry.is_file() and entry.suffix == ".py":
            yield f"{dotted}.{entry.stem}"
        elif entry.is_dir() and (entry / "__init__.py").exists():
            yield from _walk_package(entry, f"{dotted}.{entry.name}")


def _enumerate_modules() -> Iterator[str]:
    yield from _walk_package(_SRC, "ccgram")


_MODULES = sorted(set(_enumerate_modules()))


@pytest.mark.parametrize("module", _MODULES)
def test_module_imports_in_clean_interpreter(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Importing {module} failed in a clean interpreter.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# F3: the multiplexer seam is enumerated above (the walk covers
# ``ccgram.multiplexer`` and its submodules); these assertions pin the package
# coverage and prove the core contract layer pulls in no backend.


def test_multiplexer_package_is_covered() -> None:
    expected = {
        "ccgram.multiplexer",
        "ccgram.multiplexer.base",
        "ccgram.multiplexer.tmux",
        "ccgram.multiplexer.herdr",
        "ccgram.multiplexer.registry",
        "ccgram.multiplexer.vim_state",
        "ccgram.multiplexer.window_ops",
    }
    assert expected <= set(_MODULES)


def test_multiplexer_base_imports_no_backend_in_clean_interpreter() -> None:
    """Importing ``multiplexer.base`` must not pull in a backend or libtmux."""
    probe = (
        "import sys; import ccgram.multiplexer.base; "
        "bad=[m for m in sys.modules "
        "if m.startswith('ccgram.multiplexer.tmux') "
        "or m.startswith('ccgram.multiplexer.herdr') "
        "or m=='libtmux']; "
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "multiplexer.base must import no backend/libtmux.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
