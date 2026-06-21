"""No-tty drift gate — shell + provider detection never reach past the seam.

Task 1 of the herdr-shell-provider plan made ``Multiplexer.foreground(window_id)``
the single source of foreground-process truth and moved ``ps -t <tty>`` into the
tmux backend as a private detail. This audit (modelled on
``test_window_state_access_audit.py`` / ``test_multiplexer_boundary.py``) locks
that boundary: outside the tmux backend — and the separate Claude-Code hook
subprocess that bootstraps tmux identity — no module under ``src/ccgram`` may

- read a pane/identity ``.pane_tty`` attribute (a caller dereferencing a tty),
- reference the removed ``get_foreground_args`` ps-helper, or
- fork ``ps -t`` directly (the ``["ps", "-t", ...]`` argv).

Why ``.pane_tty`` (the *dotted attribute read*) and not the bare token
``pane_tty``: the neutral seam type names the field (``WindowRef.pane_tty``) and
the sanctioned capability gate is ``MultiplexerCapabilities.exposes_pane_tty`` —
shell/detection code is *expected* to branch on ``caps.exposes_pane_tty`` (Task
3). A bare-token forbid would false-positive on that gate. The leak we forbid is
a caller reaching into an object to pull its tty (``window.pane_tty``), the
``ps -t`` syscall, and the dead ps-helper name.

Ceiling: the ``ps -t`` check matches the argv form (``"ps", "-t"``) this
codebase actually uses, not prose like ``"reads ``ps -t``"`` in a docstring. A
hypothetical ``shell=True`` ``"ps -t ..."`` string would slip the ps check, but
it still needs a tty — and the only tty source is ``.pane_tty``, which the
attribute gate catches. Upgrade trigger: a non-argv ps invocation appears.

Allow-list — files that legitimately own foreground/tty internals:

- ``multiplexer/tmux.py`` — the tmux backend; owns ``pane_tty`` + ``ps -t``.
- ``hook.py`` — the separate Claude-Code hook subprocess that resolves tmux
  identity (nested-session detection, provider-from-tty) before the runtime
  seam is wired; tmux-specific and orthogonal to the foreground seam.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "ccgram"

# Files allowed to touch tty/ps foreground internals directly.
ALLOWED_FILES: frozenset[Path] = frozenset(
    {
        SRC_ROOT / "multiplexer" / "tmux.py",
        SRC_ROOT / "hook.py",
    }
)

# The removed ps-helper name — reintroducing it (even in a docstring) is drift.
_GET_FG_ARGS = re.compile(r"\bget_foreground_args\b")

# The ``ps -t`` syscall in argv form: ``"ps", "-t"`` / ``'ps', '-t'``.
_PS_T_ARGV = re.compile(r"""['"]ps['"]\s*,\s*['"]-t['"]""")


def _offenders_in_source(source: str, rel: str) -> list[tuple[int, str]]:
    """Return ``(lineno, reason)`` for each no-tty violation in *source*."""
    offenders: list[tuple[int, str]] = []

    # AST: a dotted attribute read of a tty (``window.pane_tty``). The
    # capability gate ``caps.exposes_pane_tty`` has a different ``attr`` and is
    # intentionally not matched; the field *definition* (``pane_tty: str``) is an
    # AnnAssign target, not an Attribute, so the seam type itself stays clean.
    tree = ast.parse(source, filename=rel)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "pane_tty":
            offenders.append(
                (node.lineno, "reads .pane_tty (use Multiplexer.foreground)")
            )

    # Source-text: the dead helper name and the ps-by-tty argv.
    for lineno, line in enumerate(source.splitlines(), start=1):
        if _GET_FG_ARGS.search(line):
            offenders.append((lineno, "references get_foreground_args (removed)"))
        if _PS_T_ARGV.search(line):
            offenders.append((lineno, "forks `ps -t` (use the foreground seam)"))

    return offenders


def _audited_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if p not in ALLOWED_FILES)


# ── Enforced gate ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path", _audited_files(), ids=lambda p: str(p.relative_to(SRC_ROOT))
)
def test_no_tty_dependency_outside_backend(path: Path) -> None:
    offenders = _offenders_in_source(path.read_text(encoding="utf-8"), str(path))
    assert not offenders, (
        f"{path.relative_to(SRC_ROOT)} reaches past the multiplexer seam for "
        "foreground/tty info:\n"
        + "\n".join(f"  line {ln}: {why}" for ln, why in offenders)
        + "\nGet the foreground process via Multiplexer.foreground(window_id) and "
        "gate tty-dependent branches on caps.exposes_pane_tty instead."
    )


def test_allow_list_files_exist() -> None:
    """Guard against rename drift in the allow-list."""
    for path in ALLOWED_FILES:
        assert path.is_file(), f"Allow-listed file missing: {path}"


def test_audit_actually_walks_files() -> None:
    """Sanity check: the walker finds the codebase, not an empty set."""
    assert len(_audited_files()) > 50


# ── Self-test: the gate catches planted violations ─────────────────────


def test_audit_flags_planted_pane_tty_read() -> None:
    planted = "def leak(window):\n    return window.pane_tty\n"
    offenders = _offenders_in_source(planted, "fake.py")
    assert [why for _, why in offenders] == [
        "reads .pane_tty (use Multiplexer.foreground)"
    ]


def test_audit_flags_planted_get_foreground_args() -> None:
    planted = "from .process_detection import get_foreground_args\nx = get_foreground_args(tty)\n"
    offenders = _offenders_in_source(planted, "fake.py")
    assert any("get_foreground_args" in why for _, why in offenders)


def test_audit_flags_planted_ps_t_argv() -> None:
    planted = 'args = ["ps", "-t", tty, "-o", "command="]\n'
    offenders = _offenders_in_source(planted, "fake.py")
    assert [why for _, why in offenders] == ["forks `ps -t` (use the foreground seam)"]


def test_audit_allows_capability_gate() -> None:
    """The sanctioned ``caps.exposes_pane_tty`` gate (Task 3) is not a leak."""
    clean = (
        "def detect(caps, window):\n"
        "    if not caps.exposes_pane_tty:\n"
        "        return None\n"
        "    return window\n"
    )
    assert _offenders_in_source(clean, "fake.py") == []


def test_audit_allows_prose_and_field_definition() -> None:
    """Docstrings mentioning ``ps -t``/``pane_tty`` and the field def are clean."""
    clean = (
        '"""Foreground via ``pane_tty`` + ``ps -t <tty>`` on tmux."""\n'
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class WindowRef:\n"
        '    pane_tty: str = ""\n'
    )
    assert _offenders_in_source(clean, "fake.py") == []
