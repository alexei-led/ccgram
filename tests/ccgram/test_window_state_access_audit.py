"""Baseline audit of raw WindowState field access across src/ccgram.

Captures the current footprint so subsequent feature-port migrations can
shrink it. The audit walks every ``.py`` under ``src/ccgram`` (excluding
the persistence kernel, the SessionManager facade, and the existing read
projection layer) and reports each direct read or write of a
``WindowState`` feature field on receivers that look syntactically like
``WindowState`` instances.

Two assertions:

1. Every reported (path, field, kind) site is present in
   ``BASELINE_RAW_ACCESS`` — flags any new direct field access added
   outside the approved seams.
2. ``BASELINE_RAW_ACCESS`` contains no stale entries — flags fields the
   migration has already lifted.

Kind is split between ``read`` and ``write`` so later tasks can shrink
the read column without quietly hiding any write coordination.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "ccgram"

# Files allowed to touch raw WindowState fields without going through a
# feature port. ``window_state_store`` owns persistence, ``session`` is
# the SessionManager facade, ``window_query`` is the existing read
# projection layer.
EXCLUDED_FILES: frozenset[Path] = frozenset(
    {
        SRC_ROOT / "window_state_store.py",
        SRC_ROOT / "session.py",
        SRC_ROOT / "window_query.py",
        # Feature ports — the approved seam between handlers and raw WindowState
        # fields. Pyport access here is intentional.
        SRC_ROOT / "window_state_ports" / "__init__.py",
        SRC_ROOT / "window_state_ports" / "pane_state.py",
        SRC_ROOT / "window_state_ports" / "identity_state.py",
        SRC_ROOT / "window_state_ports" / "worktree_state.py",
        SRC_ROOT / "window_state_ports" / "tool_state.py",
        SRC_ROOT / "window_state_ports" / "lifecycle_state.py",
    }
)

# WindowState feature fields tracked by the audit. The PaneInfo dict is
# accessed via the ``panes`` field — pane field reads are not part of
# the audit because PaneInfo attributes (``state``, ``name``, ...) clash
# with too many unrelated identifiers to be reported reliably with AST
# alone.
WINDOW_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "session_id",
        "cwd",
        "window_name",
        "transcript_path",
        "provider_name",
        "approval_mode",
        "batch_mode",
        "tool_call_visibility",
        "external",
        "origin",
        "panes",
        "pane_lifecycle_notify",
        "rc_probe_state",
        "rc_armed_at",
        "worktree_path",
        "worktree_branch",
        "gemini_external_warned",
    }
)


def _is_ws_source_expr(node: ast.AST) -> bool:
    """True when ``node`` syntactically yields a WindowState instance.

    Covers:

    - ``<x>.window_states[<key>]``
    - ``<x>.window_states.get(<key>, ...)``
    - ``<x>.get_window_state(<args>)``
    """
    if isinstance(node, ast.Subscript):
        val = node.value
        if isinstance(val, ast.Attribute) and val.attr == "window_states":
            return True
        if isinstance(val, ast.Name) and val.id == "window_states":
            return True
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "get_window_state":
                return True
            if (
                func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "window_states"
            ):
                return True
    return False


def _add_target_attrs(target: ast.AST, write_ids: set[int]) -> None:
    for sub in ast.walk(target):
        if isinstance(sub, ast.Attribute):
            write_ids.add(id(sub))


def _collect_write_targets(scope: ast.AST) -> set[int]:
    """Return ``id(node)`` for every Attribute used as an assignment target."""
    write_ids: set[int] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                _add_target_attrs(tgt, write_ids)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            if isinstance(node.target, ast.Attribute):
                write_ids.add(id(node.target))
        elif isinstance(node, ast.Delete):
            for tgt in node.targets:
                _add_target_attrs(tgt, write_ids)
    return write_ids


def _collect_ws_bindings(scope: ast.AST) -> set[str]:
    """Collect local names assigned from a ws-source expression."""
    bound: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and _is_ws_source_expr(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_ws_source_expr(node.value)
            and isinstance(node.target, ast.Name)
        ):
            bound.add(node.target.id)
    return bound


def _audit_file(path: Path) -> list[tuple[str, str, int, str]]:
    """Return list of (path_str, field, lineno, kind) hits for ``path``."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    rel = str(path.relative_to(SRC_ROOT.parent))

    # Treat the whole module as one scope. Local-variable shadowing
    # inside nested functions is rare in this codebase and acceptable
    # for a baseline audit; later tasks tighten the boundary instead of
    # relying on perfect scope tracking here.
    bound = _collect_ws_bindings(tree)
    write_ids = _collect_write_targets(tree)

    hits: list[tuple[str, str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in WINDOW_STATE_FIELDS:
            continue
        recv = node.value
        matched = False
        if isinstance(recv, ast.Name) and recv.id in bound or _is_ws_source_expr(recv):
            matched = True
        if not matched:
            continue
        kind = "write" if id(node) in write_ids else "read"
        hits.append((rel, node.attr, node.lineno, kind))
    return hits


def _iter_audited_files() -> list[Path]:
    paths: list[Path] = []
    for p in sorted(SRC_ROOT.rglob("*.py")):
        if p in EXCLUDED_FILES:
            continue
        paths.append(p)
    return paths


def _collect_all_hits() -> list[tuple[str, str, int, str]]:
    out: list[tuple[str, str, int, str]] = []
    for p in _iter_audited_files():
        out.extend(_audit_file(p))
    return out


# ── Baseline allowlist ────────────────────────────────────────────────
#
# Keys: (path_relative_to_src, field, kind).
# Values: a short comment naming the consumer for human readability.
#
# When a feature-port migration lands, remove the corresponding
# (path, field, kind) entries from this map; the second test asserts
# that no stale entries linger.
BASELINE_RAW_ACCESS: dict[tuple[str, str, str], str] = {
    # ── handlers/status/rc_probe.py ──────────────────────────────────
    ("ccgram/handlers/status/rc_probe.py", "rc_probe_state", "read"): (
        "rc probe lifecycle"
    ),
    ("ccgram/handlers/status/rc_probe.py", "rc_probe_state", "write"): (
        "rc probe lifecycle"
    ),
    ("ccgram/handlers/status/rc_probe.py", "rc_armed_at", "write"): (
        "rc probe lifecycle"
    ),
    # ── session_map.py (coordination seam between hook data and state) ──
    ("ccgram/session_map.py", "cwd", "read"): "session map sync",
    ("ccgram/session_map.py", "cwd", "write"): "session map sync",
    ("ccgram/session_map.py", "external", "read"): "session map sync",
    ("ccgram/session_map.py", "external", "write"): "session map sync",
    ("ccgram/session_map.py", "origin", "read"): "session map sync",
    ("ccgram/session_map.py", "origin", "write"): "session map sync",
    ("ccgram/session_map.py", "provider_name", "read"): "session map sync",
    ("ccgram/session_map.py", "provider_name", "write"): "session map sync",
    ("ccgram/session_map.py", "session_id", "read"): "session map sync",
    ("ccgram/session_map.py", "session_id", "write"): "session map sync",
    ("ccgram/session_map.py", "transcript_path", "read"): "session map sync",
    ("ccgram/session_map.py", "transcript_path", "write"): "session map sync",
    ("ccgram/session_map.py", "window_name", "read"): "session map sync",
    ("ccgram/session_map.py", "window_name", "write"): "session map sync",
}


def _grouped_hits() -> dict[tuple[str, str, str], list[int]]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for rel, field, lineno, kind in _collect_all_hits():
        grouped[(rel, field, kind)].append(lineno)
    return grouped


def test_no_new_raw_access_sites_added() -> None:
    """Every (path, field, kind) currently in source must be in the baseline."""
    grouped = _grouped_hits()
    unexpected = {
        key: lines for key, lines in grouped.items() if key not in BASELINE_RAW_ACCESS
    }
    assert not unexpected, (
        "New raw WindowState field access detected outside the approved seams. "
        "Migrate the new site to a window_state_ports feature port, or — if it "
        "is genuinely a coordination seam — add an explicit entry to "
        "BASELINE_RAW_ACCESS with a one-line justification. New hits:\n"
        + "\n".join(
            f"  {key} at lines {lines}" for key, lines in sorted(unexpected.items())
        )
    )


def test_no_stale_baseline_entries() -> None:
    """Every baseline entry must still correspond to a real access site."""
    grouped = _grouped_hits()
    stale = sorted(key for key in BASELINE_RAW_ACCESS if key not in grouped)
    assert not stale, (
        "Stale BASELINE_RAW_ACCESS entries — these (path, field, kind) tuples "
        "no longer match any source access and should be deleted:\n"
        + "\n".join(f"  {key}" for key in stale)
    )


def test_baseline_keys_use_known_fields() -> None:
    """Catch typos in the baseline by validating field names and kinds."""
    bad_fields = sorted(
        key for key in BASELINE_RAW_ACCESS if key[1] not in WINDOW_STATE_FIELDS
    )
    bad_kinds = sorted(
        key for key in BASELINE_RAW_ACCESS if key[2] not in {"read", "write"}
    )
    assert not bad_fields, f"Unknown field names in baseline: {bad_fields}"
    assert not bad_kinds, f"Unknown access kinds in baseline: {bad_kinds}"


def test_at_least_one_audited_file_exists() -> None:
    """Sanity check: the walker actually finds files."""
    files = _iter_audited_files()
    assert len(files) > 50, f"expected many audited files, got {len(files)}"


@pytest.mark.parametrize(
    "excluded",
    sorted(str(p.relative_to(SRC_ROOT)) for p in EXCLUDED_FILES),
)
def test_excluded_files_exist(excluded: str) -> None:
    """Guard against rename drift in the excluded set."""
    assert (SRC_ROOT / excluded).is_file(), (
        f"Excluded file {excluded} missing — update EXCLUDED_FILES."
    )
