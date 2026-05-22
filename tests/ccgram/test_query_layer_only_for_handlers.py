"""Structural test enforcing the read-path/write-path split for handlers.

Round-5 Task 2 invariant: handlers depend on a small read contract
(``window_query`` / ``session_query``) for read-only state and on
``SessionManager``'s explicit write/admin surface for mutations. A read
access slipping back onto ``session_manager.*`` is a hard fail — that
re-couples handlers to the full SessionManager surface and reverses the
strength reduction Task 2 achieved.

The test walks every ``.py`` file under ``src/ccgram/handlers/`` with
``ast``, finds every ``Attribute`` node whose ``.value`` is
``Name('session_manager')``, and asserts the attribute is in the
allow-list below. Each entry is a write or admin method on
``SessionManager`` — anything else (``view_window``, ``window_states``,
``iter_window_ids``, ``get_approval_mode`` etc.) must go through the
query layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ── Write/admin allow-list ────────────────────────────────────────────
#
# These attributes are the documented mutation surface of SessionManager
# and may be accessed directly from handler modules. Every other
# attribute access on ``session_manager`` is a read and must go through
# ``window_query`` / ``session_query``.
#
# Counts after Task 2 (verified via grep):
#   set_window_provider × 10
#   set_window_origin × 4
#   set_window_approval_mode × 3
#   set_window_cwd × 1
#   set_window_worktree × 1
#   set_display_name × 1
#   cycle_batch_mode × 1
#   cycle_tool_call_visibility × 1
#   sync_display_names × 2
#   prune_stale_state × 2
#   prune_stale_window_states × 1
#   audit_state × 3
ALLOWED_SESSION_MANAGER_ATTRS: frozenset[str] = frozenset(
    {
        "set_window_provider",
        "set_window_origin",
        "set_window_approval_mode",
        "set_window_cwd",
        "set_window_worktree",
        "set_display_name",
        "set_batch_mode",
        "set_tool_call_visibility",
        "cycle_batch_mode",
        "cycle_tool_call_visibility",
        "sync_display_names",
        "prune_stale_state",
        "prune_stale_window_states",
        "audit_state",
    }
)


HANDLERS_ROOT = Path(__file__).resolve().parents[2] / "src" / "ccgram" / "handlers"


def _iter_handler_files() -> list[Path]:
    return sorted(HANDLERS_ROOT.rglob("*.py"))


def _collect_session_manager_attrs(tree: ast.AST) -> list[tuple[str, int]]:
    """Return (attr_name, lineno) for every ``session_manager.X`` access."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        value = node.value
        if isinstance(value, ast.Name) and value.id == "session_manager":
            found.append((node.attr, node.lineno))
    return found


@pytest.mark.parametrize("path", _iter_handler_files(), ids=lambda p: p.name)
def test_handler_session_manager_access_is_write_or_admin(path: Path) -> None:
    """Every ``session_manager.X`` access in handlers must be a write/admin call."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations = [
        (attr, lineno)
        for attr, lineno in _collect_session_manager_attrs(tree)
        if attr not in ALLOWED_SESSION_MANAGER_ATTRS
    ]
    assert not violations, (
        f"{path.relative_to(HANDLERS_ROOT.parents[1])} accesses session_manager "
        "for reads — migrate to ccgram.window_query / ccgram.session_query. "
        f"Violations: {violations}"
    )


def test_allow_list_only_contains_real_session_manager_methods() -> None:
    """Every entry in the allow-list must exist on SessionManager.

    Catches typos and refactor drift — if a method is removed or
    renamed, the allow-list must follow.
    """
    from ccgram.session import SessionManager

    public = {name for name in dir(SessionManager) if not name.startswith("_")}
    missing = ALLOWED_SESSION_MANAGER_ATTRS - public
    assert not missing, (
        f"Allow-list contains entries not on SessionManager: {sorted(missing)}"
    )
