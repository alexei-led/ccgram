"""Fitness gate: handlers must not import session-state internals for reads.

Task 2 invariant: volatile session-state reads (task snapshot, wait header,
has-snapshot check) must go through ``session_state_ports.live_session_state``.
Handlers that need these reads had a direct dependency on
``claude_task_state`` or ``session_lifecycle`` — this test catches regressions.

**What is banned** (read symbols that the port now owns):
  - ``get_claude_task_snapshot``  (use ``session_state_ports.live_session_state.get_task_snapshot``)
  - ``get_claude_wait_header``    (use ``session_state_ports.live_session_state.get_wait_header``)
  - ``claude_task_state.has_snapshot``       (attr-access style; use ``has_task_snapshot``)
  - ``claude_task_state.get_snapshot``       (attr-access style; use ``get_task_snapshot``)
  - ``claude_task_state.get_claude_task_snapshot``  (attr-access style)
  - ``claude_task_state.get_claude_wait_header``    (attr-access style)
  - ``session_lifecycle.resolve_session_id`` (attr-access style; use ``live_session_state.get_session_id``)

**What is explicitly allowed** (write/mutation authority):
  From ``claude_task_state``:
    ``classify_wait_message``, ``claude_task_state`` (for format_completion_text,
    clear_wait_header, set_last_status — write methods called by mutation paths),
    ``build_subagent_label``, ``get_subagent_names``, ``add_subagent``,
    ``remove_subagent``, ``clear_subagents``, ``clear_claude_task_window``,
    ``IDLE_STATUS_TEXT``, ``ClaudeTaskItem``, ``ClaudeTaskSnapshot``

  From ``session_lifecycle``:
    ``session_lifecycle`` (mutation methods: handle_notification_wait,
    handle_stop_task_state, handle_subagent_start/stop, handle_session_end,
    handle_task_completed, initialize, reconcile)

  From ``session_map`` / ``session_map_sync``:
    All — session_map_sync is write/admin-only in handler contexts.

The test detects the banned import names in the module-level ``import``
statements of every Python file under ``src/ccgram/handlers/``.  It does
not ban the modules themselves (they have legitimate write callers).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ── Banned read symbols ────────────────────────────────────────────────────
#
# These are the specific names handlers must no longer import from the
# session-state internals. The port owns them now.
#
BANNED_FROM_CLAUDE_TASK_STATE: frozenset[str] = frozenset(
    {
        "get_claude_task_snapshot",
        "get_claude_wait_header",
    }
)

# Method-call style ban: ``claude_task_state.<attr>(...)`` — detected as an
# Attribute access on a Name node named "claude_task_state" with attr in this
# set.  Any import of ``claude_task_state`` is fine as long as it's only used
# for write methods; this check catches the read-path attr forms.
BANNED_ATTRIBUTE_READS: frozenset[str] = frozenset(
    {
        "has_snapshot",
        "get_snapshot",
        "get_claude_task_snapshot",  # attr-access form
        "get_claude_wait_header",  # attr-access form
    }
)

# Attr-access ban for session_lifecycle: handlers must not call
# session_lifecycle.resolve_session_id() directly; use
# session_state_ports.live_session_state.get_session_id() instead.
BANNED_SESSION_LIFECYCLE_ATTR_READS: frozenset[str] = frozenset({"resolve_session_id"})

# ── File scope ─────────────────────────────────────────────────────────────

HANDLERS_ROOT = Path(__file__).resolve().parents[2] / "src" / "ccgram" / "handlers"
# Scope: handlers/ only. miniapp/ was checked manually and contains no banned reads.
# If miniapp/ grows session-state read paths, extend _iter_handler_files() to include it.


def _iter_handler_files() -> list[Path]:
    return sorted(HANDLERS_ROOT.rglob("*.py"))


# ---------------------------------------------------------------------------
# Banned import-name check
# ---------------------------------------------------------------------------


def _collect_banned_imports(
    tree: ast.AST, banned: frozenset[str]
) -> list[tuple[str, int]]:
    """Return (name, lineno) for every top-level import of a banned symbol."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            if name in banned:
                found.append((name, node.lineno))
    return found


@pytest.mark.parametrize("path", _iter_handler_files(), ids=lambda p: p.name)
def test_no_banned_read_imports_from_claude_task_state(path: Path) -> None:
    """Handlers must not import session-state read symbols directly."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations = _collect_banned_imports(tree, BANNED_FROM_CLAUDE_TASK_STATE)
    assert not violations, (
        f"{path.relative_to(HANDLERS_ROOT.parents[1])} imports banned read "
        "symbols — use session_state_ports.live_session_state instead. "
        f"Violations: {violations}"
    )


# ---------------------------------------------------------------------------
# Banned attribute-read check: claude_task_state.has_snapshot(...)
# ---------------------------------------------------------------------------


def _collect_banned_attr_reads(
    tree: ast.AST, obj_name: str, banned_attrs: frozenset[str]
) -> list[tuple[str, int]]:
    """Return (attr, lineno) for ``obj_name.attr`` where attr is banned."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id == obj_name and node.attr in banned_attrs:
            found.append((node.attr, node.lineno))
    return found


@pytest.mark.parametrize("path", _iter_handler_files(), ids=lambda p: p.name)
def test_no_banned_read_attr_calls_on_claude_task_state(path: Path) -> None:
    """Handlers must not call banned claude_task_state read attrs directly."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations = _collect_banned_attr_reads(
        tree, "claude_task_state", BANNED_ATTRIBUTE_READS
    )
    assert not violations, (
        f"{path.relative_to(HANDLERS_ROOT.parents[1])} calls banned "
        "claude_task_state read method — use "
        "session_state_ports.live_session_state instead. "
        f"Violations: {violations}"
    )


@pytest.mark.parametrize("path", _iter_handler_files(), ids=lambda p: p.name)
def test_no_banned_attr_reads_on_session_lifecycle(path: Path) -> None:
    """Handlers must not call session_lifecycle.resolve_session_id() directly."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations = _collect_banned_attr_reads(
        tree, "session_lifecycle", BANNED_SESSION_LIFECYCLE_ATTR_READS
    )
    assert not violations, (
        f"{path.relative_to(HANDLERS_ROOT.parents[1])} calls "
        "session_lifecycle.resolve_session_id() — use "
        "session_state_ports.live_session_state.get_session_id() instead. "
        f"Violations: {violations}"
    )
