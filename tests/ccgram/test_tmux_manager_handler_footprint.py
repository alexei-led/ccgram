"""Architecture-fitness test guarding the tmux_manager handler footprint.

Task 2 of the cmux sidecar MVP introduces ``TerminalBackend`` and the
``terminal_operations`` service. New handler code must route through
the backend seam, not import ``tmux_manager`` directly. To keep the
plan incremental, this test does not (yet) require existing handlers
to migrate — it freezes the current direct-import set as an allow-list
and rejects any growth.

Two invariants:

* The migrated paths in Task 2 (``handlers/text/text_handler.py`` for
  the primary text-send, ``handlers/live/screenshot_callbacks.py`` for
  the ``/screenshot`` capture flow) MUST import ``terminal_operations``.
* No handler or Mini App module outside the allow-list may import
  ``tmux_manager`` — adding a new direct caller fails this test and
  asks the author to route through ``terminal_operations`` instead.

The allow-list is intentionally large today and should shrink as
follow-up tasks migrate flows. Each removal of a file from the
allow-list is a Task verification step in the plan.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "ccgram"
HANDLERS_ROOT = SRC_ROOT / "handlers"
MINIAPP_ROOT = SRC_ROOT / "miniapp"


# Files currently allowed to import ``tmux_manager`` directly. Each
# entry is a future migration candidate; removing one is the goal of
# subsequent terminal-operations migrations.
TMUX_MANAGER_ALLOW_LIST: frozenset[str] = frozenset(
    {
        "handlers/agent_command.py",
        "handlers/commands/failure_probe.py",
        "handlers/commands/forward.py",
        "handlers/file_handler.py",
        "handlers/interactive/interactive_callbacks.py",
        "handlers/interactive/interactive_ui.py",
        "handlers/last_reply.py",
        "handlers/live/live_view.py",
        "handlers/live/pane_callbacks.py",
        "handlers/live/screenshot_callbacks.py",
        "handlers/messaging/msg_broker.py",
        "handlers/messaging/msg_spawn.py",
        "handlers/polling/periodic_tasks.py",
        "handlers/polling/polling_coordinator.py",
        "handlers/polling/polling_state.py",
        "handlers/polling/window_tick/__init__.py",
        "handlers/polling/window_tick/apply.py",
        "handlers/polling/window_tick/observe.py",
        "handlers/recovery/history_callbacks.py",
        "handlers/recovery/recovery_banner.py",
        "handlers/recovery/restore_command.py",
        "handlers/recovery/resume_command.py",
        "handlers/recovery/transcript_discovery.py",
        "handlers/sessions_dashboard.py",
        "handlers/shell/shell_capture.py",
        "handlers/shell/shell_commands.py",
        "handlers/status/rc_probe.py",
        "handlers/status/status_bar_actions.py",
        "handlers/sync_command.py",
        "handlers/text/text_handler.py",
        "handlers/toolbar/toolbar_callbacks.py",
        "handlers/topics/directory_callbacks.py",
        "handlers/topics/topic_lifecycle.py",
        "handlers/topics/topic_orchestration.py",
        "handlers/topics/window_callbacks.py",
        "handlers/voice/voice_callbacks.py",
        "miniapp/api/terminal.py",
    }
)


# Files that MUST route through ``terminal_operations`` after Task 2.
# Adding new entries here is the standard mechanism for advertising a
# completed migration step in future tasks.
TERMINAL_OPERATIONS_REQUIRED: frozenset[str] = frozenset(
    {
        "handlers/text/text_handler.py",
        "handlers/live/screenshot_callbacks.py",
    }
)


def _iter_candidate_files() -> list[Path]:
    return sorted([*HANDLERS_ROOT.rglob("*.py"), *MINIAPP_ROOT.rglob("*.py")])


def _module_imports(tree: ast.AST) -> set[str]:
    """Collect every fully-qualified import target referenced in ``tree``.

    Picks up both ``import X`` (Import node) and ``from X import Y``
    (ImportFrom node), normalising ``from ...tmux_manager import …`` to
    the package-qualified name when the file lives inside ``ccgram``.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            found.add(mod)
            for alias in node.names:
                found.add(f"{mod}.{alias.name}" if mod else alias.name)
    return found


def _imports_tmux_manager(tree: ast.AST) -> bool:
    for imp in _module_imports(tree):
        # Matches both absolute (``ccgram.tmux_manager``) and relative
        # forms — relative imports lose their leading dots in ``ast``,
        # so we look for the bare ``tmux_manager`` token anywhere in
        # the dotted path.
        if imp == "tmux_manager" or imp.endswith(".tmux_manager"):
            return True
        if imp.startswith("tmux_manager.") or ".tmux_manager." in imp:
            return True
    return False


def _imports_terminal_operations(tree: ast.AST) -> bool:
    for imp in _module_imports(tree):
        if imp == "terminal_operations" or imp.endswith(".terminal_operations"):
            return True
        if imp.startswith("terminal_operations.") or ".terminal_operations." in imp:
            return True
    return False


def _rel(path: Path) -> str:
    return str(path.relative_to(SRC_ROOT)).replace("\\", "/")


def test_tmux_manager_imports_are_within_allow_list() -> None:
    """No new direct ``tmux_manager`` callers in handlers / Mini App.

    Fails when a handler or Mini App module that is not in the
    allow-list imports ``tmux_manager``. Author should route through
    ``ccgram.terminal_operations`` instead. If a migration legitimately
    needs to leave the call in place, add the file to
    ``TMUX_MANAGER_ALLOW_LIST`` and record the reason in the plan.
    """
    violations: list[str] = []
    for path in _iter_candidate_files():
        rel = _rel(path)
        if not _imports_tmux_manager(ast.parse(path.read_text(encoding="utf-8"))):
            continue
        if rel not in TMUX_MANAGER_ALLOW_LIST:
            violations.append(rel)
    assert not violations, (
        "New direct ``tmux_manager`` imports detected — route through "
        "ccgram.terminal_operations instead. Offending files: " + repr(violations)
    )


def test_allow_list_stays_current() -> None:
    """Allow-list entries must still import ``tmux_manager``.

    If a file in the allow-list no longer imports ``tmux_manager`` the
    entry is dead and should be removed so the next handler that drifts
    onto a direct call gets caught. Forces deliberate maintenance of
    the migration ledger.
    """
    stale: list[str] = []
    for rel in sorted(TMUX_MANAGER_ALLOW_LIST):
        path = SRC_ROOT / rel
        assert path.exists(), f"allow-list points at missing file: {rel}"
        if not _imports_tmux_manager(ast.parse(path.read_text(encoding="utf-8"))):
            stale.append(rel)
    assert not stale, (
        "Allow-list contains files that no longer import tmux_manager — "
        "remove them to keep the migration ledger honest. Stale entries: "
        f"{stale}"
    )


def test_migrated_paths_use_terminal_operations() -> None:
    """Migrated files must depend on ``terminal_operations``.

    Adding a file here advertises a completed migration step. The test
    fails if a file in this set drops its ``terminal_operations``
    dependency — which would silently undo the migration.
    """
    missing: list[str] = []
    for rel in sorted(TERMINAL_OPERATIONS_REQUIRED):
        path = SRC_ROOT / rel
        assert path.exists(), f"required-migration file missing: {rel}"
        if not _imports_terminal_operations(
            ast.parse(path.read_text(encoding="utf-8"))
        ):
            missing.append(rel)
    assert not missing, (
        "Files required to route through terminal_operations no longer "
        f"import it: {missing}"
    )
