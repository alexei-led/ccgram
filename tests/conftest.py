"""Root conftest — sets env vars BEFORE any ccgram module is imported.

The config.py module-level singleton requires TELEGRAM_BOT_TOKEN and
ALLOWED_USERS at import time, so these must be set before pytest
discovers any test that transitively imports ccgram.
"""

import contextlib
import os
import subprocess
import tempfile
from collections.abc import Iterator

import pytest

# Strip ambient ccgram config env (a running ccgram instance exports
# CCGRAM_GROUP_ID, CCGRAM_CLAUDE_COMMAND, MONITOR_POLL_INTERVAL, … which would
# otherwise leak into tests asserting config defaults and into import-time
# state like bot._group_filter). Cleared before the config singleton is built.
# The CCGRAM_ prefix is scrubbed, plus the non-prefixed vars Config reads directly.
_CONFIG_ENV_PREFIXES = ("CCGRAM_",)
_NON_PREFIXED_CONFIG_ENV = (
    "AUTOCLOSE_DEAD_MINUTES",
    "AUTOCLOSE_DONE_MINUTES",
    "CLAUDE_CONFIG_DIR",
    "MONITOR_POLL_INTERVAL",
    "TMUX_SESSION_NAME",
)
for _key in list(os.environ):
    if _key.startswith(_CONFIG_ENV_PREFIXES) or _key in _NON_PREFIXED_CONFIG_ENV:
        del os.environ[_key]

# Force-set (not setdefault) to prevent real env vars from leaking into tests
os.environ["TELEGRAM_BOT_TOKEN"] = "test:0000000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
os.environ["ALLOWED_USERS"] = "12345"
os.environ["CCGRAM_DIR"] = tempfile.mkdtemp(prefix="ccgram-test-")


@pytest.fixture(autouse=True)
def _clear_window_store():
    from ccgram.claude_task_state import claude_task_state
    from ccgram.window_state_store import get_window_store

    def _clear() -> None:
        # SessionManager hasn't been built in this test — nothing to clear.
        with contextlib.suppress(RuntimeError):
            get_window_store().window_states.clear()

    claude_task_state.reset()
    _clear()
    yield
    claude_task_state.reset()
    _clear()


@pytest.fixture(autouse=True)
def _wire_multiplexer():
    """Wire the multiplexer proxy to the tmux backend for the duration of a test.

    Production wires the proxy in ``bootstrap_application``; unit tests don't run
    bootstrap, so callers using the ``multiplexer`` proxy would hit the unwired
    error. Installing the tmux backend mirrors the default config and keeps the
    proxy forwarding to the same ``multiplexer.tmux.tmux_manager`` singleton that
    tests patch. Tests that replace the whole proxy via ``patch(...)`` are
    unaffected (the patch shadows this wiring).
    """
    from ccgram.multiplexer import (
        _reset_multiplexer_for_testing,
        get_multiplexer,
        install_multiplexer,
    )

    install_multiplexer(get_multiplexer("tmux"))
    yield
    _reset_multiplexer_for_testing()


def _close_created_windows(created: list[tuple[str, str]]) -> None:
    """Close recorded multiplexer windows via the backend CLI (best-effort)."""
    closed_workspaces: set[str] = set()
    for backend, window_id in created:
        if backend == "tmux":
            # tmux window ids (``@N``) are server-global, so target directly.
            subprocess.run(
                ["tmux", "kill-window", "-t", window_id], capture_output=True
            )
            continue
        # herdr: close the tab and the (often auto-created, empty) workspace.
        subprocess.run(["herdr", "tab", "close", window_id], capture_output=True)
        workspace_id = window_id.split(":")[0]
        if workspace_id in closed_workspaces:
            continue
        closed_workspaces.add(workspace_id)
        # A worktree-backed workspace needs `worktree remove` (deletes the
        # checkout + closes); a plain one just `workspace close`. Try both.
        subprocess.run(
            ["herdr", "worktree", "remove", "--workspace", workspace_id, "--force"],
            capture_output=True,
        )
        subprocess.run(
            ["herdr", "workspace", "close", workspace_id], capture_output=True
        )


@pytest.fixture(autouse=True)
def _cleanup_created_windows(request, monkeypatch) -> Iterator[None]:
    """Close multiplexer windows/tabs created by integration/e2e tests.

    Real-window tests spin up tmux windows and herdr tabs (plus the workspaces
    herdr auto-creates per cwd); when a test forgets — or fails before — its own
    teardown, they leak and must be closed by hand. Wrap ``create_window`` /
    ``create_worktree_window`` on both backends to record the ids THIS test
    creates, then close them after. Recording only this-test ids keeps it
    race-free under ``-n auto`` (it never touches another worker's windows).

    Gated on the ``integration`` / ``e2e`` markers: unit tests drive the backends
    with fake runners returning fake ids (e.g. ``"w2:t9"``) that must never reach
    the live herdr socket.
    """
    if not (
        request.node.get_closest_marker("integration")
        or request.node.get_closest_marker("e2e")
    ):
        yield
        return

    created: list[tuple[str, str]] = []

    def _wrap(cls: type, attr: str, backend: str) -> None:
        orig = getattr(cls, attr, None)
        if orig is None:
            return

        async def wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN202
            result = await orig(self, *args, **kwargs)
            if (
                isinstance(result, tuple)
                and len(result) >= 4
                and result[0]
                and isinstance(result[3], str)
                and result[3]
            ):
                created.append((backend, result[3]))
            return result

        monkeypatch.setattr(cls, attr, wrapper)

    from ccgram.multiplexer.herdr import HerdrManager
    from ccgram.multiplexer.tmux import TmuxManager

    _wrap(TmuxManager, "create_window", "tmux")
    _wrap(HerdrManager, "create_window", "herdr")
    _wrap(HerdrManager, "create_worktree_window", "herdr")
    try:
        yield
    finally:
        _close_created_windows(created)
