"""Root conftest — sets env vars BEFORE any ccgram module is imported.

The config.py module-level singleton requires TELEGRAM_BOT_TOKEN and
ALLOWED_USERS at import time, so these must be set before pytest
discovers any test that transitively imports ccgram.
"""

import atexit
import contextlib
import warnings
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterator

import pytest

from _native_cleanup import tmux_test_namespace

# Strip ambient ccgram config env (a running ccgram instance exports
# CCGRAM_GROUP_ID, CCGRAM_CLAUDE_COMMAND, MONITOR_POLL_INTERVAL, … which would
# otherwise leak into tests asserting config defaults and into import-time
# state like bot._group_filter). Cleared before the config singleton is built.
# The CCGRAM_ prefix is scrubbed, plus the non-prefixed vars Config reads directly.
_CONFIG_ENV_PREFIXES = ("CCGRAM_",)
_NON_PREFIXED_CONFIG_ENV = (
    "UNBOUND_WINDOW_TTL_MINUTES",
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
# Pin to tmux so tests are hermetic: auto-detect would pick the ambient
# terminal (agterm/herdr/tmux) and change config.multiplexer_name, which
# breaks every test that uses "ccgram:@0" session-map key format.
os.environ["CCGRAM_MULTIPLEXER"] = "tmux"


# Even an incompletely mocked test must never reach the user's default server.

_tmux_test_root, _close_tmux_test_root = tmux_test_namespace()
os.environ["TMUX_TMPDIR"] = str(_tmux_test_root)
os.environ.pop("TMUX", None)
os.environ.pop("TMUX_PANE", None)
atexit.register(_close_tmux_test_root)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmux_test_namespace():
    yield
    _close_tmux_test_root()


@pytest.fixture(autouse=True)
def _clean_pi_hook_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep hooks independent of the Pi process launching pytest."""
    monkeypatch.delenv("PI_CODING_AGENT", raising=False)
    monkeypatch.delenv("PI_HOOK_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("PI_SUBAGENT_CHILD", raising=False)
    monkeypatch.delenv("AGTERM_PANE", raising=False)
    monkeypatch.delenv("AGTERM_PANE_ID", raising=False)


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


def _close_created_windows(created: list[Callable[[], None]]) -> None:
    """Run captured cleanup only; never infer workspace or checkout ownership."""
    for close in created:
        try:
            close()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            warnings.warn(
                f"Native test resource cleanup failed: {type(exc).__name__}",
                RuntimeWarning,
                stacklevel=2,
            )


@pytest.fixture(autouse=True)
def _cleanup_created_windows(request, monkeypatch) -> Iterator[None]:
    """Close only resources captured on their creating backend's namespace.

    A private tmux @ID is not an address on the default server. Herdr session
    creation does not imply ownership of any pre-existing workspace/checkout.
    Unit-test fake IDs must never reach native cleanup.
    """
    if not (
        request.node.get_closest_marker("integration")
        or request.node.get_closest_marker("e2e")
    ):
        yield
        return

    from _native_cleanup import herdr_window_cleanup, tmux_window_cleanup

    created: list[Callable[[], None]] = []

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
                close = (
                    tmux_window_cleanup(self.server, result[3])
                    if backend == "tmux"
                    else herdr_window_cleanup(self, result[3])
                )
                if close is not None:
                    created.append(close)
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
