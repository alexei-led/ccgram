"""Root conftest — sets env vars BEFORE any ccgram module is imported.

The config.py module-level singleton requires TELEGRAM_BOT_TOKEN and
ALLOWED_USERS at import time, so these must be set before pytest
discovers any test that transitively imports ccgram.
"""

import contextlib
import os
import tempfile

import pytest

# Strip ambient ccgram config env (a running ccgram instance exports
# CCGRAM_GROUP_ID, CCGRAM_CLAUDE_COMMAND, MONITOR_POLL_INTERVAL, … which would
# otherwise leak into tests asserting config defaults and into import-time
# state like bot._group_filter). Cleared before the config singleton is built.
# Both the CCGRAM_ prefix and the legacy CCBOT_ fallback (Config._env_with_fallback)
# are scrubbed, plus the non-prefixed vars Config reads directly.
_CONFIG_ENV_PREFIXES = ("CCGRAM_", "CCBOT_")
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
