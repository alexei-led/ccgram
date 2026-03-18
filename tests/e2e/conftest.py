"""E2E test fixtures — real PTB app, real tmux, real agent CLIs.

Provides:
  - e2e_state_dir: isolated state files in tmp_path
  - e2e_tmux: dedicated tmux session (ccgram-e2e)
  - intercepted_calls: API call recorder
  - e2e_app: full Application lifecycle with mocked Bot API
  - work_dir: minimal agent working directory
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from ._helpers import TEST_CHAT_ID, _bump_message_id

pytestmark = pytest.mark.e2e

# ---------------------------------------------------------------------------
# State directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_state_dir(tmp_path, monkeypatch):
    """Create isolated state files and point config at them."""
    state_dir = tmp_path / "ccgram-state"
    state_dir.mkdir()

    state_file = state_dir / "state.json"
    session_map = state_dir / "session_map.json"
    events_file = state_dir / "events.jsonl"
    monitor_state = state_dir / "monitor_state.json"

    state_file.write_text("{}")
    session_map.write_text("{}")
    events_file.write_text("")
    monitor_state.write_text("{}")

    from ccgram.config import config

    monkeypatch.setattr(config, "config_dir", state_dir)
    monkeypatch.setattr(config, "state_file", state_file)
    monkeypatch.setattr(config, "session_map_file", session_map)
    monkeypatch.setattr(config, "events_file", events_file)
    monkeypatch.setattr(config, "monitor_state_file", monitor_state)

    return state_dir


# ---------------------------------------------------------------------------
# Tmux session fixture
# ---------------------------------------------------------------------------

E2E_TMUX_SESSION = "ccgram-e2e"


@pytest.fixture
def e2e_tmux(monkeypatch):
    """Create a dedicated tmux session for E2E tests."""
    import libtmux

    from ccgram.config import config
    from ccgram.tmux_manager import TmuxManager

    monkeypatch.setattr(config, "tmux_session_name", E2E_TMUX_SESSION)

    server = libtmux.Server()

    # Kill stale session if it exists
    existing = server.sessions.get(session_name=E2E_TMUX_SESSION, default=None)
    if existing:
        existing.kill()

    # Create fresh session
    session = server.new_session(
        session_name=E2E_TMUX_SESSION,
        start_directory=str(Path.home()),
    )
    if session.windows:
        session.windows[0].rename_window("__main__")

    # Replace the global tmux_manager singleton in every importing module.
    manager = TmuxManager(session_name=E2E_TMUX_SESSION)

    _tm_modules = [
        "ccgram.tmux_manager",
        "ccgram.bot",
        "ccgram.session",
        "ccgram.session_monitor",
        "ccgram.handlers.text_handler",
        "ccgram.handlers.directory_callbacks",
        "ccgram.handlers.status_polling",
        "ccgram.handlers.message_queue",
        "ccgram.handlers.recovery_callbacks",
        "ccgram.handlers.sessions_dashboard",
        "ccgram.handlers.screenshot_callbacks",
        "ccgram.handlers.interactive_ui",
        "ccgram.handlers.interactive_callbacks",
        "ccgram.handlers.window_callbacks",
        "ccgram.handlers.restore_command",
        "ccgram.handlers.resume_command",
        "ccgram.handlers.history_callbacks",
        "ccgram.handlers.sync_command",
    ]
    import importlib

    for mod_name in _tm_modules:
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, "tmux_manager", manager)

    yield manager

    # Teardown: kill the entire test session
    test_session = server.sessions.get(session_name=E2E_TMUX_SESSION, default=None)
    if test_session:
        test_session.kill()


# ---------------------------------------------------------------------------
# API interceptor
# ---------------------------------------------------------------------------


@pytest.fixture
def intercepted_calls():
    """Returns a list that records all intercepted Bot API calls."""
    return []


_GET_ME_RESPONSE = {
    "id": 1,
    "first_name": "E2EBot",
    "is_bot": True,
    "username": "e2e_testbot",
    "can_join_groups": True,
    "can_read_all_group_messages": True,
    "supports_inline_queries": False,
}

_GET_CHAT_RESPONSE = {
    "id": TEST_CHAT_ID,
    "type": "supergroup",
    "title": "Test",
    "is_forum": True,
}

_PASSTHROUGH_ENDPOINTS = frozenset(
    {
        "setMyCommands",
        "deleteMyCommands",
        "editForumTopic",
        "sendChatAction",
        "unpinAllForumTopicMessages",
        "deleteMessage",
        "answerCallbackQuery",
    }
)


def _parse_request_data(request_data):
    if request_data is None:
        return {}
    if hasattr(request_data, "parameters"):
        return dict(request_data.parameters)
    if isinstance(request_data, dict):
        return request_data
    return {}


def _make_message_response(data, msg_id=None):
    return {
        "message_id": msg_id or data.get("message_id", 1),
        "date": 0,
        "chat": {"id": data.get("chat_id", TEST_CHAT_ID), "type": "supergroup"},
        "text": data.get("text", ""),
    }


def _make_api_router(calls: list):
    """Build a side_effect function that routes Bot API calls by endpoint."""

    async def router(*, endpoint: str, data=None, **_kw):
        data = _parse_request_data(data)
        calls.append((endpoint, data))

        if endpoint == "getMe":
            return _GET_ME_RESPONSE
        if endpoint == "getChat":
            return _GET_CHAT_RESPONSE
        if endpoint in ("sendMessage", "sendPhoto", "sendDocument"):
            return _make_message_response(data, msg_id=_bump_message_id())
        if endpoint in ("editMessageText", "editMessageReplyMarkup"):
            return _make_message_response(data)
        if endpoint == "createForumTopic":
            return {
                "message_thread_id": _bump_message_id(),
                "name": data.get("name", "test-topic"),
                "icon_color": 0,
            }
        if endpoint in _PASSTHROUGH_ENDPOINTS:
            return True
        return True

    return router


# ---------------------------------------------------------------------------
# Application fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def e2e_app(e2e_state_dir, e2e_tmux, intercepted_calls, monkeypatch):
    """Full PTB Application with real handlers, mocked Bot API, real tmux."""
    from ccgram.session import SessionManager

    fresh_manager = SessionManager()

    # Patch session_manager in every module that imports it at module level.
    # Python's `from mod import name` creates a local binding, so we must
    # patch each importing module individually.
    _sm_modules = [
        "ccgram.session",
        "ccgram.bot",
        "ccgram.handlers.text_handler",
        "ccgram.handlers.directory_callbacks",
        "ccgram.handlers.directory_browser",
        "ccgram.handlers.status_polling",
        "ccgram.handlers.message_queue",
        "ccgram.handlers.recovery_callbacks",
        "ccgram.handlers.callback_helpers",
        "ccgram.handlers.sessions_dashboard",
        "ccgram.handlers.screenshot_callbacks",
        "ccgram.handlers.interactive_ui",
        "ccgram.handlers.history",
        "ccgram.handlers.hook_events",
        "ccgram.handlers.file_handler",
        "ccgram.handlers.voice_handler",
        "ccgram.handlers.voice_callbacks",
        "ccgram.handlers.window_callbacks",
        "ccgram.handlers.restore_command",
        "ccgram.handlers.resume_command",
        "ccgram.handlers.sync_command",
    ]
    import importlib

    for mod_name in _sm_modules:
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, "session_manager", fresh_manager)

    from ccgram.bot import create_bot

    app = create_bot()

    router = _make_api_router(intercepted_calls)

    with patch.object(type(app.bot), "_do_post", side_effect=router):
        async with app:
            # post_init is only called by run_polling(), not __aenter__
            # Call it manually to start monitors, register commands, etc.
            if app.post_init:
                await app.post_init(app)
            yield app, intercepted_calls, e2e_tmux, fresh_manager
            # post_shutdown likewise only runs in run_polling()
            if app.post_shutdown:
                await app.post_shutdown(app)


# ---------------------------------------------------------------------------
# Work directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def work_dir(tmp_path):
    """Create a minimal working directory for agents."""
    readme = tmp_path / "README.md"
    readme.write_text("# E2E Test Project\n\nThis is a test project for E2E tests.\n")
    return tmp_path
