"""Shared fixtures for integration tests.

Provides reusable fixtures for state directories, config patching,
and session_map/events.jsonl file management.
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Temp directory with empty state files and config patched to use it."""
    (tmp_path / "session_map.json").write_text("{}")
    (tmp_path / "events.jsonl").write_text("")
    (tmp_path / "state.json").write_text("{}")
    (tmp_path / "monitor_state.json").write_text("{}")

    monkeypatch.setattr(
        "ccgram.config.config.session_map_file", tmp_path / "session_map.json"
    )
    monkeypatch.setattr("ccgram.config.config.events_file", tmp_path / "events.jsonl")
    monkeypatch.setattr(
        "ccgram.config.config.tmux_session_name",
        "ccgram",
    )

    return tmp_path


@pytest.fixture
def write_session_map(state_dir):
    """Factory: write entries to session_map.json."""

    def _write(entries: dict) -> Path:
        path = state_dir / "session_map.json"
        path.write_text(json.dumps(entries))
        return path

    return _write


@pytest.fixture
def append_event(state_dir):
    """Factory: append a hook event line to events.jsonl."""

    def _append(
        event_type: str,
        window_key: str = "ccgram:@0",
        session_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        data: dict | None = None,
        timestamp: float | None = None,
    ) -> None:
        line = json.dumps(
            {
                "ts": timestamp or time.time(),
                "event": event_type,
                "window_key": window_key,
                "session_id": session_id,
                "data": data or {},
            },
            separators=(",", ":"),
        )
        events_file = state_dir / "events.jsonl"
        with open(events_file, "a") as f:
            f.write(line + "\n")

    return _append


@pytest.fixture
def mock_bot():
    """A mock Telegram Bot with common methods stubbed."""
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.edit_message_text = AsyncMock()
    bot.create_forum_topic = AsyncMock()
    bot.defaults = None
    bot.local_mode = False
    bot.base_url = "https://api.telegram.org/bot"
    bot.base_file_url = "https://api.telegram.org/file/bot"
    return bot
