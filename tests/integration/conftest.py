"""Shared fixtures for integration tests.

Two groups:

- state-file scaffolding (``state_dir``, ``append_event``) for the monitor /
  hook pipeline tests;
- the PTB harness (``make_ptb_app``, ``dispatch_app``) plus Telegram ``Update``
  factories for the dispatch tests, which drive a real ``Application`` with the
  Bot API transport stubbed out.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from telegram import CallbackQuery, Chat, Message, MessageEntity, Update, User
from telegram.ext import Application

# Trigger SessionManager construction so the window_store / thread_router /
# session_map_sync proxies are wired before any integration test imports
# session_monitor or related modules in isolation.  When the whole suite runs,
# some other test usually imports ccgram.session first; explicit import here
# guarantees per-file isolation.
import ccgram.session  # noqa: F401  (import-for-side-effects)
from ccgram.session import SessionManager
from ccgram.session_monitor import SessionMonitor

TEST_USER_ID = 12345
TEST_CHAT_ID = -100999
TEST_THREAD_ID = 42

_GET_ME_RESPONSE = {
    "id": 1,
    "first_name": "Bot",
    "is_bot": True,
    "username": "testbot",
}


@pytest.fixture(autouse=True)
def _default_replace_prompt_mode():
    """Default to replace mode so existing tests using ccgram:N❯ markers pass."""
    from ccgram.config import config

    original = config.prompt_mode
    config.prompt_mode = "replace"
    yield
    config.prompt_mode = original


# ── State files ──────────────────────────────────────────────────────────


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
def make_monitor(state_dir):
    """Factory: a SessionMonitor reading the state files in ``state_dir``.

    Calling it twice models a monitor restart against the same on-disk offsets.
    """

    def _make() -> SessionMonitor:
        return SessionMonitor(
            projects_path=state_dir / "projects",
            poll_interval=0.1,
            state_file=state_dir / "monitor_state.json",
        )

    return _make


@pytest.fixture
def make_session_manager(tmp_path, monkeypatch):
    """Factory: a SessionManager whose state files live under tmp_path.

    Calling it a second time re-reads the same files — that is how the
    round-trip tests simulate a restart.
    """

    def _make() -> SessionManager:
        monkeypatch.setattr("ccgram.config.config.state_file", tmp_path / "state.json")
        monkeypatch.setattr(
            "ccgram.config.config.session_map_file", tmp_path / "session_map.json"
        )
        return SessionManager()

    return _make


@pytest.fixture
def session_manager(make_session_manager) -> SessionManager:
    """A single SessionManager with isolated state files."""
    return make_session_manager()


# ── Telegram update factories ────────────────────────────────────────────


def _command_entities(text: str | None) -> list[MessageEntity] | None:
    if not text or not text.startswith("/"):
        return None
    end = text.index(" ") if " " in text else len(text)
    return [MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=end)]


def _bind_bot(bot, update: Update, *objects) -> None:
    if bot is None:
        return
    update.set_bot(bot)
    for obj in objects:
        obj.set_bot(bot)


@pytest.fixture
def make_text_update() -> Callable[..., Update]:
    """Factory: a text Update in a forum topic.

    A ``/``-prefixed text gets the ``bot_command`` entity PTB's ``CommandHandler``
    and ``filters.COMMAND`` need, so command routing is exercised for real.
    """
    ids = itertools.count(1)

    def _make(
        text: str | None,
        *,
        bot=None,
        thread_id: int = TEST_THREAD_ID,
        user_id: int = TEST_USER_ID,
        chat_id: int = TEST_CHAT_ID,
    ) -> Update:
        update_id = next(ids)
        user = User(id=user_id, first_name="Test", is_bot=False)
        chat = Chat(id=chat_id, type="supergroup")
        entities = _command_entities(text)
        message = Message(
            message_id=update_id,
            date=datetime.now(),
            chat=chat,
            from_user=user,
            text=text,
            entities=entities,
            message_thread_id=thread_id,
        )
        update = Update(update_id=update_id, message=message)
        _bind_bot(bot, update, message, *(entities or []))
        return update

    return _make


@pytest.fixture
def make_callback_update() -> Callable[..., Update]:
    """Factory: a CallbackQuery Update carrying its source message."""
    ids = itertools.count(1)

    def _make(
        data: str,
        *,
        bot=None,
        thread_id: int = TEST_THREAD_ID,
        user_id: int = TEST_USER_ID,
        chat_id: int = TEST_CHAT_ID,
        message_text: str = "(callback source)",
    ) -> Update:
        update_id = next(ids)
        user = User(id=user_id, first_name="Test", is_bot=False)
        chat = Chat(id=chat_id, type="supergroup")
        message = Message(
            message_id=update_id,
            date=datetime.now(),
            chat=chat,
            from_user=user,
            text=message_text,
            message_thread_id=thread_id,
        )
        query = CallbackQuery(
            id=str(update_id),
            from_user=user,
            chat_instance="test",
            data=data,
            message=message,
        )
        update = Update(update_id=update_id, callback_query=query)
        _bind_bot(bot, update, message, query)
        return update

    return _make


# ── PTB application harness ──────────────────────────────────────────────


@pytest.fixture
async def make_ptb_app():
    """Factory: build and start a real PTB Application with a stubbed transport.

    ``register(application)`` wires whichever handlers the test needs; the
    ``_do_post`` transport is replaced before startup so no request ever leaves
    the process. Startup and shutdown are handled by the fixture.
    """
    async with AsyncExitStack() as stack:

        async def _make(register: Callable[[Application], None]) -> Application:
            application = (
                Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
            )
            register(application)
            stack.enter_context(
                patch.object(
                    type(application.bot),
                    "_do_post",
                    AsyncMock(return_value=_GET_ME_RESPONSE),
                )
            )
            await stack.enter_async_context(application)
            return application

        yield _make


def _register_ccgram_handlers(application: Application) -> None:
    """Wire the production handler set (mirrors ``handlers.registry``)."""
    from telegram.ext import (
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    from ccgram.bot import history_command, new_command, text_handler
    from ccgram.handlers.callback_registry import (
        dispatch as callback_dispatch,
        load_handlers,
    )
    from ccgram.handlers.commands import forward_command_handler
    from ccgram.handlers.sessions_dashboard import sessions_command
    from ccgram.handlers.topics.topic_lifecycle import topic_closed_handler

    load_handlers()
    application.add_handler(CommandHandler("start", new_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("sessions", sessions_command))
    application.add_handler(CallbackQueryHandler(callback_dispatch))
    application.add_handler(
        MessageHandler(filters.StatusUpdate.FORUM_TOPIC_CLOSED, topic_closed_handler)
    )
    application.add_handler(MessageHandler(filters.COMMAND, forward_command_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
    )


@pytest.fixture
async def dispatch_app(make_ptb_app) -> Application:
    """Real PTB Application with the full ccgram handler set registered."""
    return await make_ptb_app(_register_ccgram_handlers)
