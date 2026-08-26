"""Tests for the directory-flow vs handle_new_window race guard.

See `topic_orchestration._pending_user_creations` for the bug context (MC-2967):
when the directory flow creates a tmux window, the SessionStart hook can fire
before `thread_router.bind_thread()` runs, which previously caused
`handle_new_window` to spawn a duplicate Telegram topic. The pending-creation
set lets `handle_new_window` defer to the in-flight directory flow.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.handlers.topics.topic_orchestration import (  # noqa: E501
    _is_pending_user_creation,
    _pending_creation_transactions,
    _pending_user_creations,
    clear_pending_creation,
    pending_creation_transaction,
    handle_new_window,
    register_pending_creation,
)
from ccgram.handlers.topics import topic_orchestration  # type: ignore[attr-defined]
from ccgram.session_monitor import NewWindowEvent


@pytest.fixture(autouse=True)
def _clear_pending_state():
    _pending_user_creations.clear()
    _pending_creation_transactions.clear()
    yield
    _pending_user_creations.clear()
    _pending_creation_transactions.clear()


def _make_event(window_id: str = "@42") -> NewWindowEvent:
    return NewWindowEvent(
        window_id=window_id,
        window_name="test-window",
        cwd="/tmp",
        session_id="some-session-id",
    )


def test_register_pending_creation_persists_until_clear():
    register_pending_creation("@42")
    assert _is_pending_user_creation("@42")
    clear_pending_creation("@42")
    assert not _is_pending_user_creation("@42")


def test_register_pending_creation_is_idempotent():
    register_pending_creation("@42")
    register_pending_creation("@42")
    clear_pending_creation("@42")
    assert not _is_pending_user_creation("@42")


def test_pending_creation_expires_after_ttl(monkeypatch):
    base = time.monotonic()
    monkeypatch.setattr(topic_orchestration.time, "monotonic", lambda: base)
    register_pending_creation("@42")
    assert _is_pending_user_creation("@42")
    monkeypatch.setattr(
        topic_orchestration.time,
        "monotonic",
        lambda: base + topic_orchestration._PENDING_CREATION_TTL_S + 1,
    )
    assert not _is_pending_user_creation("@42")
    assert "@42" not in _pending_user_creations


def test_clear_pending_creation_ignores_unknown_window():
    register_pending_creation("@42")
    clear_pending_creation("@nonexistent")
    assert _is_pending_user_creation("@42")


def test_creation_transaction_defers_unbound_window_adoption():
    with pending_creation_transaction():
        assert _is_pending_user_creation("unrelated-window")
    assert not _is_pending_user_creation("unrelated-window")


def test_creation_transaction_releases_on_exception():
    with (
        pytest.raises(RuntimeError, match="launch failed"),
        pending_creation_transaction(),
    ):
        raise RuntimeError("launch failed")
    assert not _is_pending_user_creation("unrelated-window")


def test_register_pending_creation_ignores_blank_window_id():
    register_pending_creation("")
    assert not _is_pending_user_creation("")


async def test_handle_new_window_skips_when_pending(monkeypatch):
    register_pending_creation("@42")

    create_topic_mock = AsyncMock()
    rebind_mock = AsyncMock(return_value=False)
    bound_mock = MagicMock(return_value=False)
    monkeypatch.setattr(topic_orchestration, "_is_window_already_bound", bound_mock)
    monkeypatch.setattr(topic_orchestration, "create_topic_in_chat", create_topic_mock)
    monkeypatch.setattr(
        topic_orchestration, "_rebind_existing_topic_by_name", rebind_mock
    )
    monkeypatch.setattr(topic_orchestration, "_auto_detect_provider", AsyncMock())

    client = MagicMock()
    await handle_new_window(_make_event(), client)

    create_topic_mock.assert_not_awaited()
    rebind_mock.assert_not_awaited()


async def test_handle_new_window_proceeds_when_not_pending(monkeypatch):
    create_topic_mock = AsyncMock()
    rebind_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(
        topic_orchestration, "_is_window_already_bound", lambda _wid: False
    )
    monkeypatch.setattr(topic_orchestration, "create_topic_in_chat", create_topic_mock)
    monkeypatch.setattr(
        topic_orchestration, "_rebind_existing_topic_by_name", rebind_mock
    )
    monkeypatch.setattr(topic_orchestration, "_auto_detect_provider", AsyncMock())
    monkeypatch.setattr(topic_orchestration, "collect_target_chats", lambda _wid: {123})

    client = MagicMock()
    await handle_new_window(_make_event(), client)

    rebind_mock.assert_awaited_once()
    create_topic_mock.assert_awaited_once()


async def test_handle_new_window_skips_when_already_bound_takes_priority(monkeypatch):
    """already-bound check runs first; pending-creation check is a fallback."""
    register_pending_creation("@42")  # also pending
    bound_mock = MagicMock(return_value=True)
    create_topic_mock = AsyncMock()
    rebind_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(topic_orchestration, "_is_window_already_bound", bound_mock)
    monkeypatch.setattr(topic_orchestration, "create_topic_in_chat", create_topic_mock)
    monkeypatch.setattr(
        topic_orchestration, "_rebind_existing_topic_by_name", rebind_mock
    )
    monkeypatch.setattr(topic_orchestration, "_auto_detect_provider", AsyncMock())

    client = MagicMock()
    await handle_new_window(_make_event(), client)

    bound_mock.assert_called_once_with("@42")
    create_topic_mock.assert_not_awaited()
