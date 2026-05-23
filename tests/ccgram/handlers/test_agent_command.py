from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers import agent_command as ac
from ccgram.handlers.callback_data import CB_AGENT_CANCEL, CB_AGENT_SET
from ccgram.session import session_manager
from ccgram.window_state_ports import identity_state
from ccgram.window_state_store import WindowState, window_store


@pytest.fixture
def clear_map_mock(monkeypatch):
    # Patch on the real session_map_sync singleton — both ``_commit_switch``
    # (for hookful providers) and the WindowStateStore hookless-switch
    # callback go through this same instance, so one patch covers both.
    from ccgram.session_map import session_map_sync as real_sms

    mock = MagicMock()
    monkeypatch.setattr(real_sms, "clear_session_map_entry", mock)
    return mock


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, clear_map_mock):
    # ``SessionManager.set_window_provider`` looks up provider capabilities
    # via the registry — the default ``supports_hook=True`` fallback would
    # silently swallow the hookless-switch path tested below.
    from ccgram.providers import _ensure_registered

    _ensure_registered()
    window_store.window_states.clear()
    window_store.window_states["@7"] = WindowState(
        session_id="OLDSID",
        cwd="/tmp/proj",
        window_name="proj",
        transcript_path="/tmp/old.jsonl",
        provider_name="claude",
    )

    monkeypatch.setattr(ac.thread_router, "get_window_for_thread", lambda u, t: "@7")
    # Patch at the Config class level (not the singleton instance) so teardown
    # cleanly restores the descriptor — instance-level setattr would leak a
    # bound-method instance attribute that shadows the class method for
    # later tests (e.g. handlers/topics/test_topic_close.py).
    with patch("ccgram.config.Config.is_user_allowed", return_value=True):
        yield

    window_store.window_states.clear()


def _make_update(text: str = "/agent"):
    update = MagicMock()
    user = MagicMock()
    user.id = 42
    update.effective_user = user
    msg = MagicMock()
    msg.text = text
    msg.message_thread_id = 99
    msg.from_user = user
    msg.chat.type = "supergroup"
    msg.chat.id = -100
    update.message = msg
    update.callback_query = None
    return update


def _make_query(data: str):
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    user = MagicMock()
    user.id = 42
    query.from_user = user
    msg = MagicMock()
    msg.message_thread_id = 99
    msg.chat.type = "supergroup"
    msg.chat.id = -100
    query.message = msg
    update = MagicMock()
    update.callback_query = query
    update.effective_user = user
    update.effective_chat = msg.chat
    update.message = None
    return update


async def test_bare_command_shows_picker():
    captured: dict[str, object] = {}

    async def fake_safe_reply(_msg, text, reply_markup=None):
        captured["text"] = text
        captured["markup"] = reply_markup

    with patch("ccgram.handlers.agent_command.safe_reply", side_effect=fake_safe_reply):
        await ac.agent_command(_make_update(), MagicMock())

    assert "claude" in str(captured["text"]).lower()
    markup = captured["markup"]
    keyboard = getattr(markup, "inline_keyboard")  # noqa: B009
    callbacks = [b.callback_data for row in keyboard for b in row]
    assert any(c.startswith(CB_AGENT_SET) for c in callbacks)
    assert any(c == f"{CB_AGENT_SET}@7:shell" for c in callbacks)
    assert any(c == f"{CB_AGENT_SET}@7:auto" for c in callbacks)
    assert any(c == f"{CB_AGENT_CANCEL}@7" for c in callbacks)


async def test_arg_shell_switches_clears_state_and_marks_override(
    monkeypatch, clear_map_mock
):
    ensure_setup_mock = AsyncMock()
    monkeypatch.setattr(
        "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
        ensure_setup_mock,
    )
    monkeypatch.setattr(
        "ccgram.handlers.shell.shell_capture.clear_shell_monitor_state",
        MagicMock(),
    )
    monkeypatch.setattr(
        "ccgram.handlers.shell.shell_prompt_orchestrator.clear_state",
        MagicMock(),
    )

    sent: dict[str, str] = {}

    async def fake_safe_reply(_msg, text, reply_markup=None):
        sent["text"] = text

    with patch("ccgram.handlers.agent_command.safe_reply", side_effect=fake_safe_reply):
        await ac.agent_command(_make_update("/agent shell"), MagicMock())

    state = window_store.window_states["@7"]
    assert state.provider_name == "shell"
    assert state.transcript_path == ""
    assert state.provider_manual_override is True
    clear_map_mock.assert_called_once_with("@7")
    ensure_setup_mock.assert_awaited_once_with("@7", "provider_switch")
    assert "shell" in sent["text"]


async def test_arg_unknown_is_rejected():
    sent: dict[str, str] = {}

    async def fake_safe_reply(_msg, text, reply_markup=None):
        sent["text"] = text

    with patch("ccgram.handlers.agent_command.safe_reply", side_effect=fake_safe_reply):
        await ac.agent_command(_make_update("/agent garbage"), MagicMock())

    assert "Unknown agent" in sent["text"]
    # State unchanged
    assert window_store.window_states["@7"].provider_name == "claude"


async def test_auto_clears_override_and_redetects(monkeypatch):
    # Pre-mark as manual override so we can verify it gets cleared.
    identity_state.set_provider_manual_override("@7", value=True)

    fake_window = MagicMock()
    fake_window.pane_current_command = "codex"
    fake_window.pane_tty = "/dev/ttys00"
    monkeypatch.setattr(
        "ccgram.tmux_manager.tmux_manager.find_window_by_id",
        AsyncMock(return_value=fake_window),
    )
    monkeypatch.setattr(
        "ccgram.providers.detect_provider_from_pane",
        AsyncMock(return_value="codex"),
    )

    sent: dict[str, str] = {}

    async def fake_safe_reply(_msg, text, reply_markup=None):
        sent["text"] = text

    with patch("ccgram.handlers.agent_command.safe_reply", side_effect=fake_safe_reply):
        await ac.agent_command(_make_update("/agent auto"), MagicMock())

    state = window_store.window_states["@7"]
    assert state.provider_name == "codex"
    assert state.provider_manual_override is False
    assert "codex" in sent["text"]


async def test_auto_falls_back_to_shell_when_detection_empty(monkeypatch):
    fake_window = MagicMock()
    fake_window.pane_current_command = "ralphex"
    fake_window.pane_tty = "/dev/ttys00"
    monkeypatch.setattr(
        "ccgram.tmux_manager.tmux_manager.find_window_by_id",
        AsyncMock(return_value=fake_window),
    )
    monkeypatch.setattr(
        "ccgram.providers.detect_provider_from_pane",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "ccgram.handlers.shell.shell_capture.clear_shell_monitor_state",
        MagicMock(),
    )
    monkeypatch.setattr(
        "ccgram.handlers.shell.shell_prompt_orchestrator.clear_state",
        MagicMock(),
    )

    sent: dict[str, str] = {}

    async def fake_safe_reply(_msg, text, reply_markup=None):
        sent["text"] = text

    with patch("ccgram.handlers.agent_command.safe_reply", side_effect=fake_safe_reply):
        await ac.agent_command(_make_update("/agent auto"), MagicMock())

    assert window_store.window_states["@7"].provider_name == "shell"
    assert "shell" in sent["text"]


async def test_callback_cancel_keeps_state(monkeypatch):
    monkeypatch.setattr(
        "ccgram.handlers.callback_helpers.user_owns_window", lambda u, w: True
    )
    edited: dict[str, str] = {}

    async def fake_safe_edit(_query, text, reply_markup=None):
        edited["text"] = text

    with patch("ccgram.handlers.agent_command.safe_edit", side_effect=fake_safe_edit):
        update = _make_query(f"{CB_AGENT_CANCEL}@7")
        await ac._dispatch(update, MagicMock())

    assert window_store.window_states["@7"].provider_name == "claude"
    assert "Cancelled" in edited["text"]


async def test_callback_set_provider_clears_transcript(monkeypatch, clear_map_mock):
    monkeypatch.setattr(
        "ccgram.handlers.agent_command.user_owns_window", lambda u, w: True
    )
    edited: dict[str, str] = {}

    async def fake_safe_edit(_query, text, reply_markup=None):
        edited["text"] = text

    with patch("ccgram.handlers.agent_command.safe_edit", side_effect=fake_safe_edit):
        update = _make_query(f"{CB_AGENT_SET}@7:gemini")
        await ac._dispatch(update, MagicMock())

    state = window_store.window_states["@7"]
    assert state.provider_name == "gemini"
    assert state.transcript_path == ""
    assert state.provider_manual_override is True
    clear_map_mock.assert_called_once_with("@7")


async def test_callback_rejects_foreign_user(monkeypatch):
    monkeypatch.setattr(
        "ccgram.handlers.agent_command.user_owns_window", lambda u, w: False
    )
    update = _make_query(f"{CB_AGENT_SET}@7:shell")
    await ac._dispatch(update, MagicMock())
    update.callback_query.answer.assert_awaited_once_with("Not your window")
    assert window_store.window_states["@7"].provider_name == "claude"


async def test_manual_override_blocks_auto_detect(monkeypatch):
    """Regression: _detect_and_apply_provider must skip overridden windows."""
    from ccgram.handlers.recovery import transcript_discovery

    identity_state.set_provider_manual_override("@7", value=True)
    # If detection ran, it would try to change provider — capture that.
    monkeypatch.setattr(
        "ccgram.providers.detect_provider_from_pane",
        AsyncMock(return_value="codex"),
    )
    set_provider_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        session_manager,
        "set_window_provider",
        lambda window_id, provider_name, cwd=None: set_provider_calls.append(
            (window_id, provider_name)
        ),
    )

    fake_w = MagicMock()
    fake_w.pane_current_command = "codex"
    fake_w.pane_tty = "/dev/ttys00"
    fake_w.cwd = "/tmp/proj"

    identity = identity_state.get_identity("@7")
    assert identity is not None
    await transcript_discovery._detect_and_apply_provider(
        "@7", identity, fake_w, client=None, chat_id=0, thread_id=0
    )

    assert set_provider_calls == []
    assert window_store.window_states["@7"].provider_name == "claude"
