from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from ccgram.handlers import agent_command as ac
from ccgram.handlers.callback_data import CB_AGENT_CANCEL, CB_AGENT_SET
from ccgram.handlers.provider_display import provider_label
from ccgram.config import config
from ccgram.session_map import session_map_prefix, session_map_sync
from ccgram.session import session_manager
from ccgram.window_state_ports import identity_state
from ccgram.window_state_store import WindowState, window_store


@pytest.fixture
def clear_map_mock(monkeypatch, tmp_path):
    # Patch on the real session_map_sync singleton — both ``_commit_switch``
    # (for hookful providers) and the WindowStateStore hookless-switch
    # callback go through this same instance, so one patch covers both.
    from ccgram.session_map import session_map_sync as real_sms

    path = tmp_path / "session-map.json"
    path.write_text(json.dumps({}))
    monkeypatch.setattr(config, "session_map_file", path)
    mock = MagicMock(wraps=real_sms.clear_session_map_entry)
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


@pytest.fixture
def reply() -> Iterator[AsyncMock]:
    """Capture ``safe_reply(message, text, reply_markup=...)`` calls."""
    with patch(
        "ccgram.handlers.agent_command.safe_reply", new_callable=AsyncMock
    ) as mock:
        yield mock


@pytest.fixture
def edit() -> Iterator[AsyncMock]:
    """Capture ``safe_edit(query, text, reply_markup=...)`` calls."""
    with patch(
        "ccgram.handlers.agent_command.safe_edit", new_callable=AsyncMock
    ) as mock:
        yield mock


def _sent_text(mock: AsyncMock) -> str:
    return mock.call_args.args[1]


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
    update.get_bot.return_value = AsyncMock()
    return update


def _make_query(data: str):
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.get_bot.return_value = AsyncMock()
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


async def test_bare_command_shows_picker(reply):
    await ac.agent_command(_make_update(), MagicMock())

    assert "claude" in _sent_text(reply).lower()
    keyboard = reply.call_args.kwargs["reply_markup"].inline_keyboard
    callbacks = [b.callback_data for row in keyboard for b in row]
    assert any(c.startswith(CB_AGENT_SET) for c in callbacks)
    assert any(c == f"{CB_AGENT_SET}@7:shell" for c in callbacks)
    assert any(c == f"{CB_AGENT_SET}@7:auto" for c in callbacks)
    assert any(c == f"{CB_AGENT_CANCEL}@7" for c in callbacks)


async def test_arg_shell_switch_clears_mismatched_hook_map(
    monkeypatch, clear_map_mock, reply
):
    """A Terminal switch clears a mapped agent transcript."""
    monkeypatch.setattr(ac, "_redetect_provider", AsyncMock(return_value="shell"))
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
    config.session_map_file.write_text(
        json.dumps(
            {
                f"{session_map_prefix()}@7": {
                    "schema_version": 1,
                    "session_id": "OLDSID",
                    "cwd": "/tmp/proj",
                    "window_name": "proj",
                    "transcript_path": "/tmp/old.jsonl",
                    "provider_name": "claude",
                }
            }
        )
    )

    await ac.agent_command(_make_update("/agent shell"), MagicMock())

    state = window_store.window_states["@7"]
    assert state.provider_name == "shell"
    assert state.transcript_path == ""
    assert state.provider_manual_override is True
    assert clear_map_mock.call_args == call("@7")
    assert "Terminal" in _sent_text(reply)
    assert ensure_setup_mock.await_count == 1
    args, kwargs = ensure_setup_mock.call_args
    assert args == ("@7", "provider_switch")
    assert kwargs["chat_id"] == -100
    assert kwargs["thread_id"] == 99
    assert kwargs["client"] is not None
    assert "Terminal" in _sent_text(reply)
    assert "will install" not in _sent_text(reply)
    assert state.session_id == ""
    assert json.loads(config.session_map_file.read_text()) == {}


async def test_arg_unknown_is_rejected(reply):
    await ac.agent_command(_make_update("/agent garbage"), MagicMock())

    assert "Unknown agent" in _sent_text(reply)
    # State unchanged
    assert window_store.window_states["@7"].provider_name == "claude"


def _confirm_live(monkeypatch, window_id: str) -> None:
    """Report *window_id* live through the reconciliation seam.

    /agent auto persists what detection returns and clears transcript
    bookkeeping, so it refuses to resolve at all unless the window is
    confirmed present.
    """
    from ccgram.multiplexer.base import WindowRef

    monkeypatch.setattr(
        "ccgram.multiplexer.tmux.tmux_manager.list_windows_for_reconciliation",
        AsyncMock(
            return_value=[
                WindowRef(window_id=window_id, window_name="proj", cwd="/proj")
            ]
        ),
    )


async def test_auto_clears_override_and_redetects(monkeypatch, reply):
    # Pre-mark as manual override so we can verify it gets cleared.
    identity_state.set_provider_manual_override("@7", value=True)

    fake_window = MagicMock()
    fake_window.pane_current_command = "codex"
    fake_window.pane_tty = "/dev/ttys00"
    monkeypatch.setattr(
        "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
        AsyncMock(return_value=fake_window),
    )
    _confirm_live(monkeypatch, "@7")
    monkeypatch.setattr(
        "ccgram.providers.detect_provider_from_pane",
        AsyncMock(return_value="codex"),
    )

    await ac.agent_command(_make_update("/agent auto"), MagicMock())

    state = window_store.window_states["@7"]
    assert state.provider_name == "codex"
    assert state.provider_manual_override is False
    assert "**Codex**" in _sent_text(reply)


async def test_auto_falls_back_to_shell_and_triggers_ensure_setup(monkeypatch, reply):
    """When /agent auto resolves to shell, ensure_setup must run so the
    'Set up / Skip' offer keyboard can render — same UX as an explicit
    /agent shell. Regression for: auto→shell silently skipped the offer."""
    fake_window = MagicMock()
    fake_window.pane_current_command = "ralphex"
    fake_window.pane_tty = "/dev/ttys00"
    monkeypatch.setattr(
        "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
        AsyncMock(return_value=fake_window),
    )
    _confirm_live(monkeypatch, "@7")
    monkeypatch.setattr(
        "ccgram.providers.detect_provider_from_pane",
        AsyncMock(return_value=""),
    )
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

    await ac.agent_command(_make_update("/agent auto"), MagicMock())

    assert window_store.window_states["@7"].provider_name == "shell"
    assert "shell" in _sent_text(reply)
    assert ensure_setup_mock.await_count == 1
    args, kwargs = ensure_setup_mock.call_args
    assert args == ("@7", "provider_switch")
    assert kwargs["chat_id"] == -100
    assert kwargs["thread_id"] == 99


async def test_callback_cancel_keeps_state(monkeypatch, edit):
    monkeypatch.setattr(
        "ccgram.handlers.agent_command.user_owns_window", lambda u, w: True
    )
    update = _make_query(f"{CB_AGENT_CANCEL}@7")
    await ac._dispatch(update, MagicMock())

    assert window_store.window_states["@7"].provider_name == "claude"
    assert "Cancelled" in _sent_text(edit)


async def test_callback_set_provider_clears_previous_provider_map(
    monkeypatch, clear_map_mock, edit
):
    """A provider change removes the old provider's mapped transcript."""
    monkeypatch.setattr(ac, "_redetect_provider", AsyncMock(return_value="gemini"))
    monkeypatch.setattr(
        "ccgram.handlers.agent_command.user_owns_window", lambda u, w: True
    )
    config.session_map_file.write_text(
        json.dumps(
            {
                f"{session_map_prefix()}@7": {
                    "schema_version": 1,
                    "session_id": "OLDSID",
                    "cwd": "/tmp/proj",
                    "window_name": "proj",
                    "transcript_path": "/tmp/old.jsonl",
                    "provider_name": "claude",
                }
            }
        )
    )
    update = _make_query(f"{CB_AGENT_SET}@7:gemini")
    await ac._dispatch(update, MagicMock())

    state = window_store.window_states["@7"]
    assert state.provider_name == "gemini"
    assert state.transcript_path == ""
    assert state.provider_manual_override is True
    assert clear_map_mock.called
    assert clear_map_mock.call_args.args == ("@7",)
    assert json.loads(config.session_map_file.read_text()) == {}


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


async def test_command_outside_bound_topic_replies_hint(monkeypatch, reply):
    monkeypatch.setattr(ac.thread_router, "get_window_for_thread", lambda u, t: "")
    await ac.agent_command(_make_update("/agent shell"), MagicMock())

    assert "bound topic" in _sent_text(reply)
    # No state change — state still has the prior provider.
    assert window_store.window_states["@7"].provider_name == "claude"


async def test_same_provider_readmission_preserves_primary_map_entry(
    monkeypatch, clear_map_mock, reply
):
    """Same-provider lock-in preserves its primary session-map entry."""
    monkeypatch.setattr(ac, "_redetect_provider", AsyncMock(return_value="claude"))
    path = config.session_map_file
    path.write_text(
        json.dumps(
            {
                f"{session_map_prefix()}@7": {
                    "schema_version": 1,
                    "session_id": "OLDSID",
                    "cwd": "/tmp/proj",
                    "window_name": "proj",
                    "transcript_path": "/tmp/old.jsonl",
                    "provider_name": "claude",
                }
            }
        )
    )
    await ac.agent_command(_make_update("/agent claude"), MagicMock())

    state = window_store.window_states["@7"]
    assert state.provider_name == "claude"
    # Transcript bookkeeping must survive — same-provider switch is a flag-only op.
    assert state.transcript_path == "/tmp/old.jsonl"
    assert state.provider_manual_override is True
    assert state.session_id == "OLDSID"
    clear_map_mock.assert_called_once_with("@7")
    saved = json.loads(path.read_text())
    assert saved[f"{session_map_prefix()}@7"]["session_id"] == "OLDSID"


def test_hookless_provider_waits_for_transcript_discovery():
    assert "transcript discovery" in ac._tracking_wait_message("antigravity")


@pytest.mark.parametrize(
    ("live_provider", "requested_provider"),
    [("shell", "pi"), ("pi", "shell")],
)
async def test_manual_provider_mismatch_is_rejected_without_state_or_map_changes(
    monkeypatch, clear_map_mock, live_provider, requested_provider
):
    state = window_store.window_states["@7"]
    before = state.to_dict()
    monkeypatch.setattr(ac, "_redetect_provider", AsyncMock(return_value=live_provider))

    provider, reply_text = await ac._apply_switch("@7", requested_provider)

    assert provider == "claude"
    assert f"{provider_label(live_provider)} is running" in reply_text
    assert "Provider unchanged" in reply_text
    assert state.to_dict() == before
    clear_map_mock.assert_not_called()


async def test_manual_switch_rejects_unidentified_foreground_process(
    monkeypatch, clear_map_mock
):
    state = window_store.window_states["@7"]
    before = state.to_dict()
    fake_window = MagicMock()
    fake_window.pane_current_command = ""
    monkeypatch.setattr(
        "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
        AsyncMock(return_value=fake_window),
    )
    _confirm_live(monkeypatch, "@7")

    provider, reply_text = await ac._apply_switch("@7", "shell")

    assert provider == "claude"
    assert "Could not verify" in reply_text
    assert state.to_dict() == before
    clear_map_mock.assert_not_called()


async def test_manual_switch_fails_closed_when_live_provider_is_unknown(
    monkeypatch, clear_map_mock
):
    state = window_store.window_states["@7"]
    before = state.to_dict()
    monkeypatch.setattr(ac, "_redetect_provider", AsyncMock(return_value=None))

    provider, reply_text = await ac._apply_switch("@7", "shell")

    assert provider == "claude"
    assert "Could not verify" in reply_text
    assert state.to_dict() == before
    clear_map_mock.assert_not_called()


@pytest.mark.parametrize("target", ["shell", "pi"])
@pytest.mark.parametrize(
    "pinned,claimed", [("claude", "claude"), ("shell", "claude"), ("claude", "pi")]
)
async def test_auto_shell_clears_old_agent_session_before_poll_reload(
    monkeypatch, clear_map_mock, tmp_path, target, pinned, claimed
):
    state = window_store.window_states["@7"]
    state.provider_name = "claude"
    state.session_id = "old-claude-session"
    state.transcript_path = "/Users/test/.claude/projects/project/old.jsonl"
    state.provider_manual_override = False
    raw_path = tmp_path / "session_map.json"
    raw_path.write_text(
        json.dumps(
            {
                f"{session_map_prefix()}@7": {
                    "schema_version": 1,
                    "session_id": "old-claude-session",
                    "cwd": "/tmp/proj",
                    "window_name": "proj",
                    "transcript_path": state.transcript_path,
                    "provider_name": claimed,
                }
            }
        )
    )
    monkeypatch.setattr(config, "session_map_file", raw_path)
    session_manager.set_window_provider("@7", pinned, preserve_session_map=True)
    state.provider_manual_override = True
    assert raw_path.exists()
    monkeypatch.setattr(ac, "_redetect_provider", AsyncMock(return_value=target))

    provider, reply_text = await ac._apply_switch("@7", "auto")

    assert provider == target
    assert state.provider_name == target and not state.provider_manual_override
    assert state.session_id == ""
    assert state.transcript_path == ""
    assert "Existing session restored" not in reply_text
    assert json.loads(raw_path.read_text()) == {}
    clear_map_mock.assert_called_once_with("@7")
    await session_map_sync.load_session_map()
    assert state.provider_name == target and state.session_id == ""


async def test_auto_keeps_provider_pinned_when_map_cleanup_is_unconfirmed(monkeypatch):
    state = window_store.window_states["@7"]
    state.provider_name = "claude"
    state.provider_manual_override = True
    monkeypatch.setattr(ac, "_redetect_provider", AsyncMock(return_value="shell"))
    monkeypatch.setattr(
        "ccgram.session_map.session_map_sync.clear_session_map_entry",
        lambda _wid: None,
    )
    monkeypatch.setattr(
        "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
        AsyncMock(),
    )

    provider, reply_text = await ac._apply_switch("@7", "auto")

    assert provider == "shell"
    assert state.provider_name == "shell" and state.provider_manual_override
    assert "remains pinned" in reply_text


async def test_auto_restores_exact_hook_session_saved_during_manual_override(
    monkeypatch, clear_map_mock, tmp_path
):
    state = window_store.window_states["@7"]
    state.provider_name = "pi"
    state.session_id = "live-pi-session"
    state.transcript_path = (
        "/Users/test/.pi/agent/sessions/project/live-pi-session.jsonl"
    )
    state.provider_manual_override = False
    raw_path = tmp_path / "session_map.json"
    raw_path.write_text(
        json.dumps(
            {
                f"{session_map_prefix()}@7": {
                    "schema_version": 1,
                    "session_id": "live-pi-session",
                    "cwd": "/tmp/proj",
                    "window_name": "proj",
                    "transcript_path": state.transcript_path,
                    "provider_name": "pi",
                }
            }
        )
    )
    monkeypatch.setattr(config, "session_map_file", raw_path)
    session_manager.set_window_provider("@7", "shell", preserve_session_map=True)
    state.provider_manual_override = True
    assert state.session_id == ""
    assert raw_path.exists()
    monkeypatch.setattr(ac, "_redetect_provider", AsyncMock(return_value="pi"))

    provider, reply_text = await ac._apply_switch("@7", "auto")

    assert provider == "pi"
    assert state.provider_name == "pi" and not state.provider_manual_override
    assert state.session_id == "live-pi-session"
    assert (
        state.transcript_path
        == "/Users/test/.pi/agent/sessions/project/live-pi-session.jsonl"
    )
    assert "Existing session restored" in reply_text
    clear_map_mock.assert_called_once_with("@7")


async def test_auto_resolving_to_same_provider_clears_override(
    monkeypatch, clear_map_mock, reply
):
    """/agent auto on a window whose foreground matches the stored provider
    must preserve the session_map entry (no actual transition) and still
    clear the manual-override flag."""
    identity_state.set_provider_manual_override("@7", value=True)

    fake_window = MagicMock()
    fake_window.pane_current_command = "claude"
    fake_window.pane_tty = "/dev/ttys00"
    monkeypatch.setattr(
        "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
        AsyncMock(return_value=fake_window),
    )
    _confirm_live(monkeypatch, "@7")
    monkeypatch.setattr(
        "ccgram.providers.detect_provider_from_pane",
        AsyncMock(return_value="claude"),
    )
    await ac.agent_command(_make_update("/agent auto"), MagicMock())

    state = window_store.window_states["@7"]
    assert state.provider_name == "claude"
    assert state.provider_manual_override is False
    # No actual transition — session_map untouched, transcript preserved.
    assert state.transcript_path == "/tmp/old.jsonl"
    clear_map_mock.assert_called_once_with("@7")


async def test_disallowed_user_is_rejected(reply):
    with patch("ccgram.config.Config.is_user_allowed", return_value=False):
        await ac.agent_command(_make_update("/agent claude"), MagicMock())

    assert "bound topic" in _sent_text(reply)
    assert window_store.window_states["@7"].provider_name == "claude"


async def test_picker_text_shows_manual_override_badge(reply):
    identity_state.set_provider_manual_override("@7", value=True)

    await ac.agent_command(_make_update(), MagicMock())

    assert "Manual" in _sent_text(reply)
    assert "@7" not in _sent_text(reply)


@pytest.mark.parametrize(
    ("data", "owns_window", "expected_answer"),
    [
        pytest.param(
            f"{CB_AGENT_CANCEL}@7", False, "Not your window", id="cancel-foreign-user"
        ),
        pytest.param(
            f"{CB_AGENT_SET}@7:shell", False, "Not your window", id="set-foreign-user"
        ),
        pytest.param(
            f"{CB_AGENT_SET}garbled", True, "Bad callback", id="missing-separator"
        ),
        pytest.param(
            f"{CB_AGENT_SET}@7:bogus", True, "Unknown provider", id="unknown-provider"
        ),
    ],
)
async def test_rejected_callbacks_leave_provider_untouched(
    monkeypatch, data: str, owns_window: bool, expected_answer: str
):
    monkeypatch.setattr(
        "ccgram.handlers.agent_command.user_owns_window", lambda u, w: owns_window
    )

    update = _make_query(data)
    await ac._dispatch(update, MagicMock())

    update.callback_query.answer.assert_awaited_once_with(expected_answer)
    assert window_store.window_states["@7"].provider_name == "claude"


class TestAgentAutoNeedsConfirmedLiveness:
    """/agent auto persists what it resolves, so an unknown answer must not.

    find_window_by_id returns None both for a window that is gone and for a
    backend that could not be reached. Resolving that to shell rewrites a
    working session's provider and clears its transcript bookkeeping.
    """

    @staticmethod
    async def _run(monkeypatch, reply, *, listing):
        identity_state.set_provider_manual_override("@7", value=True)
        monkeypatch.setattr(
            "ccgram.multiplexer.tmux.tmux_manager.list_windows_for_reconciliation",
            AsyncMock(return_value=listing),
        )
        monkeypatch.setattr(
            "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
            AsyncMock(return_value=None),
        )
        await ac.agent_command(_make_update("/agent auto"), MagicMock())
        return window_store.window_states["@7"]

    async def test_unreachable_backend_changes_nothing(self, monkeypatch, reply):
        state = await self._run(monkeypatch, reply, listing=None)

        assert state.provider_name == "claude"
        assert state.provider_manual_override is True
        assert "Could not reach" in _sent_text(reply)

    async def test_confirmed_absence_changes_nothing(self, monkeypatch, reply):
        state = await self._run(monkeypatch, reply, listing=[])

        assert state.provider_name == "claude"
        assert state.provider_manual_override is True
        assert "Could not reach" in _sent_text(reply)
