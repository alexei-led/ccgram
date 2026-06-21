from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ccgram.handlers.status import rc_probe
from ccgram.handlers.status.rc_probe import (
    RCOutcome,
    RCOutcomeKind,
    arm_rc_probe,
    classify_rc_output,
)
from ccgram.session import SessionManager
from ccgram.telegram_client import FakeTelegramClient
from ccgram.thread_router import thread_router
from ccgram.window_state_store import window_store


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    thread_router.reset()
    window_store.window_states.clear()
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


def _claude_provider(*_a, **_k):
    return SimpleNamespace(capabilities=SimpleNamespace(name="claude"))


def _codex_provider(*_a, **_k):
    return SimpleNamespace(capabilities=SimpleNamespace(name="codex"))


class TestClassifyRcOutput:
    def test_success_claude_url(self) -> None:
        text = "Starting…\nOpen https://claude.ai/remote/abc123 to control"
        out = classify_rc_output(text)
        assert out.kind is RCOutcomeKind.SUCCESS
        assert out.detail == "https://claude.ai/remote/abc123"

    def test_success_generic_remote_url(self) -> None:
        out = classify_rc_output("visit https://example.com/remote-session/x")
        assert out.kind is RCOutcomeKind.SUCCESS
        assert "remote" in out.detail

    def test_success_url_wins_over_error_line(self) -> None:
        text = "an error happened earlier\nhttps://claude.ai/remote/z9"
        out = classify_rc_output(text)
        assert out.kind is RCOutcomeKind.SUCCESS
        assert out.detail == "https://claude.ai/remote/z9"

    @pytest.mark.parametrize(
        "phrase",
        [
            "Remote Control is not available on this plan",
            "This requires a Max subscription",
            "Please upgrade to use this feature",
            "permission denied",
            "Unknown command: /remote-control",
        ],
    )
    def test_unavailable_phrases(self, phrase: str) -> None:
        out = classify_rc_output(f"$ /remote-control\n{phrase}")
        assert out.kind is RCOutcomeKind.UNAVAILABLE
        assert out.detail == phrase

    def test_failed_error_line(self) -> None:
        out = classify_rc_output("/remote-control\nInternal error: boom")
        assert out.kind is RCOutcomeKind.FAILED
        assert "error" in out.detail.lower()

    def test_failed_keyword(self) -> None:
        out = classify_rc_output("/remote-control\nconnection failed after retry")
        assert out.kind is RCOutcomeKind.FAILED

    def test_unanchored_error_in_scrollback_is_pending(self) -> None:
        text = "TypeError: boom\nthe build failed\nrunning tests..."
        assert classify_rc_output(text).kind is RCOutcomeKind.PENDING

    def test_error_before_anchor_not_failed(self) -> None:
        text = "compilation error in foo.py\n/remote-control\nConnecting…"
        assert classify_rc_output(text).kind is RCOutcomeKind.PENDING

    def test_error_after_anchor_is_failed(self) -> None:
        text = "all good\n/rc\nFatal error: connection refused"
        out = classify_rc_output(text)
        assert out.kind is RCOutcomeKind.FAILED
        assert "error" in out.detail.lower()

    def test_pending_no_match(self) -> None:
        out = classify_rc_output("just some normal terminal output\n$ ")
        assert out.kind is RCOutcomeKind.PENDING

    def test_pending_empty(self) -> None:
        assert classify_rc_output("").kind is RCOutcomeKind.PENDING


class TestFormatReply:
    def test_success_with_url_monospace(self) -> None:
        msg = rc_probe._format_reply(
            RCOutcome(RCOutcomeKind.SUCCESS, "https://claude.ai/remote/x")
        )
        assert "`https://claude.ai/remote/x`" in msg

    def test_success_without_url(self) -> None:
        msg = rc_probe._format_reply(RCOutcome(RCOutcomeKind.SUCCESS))
        assert msg == "\U0001f4e1 Remote Control active."

    def test_unavailable(self) -> None:
        msg = rc_probe._format_reply(
            RCOutcome(RCOutcomeKind.UNAVAILABLE, "not available")
        )
        assert msg == "\U0001f4e1 Remote Control unavailable — not available."

    def test_failed(self) -> None:
        msg = rc_probe._format_reply(RCOutcome(RCOutcomeKind.FAILED, "boom error"))
        assert msg == "\U0001f4e1 Remote Control failed — boom error."

    def test_timeout(self) -> None:
        msg = rc_probe._format_reply(RCOutcome(RCOutcomeKind.PENDING))
        assert "No response" in msg


class TestArmRcProbeGate:
    async def test_double_tap_guard(self, mgr, monkeypatch) -> None:
        monkeypatch.setattr(rc_probe, "get_provider_for_window", _claude_provider)
        calls: list[str] = []

        async def fake_loop(window_id: str, client) -> None:
            calls.append(window_id)

        monkeypatch.setattr(rc_probe, "_classify_loop", fake_loop)
        client = FakeTelegramClient()

        arm_rc_probe("@3", client)
        state = window_store.get_window_state("@3")
        assert state.rc_probe_state == "armed"
        assert state.rc_armed_at is not None

        arm_rc_probe("@3", client)
        await asyncio.sleep(0)
        assert calls == ["@3"]
        assert window_store.get_window_state("@3").rc_probe_state == "armed"

    async def test_capability_gate_non_claude(self, mgr, monkeypatch) -> None:
        monkeypatch.setattr(rc_probe, "get_provider_for_window", _codex_provider)
        spawned: list[str] = []

        async def fake_loop(window_id: str, client) -> None:
            spawned.append(window_id)

        monkeypatch.setattr(rc_probe, "_classify_loop", fake_loop)

        arm_rc_probe("@4", FakeTelegramClient())

        assert spawned == []
        assert window_store.get_window_state("@4").rc_probe_state is None


class TestClassifyLoop:
    async def test_sends_url_on_second_poll(self, mgr, monkeypatch) -> None:
        monkeypatch.setattr(rc_probe, "_FIRST_CAPTURE_DELAY", 0.0)
        monkeypatch.setattr(rc_probe, "_RETRY_INTERVAL", 0.0)
        monkeypatch.setattr(rc_probe, "_TOTAL_TIMEOUT", 5.0)
        monkeypatch.setattr(
            "ccgram.multiplexer.tmux.tmux_manager.capture_pane",
            AsyncMock(
                side_effect=[
                    "still working\n$ ",
                    "done\nhttps://claude.ai/remote/win7",
                ]
            ),
        )
        monkeypatch.setattr(
            thread_router, "iter_thread_bindings", lambda: iter([(111, 42, "@7")])
        )
        monkeypatch.setattr(thread_router, "resolve_chat_id", lambda *_a, **_k: 999)

        state = window_store.get_window_state("@7")
        state.rc_probe_state = "armed"
        client = FakeTelegramClient()

        await rc_probe._classify_loop("@7", client)

        assert client.call_count("send_message") == 1
        call = client.last_call("send_message")
        assert call is not None
        assert "https://claude.ai/remote/win7" in call.kwargs["text"]
        assert call.kwargs["chat_id"] == 999
        assert call.kwargs["message_thread_id"] == 42
        assert window_store.get_window_state("@7").rc_probe_state == "classified"

    async def test_timeout_resets_state(self, mgr, monkeypatch) -> None:
        monkeypatch.setattr(rc_probe, "_FIRST_CAPTURE_DELAY", 0.0)
        monkeypatch.setattr(rc_probe, "_RETRY_INTERVAL", 0.0)
        monkeypatch.setattr(rc_probe, "_TOTAL_TIMEOUT", 0.0)
        monkeypatch.setattr(
            "ccgram.multiplexer.tmux.tmux_manager.capture_pane",
            AsyncMock(return_value="nothing"),
        )
        monkeypatch.setattr(
            thread_router, "iter_thread_bindings", lambda: iter([(111, 42, "@8")])
        )
        monkeypatch.setattr(thread_router, "resolve_chat_id", lambda *_a, **_k: 999)

        state = window_store.get_window_state("@8")
        state.rc_probe_state = "armed"
        client = FakeTelegramClient()

        await rc_probe._classify_loop("@8", client)

        last = client.last_call("send_message")
        assert last is not None
        assert "No response" in last.kwargs["text"]
        assert window_store.get_window_state("@8").rc_probe_state == "classified"


class TestNoTelegramBotImport:
    def test_module_does_not_import_telegram_bot(self) -> None:
        src = Path(rc_probe.__file__).read_text(encoding="utf-8")
        assert "from telegram import Bot" not in src
        assert "telegram.Bot" not in src
