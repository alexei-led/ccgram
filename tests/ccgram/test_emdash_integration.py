"""Tests for emdash integration — foreign window support."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.handlers.directory_browser import _window_label, build_window_picker
from ccgram.session import (
    SessionManager,
    WindowState,
    parse_emdash_provider,
)
from ccgram.window_resolver import EMDASH_SESSION_PREFIX, is_foreign_window

# ── Pure helpers ──────────────────────────────────────────────────────


class TestIsForeignWindow:
    @pytest.mark.parametrize(
        ("window_id", "expected"),
        [
            ("@0", False),
            ("@12", False),
            ("emdash-claude-main-abc123:@0", True),
            ("emdash-codex-chat-def456:@0", True),
            ("other-session:@3", True),
            ("", False),
        ],
        ids=["native", "native-12", "emdash-claude", "emdash-codex", "other", "empty"],
    )
    def test_detection(self, window_id: str, expected: bool) -> None:
        assert is_foreign_window(window_id) is expected


class TestParseEmdashProvider:
    @pytest.mark.parametrize(
        ("session_name", "expected"),
        [
            ("emdash-claude-main-abc123", "claude"),
            ("emdash-codex-main-def456", "codex"),
            ("emdash-gemini-chat-xyz789", "gemini"),
            ("emdash-amp-main-111", "amp"),
            ("emdash-unknown", ""),
            ("not-emdash", ""),
            ("", ""),
        ],
        ids=["claude", "codex", "gemini", "amp", "no-sep", "wrong-prefix", "empty"],
    )
    def test_provider_extraction(self, session_name: str, expected: str) -> None:
        assert parse_emdash_provider(session_name) == expected


class TestEmdashSessionPrefix:
    def test_value(self) -> None:
        assert EMDASH_SESSION_PREFIX == "emdash-"


# ── WindowState.external ──────────────────────────────────────────────


class TestWindowStateExternal:
    def test_default_is_false(self) -> None:
        ws = WindowState()
        assert ws.external is False

    def test_to_dict_omits_when_false(self) -> None:
        ws = WindowState(session_id="s1", cwd="/tmp")
        assert "external" not in ws.to_dict()

    def test_to_dict_includes_when_true(self) -> None:
        ws = WindowState(session_id="s1", cwd="/tmp", external=True)
        d = ws.to_dict()
        assert d["external"] is True

    def test_from_dict_reads_external(self) -> None:
        ws = WindowState.from_dict(
            {"session_id": "s1", "cwd": "/tmp", "external": True}
        )
        assert ws.external is True

    def test_from_dict_defaults_false(self) -> None:
        ws = WindowState.from_dict({"session_id": "s1", "cwd": "/tmp"})
        assert ws.external is False


# ── load_session_map with emdash entries ──────────────────────────────


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


class TestLoadSessionMapEmdash:
    async def test_picks_up_emdash_entries(
        self, mgr: SessionManager, tmp_path: Path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "emdash-claude-main-abc123:@0": {
                        "session_id": "sid-emdash",
                        "cwd": "/tmp/project",
                        "window_name": "proj",
                        "transcript_path": "/home/user/.claude/sessions/sid.jsonl",
                        "provider_name": "claude",
                    }
                }
            )
        )
        monkeypatch.setattr("ccgram.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccgram.session.config.tmux_session_name", "ccgram")

        await mgr.load_session_map()

        wid = "emdash-claude-main-abc123:@0"
        assert wid in mgr.window_states
        state = mgr.window_states[wid]
        assert state.session_id == "sid-emdash"
        assert state.cwd == "/tmp/project"
        assert state.external is True
        assert state.provider_name == "claude"

    async def test_infers_provider_from_session_name(
        self, mgr: SessionManager, tmp_path: Path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "emdash-codex-main-def456:@0": {
                        "session_id": "sid-codex",
                        "cwd": "/tmp/codex-project",
                    }
                }
            )
        )
        monkeypatch.setattr("ccgram.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccgram.session.config.tmux_session_name", "ccgram")

        await mgr.load_session_map()

        wid = "emdash-codex-main-def456:@0"
        assert mgr.window_states[wid].provider_name == "codex"
        assert mgr.window_states[wid].external is True

    async def test_coexists_with_native_entries(
        self, mgr: SessionManager, tmp_path: Path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccgram:@1": {
                        "session_id": "sid-native",
                        "cwd": "/tmp/native",
                    },
                    "emdash-claude-main-abc:@0": {
                        "session_id": "sid-emdash",
                        "cwd": "/tmp/emdash",
                    },
                }
            )
        )
        monkeypatch.setattr("ccgram.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccgram.session.config.tmux_session_name", "ccgram")

        await mgr.load_session_map()

        assert "@1" in mgr.window_states
        assert mgr.window_states["@1"].external is False
        assert "emdash-claude-main-abc:@0" in mgr.window_states
        assert mgr.window_states["emdash-claude-main-abc:@0"].external is True

    async def test_emdash_entries_not_pruned_as_stale(
        self, mgr: SessionManager, tmp_path: Path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "emdash-claude-main-abc:@0": {
                        "session_id": "sid-emdash",
                        "cwd": "/tmp/emdash",
                    }
                }
            )
        )
        monkeypatch.setattr("ccgram.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccgram.session.config.tmux_session_name", "ccgram")

        await mgr.load_session_map()
        # Second load should not prune emdash entries
        await mgr.load_session_map()

        assert "emdash-claude-main-abc:@0" in mgr.window_states


class TestGetSessionMapWindowIds:
    def test_includes_emdash_entries(
        self, mgr: SessionManager, tmp_path: Path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccgram:@1": {"session_id": "s1", "cwd": "/a"},
                    "emdash-claude-main-abc:@0": {"session_id": "s2", "cwd": "/b"},
                    "other:@9": {"session_id": "s3", "cwd": "/c"},
                }
            )
        )
        monkeypatch.setattr("ccgram.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccgram.session.config.tmux_session_name", "ccgram")

        result = mgr._get_session_map_window_ids()
        assert "@1" in result
        assert "emdash-claude-main-abc:@0" in result
        assert "@9" not in result  # different session prefix


class TestPruneStaleWindowStatesEmdash:
    def test_keeps_emdash_states_in_session_map(
        self, mgr: SessionManager, tmp_path: Path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "emdash-claude-main-abc:@0": {
                        "session_id": "s-emdash",
                        "cwd": "/tmp",
                    }
                }
            )
        )
        monkeypatch.setattr("ccgram.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccgram.session.config.tmux_session_name", "ccgram")

        emdash_wid = "emdash-claude-main-abc:@0"
        mgr.window_states[emdash_wid] = WindowState(
            session_id="s-emdash", cwd="/tmp", external=True
        )
        changed = mgr.prune_stale_window_states(live_window_ids=set())
        assert not changed
        assert emdash_wid in mgr.window_states


# ── Thread binding with foreign windows ───────────────────────────────


class TestForeignWindowBindings:
    def test_bind_and_resolve_foreign_window(self, mgr: SessionManager) -> None:
        emdash_wid = "emdash-claude-main-abc:@0"
        mgr.bind_thread(100, 42, emdash_wid, window_name="proj")
        assert mgr.get_window_for_thread(100, 42) == emdash_wid

    def test_iter_includes_foreign_bindings(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.bind_thread(100, 2, "emdash-claude-main-abc:@0")
        result = set(mgr.iter_thread_bindings())
        assert (100, 2, "emdash-claude-main-abc:@0") in result


# ── TmuxManager foreign window support ────────────────────────────────


class TestKillWindowForeignGuard:
    async def test_skip_kill_for_foreign_window(self) -> None:
        from ccgram.tmux_manager import TmuxManager

        tm = TmuxManager(session_name="test")
        result = await tm.kill_window("emdash-claude-main-abc:@0")
        assert result is False

    async def test_allows_kill_for_native_window(self) -> None:
        from ccgram.tmux_manager import TmuxManager

        tm = TmuxManager(session_name="test")
        with patch.object(tm, "get_session", return_value=None):
            result = await tm.kill_window("@5")
        assert result is False  # no session, but didn't skip


# ── Window picker with emdash windows ─────────────────────────────────


class TestWindowLabel:
    def test_native_window(self) -> None:
        icon, name = _window_label("@5", "myproject")
        assert icon == "🖥"
        assert name == "myproject"

    def test_emdash_claude_window(self) -> None:
        icon, name = _window_label("emdash-claude-main-abc:@0", "claude-main-abc")
        assert icon == "📎"
        assert "claude" in name

    def test_emdash_codex_window(self) -> None:
        icon, name = _window_label("emdash-codex-main-xyz:@0", "codex-main-xyz")
        assert icon == "📎"
        assert "codex" in name


class TestBuildWindowPickerEmdash:
    def test_includes_emdash_in_picker(self) -> None:
        windows = [
            ("@5", "native-proj", "/tmp/native"),
            ("emdash-claude-main-abc:@0", "claude-main-abc", "/tmp/emdash"),
        ]
        text, keyboard, win_ids = build_window_picker(windows)

        assert "emdash-claude-main-abc:@0" in win_ids
        assert "@5" in win_ids
        assert "📎" in text
        assert "🖥" in text


# ── Resolve stale IDs skips foreign windows ───────────────────────────


class TestResolveStaleIdsEmdash:
    async def test_preserves_foreign_bindings(
        self, mgr: SessionManager, monkeypatch
    ) -> None:
        emdash_wid = "emdash-claude-main-abc:@0"
        mgr.bind_thread(100, 1, "@1", window_name="alive-proj")
        mgr.bind_thread(100, 2, emdash_wid, window_name="emdash-proj")
        mgr.window_states[emdash_wid] = WindowState(
            session_id="s-emdash", cwd="/tmp/emdash", external=True
        )

        alive = SimpleNamespace(window_id="@1", window_name="alive-proj")
        from ccgram.tmux_manager import tmux_manager

        monkeypatch.setattr(
            "ccgram.session.config.session_map_file",
            Path("/nonexistent/session_map.json"),
        )
        with patch.object(
            tmux_manager, "list_windows", AsyncMock(return_value=[alive])
        ):
            await mgr.resolve_stale_ids()

        # Foreign binding preserved (not re-resolved)
        assert mgr.get_window_for_thread(100, 2) == emdash_wid
        assert emdash_wid in mgr.window_states
