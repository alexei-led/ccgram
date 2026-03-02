"""Tests for SessionManager pure dict operations."""

import json

import pytest

from ccbot.session import SessionManager, WindowState


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


class TestThreadBindings:
    def test_bind_and_get(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        assert mgr.get_window_for_thread(100, 1) == "@1"

    def test_bind_unbind_get_returns_none(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.unbind_thread(100, 1)
        assert mgr.get_window_for_thread(100, 1) is None

    def test_unbind_nonexistent_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.unbind_thread(100, 999) is None

    def test_get_thread_for_window(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 42, "@5")
        assert mgr.get_thread_for_window(100, "@5") == 42

    def test_iter_thread_bindings(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.bind_thread(100, 2, "@2")
        mgr.bind_thread(200, 3, "@3")
        result = set(mgr.iter_thread_bindings())
        assert result == {(100, 1, "@1"), (100, 2, "@2"), (200, 3, "@3")}


class TestResolveChatId:
    def test_with_stored_group_id(self, mgr: SessionManager) -> None:
        mgr.set_group_chat_id(100, 1, -999)
        assert mgr.resolve_chat_id(100, 1) == -999

    def test_without_group_id_falls_back(self, mgr: SessionManager) -> None:
        assert mgr.resolve_chat_id(100, 1) == 100

    def test_none_thread_id_falls_back(self, mgr: SessionManager) -> None:
        mgr.set_group_chat_id(100, 1, -999)
        assert mgr.resolve_chat_id(100) == 100


class TestWindowState:
    def test_get_creates_new(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@0")
        assert state.session_id == ""
        assert state.cwd == ""

    def test_get_returns_existing(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@1")
        state.session_id = "abc"
        assert mgr.get_window_state("@1").session_id == "abc"

    def test_clear_window_session(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@1")
        state.session_id = "abc"
        mgr.clear_window_session("@1")
        assert mgr.get_window_state("@1").session_id == ""


class TestResolveWindowForThread:
    def test_none_thread_id_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.resolve_window_for_thread(100, None) is None

    def test_unbound_thread_returns_none(self, mgr: SessionManager) -> None:
        assert mgr.resolve_window_for_thread(100, 42) is None

    def test_bound_thread_returns_window(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 42, "@3")
        assert mgr.resolve_window_for_thread(100, 42) == "@3"


class TestDisplayNames:
    def test_get_display_name_fallback(self, mgr: SessionManager) -> None:
        """get_display_name returns window_id when no display name is set."""
        assert mgr.get_display_name("@99") == "@99"

    def test_set_and_get_display_name(self, mgr: SessionManager) -> None:
        mgr.set_display_name("@1", "myproject")
        assert mgr.get_display_name("@1") == "myproject"

    def test_set_display_name_update(self, mgr: SessionManager) -> None:
        mgr.set_display_name("@1", "old-name")
        mgr.set_display_name("@1", "new-name")
        assert mgr.get_display_name("@1") == "new-name"

    def test_bind_thread_sets_display_name(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="proj")
        assert mgr.get_display_name("@1") == "proj"

    def test_bind_thread_without_name_no_display(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        # No display name set, fallback to window_id
        assert mgr.get_display_name("@1") == "@1"


class TestIsWindowId:
    def test_valid_ids(self, mgr: SessionManager) -> None:
        assert mgr._is_window_id("@0") is True
        assert mgr._is_window_id("@12") is True
        assert mgr._is_window_id("@999") is True

    def test_invalid_ids(self, mgr: SessionManager) -> None:
        assert mgr._is_window_id("myproject") is False
        assert mgr._is_window_id("@") is False
        assert mgr._is_window_id("") is False
        assert mgr._is_window_id("@abc") is False


class TestFindUsersForSession:
    @staticmethod
    def _ws(session_id: str):

        return WindowState(session_id=session_id, cwd="/tmp")

    def test_returns_matching_users(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.window_states["@1"] = self._ws("sid-1")
        result = mgr.find_users_for_session("sid-1")
        assert result == [(100, "@1", 1)]

    def test_no_match_returns_empty(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.window_states["@1"] = self._ws("sid-1")
        assert mgr.find_users_for_session("sid-other") == []

    def test_multiple_users_same_session(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.bind_thread(200, 2, "@2")
        mgr.window_states["@1"] = self._ws("sid-shared")
        mgr.window_states["@2"] = self._ws("sid-shared")
        result = mgr.find_users_for_session("sid-shared")
        assert len(result) == 2
        assert {r[0] for r in result} == {100, 200}

    def test_ignores_windows_without_state(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        assert mgr.find_users_for_session("sid-1") == []


class TestLoadSessionMapDisplayName:
    async def test_preserves_existing_display_name_on_stale_session_map(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:

        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@1": {
                        "session_id": "sid-1",
                        "cwd": "/tmp/project",
                        "window_name": "bun",
                    }
                }
            )
        )

        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccbot.session.config.tmux_session_name", "ccbot")

        mgr.window_display_names["@1"] = "ccbot"
        mgr.window_states["@1"] = WindowState(
            session_id="sid-1", cwd="/tmp/project", window_name="ccbot"
        )

        await mgr.load_session_map()

        assert mgr.get_display_name("@1") == "ccbot"
        assert mgr.window_states["@1"].window_name == "ccbot"

    async def test_initializes_display_name_when_missing(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@2": {
                        "session_id": "sid-2",
                        "cwd": "/tmp/project-2",
                        "window_name": "project-2",
                    }
                }
            )
        )

        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccbot.session.config.tmux_session_name", "ccbot")

        await mgr.load_session_map()

        assert mgr.get_display_name("@2") == "project-2"
        assert mgr.window_states["@2"].window_name == "project-2"


class TestParseSessionMap:
    def test_filters_by_prefix(self) -> None:
        from ccbot.session import parse_session_map

        raw = {
            "ccbot:win-a": {"session_id": "s1", "cwd": "/a"},
            "other:win-b": {"session_id": "s2", "cwd": "/b"},
        }
        result = parse_session_map(raw, "ccbot:")
        assert "win-a" in result
        assert "win-b" not in result

    def test_skips_empty_session_id(self) -> None:
        from ccbot.session import parse_session_map

        raw = {"ccbot:win-a": {"session_id": "", "cwd": "/a"}}
        assert parse_session_map(raw, "ccbot:") == {}

    def test_empty_input(self) -> None:
        from ccbot.session import parse_session_map

        assert parse_session_map({}, "ccbot:") == {}

    def test_extracts_cwd(self) -> None:
        from ccbot.session import parse_session_map

        raw = {"ccbot:win-a": {"session_id": "s1", "cwd": "/home/user/proj"}}
        result = parse_session_map(raw, "ccbot:")
        assert result["win-a"]["cwd"] == "/home/user/proj"

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param("a string", id="string-value"),
            pytest.param(42, id="int-value"),
            pytest.param(None, id="none-value"),
            pytest.param(["a", "list"], id="list-value"),
        ],
    )
    def test_non_dict_values_skipped(self, bad_value) -> None:
        from ccbot.session import parse_session_map

        raw = {
            "ccbot:good": {"session_id": "s1", "cwd": "/a"},
            "ccbot:bad": bad_value,
        }
        result = parse_session_map(raw, "ccbot:")
        assert "good" in result
        assert "bad" not in result


class TestPruneSessionMap:
    def test_removes_dead_windows(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:

        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps(
                {
                    "ccbot:@1": {"session_id": "sid-1", "cwd": "/a"},
                    "ccbot:@2": {"session_id": "sid-2", "cwd": "/b"},
                    "ccbot:@3": {"session_id": "sid-3", "cwd": "/c"},
                    "other:@9": {"session_id": "sid-9", "cwd": "/x"},
                }
            )
        )

        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccbot.session.config.tmux_session_name", "ccbot")

        mgr.window_states["@1"] = WindowState(session_id="sid-1", cwd="/a")
        mgr.window_states["@2"] = WindowState(session_id="sid-2", cwd="/b")
        mgr.window_states["@3"] = WindowState(session_id="sid-3", cwd="/c")

        mgr.prune_session_map(live_window_ids={"@1"})

        result = json.loads(session_map_file.read_text())
        assert "ccbot:@1" in result
        assert "ccbot:@2" not in result
        assert "ccbot:@3" not in result
        assert "other:@9" in result

        assert "@1" in mgr.window_states
        assert "@2" not in mgr.window_states
        assert "@3" not in mgr.window_states

    def test_noop_when_all_alive(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps({"ccbot:@1": {"session_id": "sid-1", "cwd": "/a"}})
        )

        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccbot.session.config.tmux_session_name", "ccbot")

        mgr.prune_session_map(live_window_ids={"@1"})

        result = json.loads(session_map_file.read_text())
        assert "ccbot:@1" in result

    def test_noop_when_file_missing(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        missing = tmp_path / "nonexistent.json"
        monkeypatch.setattr("ccbot.session.config.session_map_file", missing)

        mgr.prune_session_map(live_window_ids=set())

        assert not missing.exists()

    def test_handles_malformed_json(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text("{ invalid json")

        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)

        mgr.prune_session_map(live_window_ids={"@1"})

    def test_prunes_entry_without_window_state(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps({"ccbot:@5": {"session_id": "sid-5", "cwd": "/a"}})
        )

        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccbot.session.config.tmux_session_name", "ccbot")

        mgr.prune_session_map(live_window_ids=set())

        result = json.loads(session_map_file.read_text())
        assert "ccbot:@5" not in result


class TestWindowStateProviderName:
    def test_default_provider_name_is_empty(self) -> None:

        ws = WindowState()
        assert ws.provider_name == ""

    def test_to_dict_omits_empty_provider(self) -> None:

        ws = WindowState(session_id="s1", cwd="/tmp")
        d = ws.to_dict()
        assert "provider_name" not in d

    def test_to_dict_includes_provider_when_set(self) -> None:

        ws = WindowState(session_id="s1", cwd="/tmp", provider_name="codex")
        d = ws.to_dict()
        assert d["provider_name"] == "codex"

    def test_from_dict_reads_provider(self) -> None:

        ws = WindowState.from_dict(
            {"session_id": "s1", "cwd": "/tmp", "provider_name": "gemini"}
        )
        assert ws.provider_name == "gemini"

    def test_from_dict_defaults_to_empty(self) -> None:

        ws = WindowState.from_dict({"session_id": "s1", "cwd": "/tmp"})
        assert ws.provider_name == ""

    def test_round_trip_serialization(self) -> None:

        original = WindowState(
            session_id="s1",
            cwd="/tmp",
            window_name="proj",
            provider_name="codex",
        )
        restored = WindowState.from_dict(original.to_dict())
        assert restored.provider_name == "codex"
        assert restored.session_id == "s1"


class TestGlobFallbackCwdUpdate:
    @pytest.fixture(autouse=True)
    def _mock_provider(self, monkeypatch):
        from ccbot.providers.claude import ClaudeProvider

        monkeypatch.setattr(
            "ccbot.session.get_provider_for_window",
            lambda _wid: ClaudeProvider(),
        )

    async def test_glob_fallback_updates_cwd_when_dir_exists(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        from pathlib import Path
        from unittest.mock import patch

        # Simulate: encoded dir "-data-code-proj" → decoded "/data/code/proj"
        projects_path = tmp_path / "projects"
        encoded_dir = projects_path / "-data-code-proj"
        encoded_dir.mkdir(parents=True)
        session_file = encoded_dir / "session-abc.jsonl"
        session_file.write_text('{"type":"summary","summary":"test"}\n')

        monkeypatch.setattr("ccbot.session.config.claude_projects_path", projects_path)

        mgr.window_states["@1"] = WindowState(
            session_id="session-abc", cwd="/wrong/path"
        )

        # Mock Path.is_dir to return True for the decoded cwd
        _orig_is_dir = Path.is_dir

        def _mock_is_dir(self):
            if str(self) == "/data/code/proj":
                return True
            return _orig_is_dir(self)

        with patch.object(Path, "is_dir", _mock_is_dir):
            session = await mgr._get_session_direct("session-abc", "/wrong/path", "@1")

        assert session is not None
        assert mgr.window_states["@1"].cwd == "/data/code/proj"

    async def test_glob_fallback_skips_update_for_nonexistent_decoded_path(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        # Use a path with hyphens — decoded cwd won't be a real directory
        # e.g., -tmp-my-project decodes to /tmp/my/project (doesn't exist)
        projects_path = tmp_path / "projects"
        encoded_dir = projects_path / "-tmp-my-project"
        encoded_dir.mkdir(parents=True)
        session_file = encoded_dir / "sid-456.jsonl"
        session_file.write_text('{"type":"summary","summary":"test"}\n')

        monkeypatch.setattr("ccbot.session.config.claude_projects_path", projects_path)

        mgr.window_states["@2"] = WindowState(session_id="sid-456", cwd="/wrong/path")

        session = await mgr._get_session_direct("sid-456", "/wrong/path", "@2")

        assert session is not None
        # cwd NOT updated because decoded path doesn't exist as directory
        assert mgr.window_states["@2"].cwd == "/wrong/path"

    async def test_glob_fallback_no_update_without_window_id(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        projects_path = tmp_path / "projects"
        encoded_dir = projects_path / "-tmp-myproj"
        encoded_dir.mkdir(parents=True)
        session_file = encoded_dir / "sid-123.jsonl"
        session_file.write_text('{"type":"summary","summary":"test"}\n')

        monkeypatch.setattr("ccbot.session.config.claude_projects_path", projects_path)

        # No window state before the call
        session = await mgr._get_session_direct("sid-123", "/wrong/path")

        assert session is not None
        # No window state created without window_id
        assert not mgr.window_states


class TestSetWindowProvider:
    def test_set_and_get(self, mgr: SessionManager) -> None:
        mgr.set_window_provider("@1", "codex")
        assert mgr.window_states["@1"].provider_name == "codex"

    def test_get_unset_returns_empty(self, mgr: SessionManager) -> None:
        state = mgr.window_states.get("@99")
        assert state is None

    def test_set_empty_resets(self, mgr: SessionManager) -> None:
        mgr.set_window_provider("@1", "codex")
        mgr.set_window_provider("@1", "")
        assert mgr.window_states["@1"].provider_name == ""

    def test_creates_window_state_if_missing(self, mgr: SessionManager) -> None:
        mgr.set_window_provider("@5", "gemini")
        assert "@5" in mgr.window_states
        assert mgr.window_states["@5"].provider_name == "gemini"


class TestSyncDisplayNames:
    def test_updates_drifted_name(self, mgr: SessionManager) -> None:
        mgr.window_display_names["@1"] = "old-name"
        changed = mgr.sync_display_names([("@1", "new-name")])
        assert changed is True
        assert mgr.get_display_name("@1") == "new-name"

    def test_updates_window_state_too(self, mgr: SessionManager) -> None:
        mgr.window_display_names["@1"] = "old-name"
        mgr.window_states["@1"] = WindowState(window_name="old-name")
        mgr.sync_display_names([("@1", "new-name")])
        assert mgr.window_states["@1"].window_name == "new-name"

    def test_noop_when_names_match(self, mgr: SessionManager) -> None:
        mgr.window_display_names["@1"] = "same"
        changed = mgr.sync_display_names([("@1", "same")])
        assert changed is False

    def test_skips_unknown_windows(self, mgr: SessionManager) -> None:
        changed = mgr.sync_display_names([("@99", "new-proj")])
        assert changed is False
        assert "@99" not in mgr.window_display_names

    def test_multiple_windows(self, mgr: SessionManager) -> None:
        mgr.window_display_names["@1"] = "a"
        mgr.window_display_names["@2"] = "b"
        changed = mgr.sync_display_names([("@1", "a-renamed"), ("@2", "b")])
        assert changed is True
        assert mgr.get_display_name("@1") == "a-renamed"
        assert mgr.get_display_name("@2") == "b"


class TestPruneStaleState:
    def test_removes_orphaned_display_names(self, mgr: SessionManager) -> None:
        mgr.window_display_names["@1"] = "alive"
        mgr.window_display_names["@2"] = "dead"
        changed = mgr.prune_stale_state(live_window_ids={"@1"})
        assert changed is True
        assert "@1" in mgr.window_display_names
        assert "@2" not in mgr.window_display_names

    def test_keeps_display_name_if_bound(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@2", window_name="bound-proj")
        changed = mgr.prune_stale_state(live_window_ids=set())
        assert changed is False
        assert "@2" in mgr.window_display_names

    def test_keeps_display_name_if_has_window_state(self, mgr: SessionManager) -> None:
        mgr.window_display_names["@3"] = "with-state"
        mgr.window_states["@3"] = WindowState(session_id="sid")
        changed = mgr.prune_stale_state(live_window_ids=set())
        assert changed is False
        assert "@3" in mgr.window_display_names

    def test_removes_orphaned_group_chat_ids(self, mgr: SessionManager) -> None:
        mgr.set_group_chat_id(100, 1, -999)
        mgr.set_group_chat_id(100, 2, -888)
        mgr.bind_thread(100, 1, "@1")
        changed = mgr.prune_stale_state(live_window_ids={"@1"})
        assert changed is True
        assert "100:1" in mgr.group_chat_ids
        assert "100:2" not in mgr.group_chat_ids

    def test_noop_when_nothing_stale(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="proj")
        mgr.set_group_chat_id(100, 1, -999)
        changed = mgr.prune_stale_state(live_window_ids={"@1"})
        assert changed is False

    def test_prunes_both_display_and_chat(self, mgr: SessionManager) -> None:
        mgr.window_display_names["@dead"] = "gone"
        mgr.group_chat_ids["200:99"] = -777
        changed = mgr.prune_stale_state(live_window_ids=set())
        assert changed is True
        assert "@dead" not in mgr.window_display_names
        assert "200:99" not in mgr.group_chat_ids


class TestUnbindThreadCleanup:
    def test_cleans_up_group_chat_id(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        mgr.set_group_chat_id(100, 1, -999)
        mgr.unbind_thread(100, 1)
        assert "100:1" not in mgr.group_chat_ids

    def test_removes_display_name_when_no_refs(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="proj")
        assert "@1" in mgr.window_display_names
        mgr.unbind_thread(100, 1)
        assert "@1" not in mgr.window_display_names

    def test_keeps_display_name_when_other_thread_bound(
        self, mgr: SessionManager
    ) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="proj")
        mgr.bind_thread(200, 2, "@1")
        mgr.unbind_thread(100, 1)
        assert "@1" in mgr.window_display_names

    def test_keeps_display_name_when_window_state_exists(
        self, mgr: SessionManager
    ) -> None:
        mgr.bind_thread(100, 1, "@1", window_name="proj")
        mgr.window_states["@1"] = WindowState(session_id="sid")
        mgr.unbind_thread(100, 1)
        assert "@1" in mgr.window_display_names

    def test_group_chat_id_absent_is_safe(self, mgr: SessionManager) -> None:
        mgr.bind_thread(100, 1, "@1")
        result = mgr.unbind_thread(100, 1)
        assert result == "@1"


class TestRegisterHooklessSession:
    def test_updates_window_state(self, mgr: SessionManager) -> None:
        mgr.register_hookless_session(
            window_id="@7",
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/home/.codex/sessions/2026/03/02/test.jsonl",
            provider_name="codex",
        )

        state = mgr.window_states["@7"]
        assert state.session_id == "uuid-abc"
        assert state.cwd == "/my/project"
        assert state.transcript_path == "/home/.codex/sessions/2026/03/02/test.jsonl"
        assert state.provider_name == "codex"


class TestWriteHooklessSessionMap:
    def test_writes_session_map_json(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccbot.session.config.tmux_session_name", "ccbot")

        mgr.set_display_name("@7", "pumba-codex")
        mgr.write_hookless_session_map(
            window_id="@7",
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            provider_name="codex",
        )

        raw = json.loads(session_map_file.read_text())
        entry = raw["ccbot:@7"]
        assert entry["session_id"] == "uuid-abc"
        assert entry["cwd"] == "/my/project"
        assert entry["transcript_path"] == "/path/to/transcript.jsonl"
        assert entry["provider_name"] == "codex"
        assert entry["window_name"] == "pumba-codex"

    def test_preserves_existing_session_map_entries(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        session_map_file.write_text(
            json.dumps({"ccbot:@1": {"session_id": "sid-1", "cwd": "/a"}})
        )
        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccbot.session.config.tmux_session_name", "ccbot")

        mgr.write_hookless_session_map(
            window_id="@7",
            session_id="uuid-new",
            cwd="/new/project",
            transcript_path="/path/new.jsonl",
            provider_name="codex",
        )

        raw = json.loads(session_map_file.read_text())
        assert "ccbot:@1" in raw
        assert "ccbot:@7" in raw

    def test_handles_missing_session_map_file(
        self, mgr: SessionManager, tmp_path, monkeypatch
    ) -> None:
        session_map_file = tmp_path / "session_map.json"
        monkeypatch.setattr("ccbot.session.config.session_map_file", session_map_file)
        monkeypatch.setattr("ccbot.session.config.tmux_session_name", "ccbot")

        mgr.write_hookless_session_map(
            window_id="@7",
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            provider_name="codex",
        )

        assert session_map_file.exists()
        raw = json.loads(session_map_file.read_text())
        assert "ccbot:@7" in raw
