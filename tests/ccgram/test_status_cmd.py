"""Tests for ccgram status command."""

import contextlib
import json

from ccgram.status_cmd import _read_json, status_main


class TestReadJson:
    def test_valid_json(self, tmp_path) -> None:
        path = tmp_path / "test.json"
        path.write_text('{"key": "value"}')
        assert _read_json(path) == {"key": "value"}

    def test_missing_file(self, tmp_path) -> None:
        assert _read_json(tmp_path / "nonexistent.json") == {}

    def test_invalid_json(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json")
        assert _read_json(path) == {}


class TestStatusMain:
    def test_no_state_files(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_SESSION_NAME", "test-session")
        monkeypatch.setattr("ccgram.status_cmd._list_tmux_windows", lambda _: [])

        with contextlib.suppress(SystemExit):
            status_main()

        captured = capsys.readouterr()
        assert "ccgram" in captured.out
        assert "test-session (0 windows)" in captured.out
        assert "Monitored sessions: 0" in captured.out

    def test_with_bound_window(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_SESSION_NAME", "ccgram")

        state = {
            "thread_bindings": {"12345": {"42": "@5"}},
            "window_display_names": {"@5": "my-project"},
        }
        (tmp_path / "state.json").write_text(json.dumps(state))

        session_map = {
            "ccgram:@5": {"session_id": "abc-123", "cwd": "/tmp"},
        }
        (tmp_path / "session_map.json").write_text(json.dumps(session_map))

        monkeypatch.setattr(
            "ccgram.status_cmd._list_tmux_windows",
            lambda _: [{"id": "@5", "name": "my-project"}],
        )

        with contextlib.suppress(SystemExit):
            status_main()

        captured = capsys.readouterr()
        assert "1 windows" in captured.out
        assert "Monitored sessions: 1" in captured.out
        assert "@5" in captured.out
        assert "my-project" in captured.out
        assert "topic 42" in captured.out
        assert "alive" in captured.out

    def test_dead_binding(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_SESSION_NAME", "ccgram")

        state = {
            "thread_bindings": {"12345": {"42": "@5"}},
            "window_display_names": {"@5": "gone-project"},
        }
        (tmp_path / "state.json").write_text(json.dumps(state))

        monkeypatch.setattr("ccgram.status_cmd._list_tmux_windows", lambda _: [])

        with contextlib.suppress(SystemExit):
            status_main()

        captured = capsys.readouterr()
        assert "dead" in captured.out
        assert "gone-project" in captured.out

    def test_unbound_window(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_SESSION_NAME", "ccgram")

        monkeypatch.setattr(
            "ccgram.status_cmd._list_tmux_windows",
            lambda _: [{"id": "@10", "name": "orphan"}],
        )

        with contextlib.suppress(SystemExit):
            status_main()

        captured = capsys.readouterr()
        assert "(unbound)" in captured.out
        assert "orphan" in captured.out

    def test_shows_provider_info(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("CCGRAM_PROVIDER", "claude")
        monkeypatch.setenv("TMUX_SESSION_NAME", "test")
        monkeypatch.setattr("ccgram.status_cmd._list_tmux_windows", lambda _: [])

        with contextlib.suppress(SystemExit):
            status_main()

        captured = capsys.readouterr()
        assert "Provider: claude" in captured.out
        assert "hook" in captured.out
        assert "resume" in captured.out

    def test_codex_provider_capabilities(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("CCGRAM_PROVIDER", "codex")
        monkeypatch.setenv("TMUX_SESSION_NAME", "test")
        monkeypatch.setattr("ccgram.status_cmd._list_tmux_windows", lambda _: [])

        with contextlib.suppress(SystemExit):
            status_main()

        captured = capsys.readouterr()
        assert "Provider: codex" in captured.out
        assert "hook" in captured.out.split("Provider:")[1].split("\n")[0]


class TestStatusMainHerdr:
    def test_herdr_counts_keys_and_lists_panes(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        # In herdr mode, session_map keys are "herdr:wN:pM" — the tmux session
        # prefix would never match them. Status must use the herdr prefix and
        # the herdr pane listing, not shell out to tmux.
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", "herdr")
        monkeypatch.setenv("TMUX_SESSION_NAME", "ccgram")

        state = {
            "thread_bindings": {"12345": {"42": "w2:p1"}},
            "window_display_names": {"w2:p1": "ws ▸ claude"},
        }
        (tmp_path / "state.json").write_text(json.dumps(state))

        session_map = {
            "herdr:w2:p1": {"session_id": "abc-123", "cwd": "/tmp"},
            "ccgram:@5": {"session_id": "stale", "cwd": "/old"},
        }
        (tmp_path / "session_map.json").write_text(json.dumps(session_map))

        monkeypatch.setattr(
            "ccgram.status_cmd._list_herdr_windows",
            lambda: [{"id": "w2:p1", "name": "ws ▸ claude"}],
        )

        with contextlib.suppress(SystemExit):
            status_main()

        captured = capsys.readouterr()
        # herdr key counted, tmux-prefixed key ignored
        assert "Monitored sessions: 1" in captured.out
        assert "Herdr: 1 pane(s)" in captured.out
        assert "Tmux session" not in captured.out
        assert "w2:p1" in captured.out
        assert "topic 42" in captured.out
        assert "alive" in captured.out

    def test_reads_multiplexer_from_config_dir_env(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        # CCGRAM_MULTIPLEXER set only in ~/.ccgram/.env (the documented config
        # path), not exported. status must load that .env like the bot does, so
        # it counts herdr: keys and lists herdr panes — not default to tmux.
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_SESSION_NAME", "ccgram")
        # Registers the key so the load_dotenv-set value is restored on teardown.
        monkeypatch.delenv("CCGRAM_MULTIPLEXER", raising=False)
        (tmp_path / ".env").write_text("CCGRAM_MULTIPLEXER=herdr\n")

        session_map = {"herdr:w2:p1": {"session_id": "abc-123", "cwd": "/tmp"}}
        (tmp_path / "session_map.json").write_text(json.dumps(session_map))

        monkeypatch.setattr(
            "ccgram.status_cmd._list_herdr_windows",
            lambda: [{"id": "w2:p1", "name": "ws ▸ claude"}],
        )

        with contextlib.suppress(SystemExit):
            status_main()

        captured = capsys.readouterr()
        assert "Herdr: 1 pane(s)" in captured.out
        assert "Tmux session" not in captured.out
        assert "Monitored sessions: 1" in captured.out

    def test_herdr_listing_degrades_to_empty_on_backend_error(
        self, monkeypatch
    ) -> None:
        # Socket unreachable / backend error must degrade to [] (best-effort),
        # not crash `ccgram status`.
        from ccgram.status_cmd import _list_herdr_windows

        def _boom(_name):
            raise RuntimeError("socket down")

        monkeypatch.setattr("ccgram.multiplexer.get_multiplexer", _boom)
        assert _list_herdr_windows() == []
