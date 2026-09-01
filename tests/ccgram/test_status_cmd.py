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
        # The key must be absent for load_dotenv to supply it (dotenv does not
        # override an existing value), and it must be registered with monkeypatch
        # or the value dotenv writes into os.environ outlives the test and every
        # later test in this worker resolves the herdr backend. delenv alone does
        # not register an absent key, so set it first: setenv always registers.
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", "tmux")
        monkeypatch.delenv("CCGRAM_MULTIPLEXER")
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

    def test_herdr_listing_reports_unknown_on_backend_error(self, monkeypatch) -> None:
        # Socket unreachable / backend error must not crash `ccgram status`,
        # and must not answer []: an unreachable herdr is not an empty one, and
        # the caller renders an empty listing as every binding dead.
        from ccgram.status_cmd import _list_herdr_windows

        def _boom(_name):
            raise RuntimeError("socket down")

        monkeypatch.setattr("ccgram.multiplexer.get_multiplexer", _boom)
        assert _list_herdr_windows() is None


class TestLiveWindowsListing:
    """The alive/dead column must read liveness, not adoptability.

    Backends narrow ``list_windows`` to what discovery may auto-adopt: agterm
    drops out-of-scope workspaces and sessions sitting at a shell, herdr drops
    internal workspaces. A bound session that becomes merely ineligible is
    still live, and printing it dead points /sync Fix at a healthy binding.
    """

    @staticmethod
    def _fake_backend(monkeypatch, *, ui: list, reconciliation) -> None:
        from ccgram.multiplexer.base import WindowRef

        class _Backend:
            async def list_windows(self) -> list[WindowRef]:
                return ui

            async def list_windows_for_reconciliation(self) -> list[WindowRef] | None:
                return reconciliation

        monkeypatch.setattr(
            "ccgram.multiplexer.get_multiplexer", lambda _name: _Backend()
        )

    def test_live_but_ineligible_window_is_listed(self, monkeypatch) -> None:
        from ccgram.multiplexer.base import WindowRef
        from ccgram.status_cmd import _live_windows

        excluded = WindowRef(
            window_id="AAAA",
            window_name="out-of-scope",
            cwd="/repo",
            topic_eligible=False,
        )
        self._fake_backend(monkeypatch, ui=[], reconciliation=[excluded])

        assert _live_windows("agterm") == [{"id": "AAAA", "name": "out-of-scope"}]

    def test_unconfirmed_listing_is_unknown_not_empty(self, monkeypatch) -> None:
        from ccgram.status_cmd import _live_windows

        self._fake_backend(monkeypatch, ui=[], reconciliation=None)

        assert _live_windows("agterm") is None


class TestUnreachableBackendNeverPrintsDead:
    """An outage means every window is unknown, not that every window is gone.

    ``dead`` is what /sync Fix offers to clean up, so printing it for a
    binding whose backend simply could not be reached invites closing live
    sessions. A *confirmed* empty listing still says dead.
    """

    @staticmethod
    def _bound(tmp_path, monkeypatch, mux: str) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", mux)
        monkeypatch.setenv("TMUX_SESSION_NAME", "ccgram")
        state = {
            "thread_bindings": {"12345": {"42": "@5"}},
            "window_display_names": {"@5": "my-project"},
        }
        (tmp_path / "state.json").write_text(json.dumps(state))

    @staticmethod
    def _run(capsys) -> str:
        with contextlib.suppress(SystemExit):
            status_main()
        return capsys.readouterr().out

    def test_seam_backend_returning_none(self, tmp_path, monkeypatch, capsys) -> None:
        self._bound(tmp_path, monkeypatch, "agterm")
        monkeypatch.setattr("ccgram.status_cmd._list_backend_windows", lambda _: None)

        out = self._run(capsys)

        assert "dead" not in out
        assert "unknown" in out
        assert "my-project" in out
        assert "unreachable" in out

    def test_tmux_listing_failure(self, tmp_path, monkeypatch, capsys) -> None:
        self._bound(tmp_path, monkeypatch, "tmux")
        monkeypatch.setattr("ccgram.status_cmd._list_tmux_windows", lambda _: None)

        out = self._run(capsys)

        assert "dead" not in out
        assert "unknown" in out
        assert "my-project" in out

    def test_confirmed_empty_listing_still_says_dead(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        self._bound(tmp_path, monkeypatch, "tmux")
        monkeypatch.setattr("ccgram.status_cmd._list_tmux_windows", lambda _: [])

        out = self._run(capsys)

        assert "dead" in out
        assert "unknown" not in out


class TestTmuxListingIsUnknownOnFailure:
    """The rendered tests stub _list_tmux_windows, so its own contract is here.

    Both failure shapes must answer None: the caller renders [] as every
    binding dead, and a tmux that is missing or wedged is not an empty server.
    """

    def test_nonzero_exit(self, monkeypatch) -> None:
        import subprocess

        from ccgram.status_cmd import _list_tmux_windows

        def _failed(*_a, **_kw):
            return subprocess.CompletedProcess([], returncode=1, stdout="", stderr="")

        monkeypatch.setattr("ccgram.status_cmd.subprocess.run", _failed)
        assert _list_tmux_windows("ccgram") is None

    def test_tmux_not_installed(self, monkeypatch) -> None:
        from ccgram.status_cmd import _list_tmux_windows

        def _boom(*_a, **_kw):
            raise FileNotFoundError("tmux")

        monkeypatch.setattr("ccgram.status_cmd.subprocess.run", _boom)
        assert _list_tmux_windows("ccgram") is None

    def test_healthy_listing_is_parsed(self, monkeypatch) -> None:
        import subprocess

        from ccgram.status_cmd import _list_tmux_windows

        def _ok(*_a, **_kw):
            return subprocess.CompletedProcess(
                [], returncode=0, stdout="@5\tmy-project\n", stderr=""
            )

        monkeypatch.setattr("ccgram.status_cmd.subprocess.run", _ok)
        assert _list_tmux_windows("ccgram") == [{"id": "@5", "name": "my-project"}]
