import io
import json
import shlex
import subprocess
import sys
from unittest.mock import patch

import pytest

from ccgram.hook import (
    _claude_settings_file,
    _closest_claude_ancestor,
    _ensure_codex_feature_flag,
    _foreground_pgid_on_tty,
    _hook_status,
    _install_hook,
    _is_nested_session,
    _provider_from_pane_tty,
    _resolve_window_id,
    _session_map_session_for,
    _uninstall_hook,
    get_installed_events,
    hook_main,
)
from ccgram.config import config
from ccgram.providers.base import UUID_RE


def _is_hook_installed(settings: dict) -> bool:
    return get_installed_events(settings).get("SessionStart", False)


def _expected_module_command() -> str:
    return f"{shlex.quote(sys.executable)} -m ccgram.main hook"


class TestCodexFeatureFlag:
    @pytest.mark.parametrize(
        ("existing", "expected", "result"),
        [
            pytest.param(
                "[features]\nhooks = true\n",
                "[features]\nhooks = true\n",
                0,
                id="current-enabled",
            ),
            pytest.param(
                "[features]\ncodex_hooks = true\n",
                "[features]\nhooks = true\n",
                0,
                id="migrate-enabled",
            ),
            pytest.param(
                "[features]\ncodex_hooks = true\nhooks = true\n",
                "[features]\nhooks = true\n",
                0,
                id="remove-duplicate-legacy-key",
            ),
            pytest.param(
                "[features]\nhooks = false\n",
                "[features]\nhooks = false\n",
                1,
                id="current-disabled",
            ),
            pytest.param(
                "[features]\ncodex_hooks = false\n",
                "[features]\nhooks = false\n",
                1,
                id="migrate-disabled",
            ),
            pytest.param(
                "[profiles.work.features]\nhooks = false\n",
                "[profiles.work.features]\nhooks = false\n\n[features]\nhooks = true\n",
                0,
                id="ignore-profile-features",
            ),
            pytest.param(
                "[features]\nunified_exec = true\n\n[[plugins]]\nhooks = false\n",
                "[features]\nhooks = true\nunified_exec = true\n\n[[plugins]]\nhooks = false\n",
                0,
                id="stop-at-array-table",
            ),
            pytest.param(
                "[features]  \nunified_exec = true\n",
                "[features]  \nhooks = true\nunified_exec = true\n",
                0,
                id="allow-trailing-header-whitespace",
            ),
        ],
    )
    def test_ensures_current_top_level_flag(
        self,
        tmp_path,
        monkeypatch,
        existing: str,
        expected: str,
        result: int,
    ) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(existing)
        monkeypatch.setattr("ccgram.hook._codex_config_file", lambda: config_file)

        assert _ensure_codex_feature_flag() == result
        assert config_file.read_text() == expected

    def test_creates_config_with_current_flag(self, tmp_path, monkeypatch) -> None:
        config_file = tmp_path / "config.toml"
        monkeypatch.setattr("ccgram.hook._codex_config_file", lambda: config_file)

        assert _ensure_codex_feature_flag() == 0
        assert config_file.read_text() == "[features]\nhooks = true\n"


class TestInstallHook:
    def test_install_into_empty_settings(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _install_hook()
        assert result == 0

        settings = json.loads(settings_file.read_text())
        session_start = settings["hooks"]["SessionStart"]
        assert len(session_start) == 1
        assert session_start[0]["hooks"][0]["command"] == _expected_module_command()

    def test_install_adds_to_existing_matcher_group(
        self, tmp_path, monkeypatch
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "matcher": ".*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "session-start.sh",
                                "timeout": 5,
                            }
                        ],
                    }
                ]
            }
        }
        settings_file.write_text(json.dumps(settings))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _install_hook()
        assert result == 0

        updated = json.loads(settings_file.read_text())
        session_start = updated["hooks"]["SessionStart"]
        assert len(session_start) == 1
        hooks_list = session_start[0]["hooks"]
        assert len(hooks_list) == 2
        assert hooks_list[1]["command"] == _expected_module_command()

    @pytest.mark.parametrize(
        "existing_command",
        [
            pytest.param("ccgram hook 2>/dev/null || true", id="shell-wrapped"),
            pytest.param("/usr/local/bin/ccgram hook", id="absolute-path"),
        ],
    )
    def test_install_rewrites_an_existing_ccgram_command_in_place(
        self, tmp_path, monkeypatch, existing_command: str
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": existing_command,
                                        "timeout": 5,
                                    }
                                ]
                            }
                        ]
                    }
                }
            )
        )
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        assert _install_hook() == 0

        updated = json.loads(settings_file.read_text())
        hooks_list = updated["hooks"]["SessionStart"][0]["hooks"]
        assert len(hooks_list) == 1
        assert hooks_list[0]["command"] == _expected_module_command()


class TestInstallMultipleEvents:
    def test_installs_all_event_types(self, tmp_path, monkeypatch) -> None:
        from ccgram.hook import _HOOK_EVENT_TYPES

        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _install_hook()
        assert result == 0

        settings = json.loads(settings_file.read_text())
        for event_type in _HOOK_EVENT_TYPES:
            assert event_type in settings["hooks"]
            hooks_list = settings["hooks"][event_type][0]["hooks"]
            assert any(
                h.get("command", "") == _expected_module_command() for h in hooks_list
            )

    def test_async_flag_on_subagent_events(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        _install_hook()

        settings = json.loads(settings_file.read_text())
        for event_type in ("SubagentStart", "SubagentStop"):
            hook_config = settings["hooks"][event_type][0]["hooks"][0]
            assert hook_config.get("async") is True

        session_hook = settings["hooks"]["SessionStart"][0]["hooks"][0]
        assert "async" not in session_hook

    def test_idempotent_install(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        _install_hook()
        _install_hook()  # Second install

        settings = json.loads(settings_file.read_text())
        for event_type in settings["hooks"]:
            entries = settings["hooks"][event_type]
            ccgram_hooks = [
                h
                for entry in entries
                for h in entry.get("hooks", [])
                if h.get("command", "") == _expected_module_command()
            ]
            assert len(ccgram_hooks) == 1, (
                f"{event_type} has {len(ccgram_hooks)} ccgram hooks"
            )


class TestUninstallMultipleEvents:
    def test_removes_all_event_types(self, tmp_path, monkeypatch) -> None:
        from ccgram.hook import _uninstall_hook, get_installed_events

        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        _install_hook()
        settings = json.loads(settings_file.read_text())
        assert all(get_installed_events(settings).values())

        result = _uninstall_hook()
        assert result == 0

        settings = json.loads(settings_file.read_text())
        assert not any(get_installed_events(settings).values())


class TestUuidRegex:
    @pytest.mark.parametrize(
        "value",
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "00000000-0000-0000-0000-000000000000",
            "abcdef01-2345-6789-abcd-ef0123456789",
        ],
        ids=["standard", "all-zeros", "all-hex"],
    )
    def test_valid_uuid_matches(self, value: str) -> None:
        assert UUID_RE.match(value) is not None

    @pytest.mark.parametrize(
        "value",
        [
            "not-a-uuid",
            "550e8400-e29b-41d4-a716",
            "550e8400-e29b-41d4-a716-44665544000g",
            "",
        ],
        ids=["gibberish", "truncated", "invalid-hex-char", "empty"],
    )
    def test_invalid_uuid_no_match(self, value: str) -> None:
        assert UUID_RE.match(value) is None


class TestIsHookInstalled:
    @staticmethod
    def _settings(command: str) -> dict:
        return {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": command, "timeout": 5}]}
                ]
            }
        }

    @pytest.mark.parametrize(
        ("command", "installed"),
        [
            pytest.param("ccgram hook", True, id="bare"),
            pytest.param("/usr/bin/ccgram hook", True, id="absolute-path"),
            pytest.param("ccgram hook 2>/dev/null || true", True, id="shell-wrapped"),
            pytest.param(
                f"{shlex.quote(sys.executable)} -m ccgram.main hook",
                True,
                id="python-module",
            ),
            pytest.param("other-tool hook", False, id="another-tool"),
        ],
    )
    def test_recognises_every_command_spelling(
        self, command: str, installed: bool
    ) -> None:
        assert _is_hook_installed(self._settings(command)) is installed

    def test_no_hooks_key(self) -> None:
        assert _is_hook_installed({}) is False


class TestHookMainValidation:
    def _run_hook_main(
        self, monkeypatch: pytest.MonkeyPatch, payload: dict, *, tmux_pane: str = ""
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["ccgram", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        if tmux_pane:
            monkeypatch.setenv("TMUX_PANE", tmux_pane)
        else:
            monkeypatch.delenv("TMUX_PANE", raising=False)
        hook_main()

    def test_missing_session_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {"cwd": "/tmp", "hook_event_name": "SessionStart"},
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_invalid_uuid_format(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "not-a-uuid",
                "cwd": "/tmp",
                "hook_event_name": "SessionStart",
            },
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_relative_cwd(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "cwd": "relative/path",
                "hook_event_name": "SessionStart",
            },
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_stop_event_writes_event_not_session_map(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_PANE", "%0")

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ccgram\t@0\tproject\n", stderr=""
        )
        with patch("ccgram.hook.subprocess.run", return_value=mock_result):
            self._run_hook_main(
                monkeypatch,
                {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "cwd": "/tmp",
                    "hook_event_name": "Stop",
                    "stop_reason": "end_turn",
                },
                tmux_pane="%0",
            )

        assert not (tmp_path / "session_map.json").exists()
        events_file = tmp_path / "events.jsonl"
        assert events_file.exists()
        event = json.loads(events_file.read_text().strip())
        assert event["event"] == "Stop"
        assert event["window_key"] == "ccgram:@0"
        assert event["data"]["stop_reason"] == "end_turn"

    @pytest.mark.parametrize(
        "stdin_text",
        [
            pytest.param("not json at all", id="invalid-json"),
            pytest.param("[]", id="json-array"),
            pytest.param('"a string"', id="json-scalar"),
            pytest.param("", id="empty-stdin"),
        ],
    )
    def test_unusable_stdin_writes_nothing_and_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, stdin_text: str
    ) -> None:
        """The hook runs inside the agent's own pane; a crash there is visible
        to the user, so a payload it cannot read must be dropped silently."""
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["ccgram", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
        monkeypatch.setenv("TMUX_PANE", "%0")

        hook_main()

        assert not (tmp_path / "session_map.json").exists()
        assert not (tmp_path / "events.jsonl").exists()

    def test_unhandled_event_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "cwd": "/tmp",
                "hook_event_name": "PreToolUse",
            },
        )
        assert not (tmp_path / "session_map.json").exists()
        assert not (tmp_path / "events.jsonl").exists()


class TestUninstallHook:
    @pytest.mark.parametrize(
        "installed_command",
        [
            pytest.param("ccgram hook", id="bare"),
            pytest.param("ccgram hook 2>/dev/null || true", id="shell-wrapped"),
            pytest.param(
                f"{shlex.quote(sys.executable)} -m ccgram.main hook",
                id="python-module",
            ),
        ],
    )
    def test_uninstall_removes_every_command_spelling(
        self, tmp_path, monkeypatch, installed_command: str
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": installed_command,
                                        "timeout": 5,
                                    }
                                ]
                            }
                        ]
                    }
                }
            )
        )
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        assert _uninstall_hook() == 0

        updated = json.loads(settings_file.read_text())
        assert not _is_hook_installed(updated)
        assert updated["hooks"]["SessionStart"] == []

    def test_uninstall_preserves_other_hooks_in_same_group(
        self, tmp_path, monkeypatch
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "matcher": ".*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "session-start.sh",
                                "timeout": 5,
                            },
                            {"type": "command", "command": "ccgram hook", "timeout": 5},
                        ],
                    }
                ]
            }
        }
        settings_file.write_text(json.dumps(settings))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        assert _uninstall_hook() == 0

        updated = json.loads(settings_file.read_text())
        session_start = updated["hooks"]["SessionStart"]
        assert len(session_start) == 1
        assert [h["command"] for h in session_start[0]["hooks"]] == ["session-start.sh"]

    @pytest.mark.parametrize(
        "settings_body",
        [
            pytest.param(None, id="no-settings-file"),
            pytest.param({"hooks": {}}, id="not-installed"),
        ],
    )
    def test_uninstall_is_a_no_op_when_nothing_is_installed(
        self, tmp_path, monkeypatch, settings_body: dict | None
    ) -> None:
        settings_file = tmp_path / "settings.json"
        if settings_body is not None:
            settings_file.write_text(json.dumps(settings_body))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        assert _uninstall_hook() == 0


class TestTabDelimitedParsing:
    _VALID_PAYLOAD = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "cwd": "/tmp/proj",
        "hook_event_name": "SessionStart",
        "transcript_path": "/tmp/transcript.jsonl",
    }

    @pytest.mark.parametrize(
        ("tmux_output", "expected_key", "expected_window_name"),
        [
            ("ccgram\t@0\tproject", "ccgram:@0", "project"),
            ("prod:v2\t@3\tmy-win", "prod:v2:@3", "my-win"),
            ("ccgram\t@1\tmy:project", "ccgram:@1", "my:project"),
            ("my-sess\t@12\twin name", "my-sess:@12", "win name"),
        ],
        ids=["normal-names", "colon-in-session", "colon-in-window", "special-chars"],
    )
    def test_colon_in_names_parsed_correctly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        tmux_output: str,
        expected_key: str,
        expected_window_name: str,
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_PANE", "%0")
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(self._VALID_PAYLOAD)))

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=tmux_output + "\n", stderr=""
        )
        with patch("ccgram.hook.subprocess.run", return_value=mock_result):
            hook_main()

        session_map = json.loads((tmp_path / "session_map.json").read_text())
        assert expected_key in session_map
        entry = session_map[expected_key]
        assert entry["session_id"] == self._VALID_PAYLOAD["session_id"]
        assert entry["window_name"] == expected_window_name

    def test_tmux_timeout_writes_no_session_map(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_PANE", "%0")
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(self._VALID_PAYLOAD)))

        with patch(
            "ccgram.hook.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="tmux", timeout=5),
        ):
            hook_main()

        session_map_file = tmp_path / "session_map.json"
        assert not session_map_file.exists()


class TestHookStatus:
    def _all_events_settings(self) -> dict:
        from ccgram.hook import _HOOK_EVENT_TYPES

        hooks: dict = {}
        for event_type in _HOOK_EVENT_TYPES:
            hooks[event_type] = [
                {"hooks": [{"type": "command", "command": "/usr/bin/ccgram hook"}]}
            ]
        return {"hooks": hooks}

    def test_all_installed(self, tmp_path, monkeypatch, capsys) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(self._all_events_settings()))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _hook_status()
        assert result == 0
        assert "All hooks installed" in capsys.readouterr().out

    def test_partial_installed(self, tmp_path, monkeypatch, capsys) -> None:
        settings_file = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "ccgram hook"}]}
                ]
            }
        }
        settings_file.write_text(json.dumps(settings))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _hook_status()
        assert result == 1
        out = capsys.readouterr().out
        assert "Missing hooks:" in out
        assert "SessionStart: installed" in out

    def test_not_installed(self, tmp_path, monkeypatch, capsys) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"hooks": {}}))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _hook_status()
        assert result == 1
        assert "Missing hooks:" in capsys.readouterr().out

    def test_no_settings_file(self, tmp_path, monkeypatch, capsys) -> None:
        settings_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _hook_status()
        assert result == 1
        assert "Not installed" in capsys.readouterr().out


class TestClaudeSettingsFile:
    def test_default_path(self, monkeypatch) -> None:
        from pathlib import Path

        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        assert _claude_settings_file() == Path.home() / ".claude" / "settings.json"

    def test_respects_env_var(self, monkeypatch, tmp_path) -> None:

        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "custom"))
        assert _claude_settings_file() == tmp_path / "custom" / "settings.json"


class TestSessionMapKeyForLinkedWindow:
    """A window linked into ccgram's session belongs to more than one tmux
    session. The hook must key session_map under the session readers resolve
    against (config.tmux_session_name), not whichever session tmux happens to
    report for the firing pane, or the binding is written and never found.
    """

    @staticmethod
    def _run(stdout: str, returncode: int = 0):
        def _fake(*_args, **_kwargs):
            return subprocess.CompletedProcess(
                args=[], returncode=returncode, stdout=stdout, stderr=""
            )

        return _fake

    def test_linked_window_keys_under_ccgram_session(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        monkeypatch.setattr(subprocess, "run", self._run("@12\n@34\n"))
        assert _session_map_session_for("@34", "agentdeck_foo_1234") == "ccgram"

    def test_resolve_window_id_uses_ccgram_session_for_linked_window(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")

        def _fake_run(args, **_kwargs):
            if args[1] == "display-message":
                return subprocess.CompletedProcess(
                    args, 0, "agentdeck\t@34\tcode\t/dev/ttys001\t2\n", ""
                )
            if args[1] == "list-windows":
                return subprocess.CompletedProcess(args, 0, "@34\n", "")
            pytest.fail(f"unexpected command: {args}")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        assert _resolve_window_id("%1") == (
            "ccgram:@34",
            "@34",
            "code",
            "/dev/ttys001",
        )

    def test_unlinked_window_keeps_pane_session(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        monkeypatch.setattr(subprocess, "run", self._run("@12\n"))
        assert (
            _session_map_session_for("@99", "agentdeck_foo_1234")
            == "agentdeck_foo_1234"
        )

    def test_pane_already_in_ccgram_session_is_unchanged(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: pytest.fail("no tmux probe needed")
        )
        assert _session_map_session_for("@12", "ccgram") == "ccgram"

    def test_tmux_failure_falls_back_to_pane_session(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        monkeypatch.setattr(subprocess, "run", self._run("", returncode=1))
        assert (
            _session_map_session_for("@34", "agentdeck_foo_1234")
            == "agentdeck_foo_1234"
        )

    def test_tmux_timeout_falls_back_to_pane_session(self, monkeypatch) -> None:
        def _boom(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="tmux", timeout=5)

        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        monkeypatch.setattr(subprocess, "run", _boom)
        assert (
            _session_map_session_for("@34", "agentdeck_foo_1234")
            == "agentdeck_foo_1234"
        )


class TestNestedSessionDetection:
    """Hook fired by a nested claude (e.g. claude-mem observer) must not
    overwrite session_map.json or write events for the bound topic.

    Process model: a tmux pane is hosting a primary claude that the user
    launched from their shell. The shell put claude in its own pgrp, so
    primary's PID == pgid == foreground PGID on the tty. Any subprocess claude
    inherits that pgid (without setpgid) but has a different PID — so its
    closest-claude-ancestor (itself) does not equal foreground PGID.
    """

    @staticmethod
    def _ps_lines(rows: list[tuple[int, int, int, str, str]]) -> str:
        return "\n".join(
            f"{pid} {ppid} {pgid} {stat} {cmd}" for pid, ppid, pgid, stat, cmd in rows
        )

    def test_foreground_claude_is_not_nested(self, monkeypatch) -> None:
        snapshot = {
            7818: (1, 7818, "Ss", "fish"),
            72211: (7818, 72211, "S+", "claude"),
            99999: (72211, 72211, "S+", "python"),
        }
        monkeypatch.setattr("ccgram.hook._ps_snapshot", lambda: snapshot)
        monkeypatch.setattr("ccgram.hook._foreground_pgid_on_tty", lambda *_: 72211)
        monkeypatch.setattr("os.getpid", lambda: 99999)
        assert _is_nested_session("/dev/ttys005") is False

    def test_observer_claude_is_nested(self, monkeypatch) -> None:
        snapshot = {
            7818: (1, 7818, "Ss", "fish"),
            72211: (7818, 72211, "S+", "claude"),
            72281: (72211, 72211, "S+", "bun"),
            80000: (72281, 72211, "S+", "claude"),
            99999: (80000, 72211, "S+", "python"),
        }
        monkeypatch.setattr("ccgram.hook._ps_snapshot", lambda: snapshot)
        monkeypatch.setattr("ccgram.hook._foreground_pgid_on_tty", lambda *_: 72211)
        monkeypatch.setattr("os.getpid", lambda: 99999)
        assert _is_nested_session("/dev/ttys005") is True

    def test_shell_wrapped_primary_claude_is_not_nested(self, monkeypatch) -> None:
        """A launcher running `bash -lc "... && claude ..."` keeps the shell as
        group leader, so the primary claude's PID never equals the foreground
        PGID. No claude sits above it, so it is still the primary."""
        snapshot = {
            14053: (1, 14053, "Ss+", "bash"),
            14058: (14053, 14053, "S+", "claude"),
            99999: (14058, 14053, "S+", "python"),
        }
        monkeypatch.setattr("ccgram.hook._ps_snapshot", lambda: snapshot)
        monkeypatch.setattr("ccgram.hook._foreground_pgid_on_tty", lambda *_: 14053)
        monkeypatch.setattr("os.getpid", lambda: 99999)
        assert _is_nested_session("/dev/ttys002") is False

    def test_observer_under_shell_wrapped_primary_is_nested(self, monkeypatch) -> None:
        """The nested case must still be caught when the primary is itself a
        child of the foreground shell rather than the group leader."""
        snapshot = {
            14053: (1, 14053, "Ss+", "bash"),
            14058: (14053, 14053, "S+", "claude"),
            14100: (14058, 14053, "S+", "bun"),
            14200: (14100, 14053, "S+", "claude"),
            99999: (14200, 14053, "S+", "python"),
        }
        monkeypatch.setattr("ccgram.hook._ps_snapshot", lambda: snapshot)
        monkeypatch.setattr("ccgram.hook._foreground_pgid_on_tty", lambda *_: 14053)
        monkeypatch.setattr("os.getpid", lambda: 99999)
        assert _is_nested_session("/dev/ttys002") is True

    def test_empty_pane_tty_fails_open(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.hook._ps_snapshot", lambda: pytest.fail("should not be called")
        )
        assert _is_nested_session("") is False

    def test_empty_snapshot_fails_open(self, monkeypatch) -> None:
        monkeypatch.setattr("ccgram.hook._ps_snapshot", lambda: {})
        assert _is_nested_session("/dev/ttys005") is False

    def test_unknown_foreground_pgid_fails_open(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.hook._ps_snapshot",
            lambda: {99999: (1, 99999, "S+", "python")},
        )
        monkeypatch.setattr("ccgram.hook._foreground_pgid_on_tty", lambda *_: None)
        assert _is_nested_session("/dev/ttys005") is False

    def test_no_claude_in_ancestry_fails_open(self, monkeypatch) -> None:
        snapshot = {
            7818: (1, 7818, "Ss", "fish"),
            99999: (7818, 7818, "S+", "python"),
        }
        monkeypatch.setattr("ccgram.hook._ps_snapshot", lambda: snapshot)
        monkeypatch.setattr("ccgram.hook._foreground_pgid_on_tty", lambda *_: 7818)
        monkeypatch.setattr("os.getpid", lambda: 99999)
        assert _is_nested_session("/dev/ttys005") is False

    def test_closest_claude_ancestor_picks_nearest(self) -> None:
        snapshot = {
            7818: (1, 7818, "Ss", "fish"),
            72211: (7818, 72211, "S+", "claude"),
            72281: (72211, 72211, "S+", "bun"),
            80000: (72281, 72211, "S+", "claude"),
            99999: (80000, 72211, "S+", "python"),
        }
        assert _closest_claude_ancestor(snapshot, 99999) == 80000
        assert _closest_claude_ancestor(snapshot, 72281) == 72211

    def test_closest_claude_ancestor_breaks_on_cycle(self) -> None:
        snapshot = {
            10: (20, 10, "S", "a"),
            20: (10, 20, "S", "b"),
        }
        assert _closest_claude_ancestor(snapshot, 10) is None

    def test_foreground_pgid_finds_plus_row(self) -> None:
        snapshot = {
            7818: (1, 7818, "Ss", "fish"),
            72211: (7818, 72211, "S+", "claude"),
        }
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="7818\n72211\n", stderr=""
        )
        with patch("ccgram.hook.subprocess.run", return_value=mock_result):
            assert _foreground_pgid_on_tty(snapshot, "/dev/ttys005") == 72211

    def test_foreground_pgid_handles_subprocess_error(self) -> None:
        with patch(
            "ccgram.hook.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ps", timeout=5),
        ):
            assert _foreground_pgid_on_tty({1: (0, 1, "S+", "x")}, "ttys005") is None


class TestNestedHookEndToEnd:
    """Drive ``hook_main`` end-to-end through ``subprocess.run`` mocking and
    assert that observer SessionStart/Stop events do NOT poison session_map
    or events.jsonl.
    """

    _OBSERVER_PAYLOAD = {
        "session_id": "35339b36-8b46-41eb-98fc-df51cd1ff498",
        "cwd": "/Users/alexei/.claude-mem/observer-sessions",
        "transcript_path": "/Users/alexei/.claude-team/projects/x/35339b36.jsonl",
        "hook_event_name": "SessionStart",
    }

    _PRIMARY_PAYLOAD = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "cwd": "/Users/alexei/Workspace/reflex",
        "transcript_path": "/Users/alexei/.claude/projects/y/550e8400.jsonl",
        "hook_event_name": "SessionStart",
    }

    def _drive_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
        payload: dict,
        *,
        hook_pid: int,
        snapshot: dict[int, tuple[int, int, str, str]],
        fg_pgid: int,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["ccgram", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        monkeypatch.setenv("TMUX_PANE", "%6")
        monkeypatch.setattr("os.getpid", lambda: hook_pid)
        monkeypatch.setattr("ccgram.hook._ps_snapshot", lambda: snapshot)
        monkeypatch.setattr("ccgram.hook._foreground_pgid_on_tty", lambda *_: fg_pgid)

        tmux_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ccgram\t@6\treflex-gh\t/dev/ttys012\n",
            stderr="",
        )
        with patch("ccgram.hook.subprocess.run", return_value=tmux_result):
            hook_main()

    def test_observer_session_start_does_not_overwrite_session_map(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        snapshot = {
            7818: (1, 7818, "Ss", "fish"),
            72211: (7818, 72211, "S+", "claude"),
            72281: (72211, 72211, "S+", "bun"),
            80000: (72281, 72211, "S+", "claude"),
            99999: (80000, 72211, "S+", "python"),
        }
        self._drive_hook(
            monkeypatch,
            self._OBSERVER_PAYLOAD,
            hook_pid=99999,
            snapshot=snapshot,
            fg_pgid=72211,
        )
        assert not (tmp_path / "session_map.json").exists()
        assert not (tmp_path / "events.jsonl").exists()

    def test_observer_stop_event_dropped(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        payload = {
            "session_id": "35339b36-8b46-41eb-98fc-df51cd1ff498",
            "cwd": "/Users/alexei/.claude-mem/observer-sessions",
            "hook_event_name": "Stop",
            "stop_reason": "end_turn",
        }
        snapshot = {
            7818: (1, 7818, "Ss", "fish"),
            72211: (7818, 72211, "S+", "claude"),
            80000: (72211, 72211, "S+", "claude"),
            99999: (80000, 72211, "S+", "python"),
        }
        self._drive_hook(
            monkeypatch,
            payload,
            hook_pid=99999,
            snapshot=snapshot,
            fg_pgid=72211,
        )
        assert not (tmp_path / "events.jsonl").exists()

    def test_primary_session_start_writes_session_map(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        snapshot = {
            7818: (1, 7818, "Ss", "fish"),
            72211: (7818, 72211, "S+", "claude"),
            99999: (72211, 72211, "S+", "python"),
        }
        self._drive_hook(
            monkeypatch,
            self._PRIMARY_PAYLOAD,
            hook_pid=99999,
            snapshot=snapshot,
            fg_pgid=72211,
        )
        session_map = json.loads((tmp_path / "session_map.json").read_text())
        assert "ccgram:@6" in session_map
        assert (
            session_map["ccgram:@6"]["session_id"]
            == self._PRIMARY_PAYLOAD["session_id"]
        )

    def test_introspection_failure_fails_open(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setattr("ccgram.hook._ps_snapshot", lambda: {})
        monkeypatch.setattr("ccgram.hook._foreground_pgid_on_tty", lambda *_: None)
        monkeypatch.setattr(sys, "argv", ["ccgram", "hook"])
        monkeypatch.setattr(
            sys, "stdin", io.StringIO(json.dumps(self._OBSERVER_PAYLOAD))
        )
        monkeypatch.setenv("TMUX_PANE", "%6")
        tmux_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ccgram\t@6\treflex-gh\t/dev/ttys012\n",
            stderr="",
        )
        with patch("ccgram.hook.subprocess.run", return_value=tmux_result):
            hook_main()
        assert (tmp_path / "session_map.json").exists()


class TestProviderFromPaneTty:
    @pytest.mark.parametrize(
        ("foreground_command", "provider"),
        [
            pytest.param("/usr/local/bin/gemini --model flash", "gemini", id="gemini"),
            pytest.param("/usr/local/bin/codex", "codex", id="codex"),
            pytest.param("/usr/local/bin/claude", "claude", id="claude"),
            pytest.param("/usr/local/bin/pi", "pi", id="pi"),
            # ``pi`` matches on the exact basename only — never a pip* tool.
            pytest.param("/usr/bin/pip install requests", None, id="pip"),
            pytest.param("/usr/local/bin/pipenv shell", None, id="pipenv"),
            pytest.param("/usr/bin/pip3 install foo", None, id="pip3"),
            pytest.param("/usr/bin/python3 script.py", None, id="unknown-process"),
        ],
    )
    def test_detects_provider_from_foreground_process(
        self, foreground_command: str, provider: str | None
    ) -> None:
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=foreground_command + "\n", stderr=""
        )
        with patch("ccgram.hook.subprocess.run", return_value=result):
            assert _provider_from_pane_tty("/dev/ttys012") == provider

    def test_empty_tty_returns_none(self) -> None:
        assert _provider_from_pane_tty("") is None
