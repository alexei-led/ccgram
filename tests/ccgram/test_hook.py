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
    _foreground_pgid_on_tty,
    _hook_status,
    _install_hook,
    _is_nested_session,
    _provider_from_pane_tty,
    _uninstall_hook,
    get_installed_events,
    hook_main,
)
from ccgram.providers.base import UUID_RE


def _is_hook_installed(settings: dict) -> bool:
    return get_installed_events(settings).get("SessionStart", False)


def _expected_module_command() -> str:
    return f"{shlex.quote(sys.executable)} -m ccgram.main hook"


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

    def test_install_rewrites_wrapped_relative_command(
        self, tmp_path, monkeypatch
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "ccgram hook 2>/dev/null || true",
                                "timeout": 5,
                            }
                        ]
                    }
                ]
            }
        }
        settings_file.write_text(json.dumps(settings))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _install_hook()
        assert result == 0

        updated = json.loads(settings_file.read_text())
        hooks_list = updated["hooks"]["SessionStart"][0]["hooks"]
        assert len(hooks_list) == 1
        assert hooks_list[0]["command"] == _expected_module_command()

    def test_install_rewrites_full_path_command(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "/usr/local/bin/ccgram hook",
                                "timeout": 5,
                            }
                        ]
                    }
                ]
            }
        }
        settings_file.write_text(json.dumps(settings))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _install_hook()
        assert result == 0

        updated = json.loads(settings_file.read_text())
        hooks_list = updated["hooks"]["SessionStart"][0]["hooks"]
        assert len(hooks_list) == 1
        assert hooks_list[0]["command"] == _expected_module_command()

    def test_install_uses_current_python_module_command(
        self, tmp_path, monkeypatch
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        _install_hook()

        updated = json.loads(settings_file.read_text())
        cmd = updated["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        assert cmd == _expected_module_command()
        assert " -m ccgram.main hook" in cmd


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
    def test_hook_present(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {"type": "command", "command": "ccgram hook", "timeout": 5}
                        ]
                    }
                ]
            }
        }
        assert _is_hook_installed(settings) is True

    def test_no_hooks_key(self) -> None:
        assert _is_hook_installed({}) is False

    def test_different_hook_command(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "other-tool hook"}]}
                ]
            }
        }
        assert _is_hook_installed(settings) is False

    def test_shell_wrapped_command_matches(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "ccgram hook 2>/dev/null || true",
                            }
                        ]
                    }
                ]
            }
        }
        assert _is_hook_installed(settings) is True

    def test_full_path_matches(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "/usr/bin/ccgram hook",
                                "timeout": 5,
                            }
                        ]
                    }
                ]
            }
        }
        assert _is_hook_installed(settings) is True

    def test_python_module_command_matches(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    f"{shlex.quote(sys.executable)} -m ccgram.main hook"
                                ),
                                "timeout": 5,
                            }
                        ]
                    }
                ]
            }
        }
        assert _is_hook_installed(settings) is True


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

    def test_non_session_start_refreshes_changed_window_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_PANE", "%0")
        session_id = "550e8400-e29b-41d4-a716-446655440000"
        (tmp_path / "session_map.json").write_text(
            json.dumps(
                {
                    "ccgram:@0": {
                        "session_id": session_id,
                        "cwd": "/tmp",
                        "window_name": "old-name",
                        "transcript_path": "/tmp/transcript.jsonl",
                        "provider_name": "claude",
                    }
                }
            )
        )

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ccgram\t@0\tnew-name\n", stderr=""
        )
        with patch("ccgram.hook.subprocess.run", return_value=mock_result):
            self._run_hook_main(
                monkeypatch,
                {
                    "session_id": session_id,
                    "cwd": "/tmp",
                    "hook_event_name": "Stop",
                    "transcript_path": "/tmp/transcript.jsonl",
                    "stop_reason": "end_turn",
                },
                tmux_pane="%0",
            )

        session_map = json.loads((tmp_path / "session_map.json").read_text())
        assert session_map["ccgram:@0"]["window_name"] == "new-name"
        assert session_map["ccgram:@0"]["session_id"] == session_id

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
    def test_uninstall_removes_hook(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {"type": "command", "command": "ccgram hook", "timeout": 5}
                        ]
                    }
                ]
            }
        }
        settings_file.write_text(json.dumps(settings))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _uninstall_hook()
        assert result == 0

        updated = json.loads(settings_file.read_text())
        assert not _is_hook_installed(updated)

    def test_uninstall_no_settings_file(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _uninstall_hook()
        assert result == 0

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

        result = _uninstall_hook()
        assert result == 0

        updated = json.loads(settings_file.read_text())
        session_start = updated["hooks"]["SessionStart"]
        assert len(session_start) == 1
        hooks_list = session_start[0]["hooks"]
        assert len(hooks_list) == 1
        assert hooks_list[0]["command"] == "session-start.sh"

    def test_uninstall_removes_wrapped_variant(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "ccgram hook 2>/dev/null || true",
                                "timeout": 5,
                            }
                        ]
                    }
                ]
            }
        }
        settings_file.write_text(json.dumps(settings))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _uninstall_hook()
        assert result == 0

        updated = json.loads(settings_file.read_text())
        assert not _is_hook_installed(updated)
        assert updated["hooks"]["SessionStart"] == []

    def test_uninstall_removes_python_module_variant(
        self, tmp_path, monkeypatch
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    f"{shlex.quote(sys.executable)} -m ccgram.main hook"
                                ),
                                "timeout": 5,
                            }
                        ]
                    }
                ]
            }
        }
        settings_file.write_text(json.dumps(settings))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _uninstall_hook()
        assert result == 0

        updated = json.loads(settings_file.read_text())
        assert not _is_hook_installed(updated)
        assert updated["hooks"]["SessionStart"] == []

    def test_uninstall_not_installed(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"hooks": {}}))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _uninstall_hook()
        assert result == 0


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
    def _run(self, stdout: str) -> object:
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=stdout, stderr=""
        )
        with patch("ccgram.hook.subprocess.run", return_value=result):
            return _provider_from_pane_tty("/dev/ttys012")

    def test_detects_gemini(self) -> None:
        assert self._run("/usr/local/bin/gemini --model flash\n") == "gemini"

    def test_detects_codex(self) -> None:
        assert self._run("/usr/local/bin/codex\n") == "codex"

    def test_detects_claude(self) -> None:
        assert self._run("/usr/local/bin/claude\n") == "claude"

    def test_detects_pi_exact_basename(self) -> None:
        assert self._run("/usr/local/bin/pi\n") == "pi"

    def test_pi_does_not_match_pip(self) -> None:
        assert self._run("/usr/bin/pip install requests\n") is None

    def test_pi_does_not_match_pipenv(self) -> None:
        assert self._run("/usr/local/bin/pipenv shell\n") is None

    def test_pi_does_not_match_pip3(self) -> None:
        assert self._run("/usr/bin/pip3 install foo\n") is None

    def test_empty_tty_returns_none(self) -> None:
        assert _provider_from_pane_tty("") is None

    def test_unknown_process_returns_none(self) -> None:
        assert self._run("/usr/bin/python3 script.py\n") is None
