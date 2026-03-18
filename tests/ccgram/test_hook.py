"""Tests for Claude Code session tracking hook."""

import io
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from ccgram.hook import (
    UUID_RE,
    _claude_settings_file,
    _hook_status,
    _install_hook,
    _is_hook_installed,
    _uninstall_hook,
    hook_main,
)


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
        assert session_start[0]["hooks"][0]["command"] == "ccgram hook"

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
        assert hooks_list[1]["command"] == "ccgram hook"

    def test_install_skips_when_already_present_with_wrapper(
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

    def test_install_skips_when_already_present_with_full_path(
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

    def test_install_uses_path_relative_command(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        _install_hook()

        updated = json.loads(settings_file.read_text())
        cmd = updated["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        assert cmd == "ccgram hook"
        assert "/" not in cmd


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
            assert any("ccgram hook" in h.get("command", "") for h in hooks_list)

    def test_async_flag_on_subagent_events(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        _install_hook()

        settings = json.loads(settings_file.read_text())
        for event_type in ("SubagentStart", "SubagentStop"):
            hook_config = settings["hooks"][event_type][0]["hooks"][0]
            assert hook_config.get("async") is True

        # SessionStart should NOT have async
        session_hook = settings["hooks"]["SessionStart"][0]["hooks"][0]
        assert "async" not in session_hook

    def test_idempotent_install(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        _install_hook()
        _install_hook()  # Second install

        settings = json.loads(settings_file.read_text())
        # Each event type should have exactly one ccgram hook entry
        for event_type in settings["hooks"]:
            entries = settings["hooks"][event_type]
            ccgram_hooks = [
                h
                for entry in entries
                for h in entry.get("hooks", [])
                if "ccgram hook" in h.get("command", "")
            ]
            assert len(ccgram_hooks) == 1, (
                f"{event_type} has {len(ccgram_hooks)} ccgram hooks"
            )


class TestUninstallMultipleEvents:
    def test_removes_all_event_types(self, tmp_path, monkeypatch) -> None:
        from ccgram.hook import _uninstall_hook, get_installed_events

        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        # Install first
        _install_hook()
        settings = json.loads(settings_file.read_text())
        assert all(get_installed_events(settings).values())

        # Uninstall
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

        # Stop events should NOT write session_map
        assert not (tmp_path / "session_map.json").exists()
        # But SHOULD write to events.jsonl
        events_file = tmp_path / "events.jsonl"
        assert events_file.exists()
        event = json.loads(events_file.read_text().strip())
        assert event["event"] == "Stop"
        assert event["window_key"] == "ccgram:@0"
        assert event["data"]["stop_reason"] == "end_turn"

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

    def test_uninstall_not_installed(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"hooks": {}}))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        result = _uninstall_hook()
        assert result == 0


class TestTabDelimitedParsing:
    """Verify tab-delimited tmux format parsing handles colons in names."""

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
        """Build settings with ccgram hook installed for all event types."""
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
