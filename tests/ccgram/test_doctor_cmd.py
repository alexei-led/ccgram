"""Tests for ccgram doctor command."""

import json

import pytest

from ccgram.doctor_cmd import (
    _check_allowed_users,
    _check_config_dir,
    _check_draft_streaming,
    _check_herdr,
    _check_herdr_hook_coexistence,
    _check_hooks,
    _check_multiplexer,
    _check_tmux,
    _find_orphaned_windows,
    doctor_main,
)
from ccgram.hook import _HOOK_EVENT_TYPES
from ccgram.telegram_draft import mark_draft_unavailable, reset_draft_state


class TestCheckTmux:
    def test_tmux_found(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which", lambda _cmd: "/usr/bin/tmux"
        )
        status, _msg = _check_tmux()
        assert status == "pass"

    def test_tmux_not_found(self, monkeypatch) -> None:
        monkeypatch.setattr("ccgram.doctor_cmd.shutil.which", lambda _cmd: None)
        status, msg = _check_tmux()
        assert status == "fail"
        assert "not found" in msg


class TestCheckConfigDir:
    def test_exists(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        status, _ = _check_config_dir()
        assert status == "pass"

    def test_missing(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "nonexistent"))
        status, _ = _check_config_dir()
        assert status == "fail"


class TestCheckAllowedUsers:
    def test_set(self, monkeypatch) -> None:
        monkeypatch.setenv("ALLOWED_USERS", "123,456")
        status, msg = _check_allowed_users()
        assert status == "pass"
        assert "2 user(s)" in msg

    def test_not_set(self, monkeypatch) -> None:
        monkeypatch.delenv("ALLOWED_USERS", raising=False)
        status, _ = _check_allowed_users()
        assert status == "fail"

    def test_invalid(self, monkeypatch) -> None:
        monkeypatch.setenv("ALLOWED_USERS", "abc")
        status, _ = _check_allowed_users()
        assert status == "fail"


class TestFindOrphanedWindows:
    def test_no_orphans(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_SESSION_NAME", "ccgram")

        state = {"thread_bindings": {"123": {"42": "@5"}}}
        (tmp_path / "state.json").write_text(json.dumps(state))

        monkeypatch.setattr(
            "ccgram.doctor_cmd._list_live_windows",
            lambda _: {"@5": "bound-window"},
        )

        assert _find_orphaned_windows() == []

    def test_finds_orphan(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_SESSION_NAME", "ccgram")

        monkeypatch.setattr(
            "ccgram.doctor_cmd._list_live_windows",
            lambda _: {"@10": "orphan-window"},
        )

        result = _find_orphaned_windows()
        assert len(result) == 1
        assert result[0] == ("@10", "orphan-window")


def _all_hooks_status() -> dict[str, bool]:
    """Return event status dict with all events installed."""
    return {event: True for event in _HOOK_EVENT_TYPES}


class TestCheckHooks:
    def test_all_installed(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        hooks: dict = {}
        for event_type in _HOOK_EVENT_TYPES:
            hooks[event_type] = [
                {"hooks": [{"type": "command", "command": "ccgram hook"}]}
            ]
        settings_file.write_text(json.dumps({"hooks": hooks}))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        status, msg, event_status = _check_hooks()
        assert status == "pass"
        assert all(event_status.values())

    def test_partial(self, tmp_path, monkeypatch) -> None:
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

        status, msg, event_status = _check_hooks()
        assert status == "warn"
        assert event_status["SessionStart"] is True
        assert event_status["Notification"] is False

    def test_none_installed(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"hooks": {}}))
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        status, msg, event_status = _check_hooks()
        assert status == "fail"
        assert not any(event_status.values())

    def test_missing_settings_file(self, tmp_path, monkeypatch) -> None:
        settings_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

        status, msg, event_status = _check_hooks()
        assert status == "fail"
        # Populated with {event: False} so doctor --fix can install them.
        assert event_status
        assert all(v is False for v in event_status.values())


class TestCheckPiHooks:
    _EVENTS = ("SessionStart", "Stop", "SessionEnd", "Notification")

    def test_missing_pi_hooks_are_reported(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)

        status, message, event_status = _check_hooks("pi")

        assert status == "fail"
        assert "hooks.json" in message
        assert event_status == dict.fromkeys(self._EVENTS, False)

    def test_flat_pi_hooks_are_checked_without_shadowing_nested_map(
        self, tmp_path, monkeypatch
    ) -> None:
        hooks_file = tmp_path / "hooks.json"
        settings = {
            **{
                event: [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "ccgram.main hook --provider pi",
                            }
                        ]
                    }
                ]
                for event in self._EVENTS
            },
            "PreToolUse": [
                {"hooks": [{"type": "command", "command": "custom-pre-tool-use"}]}
            ],
            "hookRunner": {"disabledHooks": ["custom-hook.sh"]},
            "customOption": {"keep": True},
        }
        hooks_file.write_text(json.dumps(settings))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))

        status, message, event_status = _check_hooks("pi")

        assert status == "pass"
        assert "4 hook events" in message
        assert event_status == dict.fromkeys(self._EVENTS, True)

    def test_nested_pi_hooks_take_precedence_over_flat_events(
        self, tmp_path, monkeypatch
    ) -> None:
        hooks_file = tmp_path / "hooks.json"
        settings = {
            **{
                event: [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "ccgram.main hook --provider pi",
                            }
                        ]
                    }
                ]
                for event in self._EVENTS
            },
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "custom-nested"}]}
                ]
            },
        }
        hooks_file.write_text(json.dumps(settings))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))

        status, _message, event_status = _check_hooks("pi")

        assert status == "fail"
        assert event_status == dict.fromkeys(self._EVENTS, False)

    def test_custom_agent_dir_pi_hooks_are_checked(self, tmp_path, monkeypatch) -> None:
        agent_dir = tmp_path / "custom-agent"
        hooks_file = agent_dir / "hooks.json"
        agent_dir.mkdir()
        hooks_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        event: [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "ccgram.main hook --provider pi",
                                    }
                                ]
                            }
                        ]
                        for event in self._EVENTS
                    }
                }
            )
        )
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))

        status, message, event_status = _check_hooks("pi")

        assert status == "pass"
        assert "4 hook events" in message
        assert event_status == dict.fromkeys(self._EVENTS, True)

    def test_malformed_pi_hooks_are_reported_not_raised(
        self, tmp_path, monkeypatch
    ) -> None:
        agent_dir = tmp_path / ".pi" / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "hooks.json").write_text(json.dumps({"hooks": []}))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)

        status, _message, event_status = _check_hooks("pi")

        assert status == "fail"
        assert event_status == dict.fromkeys(self._EVENTS, False)


class TestDoctorMain:
    def test_runs_without_crash(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TMUX_SESSION_NAME", "test")
        monkeypatch.setenv("ALLOWED_USERS", "123")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        tmp_path.mkdir(exist_ok=True)

        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which",
            lambda _cmd: f"/usr/bin/{_cmd}",
        )
        monkeypatch.setattr(
            "ccgram.doctor_cmd._check_tmux_session",
            lambda: ("pass", 'tmux session "test" exists'),
        )
        monkeypatch.setattr(
            "ccgram.doctor_cmd._check_hooks",
            lambda _provider="claude": (
                "pass",
                "all 5 hook events installed",
                _all_hooks_status(),
            ),
        )
        monkeypatch.setattr(
            "ccgram.doctor_cmd._find_orphaned_windows",
            lambda: [],
        )

        with pytest.raises(SystemExit) as exc_info:
            doctor_main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "\u2713" in captured.out

    def test_shows_provider_name(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("CCGRAM_PROVIDER", "claude")
        monkeypatch.setenv("TMUX_SESSION_NAME", "test")
        monkeypatch.setenv("ALLOWED_USERS", "123")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")

        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which",
            lambda _cmd: f"/usr/bin/{_cmd}",
        )
        monkeypatch.setattr(
            "ccgram.doctor_cmd._check_tmux_session",
            lambda: ("pass", "ok"),
        )
        monkeypatch.setattr(
            "ccgram.doctor_cmd._check_hooks",
            lambda _provider="claude": (
                "pass",
                "all 5 hook events installed",
                _all_hooks_status(),
            ),
        )
        monkeypatch.setattr("ccgram.doctor_cmd._find_orphaned_windows", lambda: [])

        with pytest.raises(SystemExit):
            doctor_main()

        captured = capsys.readouterr()
        assert "Provider: claude" in captured.out

    def test_reports_missing_hooks_for_codex_provider(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("CCGRAM_PROVIDER", "codex")
        monkeypatch.setenv("TMUX_SESSION_NAME", "test")
        monkeypatch.setenv("ALLOWED_USERS", "123")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")

        # Isolate from the dev machine's real ~/.codex (codex hooks may be
        # installed there) so the "not installed" path is exercised.
        monkeypatch.setattr(
            "ccgram.hook._codex_hooks_file",
            lambda: tmp_path / ".codex" / "hooks.json",
        )

        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which",
            lambda _cmd: f"/usr/bin/{_cmd}",
        )
        monkeypatch.setattr(
            "ccgram.doctor_cmd._check_tmux_session",
            lambda: ("pass", "ok"),
        )
        monkeypatch.setattr("ccgram.doctor_cmd._find_orphaned_windows", lambda: [])
        monkeypatch.setattr(
            "ccgram.hook._codex_hooks_file",
            lambda: tmp_path / ".codex" / "hooks.json",
        )

        with pytest.raises(SystemExit):
            doctor_main()

        captured = capsys.readouterr()
        assert "Provider: codex" in captured.out
        assert "hooks not installed" in captured.out


class TestCheckProviderCommand:
    def test_found(self, monkeypatch) -> None:
        from ccgram.doctor_cmd import _check_provider_command

        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which", lambda _cmd: "/usr/bin/codex"
        )
        status, msg = _check_provider_command("codex")
        assert status == "pass"
        assert "codex" in msg

    def test_not_found(self, monkeypatch) -> None:
        from ccgram.doctor_cmd import _check_provider_command

        monkeypatch.setattr("ccgram.doctor_cmd.shutil.which", lambda _cmd: None)
        status, msg = _check_provider_command("codex")
        assert status == "fail"
        assert "codex" in msg

    def test_per_provider_env_override(self, monkeypatch) -> None:
        from ccgram.doctor_cmd import _check_provider_command

        monkeypatch.setenv("CCGRAM_CODEX_COMMAND", "my-codex-wrapper")
        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which", lambda _cmd: "/usr/bin/my-codex-wrapper"
        )
        status, msg = _check_provider_command("codex")
        assert status == "pass"
        assert "my-codex-wrapper" in msg


class _FakeHerdrBackend:
    """Minimal stand-in for the herdr backend's ``ensure_session`` probe."""

    def __init__(self, fail: Exception | None = None) -> None:
        self._fail = fail

    async def ensure_session(self) -> None:
        if self._fail is not None:
            raise self._fail


class TestCheckMultiplexer:
    def test_tmux_default(self, monkeypatch) -> None:
        monkeypatch.delenv("CCGRAM_MULTIPLEXER", raising=False)
        status, msg = _check_multiplexer()
        assert status == "pass"
        assert "tmux" in msg

    def test_herdr_selected(self, monkeypatch) -> None:
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", "herdr")
        status, msg = _check_multiplexer()
        assert status == "pass"
        assert "herdr" in msg

    def test_unknown_backend_fails(self, monkeypatch) -> None:
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", "bogus")
        status, msg = _check_multiplexer()
        assert status == "fail"
        assert "bogus" in msg


class TestCheckHerdr:
    def test_binary_missing(self, monkeypatch) -> None:
        monkeypatch.setattr("ccgram.doctor_cmd.shutil.which", lambda _cmd: None)
        status, msg = _check_herdr()
        assert status == "fail"
        assert "not found" in msg

    def test_socket_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which", lambda _cmd: "/usr/bin/herdr"
        )
        monkeypatch.setattr(
            "ccgram.multiplexer.get_multiplexer", lambda _name: _FakeHerdrBackend()
        )
        status, msg = _check_herdr()
        assert status == "pass"
        assert "server reachable" in msg

    def test_server_error_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which", lambda _cmd: "/usr/bin/herdr"
        )
        boom = RuntimeError("herdr status failed: connection refused")
        monkeypatch.setattr(
            "ccgram.multiplexer.get_multiplexer",
            lambda _name: _FakeHerdrBackend(fail=boom),
        )
        status, msg = _check_herdr()
        assert status == "fail"
        assert "connection refused" in msg


class TestCheckHerdrHookCoexistence:
    def _write_claude_settings(self, tmp_path, monkeypatch, commands) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {"hooks": [{"type": "command", "command": c}]}
                            for c in commands
                        ]
                    }
                }
            )
        )
        monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

    def test_both_present(self, tmp_path, monkeypatch) -> None:
        self._write_claude_settings(
            tmp_path, monkeypatch, ["ccgram hook", "herdr integration hook"]
        )
        status, msg = _check_herdr_hook_coexistence()
        assert status == "pass"
        assert "coexist" in msg

    def test_only_ccgram_warns(self, tmp_path, monkeypatch) -> None:
        self._write_claude_settings(tmp_path, monkeypatch, ["ccgram hook"])
        status, msg = _check_herdr_hook_coexistence()
        assert status == "warn"
        assert "herdr" in msg

    def test_ccgram_missing_fails(self, tmp_path, monkeypatch) -> None:
        self._write_claude_settings(tmp_path, monkeypatch, ["herdr integration hook"])
        status, msg = _check_herdr_hook_coexistence()
        assert status == "fail"
        assert "ccgram" in msg

    def test_settings_missing_warns(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.hook._claude_settings_file",
            lambda: tmp_path / "nonexistent.json",
        )
        status, msg = _check_herdr_hook_coexistence()
        assert status == "warn"
        assert "missing" in msg


class TestDoctorMainHerdrMode:
    def test_runs_herdr_branch(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("CCGRAM_MULTIPLEXER", "herdr")
        monkeypatch.setenv("ALLOWED_USERS", "123")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        tmp_path.mkdir(exist_ok=True)

        monkeypatch.setattr(
            "ccgram.doctor_cmd.shutil.which", lambda _cmd: f"/usr/bin/{_cmd}"
        )
        # Avoid touching a real herdr socket / settings.json.
        monkeypatch.setattr(
            "ccgram.doctor_cmd._check_herdr",
            lambda: ("pass", "herdr server reachable, protocol OK"),
        )
        monkeypatch.setattr(
            "ccgram.doctor_cmd._check_herdr_hook_coexistence",
            lambda: ("pass", "ccgram + herdr Claude hooks coexist"),
        )
        monkeypatch.setattr(
            "ccgram.doctor_cmd._check_hooks",
            lambda _provider="claude": (
                "pass",
                "all 5 hook events installed",
                _all_hooks_status(),
            ),
        )

        # Orphan scan would shell out to tmux; it must be skipped on herdr.
        def _fail_orphans():
            raise AssertionError("orphan scan must not run on herdr")

        monkeypatch.setattr("ccgram.doctor_cmd._find_orphaned_windows", _fail_orphans)

        with pytest.raises(SystemExit) as exc_info:
            doctor_main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "multiplexer backend: herdr" in captured.out
        assert "tmux session" not in captured.out
        assert "orphaned" not in captured.out


class TestCheckDraftStreaming:
    def test_available_when_flag_unset(self) -> None:
        reset_draft_state()
        status, msg = _check_draft_streaming()
        assert status == "pass"
        assert "[draft-streaming]" in msg
        assert "untested" in msg

    def test_warns_when_flag_set(self) -> None:
        reset_draft_state()
        mark_draft_unavailable("Bot API <9.5")
        status, msg = _check_draft_streaming()
        assert status == "warn"
        assert "[draft-streaming]" in msg
        assert "degraded" in msg
        assert "Bot API <9.5" in msg
        reset_draft_state()

    def test_warns_with_default_reason_when_empty(self) -> None:
        reset_draft_state()
        mark_draft_unavailable("")
        status, msg = _check_draft_streaming()
        assert status == "warn"
        assert "Bot API <9.5" in msg
        reset_draft_state()


@pytest.mark.parametrize("group", [{}, {"hooks": [{}]}, {"hooks": [{"command": None}]}])
def test_pi_diagnostics_ignore_missing_hook_fields(tmp_path, monkeypatch, group):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    (tmp_path / "hooks.json").write_text(
        json.dumps({"hooks": {"SessionStart": [group]}})
    )

    status, _message, events = _check_hooks("pi")

    assert status == "fail"
    assert events and not any(events.values())


@pytest.mark.parametrize("value", [[], None, True, 7, "not-an-object"])
def test_pi_diagnostics_reject_non_object_json(tmp_path, monkeypatch, value):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    (tmp_path / "hooks.json").write_text(json.dumps(value))

    status, _message, events = _check_hooks("pi")

    assert status == "fail"
    assert events and not any(events.values())
