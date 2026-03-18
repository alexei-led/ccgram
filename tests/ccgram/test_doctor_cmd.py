"""Tests for ccgram doctor command."""

import json

import pytest

from ccgram.doctor_cmd import (
    _check_allowed_users,
    _check_config_dir,
    _check_hooks,
    _check_tmux,
    _find_orphaned_windows,
    doctor_main,
)
from ccgram.hook import _HOOK_EVENT_TYPES


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
        assert event_status == {}


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
            lambda: ("pass", "all 5 hook events installed", _all_hooks_status()),
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
            lambda: ("pass", "all 5 hook events installed", _all_hooks_status()),
        )
        monkeypatch.setattr("ccgram.doctor_cmd._find_orphaned_windows", lambda: [])

        with pytest.raises(SystemExit):
            doctor_main()

        captured = capsys.readouterr()
        assert "Provider: claude" in captured.out

    def test_skips_hook_check_for_hookless_provider(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("CCGRAM_PROVIDER", "codex")
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
        monkeypatch.setattr("ccgram.doctor_cmd._find_orphaned_windows", lambda: [])

        with pytest.raises(SystemExit):
            doctor_main()

        captured = capsys.readouterr()
        assert "Provider: codex" in captured.out
        assert "hook check skipped" in captured.out


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
