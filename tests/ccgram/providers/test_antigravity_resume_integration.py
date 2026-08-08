import importlib
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import quote

import pytest

from ccgram.providers import resolve_launch_command
from ccgram.providers.antigravity import AntigravityProvider

resume_command = importlib.import_module("ccgram.handlers.recovery.resume_command")
resume_picker = importlib.import_module("ccgram.handlers.recovery.resume_picker")


class TestAntigravityWorkspaceDiscovery:
    @pytest.mark.parametrize(
        "encoded", [False, True], ids=["literal-space", "encoded-space"]
    )
    def test_discovers_workspace_with_spaces(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoded: bool
    ) -> None:
        workspace = tmp_path / "workspace with spaces"
        workspace.mkdir()
        brain = tmp_path / "brain"
        log_dir = brain / "conversation-1" / ".system_generated" / "logs"
        log_dir.mkdir(parents=True)
        workspace_uri_path = quote(str(workspace)) if encoded else str(workspace)
        (log_dir / "transcript.jsonl").write_text(
            json.dumps({"cwd": f"file://{workspace_uri_path}"}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CCGRAM_ANTIGRAVITY_DATA_DIR", str(brain))

        sessions = AntigravityProvider().discover_resumable_sessions()

        assert [(session.session_id, session.cwd) for session in sessions] == [
            ("conversation-1", str(workspace.resolve()))
        ]

    def test_discovers_real_cli_layout_from_last_conversations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        brain = tmp_path / "brain"
        session_id = "a4186fac-9bf9-40d2-b655-02bdd21cf0dd"
        log_dir = brain / session_id / ".system_generated" / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": "hello",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cache = tmp_path / "cache" / "last_conversations.json"
        cache.parent.mkdir()
        cache.write_text(json.dumps({str(workspace): session_id}), encoding="utf-8")
        monkeypatch.setenv("CCGRAM_ANTIGRAVITY_DATA_DIR", str(brain))

        sessions = AntigravityProvider().discover_resumable_sessions(cwd=str(workspace))

        assert [(session.session_id, session.cwd) for session in sessions] == [
            (session_id, str(workspace.resolve()))
        ]

    def test_discovery_ignores_pytest_environment_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        brain = tmp_path / "brain"
        log_dir = brain / "conversation-1" / ".system_generated" / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "transcript.jsonl").write_text(
            json.dumps({"cwd": str(workspace)}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CCGRAM_ANTIGRAVITY_DATA_DIR", str(brain))
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "inherited-marker")

        sessions = AntigravityProvider().discover_resumable_sessions()

        assert [session.session_id for session in sessions] == ["conversation-1"]


class TestProviderScopedResumeDiscovery:
    def test_global_scan_uses_only_selected_provider(self, monkeypatch) -> None:
        provider = MagicMock()
        provider.discover_resumable_sessions.return_value = [
            SimpleNamespace(
                session_id="agy-session",
                summary="Antigravity",
                cwd="/workspace",
                mtime=10.0,
                msg_count=None,
                provider_name="antigravity",
            )
        ]
        get_provider = MagicMock(return_value=provider)
        monkeypatch.setattr(resume_command, "get_provider_for_window", get_provider)

        sessions = resume_command.scan_all_sessions("antigravity")

        get_provider.assert_called_once_with("", provider_name="antigravity")
        provider.discover_resumable_sessions.assert_called_once_with()
        assert [session.provider_name for session in sessions] == ["antigravity"]

    def test_cwd_scan_uses_only_selected_provider(self, tmp_path, monkeypatch) -> None:
        provider = MagicMock()
        provider.discover_resumable_sessions.return_value = [
            SimpleNamespace(
                session_id="agy-session",
                summary="Antigravity",
                cwd=str(tmp_path),
                mtime=10.0,
                msg_count=None,
                provider_name="antigravity",
            )
        ]
        get_provider = MagicMock(return_value=provider)
        monkeypatch.setattr(resume_picker, "get_provider_for_window", get_provider)

        sessions = resume_picker.scan_sessions_for_cwd(
            str(tmp_path), provider_name="antigravity"
        )

        get_provider.assert_called_once_with("", provider_name="antigravity")
        provider.discover_resumable_sessions.assert_called_once_with(
            cwd=str(tmp_path.resolve()), limit=6
        )
        assert [session.provider_name for session in sessions] == ["antigravity"]


class TestProviderScopedResumeLaunch:
    async def test_global_resume_launches_selected_provider(
        self, tmp_path, monkeypatch
    ) -> None:
        provider = MagicMock()
        provider.capabilities.name = "antigravity"
        provider.capabilities.supports_hook = False
        provider.make_launch_args.return_value = "--conversation agy-session"
        monkeypatch.setattr(
            resume_command, "get_provider_for_window", MagicMock(return_value=provider)
        )
        monkeypatch.setattr(
            resume_command, "resolve_launch_command", MagicMock(return_value="agy")
        )
        tmux_manager = MagicMock()
        tmux_manager.create_window = AsyncMock(
            return_value=(True, "created", "project", "@9")
        )
        monkeypatch.setattr(resume_command, "tmux_manager", tmux_manager)
        monkeypatch.setattr(
            resume_command.thread_router,
            "get_window_for_thread",
            MagicMock(return_value=None),
        )

        result = await resume_command._create_resume_window(
            1,
            2,
            "agy-session",
            str(tmp_path),
            provider_name="antigravity",
        )

        assert result[0] is True
        provider.make_launch_args.assert_called_once_with(resume_id="agy-session")
        tmux_manager.create_window.assert_awaited_once_with(
            str(tmp_path),
            agent_args="--conversation agy-session",
            launch_command="agy",
        )

    async def test_invalid_session_does_not_unbind_old_window(
        self, tmp_path, monkeypatch
    ) -> None:
        provider = MagicMock()
        provider.make_launch_args.side_effect = ValueError("invalid session")
        thread_router = MagicMock()
        thread_router.get_window_for_thread.return_value = "@3"
        monkeypatch.setattr(resume_command, "thread_router", thread_router)
        monkeypatch.setattr(
            resume_command.window_query,
            "view_window",
            MagicMock(return_value=SimpleNamespace(approval_mode="normal")),
        )
        monkeypatch.setattr(
            resume_command, "get_provider_for_window", MagicMock(return_value=provider)
        )

        with pytest.raises(ValueError, match="invalid session"):
            await resume_command._create_resume_window(
                1,
                2,
                "invalid/session",
                str(tmp_path),
                provider_name="antigravity",
            )

        thread_router.unbind_thread.assert_not_called()


class TestAntigravityExecutableQuoting:
    def test_automatic_executable_path_with_spaces_is_shell_safe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home with spaces"
        executable = home / ".local" / "bin" / "agy"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        monkeypatch.delenv("CCGRAM_ANTIGRAVITY_COMMAND", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: home)
        monkeypatch.setattr("shutil.which", lambda _name: None)

        command = resolve_launch_command("antigravity")

        assert shlex.split(command) == [str(executable.resolve())]
