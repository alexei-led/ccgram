"""Tests for the typed new-session entry (one message starts a session).

The entry point is stateless: an unresolved or ambiguous plan is answered with
one reply, never a keyboard, so there is no draft lifecycle to test.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.topics import typed_topic
from ccgram.handlers.topics.plan_parser import TopicPlan, parse_topic_plan
from ccgram.handlers.topics.typed_topic import (
    PlanError,
    _agent_blocked,
    _resolve_cwd,
    start_from_text,
)
from ccgram.worktrees import WorktreeEligibility


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git project under a configured projects root."""
    root = tmp_path / "Workspace"
    repo = root / "ccgram"
    (repo / ".git").mkdir(parents=True)
    from ccgram.config import config

    monkeypatch.setattr(config, "projects_roots", [root], raising=False)
    monkeypatch.setattr(config, "provider_name", "claude", raising=False)
    monkeypatch.setattr(config, "default_mode", "yolo", raising=False)
    monkeypatch.setattr(config, "dir_browser_all_dirs", False, raising=False)
    monkeypatch.setattr(
        config,
        "enabled_providers",
        ("antigravity", "claude", "codex", "pi", "shell"),
        raising=False,
    )
    return repo


def _message() -> MagicMock:
    message = MagicMock()
    message.chat.id = -100999
    message.message_id = 11
    message.reply_text = AsyncMock(return_value=MagicMock())
    return message


def _context(user_data: dict | None = None) -> MagicMock:
    context = MagicMock()
    context.user_data = {} if user_data is None else user_data
    context.bot_data = {}
    return context


async def _start(text: str, *, mru=(), message=None) -> tuple[bool, list, MagicMock]:
    launched: list = []
    message = message or _message()

    async def fake_launch(plan, **_kwargs):
        launched.append(plan)

    with (
        patch.object(typed_topic, "launch_plan", side_effect=fake_launch),
        patch.object(
            typed_topic.user_preferences, "get_user_mru", return_value=list(mru)
        ),
    ):
        handled = await start_from_text(
            user_id=100,
            thread_id=42,
            text=text,
            message=message,
            context=_context({}),
        )
    return handled, launched, message


class TestStartFromText:
    async def test_typed_line_launches_without_any_picker(self, project: Path) -> None:
        handled, launched, _message_ = await _start("codex ccgram fix the bug")

        assert handled is True
        assert len(launched) == 1
        plan = launched[0]
        assert (plan.agent, plan.project.name, plan.instruction) == (
            "codex",
            "ccgram",
            "fix the bug",
        )

    async def test_branch_is_carried_on_the_plan(self, project: Path) -> None:
        _handled, launched, _message_ = await _start("codex ccgram @feat-x do it")
        assert launched[0].branch == "feat-x"

    async def test_last_project_fills_the_missing_word(self, project: Path) -> None:
        _handled, launched, _message_ = await _start("codex fix it", mru=[str(project)])
        assert launched
        assert launched[0].project.name == "ccgram"
        assert launched[0].instruction == "fix it"

    async def test_an_ambiguous_project_gets_one_reply_naming_the_choices(
        self, project: Path
    ) -> None:
        sibling = project.parent / "ccgram-tools"
        (sibling / ".git").mkdir(parents=True)

        handled, launched, message = await _start("codex ccgr do it")

        assert handled is True
        assert launched == []
        reply = message.reply_text.call_args[0][0]
        assert "Ambiguous" in reply
        assert "ccgram" in reply

    async def test_a_missing_project_gets_one_reply_not_a_launch(
        self, project: Path
    ) -> None:
        handled, launched, message = await _start("codex nosuch @fix do it")

        assert handled is True
        assert launched == []
        assert "Project not found" in message.reply_text.call_args[0][0]

    async def test_an_unknown_mode_is_reported(self, project: Path) -> None:
        handled, launched, message = await _start("claude:sfae ccgram do it")

        assert handled is True
        assert launched == []
        assert "Unknown mode" in message.reply_text.call_args[0][0]

    async def test_disabled_agent_is_refused_with_the_enabled_list(
        self, project: Path
    ) -> None:
        handled, launched, message = await _start("gemini ccgram do it")

        assert handled is True
        assert launched == []
        reply = message.reply_text.call_args[0][0]
        assert "disabled" in reply and "claude" in reply

    async def test_plain_sentence_falls_through_to_the_browser(
        self, project: Path
    ) -> None:
        handled, launched, _message_ = await _start("hello there")

        assert handled is False
        assert launched == []


class TestAgentBlocked:
    def test_unknown_agent(self) -> None:
        assert "Unknown agent" in _agent_blocked("nosuchagent")

    def test_disabled_agent_names_the_knob(self, project: Path) -> None:
        message = _agent_blocked("gemini")
        assert "disabled" in message and "CCGRAM_PROVIDERS_ENABLED" in message

    def test_enabled_agent_passes(self, project: Path) -> None:
        assert _agent_blocked("codex") == ""


class TestResolveCwd:
    async def test_plain_project_is_used_as_is(self, project: Path) -> None:
        plan = TopicPlan(
            agent="codex", project=project, instruction="x", mode="yolo", label="l"
        )
        cwd, worktree = await _resolve_cwd(plan)
        assert cwd == str(project)
        assert worktree == {}

    async def test_branch_on_a_non_repo_is_rejected(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        plan = TopicPlan(
            agent="codex",
            project=plain,
            instruction="x",
            mode="yolo",
            branch="feat-x",
            label="l",
        )
        with pytest.raises(PlanError):
            await _resolve_cwd(plan)

    async def test_native_backend_plan_carries_a_worktree_path(
        self, project: Path
    ) -> None:
        plan = TopicPlan(
            agent="codex",
            project=project,
            instruction="x",
            mode="yolo",
            branch="feat-x",
            label="l",
        )
        native = MagicMock()
        native.capabilities.native_worktrees = True
        eligible = WorktreeEligibility(
            eligible=True,
            repo_path=project.resolve(),
            current_branch="main",
            dirty=False,
        )
        with (
            patch.object(typed_topic, "tmux_manager", native),
            patch.object(
                typed_topic, "check_worktree_eligibility", return_value=eligible
            ),
        ):
            cwd, worktree = await _resolve_cwd(plan)

        assert worktree["worktree_path"].endswith("feat-x")
        assert cwd == worktree["worktree_path"]

    async def test_a_subdirectory_target_is_materialized_by_ccgram(
        self, project: Path, tmp_path: Path
    ) -> None:
        """Native backends cannot root a session in a subdir, so ccgram does it."""
        subdir = project / "packages" / "api"
        subdir.mkdir(parents=True)
        plan = TopicPlan(
            agent="codex",
            project=subdir,
            instruction="x",
            mode="yolo",
            branch="feat-x",
            label="l",
        )
        native = MagicMock()
        native.capabilities.native_worktrees = True
        eligible = WorktreeEligibility(
            eligible=True,
            repo_path=project.resolve(),
            current_branch="main",
            dirty=False,
        )
        created: list = []

        def fake_create(repo, branch, path):
            created.append((repo, branch, path))
            (path / "packages" / "api").mkdir(parents=True)

        with (
            patch.object(typed_topic, "tmux_manager", native),
            patch.object(
                typed_topic, "check_worktree_eligibility", return_value=eligible
            ),
            patch.object(typed_topic, "create_worktree", side_effect=fake_create),
        ):
            cwd, worktree = await _resolve_cwd(plan)

        assert created
        assert cwd.endswith("packages/api")
        # ccgram made the checkout, so the backend must not make it again.
        assert worktree == {}


class TestLaunchPlan:
    async def test_yolo_only_where_the_provider_supports_it(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "default_mode", "yolo", raising=False)
        captured: list = []

        async def fake_launch(_target, _context, request):
            captured.append(request)

        plan = parse_topic_plan(
            "pi ccgram say hi", roots=[project.parent], default_agent="claude"
        ).plan
        assert plan is not None
        with (
            patch.object(typed_topic, "launch_window", side_effect=fake_launch),
            patch.object(typed_topic.user_preferences, "get_user_mru", return_value=[]),
        ):
            await typed_topic.launch_plan(
                plan,
                user_id=100,
                thread_id=42,
                message=_message(),
                context=_context({}),
            )

        assert captured[0].provider_name == "pi"
        assert captured[0].mode == "normal"

    async def test_typed_launch_ignores_other_topics_pending_state(
        self, project: Path
    ) -> None:
        """A half-finished browser flow in another topic must not leak in."""
        captured: list = []

        async def fake_launch(_target, _context, request):
            captured.append(request)

        plan = parse_topic_plan(
            "codex ccgram @feat-x do it", roots=[project.parent], default_agent="claude"
        ).plan
        assert plan is not None
        user_data = {
            "_pending_worktree_repo": "/other/repo",
            "_pending_worktree_branch": "other",
            "_pending_worktree_path": "/other/repo.worktrees/other",
            "_pending_workspace_id": "ws-from-another-topic",
        }
        with (
            patch.object(
                typed_topic, "_resolve_cwd", AsyncMock(return_value=(str(project), {}))
            ),
            patch.object(typed_topic, "launch_window", side_effect=fake_launch),
            patch.object(typed_topic.user_preferences, "get_user_mru", return_value=[]),
        ):
            await typed_topic.launch_plan(
                plan,
                user_id=100,
                thread_id=42,
                message=_message(),
                context=_context(user_data),
            )

        request = captured[0]
        assert request.ignore_pending_worktree is True
        assert request.ignore_pending_workspace is True
