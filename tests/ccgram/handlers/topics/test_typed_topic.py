"""Tests for the typed new-session entry (one message starts a session)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_PLAN_SELECT
from ccgram.handlers.topics import typed_topic
from ccgram.handlers.topics.typed_topic import (
    PlanError,
    _agent_blocked,
    _resolve_cwd,
    start_from_text,
)
from ccgram.handlers.user_state import PENDING_PLAN


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
    message.reply_text = AsyncMock(return_value=MagicMock())
    return message


def _context(user_data: dict | None = None) -> MagicMock:
    context = MagicMock()
    context.user_data = {} if user_data is None else user_data
    context.bot_data = {}
    return context


async def _start(text: str, *, user_data: dict | None = None, mru=()) -> tuple:
    launched: list = []

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
            user_data=user_data if user_data is not None else {},
            message=_message(),
            context=_context(user_data),
        )
    return handled, launched


class TestStartFromText:
    async def test_typed_line_launches_without_any_picker(self, project: Path) -> None:
        handled, launched = await _start("codex ccgram fix the hook timeout")

        assert handled is True
        assert len(launched) == 1
        plan = launched[0]
        assert (plan.agent, plan.project.name, plan.instruction) == (
            "codex",
            "ccgram",
            "fix the hook timeout",
        )

    async def test_branch_is_carried_on_the_plan(self, project: Path) -> None:
        _handled, launched = await _start("codex ccgram @feat-x do it")
        assert launched[0].branch == "feat-x"

    async def test_last_project_fills_the_missing_word(self, project: Path) -> None:
        _handled, launched = await _start("codex fix it", mru=[str(project)])
        assert launched
        assert launched[0].project.name == "ccgram"
        assert launched[0].instruction == "fix it"

    async def test_unknown_project_offers_candidates_and_stores_a_draft(
        self, project: Path
    ) -> None:
        user_data: dict = {}
        handled, launched = await _start(
            "codex nosuchproject do it", user_data=user_data
        )

        assert handled is True
        assert launched == []
        draft = user_data[PENDING_PLAN]
        assert draft["agent"] == "codex"
        assert draft["candidates"] == [str(project.resolve())]

    async def test_disabled_agent_is_refused_with_the_enabled_list(
        self, project: Path
    ) -> None:
        message = _message()
        with patch.object(
            typed_topic.user_preferences, "get_user_mru", return_value=[]
        ):
            handled = await start_from_text(
                user_id=100,
                thread_id=42,
                text="gemini ccgram do it",
                user_data={},
                message=message,
                context=_context({}),
            )

        assert handled is True
        reply = message.reply_text.call_args[0][0]
        assert "disabled" in reply
        assert "claude" in reply

    async def test_plain_sentence_offers_projects_and_keeps_the_text(
        self, project: Path
    ) -> None:
        user_data: dict = {}
        handled, launched = await _start("hello there", user_data=user_data)

        assert handled is True
        assert launched == []
        assert user_data[PENDING_PLAN]["text"] == "hello there"

    async def test_plain_sentence_with_no_projects_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.config import config

        empty = tmp_path / "empty-root"
        empty.mkdir()
        monkeypatch.setattr(config, "projects_roots", [empty], raising=False)
        handled, launched = await _start("hello there")

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
        from ccgram.handlers.topics.plan_parser import TopicPlan

        plan = TopicPlan(
            agent="codex", project=project, instruction="x", mode="yolo", label="l"
        )
        cwd, worktree = await _resolve_cwd(plan)
        assert cwd == str(project)
        assert worktree == {}

    async def test_branch_on_a_non_repo_is_rejected(self, tmp_path: Path) -> None:
        from ccgram.handlers.topics.plan_parser import TopicPlan

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


class TestPlanCallbackConstants:
    def test_select_prefix_is_stable(self) -> None:
        assert CB_PLAN_SELECT == "plan:p:"
