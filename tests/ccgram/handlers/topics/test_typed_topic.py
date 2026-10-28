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
    message.reply_text = AsyncMock(return_value=MagicMock())
    return message


def _update(thread_id: int = 42) -> MagicMock:
    update = MagicMock()
    # get_thread_id prefers update.message; a callback update carries none.
    update.message = None
    update.callback_query = MagicMock()
    update.callback_query.message = MagicMock()
    update.callback_query.message.message_thread_id = thread_id
    return update


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
        assert user_data[PENDING_PLAN]["instruction"] == "hello there"

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

    async def test_the_draft_identity_is_the_keyboard_message(
        self, project: Path
    ) -> None:
        """A callback arrives on the bot's prompt message, not the user's."""
        message = _message()
        message.message_id = 11
        reply = MagicMock(message_id=777)
        user_data: dict = {}

        with (
            patch.object(typed_topic, "safe_reply", AsyncMock(return_value=reply)),
            patch.object(typed_topic.user_preferences, "get_user_mru", return_value=[]),
        ):
            await start_from_text(
                user_id=100,
                thread_id=42,
                text="codex nosuchproject do it",
                user_data=user_data,
                message=message,
                context=_context(user_data),
            )

        draft = user_data[PENDING_PLAN]
        assert draft["message_id"] == reply.message_id
        assert draft["message_id"] != message.message_id
        # The unresolved project word stays in the instruction: it is not
        # consumed, so nothing the user typed is silently dropped.
        assert draft["instruction"] == "nosuchproject do it"


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


class TestReviewFindings:
    """Regressions for the findings the pre-merge review raised."""

    async def test_native_backend_plan_carries_a_worktree_path(
        self, project: Path
    ) -> None:
        from unittest.mock import patch

        from ccgram.handlers.topics.plan_parser import TopicPlan

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

        assert worktree["worktree_branch"] == "feat-x"
        assert worktree["worktree_repo"] == str(project.resolve())
        # Without worktree_path the launch gate skips worktree creation and
        # launches in the source checkout, so this assertion is the regression.
        assert worktree["worktree_path"].endswith("feat-x")
        assert cwd == worktree["worktree_path"]

    async def test_agent_without_a_yolo_flag_launches_normally(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "default_mode", "yolo", raising=False)
        captured: list = []

        async def fake_launch(_target, _context, request):
            captured.append(request)

        message = _message()
        plan = typed_topic.parse_topic_plan(
            "pi ccgram say hi",
            roots=[project.parent],
            default_agent="claude",
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
                message=message,
                context=_context({}),
            )

        assert captured and captured[0].provider_name == "pi"
        assert captured[0].mode == "normal"


class TestPlanDraftIdentity:
    """A keyboard may only launch the draft it was rendered for."""

    @staticmethod
    def _draft(message_id: int, candidate: Path) -> dict:
        return {
            "agent": "codex",
            "mode": "yolo",
            "instruction": "do it",
            "branch": None,
            "thread_id": 42,
            "chat_id": -100999,
            "message_id": message_id,
            "candidates": [str(candidate)],
        }

    @staticmethod
    def _query(message_id: int) -> MagicMock:
        query = AsyncMock()
        query.data = f"{CB_PLAN_SELECT}0"
        query.message = MagicMock()
        query.message.message_id = message_id
        query.message.chat.id = -100999
        return query

    async def test_an_older_keyboard_cannot_launch_the_newer_plan(
        self, project: Path
    ) -> None:
        user_data = {PENDING_PLAN: self._draft(11, project.resolve())}
        launched: list = []

        async def fake_launch(plan, **_kwargs):
            launched.append(plan)

        stale = self._query(22)
        with patch.object(typed_topic, "launch_plan", side_effect=fake_launch):
            await typed_topic._handle_plan_callback(
                stale, 100, f"{CB_PLAN_SELECT}0", _update(), _context(user_data)
            )

        assert stale.answer.await_args.args[0].startswith("Stale plan")

        assert launched == []
        assert PENDING_PLAN in user_data

    async def test_the_matching_keyboard_launches(self, project: Path) -> None:
        user_data = {PENDING_PLAN: self._draft(11, project.resolve())}
        launched: list = []

        async def fake_launch(plan, **_kwargs):
            launched.append(plan)

        with patch.object(typed_topic, "launch_plan", side_effect=fake_launch):
            await typed_topic._handle_plan_callback(
                self._query(11),
                100,
                f"{CB_PLAN_SELECT}0",
                _update(),
                _context(user_data),
            )

        assert launched and launched[0].project == project.resolve()

    async def test_start_clears_a_pending_plan(self, project: Path) -> None:
        """A resolution keyboard must fail closed after /start, like every other."""
        from ccgram.handlers.topics.new_command import new_command

        user_data: dict = {}
        await _start("codex nosuchproject do it", user_data=user_data)
        assert PENDING_PLAN in user_data

        update = MagicMock()
        update.effective_user.id = 100
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        context = _context(user_data)
        with (
            patch.object(
                new_command.__globals__["config"], "is_user_allowed", return_value=True
            ),
            patch.object(new_command.__globals__["config"], "group_id", None),
            patch("ccgram.handlers.topics.new_command.commands_command", AsyncMock()),
        ):
            await new_command(update, context)

        assert PENDING_PLAN not in user_data

    async def test_typed_plan_ignores_a_browser_flows_pending_worktree(
        self, project: Path
    ) -> None:
        """A typed launch must not inherit another flow's PENDING_WORKTREE_*."""
        from ccgram.handlers.topics.window_launch_service import (
            WindowLaunchRequest,
            _create_topic_window,
        )

        request = WindowLaunchRequest(
            user_id=100,
            thread_id=42,
            provider_name="codex",
            cwd=str(project),
            mode="yolo",
            pending_text="do it",
            ignore_pending_worktree=True,
        )
        captured: list = []
        adapter = MagicMock()

        async def create(data, **_kwargs):
            captured.append(data)
            return MagicMock(title="t", target_id="@9")

        adapter.create = AsyncMock(side_effect=create)
        user_data = {
            "browse_path": str(project),
            "_pending_worktree_repo": "/some/other/repo",
            "_pending_worktree_branch": "other-branch",
            "_pending_worktree_path": "/some/other/repo.worktrees/other-branch",
        }
        context = _context(user_data)
        with (
            patch(
                "ccgram.handlers.topics.window_launch_service.adapter_for",
                lambda _bot_data: adapter,
            ),
            patch(
                "ccgram.handlers.topics.window_launch_service.preseed_trust",
                lambda *a, **k: False,
            ),
        ):
            await _create_topic_window(
                str(project), "codex", "yolo", None, context, request
            )

        assert captured
        assert "worktree" not in captured[0]
        assert captured[0]["directory"] == str(project)
