"""Typed new-session entry: one message starts a session.

Replaces the window-picker → directory-browser → worktree → workspace →
provider → mode chain with a single parsed line::

    <agent>[:<mode>] <project|path> [@branch] <instruction...>

Nothing here holds state between messages. A plan that cannot be resolved is
answered with one short reply naming what is missing or ambiguous, and the
operator retypes with a fuller name. There is no pending draft, no keyboard
and therefore no stale-button, cross-topic or reset lifecycle to get wrong.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ...config import config
from ...multiplexer import multiplexer as tmux_manager
from ...providers import has_yolo_mode, is_known_provider, provider_launchable
from ...user_preferences import user_preferences
from ...worktrees import (
    WorktreeError,
    check_worktree_eligibility,
    create_worktree,
    slug_for_path,
    suggest_branch_name,
    worktree_path_for,
)
from ..messaging_pipeline.message_sender import interactive_edit as safe_edit
from ..messaging_pipeline.message_sender import safe_reply
from .plan_parser import PROJECT_HINTS, PlanParse, TopicPlan, parse_topic_plan
from .window_launch_service import WindowLaunchRequest, launch_window

if TYPE_CHECKING:
    from telegram import Message
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

__all__ = ["start_from_text", "launch_plan"]


class PlanError(RuntimeError):
    """A plan that parsed but cannot be launched (bad worktree, bad agent)."""


def _mru_projects(user_id: int) -> list[Path]:
    """Projects this user used recently, newest first, existing only."""
    projects: list[Path] = []
    for entry in user_preferences.get_user_mru(user_id):
        try:
            path = Path(entry)
            if path.is_dir():
                projects.append(path.resolve())
        except OSError:
            continue
    return projects


def _last_project(user_id: int) -> Path | None:
    projects = _mru_projects(user_id)
    return projects[0] if projects else None


def _agent_blocked(agent: str) -> str:
    """Reason *agent* cannot be started here, or "" when it is fine."""
    # is_known_provider registers the built-ins first; the raw registry is
    # empty until something asks for a provider.
    if not is_known_provider(agent):
        return f"Unknown agent '{agent}'."
    enabled = config.enabled_providers
    if enabled and agent not in enabled:
        return (
            f"Agent '{agent}' is disabled. Enabled: {', '.join(enabled)} "
            "(CCGRAM_PROVIDERS_ENABLED)."
        )
    if not provider_launchable(agent):
        return (
            f"Agent '{agent}' is not installed (no '{agent}' on PATH). "
            "Install it or set a command override."
        )
    return ""


async def _materialize_worktree(repo: Path, branch: str, subdir: Path) -> Path:
    """Create the checkout with git and return the directory to launch in."""
    worktree_path = worktree_path_for(repo, slug_for_path(branch))
    if worktree_path.exists():
        raise PlanError(f"Worktree {worktree_path} already exists")
    try:
        await asyncio.to_thread(create_worktree, repo, branch, worktree_path)
    except WorktreeError as exc:
        raise PlanError(
            f"Could not create worktree: {str(exc).splitlines()[0]}"
        ) from exc
    target = worktree_path / subdir if subdir.parts else worktree_path
    if target.is_dir():
        logger.info(
            "Worktree %s on %s ready for plan launch (cwd=%s)",
            worktree_path,
            branch,
            target,
        )
        return target
    logger.info(
        "Worktree subdir %s absent in the fresh checkout; rooting at %s",
        subdir,
        worktree_path,
    )
    return worktree_path


def _subdir_within(project: Path, repo: Path) -> Path:
    """Path of *project* relative to *repo*, empty when they are the same."""
    try:
        return project.resolve().relative_to(repo.resolve())
    except OSError, ValueError:
        return Path()


async def _resolve_cwd(plan: TopicPlan) -> tuple[str, dict[str, str]]:
    """Resolve the launch cwd and any worktree metadata for *plan*."""
    branch = plan.branch
    if branch is None and not config.new_topic_worktrees:
        return str(plan.project), {}
    eligibility = await asyncio.to_thread(check_worktree_eligibility, plan.project)
    if not eligibility.eligible or eligibility.repo_path is None:
        if branch is None:
            return str(plan.project), {}
        raise PlanError(
            f"{plan.project.name} cannot take a worktree "
            f"({eligibility.reason or 'not a git work tree'})"
        )
    repo = eligibility.repo_path
    if branch is None:
        # CCGRAM_NEW_TOPIC_WORKTREES=on: every session gets its own worktree.
        branch = await asyncio.to_thread(suggest_branch_name, plan.label, repo)
    worktree_path = worktree_path_for(repo, slug_for_path(branch))
    subdir = _subdir_within(plan.project, repo)
    native = tmux_manager.capabilities.native_worktrees
    if native and not subdir.parts:
        # The backend creates the checkout itself, but the launch gate still
        # requires all three fields: without the path the plan silently
        # launches in the source checkout and never creates the branch. The
        # worktree is also the cwd, so MRU and the window record point at the
        # checkout that was just used, not at the source clone.
        return str(worktree_path), {
            "worktree_repo": str(repo),
            "worktree_branch": branch,
            "worktree_path": str(worktree_path),
        }
    cwd = await _materialize_worktree(repo, branch, subdir)
    if native:
        # ccgram created the checkout itself to honor the subdirectory, so the
        # backend must not create it again: launch plainly in the target dir.
        return str(cwd), {}
    return str(cwd), {
        "worktree_repo": str(repo),
        "worktree_branch": branch,
        "worktree_path": str(worktree_path),
    }


async def launch_plan(
    plan: TopicPlan,
    *,
    user_id: int,
    thread_id: int | None,
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Start the session described by *plan* and bind it to this topic."""
    blocked = _agent_blocked(plan.agent)
    if blocked:
        await safe_reply(message, f"\u274c {blocked}")
        return
    note = await safe_reply(message, f"\u25b6 {plan.describe()}\u2026")
    if note is None:
        return
    try:
        cwd, worktree = await _resolve_cwd(plan)
    except PlanError as exc:
        await safe_edit(note, f"\u274c {exc}")
        return
    await launch_window(
        note,
        context,
        WindowLaunchRequest(
            user_id=user_id,
            thread_id=thread_id,
            provider_name=plan.agent,
            cwd=cwd,
            # Providers without a permissive flag reject mode=yolo outright,
            # so normalize here exactly as the browser flow does.
            mode=(
                "yolo"
                if plan.mode == "yolo" and has_yolo_mode(plan.agent)
                else "normal"
            ),
            pending_text=plan.instruction or None,
            chat_id=message.chat.id if message.chat else None,
            label=plan.label,
            worktree_repo=worktree.get("worktree_repo"),
            worktree_branch=worktree.get("worktree_branch"),
            worktree_path=worktree.get("worktree_path"),
            ignore_pending_worktree=True,
            ignore_pending_workspace=True,
        ),
    )


def _resolution_reply(parsed: PlanParse, candidates: list[Path]) -> str:
    """One short reply for a plan that cannot be launched as typed."""
    if parsed.reason == "mode":
        return (
            "\u274c Unknown mode. Use \u0060yolo\u0060 or \u0060safe\u0060:\n"
            "\u0060codex:safe <project> <instruction>\u0060"
        )
    if parsed.reason == "branch":
        return (
            "\u274c Missing branch name after @. Type: <project> @branch <instruction>"
        )
    if candidates:
        names = ", ".join(path.name for path in candidates)
        return f"\u274c Ambiguous project. Retype with the full name: {names}"
    return (
        "\u274c Project not found. Type: <agent> <project|path> [@branch] <instruction>\n"
        f"Looking under: {', '.join(str(root) for root in config.projects_roots) or 'nothing configured'}"
    )


async def start_from_text(
    *,
    user_id: int,
    thread_id: int | None,
    text: str,
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    """Try the typed-plan path for an unbound topic. True when handled.

    Stateless on purpose: nothing is stored between messages, so there is no
    pending draft for a later callback, a /start reset or another topic's
    flow to trip over.
    """
    last = _last_project(user_id)
    parsed = parse_topic_plan(
        text,
        roots=config.projects_roots,
        default_agent=config.provider_name,
        default_mode=config.default_mode,
        last_project=last,
        git_only=not config.dir_browser_all_dirs,
    )

    # Refused before anything is created, and before the project slot is even
    # considered: a disabled or uninstalled agent must never be replaced by the
    # default provider, and no worktree or window should appear for a command
    # that cannot run.
    if is_known_provider(parsed.agent):
        blocked = _agent_blocked(parsed.agent)
        if blocked:
            await safe_reply(message, f"\u274c {blocked}")
            return True

    if parsed.plan is not None:
        await launch_plan(
            parsed.plan,
            user_id=user_id,
            thread_id=thread_id,
            message=message,
            context=context,
        )
        return True

    if parsed.reason in ("mode", "branch"):
        await safe_reply(message, _resolution_reply(parsed, []))
        return True

    if parsed.reason == "project" and parsed.blocked:
        # A project slot was attempted (ambiguous, explicit, or followed by
        # @branch): say what is wrong instead of guessing a repository.
        await safe_reply(
            message,
            _resolution_reply(parsed, list(parsed.candidates)[:PROJECT_HINTS]),
        )
        return True

    mode_requested = parsed.mode and parsed.mode != config.default_mode
    if mode_requested or parsed.branch:
        # The directory browser applies neither, so say the line is incomplete
        # instead of dropping half of it.
        await safe_reply(message, _resolution_reply(parsed, []))
        return True

    # A plain instruction: let the caller open the directory browser.
    return False
