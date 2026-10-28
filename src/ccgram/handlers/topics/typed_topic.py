"""Typed new-session entry: one message starts a session.

Replaces the window-picker → directory-browser → worktree → workspace →
provider → mode chain with a single parsed line::

    <agent>[:<mode>] <project|path> [@branch] <instruction...>

Only two situations still ask anything: an unresolvable project word (one
message with candidate buttons) and an ambiguous one. Everything else is a
configuration default. The parsed plan is a value object, so there is no
multi-step keyboard state to go stale.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update

from ...config import config
from ...multiplexer import multiplexer as tmux_manager
from ...providers import has_yolo_mode, is_known_provider
from ...user_preferences import user_preferences
from ...worktrees import (
    WorktreeError,
    check_worktree_eligibility,
    create_worktree,
    slug_for_path,
    suggest_branch_name,
    worktree_path_for,
)
from ..callback_data import CB_PLAN_CANCEL, CB_PLAN_HOME, CB_PLAN_SELECT
from ..callback_helpers import get_thread_id
from ..callback_registry import register
from ..messaging_pipeline.message_sender import interactive_edit as safe_edit
from ..messaging_pipeline.message_sender import safe_reply
from ..user_state import PENDING_PLAN
from .plan_parser import (
    PROJECT_HINTS,
    PlanParse,
    TopicPlan,
    build_label,
    parse_topic_plan,
)
from .window_launch_service import WindowLaunchRequest, launch_window

if TYPE_CHECKING:
    from telegram import Message
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

__all__ = ["start_from_text", "launch_plan", "plan_pending"]

# Candidate buttons per keyboard row.
_PLAN_ROW_WIDTH = 2


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


def _plan_prompt(parsed: PlanParse, candidates: list[Path]) -> str:
    lines = ["*Pick a project*", ""]
    if parsed.reason == "branch":
        lines = ["*Empty branch after @*", "", "Type: project @branch instruction", ""]
    elif candidates:
        lines.append("The project word did not resolve. Candidates:")
        lines.extend(f"\u2022 {path.name}" for path in candidates)
    else:
        lines.append("No project word and no last-used project yet.")
        lines.append("Type: <agent> <project> [@branch] <instruction>")
    return "\n".join(lines)


def _plan_keyboard(candidates: list[Path], last: Path | None) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for idx, path in enumerate(candidates):
        row.append(
            InlineKeyboardButton(path.name, callback_data=f"{CB_PLAN_SELECT}{idx}")
        )
        if len(row) == _PLAN_ROW_WIDTH:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    if last is not None and last not in candidates:
        rows.append(
            [InlineKeyboardButton(f"\u23ee {last.name}", callback_data=CB_PLAN_HOME)]
        )
    rows.append([InlineKeyboardButton("Cancel", callback_data=CB_PLAN_CANCEL)])
    return InlineKeyboardMarkup(rows)


def _store_pending(
    user_data: dict | None,
    parsed: PlanParse,
    candidates: list[Path],
    thread_id: int | None,
    *,
    chat_id: int | None,
    message_id: int | None,
) -> None:
    if user_data is None:
        return
    # Identity is (chat, thread, keyboard message): one draft per user is not
    # enough, because a second unresolved plan in the same topic would answer
    # the first keyboard with the second plan's candidate index. message_id
    # is the bot's prompt message that carries the keyboard — the one the
    # callback arrives on — never the user's own message.
    user_data[PENDING_PLAN] = {
        "agent": parsed.agent,
        "mode": parsed.mode or config.default_mode,
        "instruction": parsed.instruction,
        "branch": parsed.branch,
        "thread_id": thread_id,
        "chat_id": chat_id,
        "message_id": message_id,
        "candidates": [str(path) for path in candidates],
    }


def plan_pending(user_data: dict | None) -> dict | None:
    """The pending plan draft for this user, if one is waiting on a project."""
    if not user_data:
        return None
    draft = user_data.get(PENDING_PLAN)
    return draft if isinstance(draft, dict) else None


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
        ),
    )


async def start_from_text(
    *,
    user_id: int,
    thread_id: int | None,
    text: str,
    user_data: dict | None,
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    """Try the typed-plan path for an unbound topic. True when handled."""
    last = _last_project(user_id)
    parsed = parse_topic_plan(
        text,
        roots=config.projects_roots,
        default_agent=config.provider_name,
        default_mode=config.default_mode,
        last_project=last,
        git_only=not config.dir_browser_all_dirs,
    )

    first_word = text.split(maxsplit=1)[0].lower() if text.split() else ""
    if not parsed.agent_typed and first_word:
        name = first_word.split(":", 1)[0]
        if is_known_provider(name):
            blocked = _agent_blocked(name)
            if blocked:
                await safe_reply(message, f"\u274c {blocked}")
                return True

    if parsed.plan is not None:
        if user_data is not None:
            user_data.pop(PENDING_PLAN, None)
        await launch_plan(
            parsed.plan,
            user_id=user_id,
            thread_id=thread_id,
            message=message,
            context=context,
        )
        return True

    if parsed.reason == "mode":
        await safe_reply(
            message,
            "\u274c Unknown mode. Use `yolo` or `safe`:\n"
            "`codex:safe <project> <instruction>`",
        )
        return True
    if parsed.reason not in ("project", "branch"):
        return False

    candidates = list(parsed.candidates)[:PROJECT_HINTS]
    if parsed.reason == "project" and not candidates:
        if parsed.mode and parsed.mode != config.default_mode or parsed.branch:
            # The line asked for a mode or a worktree; the directory browser
            # cannot honour either, so say so instead of dropping them.
            await safe_reply(
                message,
                "\u274c Project not found, and a mode/branch was requested.\n"
                "Type: <agent>[:<mode>] <project|path> [@branch] <instruction>",
            )
            return True
        # A plain instruction: let the caller open the directory browser.
        return False
    prompt = await safe_reply(
        message,
        _plan_prompt(parsed, candidates),
        reply_markup=_plan_keyboard(candidates, last),
    )
    # The keyboard lives on the bot's prompt message, so that message is the
    # draft identity: a callback always carries its id, never the user's.
    _store_pending(
        user_data,
        parsed,
        candidates,
        thread_id,
        chat_id=message.chat.id if message.chat else None,
        message_id=prompt.message_id if prompt is not None else None,
    )
    return True


def _draft_plan(draft: dict, project: Path) -> TopicPlan | None:
    """Rebuild the plan for *project* from the slots the parse already resolved.

    Re-parsing the original text would meet the same ambiguity again and could
    never produce a plan, so the chosen project is taken as given here.
    """
    agent = str(draft.get("agent") or "")
    if not agent:
        return None
    instruction = str(draft.get("instruction") or "")
    branch = draft.get("branch")
    return TopicPlan(
        agent=agent,
        project=project,
        instruction=instruction,
        mode=str(draft.get("mode") or config.default_mode),
        branch=str(branch) if branch else None,
        label=build_label(project, instruction),
    )


async def _draft_is_current(query, draft: dict, thread_id: int | None) -> bool:
    """True when this keyboard still owns the pending draft.

    The draft carries the chat, thread and prompt message it was created for.
    A second unresolved plan in the same topic replaces the draft, and its
    keyboard answers any older one — a tap there must not launch the newer
    plan with an index taken from the older candidate list.
    """
    if draft.get("thread_id") not in (None, thread_id):
        await query.answer("Stale plan (topic mismatch)", show_alert=True)
        return False
    prompt_message = query.message
    if prompt_message is None:
        return True
    expected_message_id = draft.get("message_id")
    if (
        expected_message_id is not None
        and prompt_message.message_id != expected_message_id
    ):
        await query.answer("Stale plan: send the line again", show_alert=True)
        return False
    expected_chat_id = draft.get("chat_id")
    prompt_chat = prompt_message.chat
    if (
        expected_chat_id is not None
        and prompt_chat is not None
        and prompt_chat.id != expected_chat_id
    ):
        await query.answer("Stale plan (chat mismatch)", show_alert=True)
        return False
    return True


async def _choice_from_callback(
    query, data: str, draft: dict, user_id: int
) -> Path | None:
    """Resolve a plan:* tap to a project, answering the query on failure."""
    if data == CB_PLAN_HOME:
        choice = _last_project(user_id)
        if choice is None:
            await query.answer("No last project", show_alert=True)
        return choice
    candidates = [Path(entry) for entry in draft.get("candidates") or []]
    try:
        idx = int(data[len(CB_PLAN_SELECT) :])
    except ValueError:
        await query.answer("Invalid choice", show_alert=True)
        return None
    if idx < 0 or idx >= len(candidates):
        await query.answer("Project list changed", show_alert=True)
        return None
    return candidates[idx]


async def _handle_plan_callback(
    query,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Dispatch plan:* taps: candidate project, last project, cancel."""
    draft = plan_pending(context.user_data)
    thread_id = get_thread_id(update)
    if draft is None:
        await query.answer("Start over: type the session line again", show_alert=True)
        return
    if not await _draft_is_current(query, draft, thread_id):
        return
    if data == CB_PLAN_CANCEL:
        if context.user_data is not None:
            context.user_data.pop(PENDING_PLAN, None)
        await query.answer("Cancelled")
        await safe_edit(query, "Cancelled")
        return
    choice = await _choice_from_callback(query, data, draft, user_id)
    if choice is None:
        return
    plan = _draft_plan(draft, choice)
    if context.user_data is not None:
        context.user_data.pop(PENDING_PLAN, None)
    await query.answer()
    if plan is None:
        await safe_edit(query, "\u274c Could not rebuild the plan. Type it again.")
        return
    message = query.message
    if message is None:
        return
    await launch_plan(
        plan,
        user_id=user_id,
        thread_id=thread_id,
        message=message,
        context=context,
    )


@register(CB_PLAN_SELECT, CB_PLAN_HOME, CB_PLAN_CANCEL)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    await _handle_plan_callback(query, user.id, query.data, update, context)
