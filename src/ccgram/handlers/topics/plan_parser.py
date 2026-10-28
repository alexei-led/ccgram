"""Parse a typed new-session line into a launch plan.

Grammar, all on the first line:

    <agent>[:<mode>] <project|path> [@branch] <instruction...>

Every slot except the instruction is optional and each omission falls back to
a configured default, so the common case is one typed message and no keyboard.
The parser is pure: it walks the filesystem with ``Path.is_dir`` only and
never touches Telegram, the multiplexer, or window state.

Matching is deliberately narrow:

* the first word is an agent only when it is a known alias (optionally
  followed by `:yolo` / `:safe`), so ordinary sentences are never
  swallowed;
* the project word must resolve under ``CCGRAM_PROJECTS_ROOTS`` (exact
  name, then unique prefix, then unique substring) or be an explicit path;
* the branch word is recognised only in ``@branch`` form.

Anything unresolved is reported as a reason plus candidate projects, never as
an exception.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Aliases the first word may use. Keep them short and unambiguous.
AGENT_ALIASES: dict[str, str] = {
    "c": "claude",
    "cc": "claude",
    "claude": "claude",
    "cx": "codex",
    "codex": "codex",
    "pi": "pi",
    "ag": "antigravity",
    "agy": "antigravity",
    "antigravity": "antigravity",
    "sh": "shell",
    "bash": "shell",
    "shell": "shell",
}

# ``<agent>:<mode>`` suffix. "safe" is the readable alias for "normal".
MODE_ALIASES: dict[str, str] = {
    "yolo": "yolo",
    "auto": "yolo",
    "safe": "normal",
    "normal": "normal",
    "std": "normal",
}

# How many alternative projects a failed resolution offers.
PROJECT_HINTS = 5

# Telegram topic titles are capped at 128 chars; keep ours short enough to
# stay readable in the topic list next to the state emoji badges.
MAX_LABEL_LEN = 48

_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class TopicPlan:
    """Everything one typed line needs to start a session."""

    agent: str
    project: Path
    instruction: str
    mode: str
    branch: str | None = None
    label: str = ""

    def describe(self) -> str:
        """One-line summary used in the start message and logs."""
        parts = [self.agent, self.project.name]
        if self.branch:
            parts.append("@" + self.branch)
        parts.append(self.mode)
        return " \u00b7 ".join(parts)


@dataclass(frozen=True, slots=True)
class PlanParse:
    """Outcome of :func:`parse_topic_plan`.

    ``plan`` is set on success. Otherwise ``reason`` names the slot that
    could not be filled ("empty", "project", "branch") and ``candidates``
    lists projects the caller can offer as one-tap alternatives.
    """

    plan: TopicPlan | None = None
    reason: str = ""
    candidates: tuple[Path, ...] = ()
    agent: str = ""
    agent_typed: bool = False
    project_typed: bool = False
    # Slots already resolved when only the project word is missing: the
    # candidate keyboard rebuilds the plan from these instead of re-parsing
    # (a re-parse would meet the same ambiguity again).
    instruction: str = ""
    mode: str = ""
    branch: str | None = None
    # True when the text named a project slot that could not be used
    # (ambiguous, explicit path, or followed by @branch). Then the caller
    # reports it; False means the word was just the first instruction word.
    blocked: bool = False


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """An agent word, with the mode suffix it carried (if any)."""

    name: str
    mode: str | None = None
    # The raw suffix when it is not a known mode. Reported instead of being
    # treated as "no suffix", which would silently launch the default (YOLO)
    # after a typo like ``claude:sfae``.
    unknown_mode: str | None = None


def parse_agent(token: str) -> AgentSpec | None:
    """Return the agent spec for *token*, or None when it is not an alias."""
    name, sep, mode = token.lower().partition(":")
    agent = AGENT_ALIASES.get(name)
    if agent is None:
        return None
    if not sep:
        return AgentSpec(agent)
    resolved = MODE_ALIASES.get(mode)
    if resolved is None:
        return AgentSpec(agent, unknown_mode=mode)
    return AgentSpec(agent, mode=resolved)


def is_git_repo(path: Path) -> bool:
    """True when *path* carries a git marker (directory or worktree file)."""
    try:
        return (path / ".git").exists()
    except OSError:
        return False


def list_projects(
    roots: list[Path] | tuple[Path, ...], *, git_only: bool = True
) -> list[Path]:
    """List candidate projects directly under *roots*, name-sorted, deduped.

    With ``git_only`` a child counts only when it holds a git marker, which
    keeps the candidate list to things an agent session can actually use.

    Entries keep the name they are listed under: a root may hold a symlink
    (``client`` → ``/repos/backend``) whose name is the word the operator
    types, and resolving it here would make that word unmatchable. Dedupe is
    by resolved path so two links to one checkout appear once.
    """
    seen: set[Path] = set()
    projects: list[Path] = []
    for root in roots:
        try:
            children = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for child in children:
            try:
                if not child.is_dir() or child.name.startswith("."):
                    continue
                resolved = child.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            if git_only and not is_git_repo(resolved):
                continue
            seen.add(resolved)
            projects.append(child)
    return projects


def _is_explicit_path(token: str) -> bool:
    return token.startswith("~") or token.startswith(os.sep) or os.sep in token


def resolve_project(
    token: str,
    roots: list[Path] | tuple[Path, ...],
    *,
    git_only: bool = True,
) -> tuple[Path | None, tuple[Path, ...]]:
    """Resolve a project word to a directory, or list the alternatives.

    An explicit path wins outright. A bare word matches exactly first, then
    by unique prefix, then by unique substring; ambiguity returns the
    matching projects so the caller can offer them as buttons.
    """
    if _is_explicit_path(token):
        path = Path(token).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            return None, ()
        return (resolved, ()) if resolved.is_dir() else (None, ())

    pool = list_projects(roots, git_only=git_only)
    word = token.lower()
    for match in (
        [p for p in pool if p.name.lower() == word],
        [p for p in pool if p.name.lower().startswith(word)],
        [p for p in pool if word in p.name.lower()],
    ):
        if len(match) == 1:
            return match[0], ()
        if match:
            return None, tuple(match[:PROJECT_HINTS])
    return None, ()


def build_label(
    project: Path, instruction: str, *, max_len: int = MAX_LABEL_LEN
) -> str:
    """Topic/window label: project name plus the first few instruction words."""
    hint = " ".join(instruction.split()[:4])
    label = f"{project.name} \u00b7 {hint}" if hint else project.name
    if len(label) <= max_len:
        return label
    return label[: max_len - 1].rstrip(" \u00b7") + "\u2026"


def _existing_dir(path: Path | None) -> Path | None:
    """Resolved *path* when it is a directory, else None."""
    if path is None:
        return None
    try:
        return path.resolve() if path.is_dir() else None
    except OSError:
        return None


def _take_project(
    tokens: list[tuple[str, int, int]],
    *,
    start: int,
    roots: list[Path] | tuple[Path, ...],
    git_only: bool,
    last_project: Path | None,
) -> tuple[Path | None, tuple[Path, ...], int, bool, bool]:
    """Consume the optional project word.

    Returns (project, candidates, next_consumed, blocked, project_typed).
    ``blocked`` means the word named something real but unusable — several
    matching projects, or an explicit path that does not exist. Falling back
    to the last project there would launch in the wrong repository, so it is
    reported instead. Only a word that matches nothing stays instruction.
    """
    project: Path | None = None
    project_typed = False
    candidates: tuple[Path, ...] = ()
    blocked = False
    consumed = start
    if consumed < len(tokens) and not tokens[consumed][0].startswith("@"):
        word = tokens[consumed][0]
        project, candidates = resolve_project(word, roots, git_only=git_only)
        if project is not None:
            project_typed = True
            consumed += 1
        else:
            # A word is a project attempt when it names something real but
            # unusable (several matches, or an explicit path), or when the
            # very next word is an @branch: <project> @branch is the only
            # grammar slot a branch can follow, so its presence proves the
            # intent. Anything else is the first instruction word.
            next_is_branch = consumed + 1 < len(tokens) and tokens[consumed + 1][
                0
            ].startswith("@")
            blocked = bool(candidates) or _is_explicit_path(word) or next_is_branch
    if project is None and not blocked:
        project = _existing_dir(last_project)
    return project, candidates, consumed, blocked, project_typed


def _take_branch(
    tokens: list[tuple[str, int, int]],
    *,
    slot: int,
    candidates: tuple[Path, ...],
    agent: str,
    agent_typed: bool,
    project_typed: bool,
    mode: str,
) -> tuple[str | None, int, int] | PlanParse:
    """Consume the optional ``@branch`` word at *slot*.

    Returns (branch, next_consumed, token_end) or a failure PlanParse. The
    branch slot follows the project slot whether that slot resolved or not,
    so a typed "@branch" never falls into the instruction.
    """
    if slot >= len(tokens) or not tokens[slot][0].startswith("@"):
        return None, -1, 0
    branch = tokens[slot][0][1:] or None
    if branch is None:
        return PlanParse(
            reason="branch",
            candidates=candidates,
            agent=agent,
            agent_typed=agent_typed,
            project_typed=project_typed,
            mode=mode,
        )
    return branch, slot + 1, tokens[slot][2]


def parse_topic_plan(
    text: str,
    *,
    roots: list[Path] | tuple[Path, ...] = (),
    default_agent: str,
    default_mode: str = "yolo",
    last_project: Path | None = None,
    git_only: bool = True,
) -> PlanParse:
    """Parse *text* into a :class:`TopicPlan`, or explain what is missing."""
    tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    if not tokens:
        return PlanParse(reason="empty")

    agent = default_agent
    mode = default_mode
    consumed = 0
    agent_typed = False
    parsed_agent = parse_agent(tokens[0][0])
    if parsed_agent is not None:
        agent = parsed_agent.name
        agent_typed = True
        consumed = 1
        if parsed_agent.unknown_mode is not None:
            # A typo in the requested approval mode must not select the
            # permissive default behind the operator's back.
            return PlanParse(
                reason="mode",
                agent=agent,
                agent_typed=True,
                mode=default_mode,
            )
        mode = parsed_agent.mode or default_mode

    project, candidates, consumed, blocked, project_typed = _take_project(
        tokens,
        start=consumed,
        roots=roots,
        git_only=git_only,
        last_project=last_project,
    )

    # Where the instruction begins if the project slot fails: after the
    # unresolved project word when there is one, else after the agent word.
    instruction_start = (
        tokens[consumed][2] if blocked else tokens[consumed - 1][2] if consumed else 0
    )
    # Only a blocked project word leaves a slot to skip: after a resolved one,
    # ``consumed`` already points past it, so the next word is the branch or
    # the start of the instruction — never another project.
    taken = _take_branch(
        tokens,
        slot=consumed + 1 if blocked else consumed,
        candidates=candidates,
        agent=agent,
        agent_typed=agent_typed,
        project_typed=project_typed,
        mode=mode,
    )
    if isinstance(taken, PlanParse):
        return taken
    branch, branch_consumed, branch_start = taken
    if branch_consumed >= 0:
        consumed = branch_consumed
        if blocked:
            # The branch is consumed even though the project was not.
            instruction_start = branch_start

    if project is None:
        # Offer the whole candidate list only when no project word was tried:
        # with a blocked slot the candidates are exactly what was matched, and
        # an empty list means "nothing by that name", not "pick anything".
        offered = candidates
        if not blocked:
            offered = tuple(list_projects(roots, git_only=git_only)[:PROJECT_HINTS])
        return PlanParse(
            reason="project",
            candidates=offered,
            agent=agent,
            agent_typed=agent_typed,
            project_typed=project_typed,
            instruction=text[instruction_start:].strip(),
            mode=mode,
            branch=branch,
            blocked=blocked,
        )

    instruction = text[tokens[consumed - 1][2] :].strip() if consumed else text.strip()
    plan = TopicPlan(
        agent=agent,
        project=project,
        instruction=instruction,
        mode=mode,
        branch=branch,
        label=build_label(project, instruction),
    )
    return PlanParse(
        plan=plan,
        agent=agent,
        agent_typed=agent_typed,
        project_typed=project_typed,
    )
