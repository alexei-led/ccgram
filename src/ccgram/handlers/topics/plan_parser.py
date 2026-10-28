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


def parse_agent(token: str) -> tuple[str, str | None] | None:
    """Return (agent, mode_override) when *token* is an agent alias."""
    name, sep, mode = token.lower().partition(":")
    agent = AGENT_ALIASES.get(name)
    if agent is None:
        return None
    if not sep:
        return agent, None
    return agent, MODE_ALIASES.get(mode)


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
            projects.append(resolved)
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
        agent, mode_override = parsed_agent
        mode = mode_override or default_mode
        agent_typed = True
        consumed = 1

    project: Path | None = None
    project_typed = False
    candidates: tuple[Path, ...] = ()
    # A word that matches several projects is ambiguous, not missing: falling
    # back to the last project there would launch in the wrong repository and
    # turn a typed "@branch" into instruction text. Only a word that matches
    # nothing at all (an ordinary sentence) falls back.
    ambiguous = False
    if consumed < len(tokens) and not tokens[consumed][0].startswith("@"):
        project, candidates = resolve_project(
            tokens[consumed][0], roots, git_only=git_only
        )
        if project is not None:
            project_typed = True
            consumed += 1
        else:
            ambiguous = bool(candidates)
    if project is None and not ambiguous:
        project = _existing_dir(last_project)

    branch: str | None = None
    if consumed < len(tokens) and tokens[consumed][0].startswith("@"):
        branch = tokens[consumed][0][1:] or None
        if branch is None:
            return PlanParse(
                reason="branch",
                candidates=candidates,
                agent=agent,
                agent_typed=agent_typed,
                project_typed=project_typed,
            )
        consumed += 1

    if project is None:
        return PlanParse(
            reason="project",
            candidates=candidates
            or tuple(list_projects(roots, git_only=git_only)[:PROJECT_HINTS]),
            agent=agent,
            agent_typed=agent_typed,
            project_typed=project_typed,
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
