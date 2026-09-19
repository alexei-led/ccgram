"""Generic on-disk command discovery shared by the pi and omp providers.

The engine here owns the filesystem machinery only: reading skills, prompt
templates, extension modules, and hook factories, and merging what it finds
with a provider's Telegram-safe builtins (builtins win, then skills, prompts,
extension commands, hook commands; de-duped by name, first source wins).

Providers own their layout — home/agent dir name, project config dir name, and
which ancestors to walk — and hand the concrete roots in as ``DiscoveryRoots``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ccgram.command_catalog import parse_frontmatter
from ccgram.providers.base import DiscoveredCommand


@dataclass(frozen=True, slots=True)
class DiscoveryRoots:
    """Filesystem roots a provider wants scanned, in priority order."""

    skills: tuple[Path, ...]
    prompts: tuple[Path, ...]
    extensions: tuple[Path, ...]
    hooks: tuple[Path, ...] = ()
    commands: tuple[Path, ...] = ()


def ancestor_dirs(base_dir: str) -> list[Path]:
    """Return *base_dir* and its ancestors, stopping at the first ``.git``.

    The walk itself is shared so pi and omp cannot drift on which ancestors a
    project's config is consulted in; providers own the config dir name.
    """
    current = Path(base_dir).resolve()
    dirs: list[Path] = []
    for parent in (current, *current.parents):
        dirs.append(parent)
        if (parent / ".git").is_dir():
            break
    return dirs


# Extension factories receive the same legacy ``pi`` API object in both agents
# — omp's ``export default function (pi) { pi.registerCommand(...) }`` is the
# pi contract verbatim, so one regex covers both.
_EXTENSION_COMMAND_RE = re.compile(
    r"pi\.registerCommand\(\s*[\"'](?P<name>[^\"']+)[\"']",
)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return ""


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _command_description(path: Path, *, fallback: str) -> str:
    frontmatter = parse_frontmatter(path)
    description = frontmatter.get("description", "")
    if description:
        hint = frontmatter.get("argument-hint", "")
        return f"{hint} — {description}" if hint else description
    if frontmatter:
        return fallback
    body = _first_nonempty_line(_safe_read_text(path))
    return body or fallback


def _skill_command(path: Path) -> DiscoveredCommand | None:
    if path.name == "SKILL.md":
        skill_dir = path.parent
        name = parse_frontmatter(path).get("name", skill_dir.name)
        description = _command_description(path, fallback=f"/{name}")
        return DiscoveredCommand(name=name, description=description, source="skill")

    if path.suffix.lower() != ".md" or path.name.startswith("."):
        return None

    name = parse_frontmatter(path).get("name", path.stem)
    description = _command_description(path, fallback=f"/{name}")
    return DiscoveredCommand(name=name, description=description, source="skill")


def _scan_skill_root(root: Path) -> list[DiscoveredCommand]:
    """Scan a single skill root directory for commands."""
    if not root.is_dir():
        return []
    # A loose .md directly inside a ``skills/`` root is a skill, not a stray
    # doc: user roots live under ``.pi``/``.omp``/agent dirs.
    allow_root_markdown = root.name == "skills" and root.parent.name in {
        ".pi",
        ".omp",
        "agent",
    }
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    found: list[DiscoveredCommand] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_file() and allow_root_markdown and entry.suffix.lower() == ".md":
            cmd = _skill_command(entry)
            if cmd:
                found.append(cmd)
            continue
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if skill_md.is_file():
            cmd = _skill_command(skill_md)
            if cmd:
                found.append(cmd)
    return found


def _discover_prompt_templates(roots: tuple[Path, ...]) -> list[DiscoveredCommand]:
    """Scan prompt-template and custom-command roots for ``*.md`` templates.

    Both families are markdown files with optional frontmatter resolved the
    same way; omp reads them from ``prompts/`` and ``commands/`` respectively.
    """
    discovered: list[DiscoveredCommand] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        try:
            files = sorted(root.glob("*.md"))
        except OSError:
            continue
        for path in files:
            if path.name.startswith("."):
                continue
            name = path.stem
            description = _command_description(path, fallback=f"/{name}")
            discovered.append(
                DiscoveredCommand(name=name, description=description, source="command")
            )
    return discovered


_EXTENSION_SKIP_DIRS = frozenset({"node_modules", "dist", "build", ".git"})


_EXTENSION_SUFFIXES = frozenset({".ts", ".js", ".mjs", ".cjs"})


def _extension_candidates(entry: Path) -> list[Path]:
    """Collect script files from a single extension root entry.

    ``os.walk`` is used instead of ``rglob`` so skip dirs are pruned before
    descent — avoids walking ``node_modules``, ``dist``, etc. on large trees.
    """
    if entry.is_file() and entry.suffix.lower() in _EXTENSION_SUFFIXES:
        return [entry]
    if not entry.is_dir():
        return []
    candidates: list[Path] = []
    try:
        for dirpath, dirnames, filenames in os.walk(entry):
            dirnames[:] = [d for d in dirnames if d not in _EXTENSION_SKIP_DIRS]
            dir_path = Path(dirpath)
            for name in filenames:
                if name.startswith("."):
                    continue
                if Path(name).suffix.lower() in _EXTENSION_SUFFIXES:
                    candidates.append(dir_path / name)
    except OSError:
        return []
    return candidates


def _extract_commands_from_file(path: Path) -> list[DiscoveredCommand]:
    """Extract registered command names from a single extension file."""
    text = _safe_read_text(path)
    if not text:
        return []
    found: list[DiscoveredCommand] = []
    for match in _EXTENSION_COMMAND_RE.finditer(text):
        name = match.group("name").strip()
        if name:
            found.append(
                DiscoveredCommand(name=name, description=f"/{name}", source="command")
            )
    return found


def _discover_script_commands(roots: tuple[Path, ...]) -> list[DiscoveredCommand]:
    """Scan extension/hook roots for ``registerCommand`` calls."""
    discovered: list[DiscoveredCommand] = []
    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(root)
        for path in _extension_candidates(root):
            discovered.extend(_extract_commands_from_file(path))
    return discovered


def discover_commands(
    *,
    roots: DiscoveryRoots,
    builtins: Mapping[str, str],
) -> list[DiscoveredCommand]:
    """Merge a provider's builtins with everything found under *roots*."""
    commands: list[DiscoveredCommand] = [
        DiscoveredCommand(name=name, description=desc, source="builtin")
        for name, desc in builtins.items()
    ]
    for root in roots.skills:
        commands.extend(_scan_skill_root(root))
    commands.extend(_discover_prompt_templates(roots.prompts))
    commands.extend(_discover_prompt_templates(roots.commands))
    commands.extend(_discover_script_commands(roots.extensions))
    commands.extend(_discover_script_commands(roots.hooks))

    deduped: list[DiscoveredCommand] = []
    seen: set[str] = set()
    for cmd in commands:
        if not cmd.name or cmd.name in seen:
            continue
        deduped.append(cmd)
        seen.add(cmd.name)
    return deduped
