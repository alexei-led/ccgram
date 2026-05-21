"""Pi command discovery for Telegram-exposed slash commands.

Discovers Telegram-suitable Pi commands from the built-in command set plus
on-disk skills, prompt templates, and extension commands.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ccgram.command_catalog import parse_frontmatter
from ccgram.providers.base import DiscoveredCommand


def _pi_home() -> Path:
    return Path.home() / ".pi" / "agent"


def _agents_home() -> Path:
    return Path.home() / ".agents"


_PI_EXTENSION_COMMAND_RE = re.compile(
    r"pi\.registerCommand\(\s*[\"'](?P<name>[^\"']+)[\"']",
)

# Telegram-friendly Pi built-ins. The forward path lets users send *any*
# /-command through to the provider, so this list is purely for discovery
# (/commands listing, menu autocomplete). Modal-TUI flows like /model,
# /login open pickers the user navigates with the inline toolbar.
#
# NOTE: /new and /resume excluded — they collide with bot-native handlers
# (handlers/topics/new_command.py, handlers/recovery/resume_command.py)
# and will never be forwarded to Pi. Same convention as codex.py.
_PI_TELEGRAM_BUILTINS: dict[str, str] = {
    "/changelog": "Show version history",
    "/clear": "Clear conversation history",
    "/clone": "Clone session from an earlier message",
    "/colors": "Show terminal color palette",
    "/compact": "Compact conversation context",
    "/copy": "Copy last response to clipboard",
    "/debug": "Show debug information",
    "/export": "Export session to HTML",
    "/fork": "Fork conversation from an earlier message",
    "/hotkeys": "Show keyboard shortcuts",
    "/import": "Import a previous session",
    "/login": "Add or update provider credentials",
    "/logout": "Remove provider credentials",
    "/model": "Pick the active model",
    "/name": "Set session display name",
    "/quit": "Exit Pi (Ctrl-C twice)",
    "/reload": "Reload extensions, skills, prompts, themes",
    "/session": "Show session info",
    "/settings": "Open settings TUI",
    "/share": "Upload as private GitHub gist",
    "/tree": "Show session tree",
}


def telegram_builtins() -> list[DiscoveredCommand]:
    """Return the Telegram-safe Pi built-in commands."""
    return [
        DiscoveredCommand(name=name, description=desc, source="builtin")
        for name, desc in _PI_TELEGRAM_BUILTINS.items()
    ]


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


def _discover_skill_roots(base_dir: Path) -> list[Path]:
    roots = [_pi_home() / "skills", _agents_home() / "skills"]
    current = base_dir.resolve()
    for parent in (current, *current.parents):
        roots.extend([parent / ".pi" / "skills", parent / ".agents" / "skills"])
        if (parent / ".git").is_dir():
            break
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        ordered.append(root)
    return ordered


def _scan_skill_root(root: Path) -> list[DiscoveredCommand]:
    """Scan a single skill root directory for commands."""
    if not root.is_dir():
        return []
    allow_root_markdown = root.name == "skills" and root.parent.name in {
        ".pi",
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


def _discover_skills(base_dir: str) -> list[DiscoveredCommand]:
    discovered: list[DiscoveredCommand] = []
    for root in _discover_skill_roots(Path(base_dir)):
        discovered.extend(_scan_skill_root(root))
    return discovered


def _discover_prompt_templates(base_dir: str) -> list[DiscoveredCommand]:
    discovered: list[DiscoveredCommand] = []
    roots = [_pi_home() / "prompts"]
    current = Path(base_dir).resolve()
    for parent in (current, *current.parents):
        roots.append(parent / ".pi" / "prompts")
        if (parent / ".git").is_dir():
            break

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
    for match in _PI_EXTENSION_COMMAND_RE.finditer(text):
        name = match.group("name").strip()
        if name:
            found.append(
                DiscoveredCommand(name=name, description=f"/{name}", source="command")
            )
    return found


def _discover_extension_commands(base_dir: str) -> list[DiscoveredCommand]:
    discovered: list[DiscoveredCommand] = []
    roots = [_pi_home() / "extensions"]
    current = Path(base_dir).resolve()
    for parent in (current, *current.parents):
        roots.append(parent / ".pi" / "extensions")
        if (parent / ".git").is_dir():
            break

    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(root)
        try:
            entries = sorted(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            for path in _extension_candidates(entry):
                discovered.extend(_extract_commands_from_file(path))
    return discovered


def discover_pi_commands(base_dir: str) -> list[DiscoveredCommand]:
    """Discover Telegram-suitable Pi commands from filesystem sources."""
    commands = []
    commands.extend(telegram_builtins())
    commands.extend(_discover_skills(base_dir))
    commands.extend(_discover_prompt_templates(base_dir))
    commands.extend(_discover_extension_commands(base_dir))

    deduped: list[DiscoveredCommand] = []
    seen: set[str] = set()
    for cmd in commands:
        if not cmd.name or cmd.name in seen:
            continue
        deduped.append(cmd)
        seen.add(cmd.name)
    return deduped
