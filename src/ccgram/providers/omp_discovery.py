"""Oh My Pi command discovery for Telegram-exposed slash commands.

omp's layer over the shared discovery engine (``command_discovery``): it owns
the omp layout — agent dir, project config dir, hook roots — and the
Telegram-safe builtin list.

omp's native extension/skill/prompt roots are cwd-only (``<cwd>/.omp`` — the
native provider does not walk ancestors). ccgram deliberately walks ancestors
anyway, exactly as it does for pi, so a topic opened in a subdirectory still
sees the project's commands. omp reads two separate ``*.md`` families —
``prompts/`` (invoked as ``/prompts:<name>``) and ``commands/`` — and both are
scanned.
"""

from __future__ import annotations

from pathlib import Path

from ccgram.providers.base import DiscoveredCommand
from ccgram.providers.command_discovery import (
    DiscoveryRoots,
    ancestor_dirs,
    discover_commands,
)


def _omp_home() -> Path:
    """omp's agent dir. ``omp --profile <name>`` relocates this per profile."""
    return Path.home() / ".omp" / "agent"


def _agents_home() -> Path:
    return Path.home() / ".agents"


# Telegram-friendly Oh My Pi built-ins, descriptions from the command registry
# in the omp binary (name/description pairs).
# The forward path lets users send *any* /-command through to the provider,
# so this list is purely for discovery (/commands listing, menu autocomplete).
#
# NOTE: /resume is excluded because it collides with ccgram's bot-native
# session picker (handlers/recovery/resume_command.py). /new and /clear are
# both kept: they mean different things in omp — /new starts a fresh session,
# /clear only drops context and keeps the same session file.
_OMP_TELEGRAM_BUILTINS: dict[str, str] = {
    "/agents": "Open the agents hub (per-agent model, prewalk, and advisor)",
    "/append": "Append a task to the todo list",
    "/branch": "Rewind to a previous message, keeping the old path as a branch",
    "/changelog": "Show changelog entries",
    "/clear": "Clear the conversation context in place, keeping the session",
    "/compact": "Manually compact the session context",
    "/context": "Show estimated context usage breakdown",
    "/copy": "Pick text or code from the conversation to copy",
    "/delete": "Delete the current session and start a new one",
    "/dirs": "List this session's workspace directories",
    "/dump": "Copy session transcript to clipboard",
    "/export": "Export session to HTML file",
    "/extensions": "Open the Extension Control Center dashboard",
    # ccgram synthetic command: sends omp's Ctrl+Q follow-up shortcut.
    "/followup": "Queue a follow-up message after the current turn finishes",
    "/fork": "Create a new fork from a previous message",
    "/git": "Open the git UI (split diff viewer, staging, commit composer)",
    "/handoff": "Hand off session context to a new session",
    "/hotkeys": "Show all keyboard shortcuts",
    "/jobs": "Show async background jobs status",
    "/login": "Login with OAuth provider",
    "/logout": "Logout from OAuth provider",
    "/mcp": "Manage MCP servers (add, list, remove, test)",
    "/model": "Switch model for this session",
    "/new": "Start a new session",
    "/open": "Open the last link from the conversation in your browser",
    "/plan": "Toggle plan mode (agent plans before executing)",
    "/queue": "Queue a message for after the agent yields",
    "/quit": "Quit the application",
    "/restart": "Restart omp with the same launch flags, resuming this session",
    "/session": "Session management commands",
    "/settings": "Open settings menu",
    "/shake": "Drop heavy content from context (tool results, large blocks)",
    "/share": "Share session via an encrypted link",
    "/stats": "Launch the local stats dashboard",
    "/switch": "Switch model for this session (fuzzy ids, provider/id, @role, :level)",
    "/todo": "View or modify the agent's todo list",
    "/tools": "Show tools currently visible to the agent",
    "/tree": "Navigate session tree (switch branches)",
    "/usage": "Show provider usage and limits",
}


def _omp_roots(base_dir: str) -> DiscoveryRoots:
    """Build omp's discovery roots: agent dir, then project ancestors."""
    agent = _omp_home()
    agents_home = _agents_home()
    skills = [agent / "skills", agents_home / "skills"]
    prompts = [agent / "prompts"]
    commands = [agent / "commands"]
    extensions = [agent / "extensions"]
    hooks = [agent / "hooks" / "pre", agent / "hooks" / "post"]

    for parent in ancestor_dirs(base_dir):
        skills.append(parent / ".omp" / "skills")
        skills.append(parent / ".agents" / "skills")
        prompts.append(parent / ".omp" / "prompts")
        commands.append(parent / ".omp" / "commands")
        extensions.append(parent / ".omp" / "extensions")
        hooks.append(parent / ".omp" / "hooks" / "pre")
        hooks.append(parent / ".omp" / "hooks" / "post")

    return DiscoveryRoots(
        skills=tuple(skills),
        prompts=tuple(prompts),
        extensions=tuple(extensions),
        hooks=tuple(hooks),
        commands=tuple(commands),
    )


def discover_omp_commands(base_dir: str) -> list[DiscoveredCommand]:
    """Discover Telegram-suitable Oh My Pi commands from filesystem sources."""
    return discover_commands(
        roots=_omp_roots(base_dir),
        builtins=_OMP_TELEGRAM_BUILTINS,
    )
