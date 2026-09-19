"""Pi command discovery for Telegram-exposed slash commands.

Pi's layer over the shared discovery engine (``command_discovery``): it owns
the pi layout — agent dir, project config dir, and which ancestors to walk —
and the Telegram-safe builtin list. All filesystem scanning lives in the
engine.
"""

from __future__ import annotations

from pathlib import Path

from ccgram.providers.base import DiscoveredCommand
from ccgram.providers.command_discovery import (
    DiscoveryRoots,
    ancestor_dirs,
    discover_commands,
)


def _pi_home() -> Path:
    return Path.home() / ".pi" / "agent"


def _agents_home() -> Path:
    return Path.home() / ".agents"


# Telegram-friendly Pi built-ins from https://pi.dev/docs/latest/usage.
# The forward path lets users send *any* /-command through to the provider,
# so this list is purely for discovery (/commands listing, menu autocomplete).
# Modal-TUI flows like /model and /settings are navigated with the inline toolbar.
#
# NOTE: /resume remains excluded because it collides with ccgram's bot-native
# session picker (handlers/recovery/resume_command.py). /clear is accepted as a
# hidden compatibility alias in forward.py, but Pi's documented reset command is /new.
_PI_TELEGRAM_BUILTINS: dict[str, str] = {
    "/changelog": "Show version history",
    "/clone": "Duplicate the active branch into a new session",
    "/compact": "Compact conversation context",
    "/copy": "Copy last assistant message to clipboard",
    "/export": "Export session to HTML",
    # ccgram synthetic command: sends Pi's Alt+Enter follow-up shortcut.
    "/followup": "Queue a follow-up message after current work finishes",
    "/fork": "Fork session from an earlier user message",
    "/hotkeys": "Show keyboard shortcuts",
    "/login": "Manage OAuth or API-key credentials",
    "/logout": "Manage OAuth or API-key credentials",
    "/model": "Switch models",
    "/name": "Set session display name",
    "/new": "Start a new Pi session",
    "/quit": "Quit Pi",
    "/reload": "Reload keybindings, extensions, skills, prompts, and context files",
    "/scoped-models": "Enable or disable models for Ctrl+P cycling",
    "/session": "Show session file, ID, messages, tokens, and cost",
    "/settings": "Open Pi settings",
    "/share": "Upload as private GitHub gist with shareable HTML link",
    "/tree": "Navigate the session tree",
}


def telegram_builtins() -> list[DiscoveredCommand]:
    """Return the Telegram-safe Pi built-in commands."""
    return [
        DiscoveredCommand(name=name, description=desc, source="builtin")
        for name, desc in _PI_TELEGRAM_BUILTINS.items()
    ]


def _pi_roots(base_dir: str) -> DiscoveryRoots:
    """Build pi's discovery roots: user agent dir, then project ancestors.

    pi has no separate custom-commands directory — its prompt templates are the
    only ``*.md`` command source — so ``commands`` stays empty.
    """
    pi_home = _pi_home()
    agents_home = _agents_home()
    skills = [pi_home / "skills", agents_home / "skills"]
    prompts = [pi_home / "prompts"]
    extensions = [pi_home / "extensions"]

    for parent in ancestor_dirs(base_dir):
        skills.append(parent / ".pi" / "skills")
        skills.append(parent / ".agents" / "skills")
        prompts.append(parent / ".pi" / "prompts")
        extensions.append(parent / ".pi" / "extensions")

    return DiscoveryRoots(
        skills=tuple(skills),
        prompts=tuple(prompts),
        extensions=tuple(extensions),
    )


def discover_pi_commands(base_dir: str) -> list[DiscoveredCommand]:
    """Discover Telegram-suitable Pi commands from filesystem sources."""
    return discover_commands(
        roots=_pi_roots(base_dir),
        builtins=_PI_TELEGRAM_BUILTINS,
    )
