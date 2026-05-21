"""Discover Claude Code commands for Telegram bot menu registration.

Scans three sources to build the command list:
  1. Built-in CC commands (always present)
  2. User-invocable skills from ~/.claude/skills/
  3. Custom commands from ~/.claude/commands/

Core components:
  - CCCommand dataclass: name, telegram_name, description, source
  - discover_cc_commands(): filesystem scanner with caching
  - register_commands(): sets Telegram bot menu (BotCommand list)
  - get_cc_name(): reverse lookup from sanitized telegram name to CC name
"""

import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, cast

from ccgram.command_catalog import (
    command_catalog,
    discover_user_defined_commands,
    parse_frontmatter as _parse_frontmatter,
)
from ccgram.providers.base import AgentProvider
from telegram import Bot, BotCommand, BotCommandScope

logger = structlog.get_logger()

# Built-in Claude Code commands. Refreshed from code.claude.com/docs/en/commands.
# Telegram menu is a discovery surface only — any /<token> still forwards
# to the provider (see commands/forward.py). /new is bot-native (excluded);
# /resume collides with bot-native /resume (excluded — ccgram has its own picker).
CC_BUILTINS: dict[str, str] = {
    "agents": "↗ Manage subagent configurations",
    "background": "↗ Detach session as background agent",
    "branch": "↗ Fork conversation at current point",
    "clear": "↗ Clear conversation context",
    "compact": "↗ Summarize history to free context",
    "context": "↗ Show context window usage",
    "copy": "↗ Copy last response to clipboard",
    "cost": "↗ Show session cost (alias /usage)",
    "diff": "↗ Open interactive diff viewer",
    "doctor": "↗ Diagnose Claude Code installation",
    "effort": "↗ Adjust thinking effort level",
    "export": "↗ Export conversation as text",
    "feedback": "↗ Submit feedback or bug report",
    "help": "↗ Show Claude Code help",
    "init": "↗ Initialize CLAUDE.md in project",
    "loop": "↗ Run prompt repeatedly on a schedule",
    "mcp": "↗ Manage MCP server connections",
    "memory": "↗ Edit CLAUDE.md memory files",
    "model": "↗ Select model and effort",
    "permissions": "↗ Manage tool permissions",
    "plan": "↗ Enter plan mode",
    "rc": "↗ Start remote control (alias)",
    "recap": "↗ Summarize current session",
    "release-notes": "↗ View changelog",
    "remote-control": "↗ Start remote control session",
    "review": "↗ Review pull request locally",
    "rewind": "↗ Rewind conversation or code",
    "security-review": "↗ Analyze pending changes for security",
    "settings": "↗ Open Settings interface",
    "skills": "↗ List available skills",
    "statusline": "↗ Configure status line",
    "status": "↗ Show session status",
    "tasks": "↗ Manage background tasks",
    "theme": "↗ Change color theme (picker)",
    "tui": "↗ Switch terminal UI renderer",
    "usage": "↗ Show usage and cost stats",
    "verify": "↗ Build and run app to verify change",
}

# Bot-native commands (registered first, not from CC)
_BOT_COMMANDS: list[tuple[str, str]] = [
    ("new", "Create new Claude session"),
    ("commands", "List commands for this topic provider"),
    ("history", "Message history for this topic"),
    ("sessions", "Sessions dashboard"),
    ("resume", "Browse and resume past sessions"),
    ("screenshot", "Capture terminal screenshot"),
    ("live", "Open auto-refreshing terminal view"),
    ("panes", "List panes in this window"),
    ("restore", "Recover a dead topic"),
    ("sync", "Audit and fix state"),
    ("unbind", "Unbind this topic"),
    ("recall", "Recall recent commands"),
    ("toolbar", "Show action toolbar"),
    ("verbose", "Toggle tool call batching"),
    ("upgrade", "Upgrade ccgram and restart"),
]

# Telegram limits: max 100 commands, descriptions max 256 chars
_MAX_TELEGRAM_COMMANDS = 100
_MAX_DESCRIPTION_LEN = 256


@dataclass(frozen=True, slots=True)
class CCCommand:
    """A discovered Claude Code command."""

    name: str  # Original CC name (e.g. "spec:work", "committing-code")
    telegram_name: str  # Sanitized for Telegram (e.g. "spec_work")
    description: str
    source: Literal["builtin", "skill", "command"]


def _sanitize_telegram_name(name: str) -> str:
    """Sanitize a CC command name for Telegram.

    Telegram allows only [a-z0-9_] in command names, max 32 chars.
    Returns empty string for unrepresentable names.
    """
    sanitized = name.lower().replace("-", "_").replace(":", "_")
    # Strip anything not alphanumeric or underscore
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    return sanitized[:32]


def _cc_desc(desc: str) -> str:
    """Ensure description has ↗ prefix for CC-forwarded commands."""
    return desc if desc.startswith("↗") else f"↗ {desc}"


def parse_frontmatter(path: Path) -> dict[str, str]:
    """Compatibility wrapper around provider-agnostic frontmatter parser."""
    return _parse_frontmatter(path)


def discover_cc_commands(claude_dir: Path | None = None) -> list[CCCommand]:
    """Scan filesystem for CC commands.

    Sources (in order):
      1. Built-in commands (CC_BUILTINS)
      2. Skills: {claude_dir}/skills/*/SKILL.md (user-invocable only)
      3. Custom commands: {claude_dir}/commands/{group}/*.md

    Commands with empty sanitized names are skipped.
    """
    if claude_dir is None:
        # Lazy: config singleton is wired late at startup; importing at top
        # would freeze test overrides that monkeypatch config attrs.
        from ccgram.config import config

        claude_dir = config.claude_config_dir

    commands: list[CCCommand] = []

    # 1. Builtins
    for name, desc in CC_BUILTINS.items():
        commands.append(
            CCCommand(
                name=name,
                telegram_name=_sanitize_telegram_name(name),
                description=desc,
                source="builtin",
            )
        )

    # 2. User-defined skills + commands
    for cmd in discover_user_defined_commands(claude_dir):
        tg_name = _sanitize_telegram_name(cmd.name)
        if not tg_name:
            continue
        commands.append(
            CCCommand(
                name=cmd.name,
                telegram_name=tg_name,
                description=_cc_desc(cmd.description),
                source=cast(Literal["builtin", "skill", "command"], cmd.source),
            )
        )

    return commands


# Module-level cache (telegram_name → cc_name, first-wins to match registration)
_name_map: dict[str, str] = {}


def _provider_base_dir(claude_dir: Path | None = None) -> str:
    """Resolve base dir for provider command discovery."""
    # Lazy: config singleton accessed only when this helper actually runs
    from ccgram.config import config as _cfg

    return str(claude_dir) if claude_dir else str(_cfg.claude_config_dir)


def discover_provider_commands(
    provider: AgentProvider,
    claude_dir: Path | None = None,
) -> list[CCCommand]:
    """Discover commands for one provider as CCCommand entries."""
    base_dir = _provider_base_dir(claude_dir)
    valid_sources = {"builtin", "skill", "command"}
    discovered = command_catalog.get_provider_commands(provider, base_dir)
    commands: list[CCCommand] = []
    for cmd in discovered:
        if not cmd.name:
            continue
        commands.append(
            CCCommand(
                name=cmd.name,
                telegram_name=_sanitize_telegram_name(cmd.name),
                description=_cc_desc(cmd.description),
                source=cast(
                    Literal["builtin", "skill", "command"],
                    cmd.source if cmd.source in valid_sources else "command",
                ),
            )
        )
    return commands


def get_provider_command_map(
    provider: AgentProvider,
    claude_dir: Path | None = None,
) -> dict[str, str]:
    """Build telegram_name -> original command mapping for a provider."""
    mapping: dict[str, str] = {}
    for cmd in discover_provider_commands(provider, claude_dir):
        if cmd.telegram_name and cmd.telegram_name not in mapping:
            mapping[cmd.telegram_name] = cmd.name
    return mapping


def get_provider_supported_commands(
    provider: AgentProvider,
    claude_dir: Path | None = None,
) -> set[str]:
    """Return normalized slash commands supported by a provider."""
    supported: set[str] = set()
    for name in get_provider_command_map(provider, claude_dir).values():
        token = name if name.startswith("/") else f"/{name}"
        supported.add(token.lower())
    for name in provider.capabilities.builtin_commands:
        if not name:
            continue
        token = name if name.startswith("/") else f"/{name}"
        supported.add(token.lower())
    return supported


def _refresh_cache(
    claude_dir: Path | None = None,
    provider: AgentProvider | None = None,
    providers: Iterable[AgentProvider] | None = None,
) -> list[CCCommand]:
    """Re-discover commands and update the cache.

    When *providers* is given, merges commands from each provider in order.
    When *provider* is given, uses ``provider.discover_commands()`` which
    returns ``list[DiscoveredCommand]``.
    Falls back to filesystem scanning via ``discover_cc_commands()`` otherwise.
    """
    global _name_map

    if providers is not None:
        commands = []
        for discovered_provider in providers:
            commands.extend(discover_provider_commands(discovered_provider, claude_dir))
    elif provider is not None:
        commands = discover_provider_commands(provider, claude_dir)
    else:
        commands = discover_cc_commands(claude_dir)
    # First-wins: matches the dedup order in register_commands
    new_map: dict[str, str] = {}
    for cmd in commands:
        if cmd.telegram_name not in new_map:
            new_map[cmd.telegram_name] = cmd.name
    _name_map = new_map
    return commands


def get_cc_name(telegram_name: str) -> str | None:
    """Look up the original CC command name from a sanitized Telegram name."""
    return _name_map.get(telegram_name)


async def register_commands(
    bot: Bot,
    claude_dir: Path | None = None,
    provider: AgentProvider | None = None,
    providers: Iterable[AgentProvider] | None = None,
    include_cc_commands: bool = True,
    scope: BotCommandScope | None = None,
) -> None:
    """Discover CC commands and register them in the Telegram bot menu.

    When *providers* is given, commands are merged from each provider in order.
    When *provider* is given, command discovery is delegated to that provider.
    Registers bot-native commands first (new, history, etc.), then up to
    the remaining Telegram limit of discovered CC commands. Deduplicates
    by telegram_name (first-wins) and excludes collisions with bot-native names.
    """
    commands = (
        _refresh_cache(claude_dir, provider=provider, providers=providers)
        if include_cc_commands
        else []
    )

    bot_commands = [BotCommand(name, desc) for name, desc in _BOT_COMMANDS]
    max_cc = _MAX_TELEGRAM_COMMANDS - len(bot_commands)

    # Pre-populate with bot-native names to avoid collisions
    seen_names: set[str] = {name for name, _ in _BOT_COMMANDS}
    cc_count = 0
    for cmd in commands:
        if cc_count >= max_cc:
            break
        # Skip empty names, duplicates, and bot-native collisions
        if not cmd.telegram_name or cmd.telegram_name in seen_names:
            continue
        seen_names.add(cmd.telegram_name)
        desc = cmd.description[:_MAX_DESCRIPTION_LEN]
        bot_commands.append(BotCommand(cmd.telegram_name, desc))
        cc_count += 1

    # Lazy: only needed when scheduling the menu-refresh task
    import asyncio

    # Lazy: only used inside the rate-limit retry path
    from telegram.error import RetryAfter

    max_startup_wait = 120  # Don't block startup longer than this

    try:
        if scope is None:
            await bot.delete_my_commands()
            await bot.set_my_commands(bot_commands)
        else:
            await bot.delete_my_commands(scope=scope)
            await bot.set_my_commands(bot_commands, scope=scope)
    except RetryAfter as e:
        retry_secs = (
            e.retry_after
            if isinstance(e.retry_after, int)
            else int(e.retry_after.total_seconds())
        )
        if retry_secs > max_startup_wait:
            logger.warning(
                "Telegram flood control: %ds cooldown, skipping command registration"
                " (will retry in 10min refresh cycle)",
                retry_secs,
            )
            return
        logger.warning(
            "Telegram flood control: retrying command registration in %ds",
            retry_secs,
        )
        await asyncio.sleep(retry_secs)
        if scope is None:
            await bot.delete_my_commands()
            await bot.set_my_commands(bot_commands)
        else:
            await bot.delete_my_commands(scope=scope)
            await bot.set_my_commands(bot_commands, scope=scope)
    logger.info("Registered %d bot commands (%d CC)", len(bot_commands), cc_count)
