"""Provider abstractions for multi-agent CLI backends.

Re-exports the protocol, event types, capability dataclass, and registry
so consumers can do ``from ccgram.providers import registry, ...``.
Also provides ``get_provider()`` for accessing the active provider singleton,
and ``resolve_capabilities()`` for lightweight CLI commands that don't
require Config (doctor, status).
"""

import structlog
import os

from ccgram.providers.base import (
    EXPANDABLE_QUOTE_END,
    EXPANDABLE_QUOTE_START,
    AgentMessage,
    AgentProvider,
    DiscoveredCommand,
    ProviderCapabilities,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.providers.registry import ProviderRegistry, UnknownProviderError, registry

logger = structlog.get_logger()

# Launch-mode constants for per-session approval behavior.
_APPROVAL_MODE_NORMAL = "normal"
_APPROVAL_MODE_YOLO = "yolo"
_YOLO_FLAGS: dict[str, str] = {
    "claude": "--dangerously-skip-permissions",
    "codex": "--dangerously-bypass-approvals-and-sandbox",
    "gemini": "--yolo",
}

# Singleton cache
_active: AgentProvider | None = None


_registered = False


def _ensure_registered() -> None:
    """Register all known providers into the global registry (once)."""
    global _registered
    if _registered:
        return
    from ccgram.providers.claude import ClaudeProvider
    from ccgram.providers.codex import CodexProvider
    from ccgram.providers.gemini import GeminiProvider

    registry.register("claude", ClaudeProvider)
    registry.register("codex", CodexProvider)
    registry.register("gemini", GeminiProvider)
    _registered = True


def get_provider() -> AgentProvider:
    """Return the active provider instance (lazy singleton).

    On first call, registers all providers into the global registry and
    resolves the provider name from config. Falls back to ``"claude"`` if
    the configured provider is unknown.
    """
    global _active
    if _active is None:
        _ensure_registered()

        from ccgram.config import config

        try:
            _active = registry.get(config.provider_name)
        except UnknownProviderError:
            logger.warning(
                "Unknown provider %r, falling back to 'claude'",
                config.provider_name,
            )
            _active = registry.get("claude")
    return _active


def _reset_provider() -> None:
    """Reset the cached provider singleton (for tests only)."""
    global _active, _registered
    _active = None
    _registered = False


def get_provider_for_window(window_id: str) -> AgentProvider:
    """Return the provider for a specific window, falling back to config default.

    Looks up provider_name from the window's WindowState. If empty or invalid,
    falls back to the global config provider (get_provider()).
    """
    _ensure_registered()

    from ccgram.session import session_manager

    state = session_manager.window_states.get(window_id)
    if state and state.provider_name and registry.is_valid(state.provider_name):
        return registry.get(state.provider_name)
    return get_provider()


def detect_provider_from_command(pane_current_command: str) -> str:
    """Detect provider name from a tmux pane's running process.

    Matches the basename of the command against known provider names
    to avoid false positives from paths containing provider names.
    Returns empty string for unrecognized or empty commands so callers
    can distinguish "no match" from a confident detection.
    """
    cmd = pane_current_command.strip().lower()
    if not cmd:
        return ""

    # Match basename only (first token) to avoid false positives
    # from paths like /home/claude/bin/vim
    basename = os.path.basename(cmd.split()[0])
    for name in ("claude", "codex", "gemini"):
        if basename == name or basename.startswith(name + "-"):
            return name

    return ""


def should_probe_pane_title_for_provider_detection(pane_current_command: str) -> bool:
    """Return True when any provider needs pane-title context to detect runtime."""
    _ensure_registered()
    for name in registry.provider_names():
        if registry.get(name).requires_pane_title_for_detection(pane_current_command):
            return True
    return False


def detect_provider_from_runtime(
    pane_current_command: str,
    *,
    pane_title: str = "",
) -> str:
    """Detect provider from process name and optional pane-title hints."""
    detected = detect_provider_from_command(pane_current_command)
    if detected or not pane_title:
        return detected

    _ensure_registered()
    for name in registry.provider_names():
        provider = registry.get(name)
        if provider.detect_from_pane_title(pane_current_command, pane_title):
            return provider.capabilities.name
    return ""


def resolve_launch_command(
    provider_name: str, *, approval_mode: str = _APPROVAL_MODE_NORMAL
) -> str:
    """Resolve launch command for a provider, with optional approval mode.

    Resolution: ``CCGRAM_<NAME>_COMMAND`` (e.g. ``CCGRAM_CLAUDE_COMMAND``) if set,
    otherwise the provider's hardcoded default (``capabilities.launch_command``).
    Falls back to legacy ``CCBOT_<NAME>_COMMAND`` env var.
    When ``approval_mode`` is ``"yolo"``, appends the provider-specific
    permissive-mode flag unless it is already present.
    """
    _ensure_registered()
    provider = provider_name.lower()
    new_env = f"CCGRAM_{provider.upper()}_COMMAND"
    old_env = f"CCBOT_{provider.upper()}_COMMAND"
    override = os.environ.get(new_env)
    if not override:
        override = os.environ.get(old_env)
        if override:
            logger.warning("%s is deprecated, use %s instead", old_env, new_env)
    if override:
        command = override
    else:
        try:
            command = registry.get(provider).capabilities.launch_command
        except UnknownProviderError:
            provider = "claude"
            command = registry.get("claude").capabilities.launch_command

    # CCGRAM_GEMINI_COMMAND overrides stay fully user-controlled.
    # For ccgram-managed Gemini launches, force stable shell mode defaults.
    if provider == "gemini" and not override:
        from ccgram.providers.gemini import build_hardened_gemini_launch_command

        command = build_hardened_gemini_launch_command(command)

    if approval_mode.lower() != _APPROVAL_MODE_YOLO:
        return command

    yolo_flag = _YOLO_FLAGS.get(provider)
    if not yolo_flag or yolo_flag in command:
        return command
    return f"{command} {yolo_flag}"


def resolve_capabilities(provider_name: str | None = None) -> ProviderCapabilities:
    """Resolve provider capabilities without requiring full Config.

    Reads ``CCGRAM_PROVIDER`` (or legacy ``CCBOT_PROVIDER``) from env when
    *provider_name* is not given.  Falls back to ``"claude"`` for unknown
    providers.  Suitable for lightweight CLI commands (doctor, status) that
    must not import Config (which requires TELEGRAM_BOT_TOKEN).
    """
    _ensure_registered()
    name = (
        provider_name
        if provider_name is not None
        else (
            os.environ.get("CCGRAM_PROVIDER")
            or os.environ.get("CCBOT_PROVIDER", "claude")
        )
    )
    try:
        return registry.get(name).capabilities
    except UnknownProviderError:
        return registry.get("claude").capabilities


__all__ = [
    "EXPANDABLE_QUOTE_END",
    "EXPANDABLE_QUOTE_START",
    "AgentMessage",
    "AgentProvider",
    "DiscoveredCommand",
    "ProviderCapabilities",
    "ProviderRegistry",
    "SessionStartEvent",
    "StatusUpdate",
    "UnknownProviderError",
    "detect_provider_from_command",
    "detect_provider_from_runtime",
    "get_provider",
    "get_provider_for_window",
    "registry",
    "resolve_capabilities",
    "resolve_launch_command",
    "should_probe_pane_title_for_provider_detection",
]
