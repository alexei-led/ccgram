"""Provider abstractions for multi-agent CLI backends.

Re-exports the protocol, event types, capability dataclass, and registry
so consumers can do ``from ccgram.providers import registry, ...``.
Also provides ``get_provider()`` for accessing the active provider singleton,
and ``resolve_capabilities()`` for lightweight CLI commands that don't
require Config (doctor, status).
"""

import re

import structlog
import os

from ccgram.providers.base import (
    AgentMessage,
    AgentProvider,
    DiscoveredCommand,
    ProviderCapabilities,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.providers.process_detection import JS_RUNTIMES
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


def has_yolo_mode(provider_name: str) -> bool:
    """Return True if the provider supports YOLO (permissive) launch mode."""
    return provider_name in _YOLO_FLAGS


# Singleton cache
_active: AgentProvider | None = None


_registered = False


def _ensure_registered() -> None:
    """Register all known providers into the global registry (once)."""
    global _registered
    if _registered:
        return
    # Lazy: provider classes register against the registry at import; defer until the registry factory runs
    from ccgram.providers.claude import ClaudeProvider

    # Lazy: provider classes register against the registry at import; defer until the registry factory runs
    from ccgram.providers.codex import CodexProvider

    # Lazy: provider classes register against the registry at import; defer until the registry factory runs
    from ccgram.providers.gemini import GeminiProvider

    # Lazy: provider classes register against the registry at import; defer until the registry factory runs
    from ccgram.providers.pi import PiProvider

    # Lazy: provider classes register against the registry at import; defer until the registry factory runs
    from ccgram.providers.shell import ShellProvider

    registry.register("claude", ClaudeProvider)
    registry.register("codex", CodexProvider)
    registry.register("gemini", GeminiProvider)
    registry.register("pi", PiProvider)
    registry.register("shell", ShellProvider)
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

        # Lazy: config singleton is wired late at startup; importing at top
        # would freeze test overrides that monkeypatch config attrs.
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


def get_provider_for_window(
    window_id: str,  # noqa: ARG001
    provider_name: str | None = None,
) -> AgentProvider:
    """Return the provider for a specific window, falling back to config default.

    Callers must supply *provider_name* (e.g. from ``window_query.get_window_provider``
    or ``view.provider_name``). When it is None or unknown, falls back to the
    config default provider.
    """
    _ensure_registered()

    if provider_name and registry.is_valid(provider_name):
        return registry.get(provider_name)
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
    for name in ("claude", "codex", "gemini", "pi"):
        if basename == name or basename.startswith(name + "-"):
            return name

    # Lazy: providers.shell pulls in shell_infra (prompt-marker machinery
    # + readline lookups) at import; only load when we have to fall
    # through to shell-process detection.
    # Lazy: shell provider is the only one that needs KNOWN_SHELLS
    from .shell import KNOWN_SHELLS

    if basename in KNOWN_SHELLS or basename.lstrip("-") in KNOWN_SHELLS:
        return "shell"

    return ""


_CLAUDE_PROJECTS_RE = re.compile(r"/\.claude[a-z0-9._-]*/projects/")


def detect_provider_from_transcript_path(transcript_path: str) -> str:
    """Infer provider name from a persisted transcript path when possible.

    Claude wrappers (`ce`, `cc-mirror`, `zai`, `claude-team`, etc.) use sibling
    config dirs of the form ``~/.claude<suffix>/projects/`` with the same JSONL
    schema, so any ``.claude*`` config dir is treated as Claude.
    """
    if not isinstance(transcript_path, str):
        return ""
    normalized = transcript_path.strip().lower().replace("\\", "/")
    if not normalized:
        return ""
    if "/.codex/sessions/" in normalized:
        return "codex"
    if _CLAUDE_PROJECTS_RE.search(normalized):
        return "claude"
    if "/.gemini/" in normalized and "/chats/" in normalized:
        return "gemini"
    if "/.pi/agent/sessions/" in normalized:
        return "pi"
    return ""


def should_probe_pane_title_for_provider_detection(pane_current_command: str) -> bool:
    """Return True when any provider needs pane-title context to detect runtime."""
    _ensure_registered()
    for name in registry.provider_names():
        if registry.get(name).requires_pane_title_for_detection(pane_current_command):
            return True
    return False


_CCGRAM_TITLE_PREFIX = "ccgram:"


def detect_provider_from_runtime(
    pane_current_command: str,
    *,
    pane_title: str = "",
) -> str:
    """Detect provider from process name and optional pane-title hints."""
    detected = detect_provider_from_command(pane_current_command)
    if detected or not pane_title:
        return detected

    # Check for ccgram title stamp (set on launch via stamp_pane_title)
    if pane_title.startswith(_CCGRAM_TITLE_PREFIX):
        stamped = pane_title[len(_CCGRAM_TITLE_PREFIX) :].strip()
        _ensure_registered()
        if registry.is_valid(stamped):
            return stamped

    _ensure_registered()
    for name in registry.provider_names():
        provider = registry.get(name)
        if provider.detect_from_pane_title(pane_current_command, pane_title):
            return provider.capabilities.name
    return ""


async def detect_provider_from_pane(
    pane_current_command: str,
    *,
    window_id: str = "",
) -> str:
    """Detect provider using fast path + foreground-process inspection.

    1. Fast path: basename match via ``detect_provider_from_command()``
    2. When the command alone is inconclusive — a JS runtime (node/bun/npx)
       hiding the real CLI, *or* empty (herdr reports no
       ``pane_current_command`` for a bare shell pane; tmux always reports
       one) — and a ``window_id`` is available, fall back to the multiplexer's
       foreground process (``Multiplexer.foreground``) with PGID cache.  The
       backend owns how the foreground process is read (tmux ``ps -t``; herdr
       ``process-info``).
    """
    detected = detect_provider_from_command(pane_current_command)
    if detected:
        return detected

    if not window_id:
        return ""

    cmd = pane_current_command.strip().lower()
    basename = os.path.basename(cmd.split()[0]) if cmd else ""
    # A non-empty, non-runtime command already failed the fast path and won't
    # be clarified by the foreground process, so stop here. Empty commands fall
    # through (herdr bare shell) so the foreground argv can classify them.
    if cmd and basename not in JS_RUNTIMES:
        return ""

    # Lazy: process_detection pulls the providers internals; only worth
    # loading on the foreground fallback.
    from .process_detection import detect_provider_cached

    # Lazy: multiplexer proxy resolves the active backend; foreground()
    # reads the pane's foreground process via the seam.
    from ..multiplexer import multiplexer

    fg = await multiplexer.foreground(window_id)
    return await detect_provider_cached(window_id, fg)


def resolve_launch_command(
    provider_name: str, *, approval_mode: str = _APPROVAL_MODE_NORMAL
) -> str:
    """Resolve launch command for a provider, with optional approval mode.

    Resolution: ``CCGRAM_<NAME>_COMMAND`` (e.g. ``CCGRAM_CLAUDE_COMMAND``) if set,
    otherwise the provider's hardcoded default (``capabilities.launch_command``).
    When ``approval_mode`` is ``"yolo"``, appends the provider-specific
    permissive-mode flag unless it is already present.
    """
    _ensure_registered()
    provider = provider_name.lower()
    env_key = f"CCGRAM_{provider.upper()}_COMMAND"
    override = os.environ.get(env_key)
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
        # Lazy: only the gemini launch path needs the hardener; importing at
        # top would pull gemini provider code on every provider resolution.
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

    Reads ``CCGRAM_PROVIDER`` from env when *provider_name* is not given.
    Falls back to ``"claude"`` for unknown providers.
    Suitable for lightweight CLI commands (doctor, status) that
    must not import Config (which requires TELEGRAM_BOT_TOKEN).
    """
    _ensure_registered()
    name = (
        provider_name
        if provider_name is not None
        else os.environ.get("CCGRAM_PROVIDER", "claude")
    )
    try:
        return registry.get(name).capabilities
    except UnknownProviderError:
        return registry.get("claude").capabilities


__all__ = [
    "AgentMessage",
    "AgentProvider",
    "DiscoveredCommand",
    "ProviderCapabilities",
    "ProviderRegistry",
    "SessionStartEvent",
    "StatusUpdate",
    "UnknownProviderError",
    "detect_provider_from_command",
    "detect_provider_from_pane",
    "detect_provider_from_transcript_path",
    "detect_provider_from_runtime",
    "get_provider",
    "get_provider_for_window",
    "has_yolo_mode",
    "registry",
    "resolve_capabilities",
    "resolve_launch_command",
    "should_probe_pane_title_for_provider_detection",
]
