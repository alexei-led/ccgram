"""Foreground process detection for tmux panes via ``ps -t``.

All supported CLIs (claude, codex, gemini) are Node.js scripts — tmux's
``pane_current_command`` shows ``bun`` or ``node`` instead of the CLI name.
This module inspects the actual foreground process group on the pane's TTY
to reliably identify which provider is running.

Detection flow:
1. Run ``ps -t <tty> -o pid=,pgid=,stat=,args=``
2. Filter for ``+`` in stat → foreground process group
3. Find the group leader (pid == pgid) among foreground processes
4. Match the leader's full args against provider patterns, skipping wrapper
   tokens (sudo, env, node, bun, …)
5. Cache by ``(window_id, fg_pgid)`` to avoid repeated subprocess calls

Exposed entry points:
- ``detect_provider_from_tty`` — uncached, single-shot detection
- ``detect_provider_cached`` — cached by foreground PGID per window
- ``clear_detection_cache`` — invalidate cache entries on cleanup
"""

from __future__ import annotations

import asyncio
import os

import structlog

from .shell import KNOWN_SHELLS as _KNOWN_SHELLS

logger = structlog.get_logger()

# Tokens that wrap the actual CLI binary — skip during classification.
_WRAPPER_TOKENS = frozenset(
    {"sudo", "env", "node", "bun", "npx", "bunx", "uv", "python", "python3"}
)

# Basename → provider name.  Checked via exact match and prefix (``claude-*``).
_PROVIDER_BASENAMES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"claude", "ce", "cc-mirror", "zai"}), "claude"),
    (frozenset({"codex"}), "codex"),
    (frozenset({"gemini"}), "gemini"),
)

# Path substrings that identify a provider when basename alone is ambiguous
# (e.g. ``cli.js`` launched by bun).
_PROVIDER_PATH_MARKERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("claude-code", "cc-team"), "claude"),
    (("@openai/codex", "/codex/", "/codex-"), "codex"),
    (("gemini-cli",), "gemini"),
)


# JS runtimes that trigger ps-based detection in the caller.
JS_RUNTIMES = frozenset({"node", "bun", "npx", "bunx"})


def _match_token(token: str) -> str:
    """Match a single argv token against provider basenames and path markers."""
    basename = os.path.basename(token).lower().lstrip("-")

    # Direct basename match
    for names, provider in _PROVIDER_BASENAMES:
        if basename in names or basename.startswith(f"{provider}-"):
            return provider
    if basename in _KNOWN_SHELLS:
        return "shell"

    # Path-based match for ambiguous basenames (e.g. cli.js)
    token_lower = token.lower()
    for markers, provider in _PROVIDER_PATH_MARKERS:
        if any(m in token_lower for m in markers):
            return provider

    return ""


def classify_provider_from_args(args: str) -> str:
    """Classify provider from a process's full argv string.

    Skips wrapper tokens (``node``, ``bun``, ``sudo``, …) and matches the
    first meaningful token against known provider names or path markers.
    Returns provider name (``"claude"``, ``"codex"``, ``"gemini"``,
    ``"shell"``) or empty string if unrecognised.
    """
    if not args:
        return ""

    for token in args.split():
        cleaned = os.path.basename(token).lower().lstrip("-")
        if cleaned in _WRAPPER_TOKENS:
            continue
        return _match_token(token)

    return ""


async def get_foreground_args(tty_path: str) -> tuple[str, int]:
    """Get the full argv and PGID of the foreground process on a TTY.

    Runs ``ps -t <tty> -o pid=,pgid=,stat=,args=`` and filters for
    processes with ``+`` in their stat field (foreground group).  Among
    those, returns the group leader's (pid == pgid) argv string.

    Returns:
        ``(args_string, fg_pgid)`` on success, ``("", 0)`` on any error.
    """
    if not tty_path:
        return "", 0

    try:
        proc = await asyncio.create_subprocess_exec(
            "ps",
            "-t",
            tty_path,
            "-o",
            "pid=,pgid=,stat=,args=",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(3.0):
            stdout, _ = await proc.communicate()
    except TimeoutError, OSError:
        return "", 0

    if proc.returncode != 0 or not stdout:
        return "", 0

    best_args = ""
    best_pgid = 0
    for line in stdout.decode("utf-8", errors="replace").strip().splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:  # noqa: PLR2004
            continue
        pid_s, pgid_s, stat, args = parts
        if "+" not in stat:
            continue
        try:
            pid = int(pid_s)
            pgid = int(pgid_s)
        except ValueError:
            continue
        # Prefer the group leader (pid == pgid)
        if pid == pgid:
            return args, pgid
        # Track any foreground process as fallback
        if not best_args:
            best_args = args
            best_pgid = pgid

    return best_args, best_pgid


async def detect_provider_from_tty(tty_path: str) -> str:
    """Detect provider from the foreground process on a TTY (uncached)."""
    args, _ = await get_foreground_args(tty_path)
    return classify_provider_from_args(args)


# ---------------------------------------------------------------------------
# PGID-based cache
# ---------------------------------------------------------------------------

# window_id → (foreground_pgid, detected_provider_name)
_pgid_cache: dict[str, tuple[int, str]] = {}


async def detect_provider_cached(window_id: str, tty_path: str) -> str:
    """Detect provider with PGID-based caching.

    Only calls ``ps`` when the foreground PGID changes (or on first call).
    """
    args, pgid = await get_foreground_args(tty_path)
    if not args or pgid == 0:
        return ""

    cached = _pgid_cache.get(window_id)
    if cached and cached[0] == pgid:
        return cached[1]

    provider = classify_provider_from_args(args)
    if provider:
        _pgid_cache[window_id] = (pgid, provider)
    return provider


def clear_detection_cache(window_id: str | None = None) -> None:
    """Clear cached detection for a window, or all windows if None."""
    if window_id is None:
        _pgid_cache.clear()
    else:
        _pgid_cache.pop(window_id, None)
