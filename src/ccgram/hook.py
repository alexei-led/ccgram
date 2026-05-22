"""Hook subcommand for Claude Code session and event tracking.

Called by Claude Code hooks (SessionStart, Notification, Stop, SubagentStart,
SubagentStop, TeammateIdle, TaskCompleted) to maintain a window↔session
mapping and an append-only event log.  Also provides `--install` to
auto-configure hooks in settings.json (respects CLAUDE_CONFIG_DIR).

This module must NOT import config.py (which requires TELEGRAM_BOT_TOKEN),
since hooks run inside tmux panes where bot env vars are not set.
Config directory resolution uses utils.ccgram_dir() (shared with config.py).
Claude settings path resolution uses CLAUDE_CONFIG_DIR env var (shared with config.py).

Key functions: hook_main() (CLI entry), _install_hook().
"""

import fcntl
import json
import logging
import os
import re
import shlex
import subprocess
import structlog
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

from ccgram.hooks.adapters import (
    detect_provider_from_payload,
    get_hook_adapter,
)
from ccgram.hooks.model import ProviderName

logger = structlog.get_logger()

# Validate session_id looks like a UUID


def _claude_settings_file() -> Path:
    """Resolve Claude settings.json path, respecting CLAUDE_CONFIG_DIR."""
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / "settings.json"
    return Path.home() / ".claude" / "settings.json"


# Current hook command uses the active Python interpreter to avoid PATH issues.
_CURRENT_HOOK_MARKER = "ccgram.main hook"
# Older installs used the console script name directly.
_PATH_HOOK_MARKER = "ccgram hook"
# Legacy marker from pre-rename ccbot — used for detection and cleanup.
_LEGACY_HOOK_MARKER = "ccbot hook"

# Expected number of parts when parsing tmux display-message output.
# Minimum is 3 (session_name\t@id\twindow_name); a fourth pane_tty field is
# optional so older test mocks keep working with a 3-part stdout.
_TMUX_FORMAT_PARTS = 3
_TMUX_FORMAT_PARTS_WITH_TTY = 4

# ps -A output is split into 5 fields: pid, ppid, pgid, stat, command.
_PS_SNAPSHOT_FIELDS = 5

# Hook event types ccgram handles (order matters for status display)
_HOOK_EVENT_TYPES: tuple[str, ...] = (
    "SessionStart",
    "Notification",
    "Stop",
    "StopFailure",
    "SessionEnd",
    "SubagentStart",
    "SubagentStop",
    "TeammateIdle",
    "TaskCompleted",
)

# Events that should not block the agent (async: true)
_ASYNC_EVENTS: frozenset[str] = frozenset(
    {
        "StopFailure",
        "SessionEnd",
        "SubagentStart",
        "SubagentStop",
        "TeammateIdle",
        "TaskCompleted",
    }
)


def _installable_events_for(provider_name: str) -> tuple[str, ...]:
    """Pull installable_events from an adapter, asserting it exists."""
    adapter = get_hook_adapter(provider_name)
    if adapter is None:
        raise AssertionError(f"no hook adapter registered for {provider_name!r}")
    return adapter.installable_events


# Source of truth: each adapter declares its installable_events. We re-export
# under the legacy names so existing call sites in doctor_cmd keep working
# without a churny import migration.
_CODEX_HOOK_EVENTS: tuple[str, ...] = _installable_events_for("codex")
_GEMINI_HOOK_EVENTS: tuple[str, ...] = _installable_events_for("gemini")


def _codex_hooks_file() -> Path:
    """Return the user-level Codex hooks.json path."""
    return Path.home() / ".codex" / "hooks.json"


def _codex_config_file() -> Path:
    """Return the user-level Codex config.toml path."""
    return Path.home() / ".codex" / "config.toml"


def _gemini_settings_file() -> Path:
    """Return the user-level Gemini settings.json path."""
    return Path.home() / ".gemini" / "settings.json"


def _current_hook_command(provider_name: str = "claude") -> str:
    """Build the hook command bound to the current Python interpreter."""
    command = f"{shlex.quote(sys.executable)} -m ccgram.main hook"
    if provider_name != "claude":
        command += f" --provider {shlex.quote(provider_name)}"
    return command


def _is_current_hook_command(command: str) -> bool:
    """Return True when the command matches the current module-based hook style."""
    return _CURRENT_HOOK_MARKER in command


def _is_any_ccgram_hook_command(command: str) -> bool:
    """Return True for current, old, or legacy hook command styles."""
    return any(
        marker in command
        for marker in (_CURRENT_HOOK_MARKER, _PATH_HOOK_MARKER, _LEGACY_HOOK_MARKER)
    )


def _has_matching_hook(
    settings: dict, event_type: str, predicate: Callable[[str], bool]
) -> bool:
    """Check if an event has a hook command matching the predicate."""
    hooks = settings.get("hooks", {})
    event_hooks = hooks.get(event_type, [])

    for entry in event_hooks:
        if not isinstance(entry, dict):
            continue
        inner_hooks = entry.get("hooks", [])
        for h in inner_hooks:
            if not isinstance(h, dict):
                continue
            cmd = h.get("command", "")
            if predicate(cmd):
                return True
    return False


def _has_ccgram_hook(settings: dict, event_type: str) -> bool:
    """Check if ccgram hook (or legacy ccbot hook) is installed."""
    return _has_matching_hook(settings, event_type, _is_any_ccgram_hook_command)


def get_installed_events(settings: dict) -> dict[str, bool]:
    """Return installation status for each expected hook event type."""
    return {event: _has_ccgram_hook(settings, event) for event in _HOOK_EVENT_TYPES}


def _replace_hook_commands(
    settings: dict, event_type: str, predicate: Callable[[str], bool], replacement: str
) -> None:
    """Replace matching hook commands for an event with the given command."""
    event_hooks = settings.get("hooks", {}).get(event_type, [])
    for entry in event_hooks:
        if not isinstance(entry, dict):
            continue
        for h in entry.get("hooks", []):
            if not isinstance(h, dict):
                continue
            cmd = h.get("command", "")
            if predicate(cmd):
                h["command"] = replacement


def _load_json_settings(path: Path) -> dict[str, Any] | None:
    """Load a JSON settings file, returning an empty dict when absent."""
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return None
    if not isinstance(parsed, dict):
        print(f"Error reading {path}: expected JSON object", file=sys.stderr)
        return None
    return parsed


def _json_hook_command_predicate(provider_name: str) -> Callable[[str], bool]:
    """Build predicate for provider-specific ccgram hook commands.

    Matches `--provider {name}` as a whole token so e.g. `--provider codex-dev`
    does not also match `--provider codex`. We append a trailing space to the
    command so a token at the very end of the string also matches the
    space-delimited needle.
    """

    needle = f" --provider {provider_name} "

    def _predicate(command: str) -> bool:
        return _is_any_ccgram_hook_command(command) and needle in f" {command} "

    return _predicate


def _hook_entry(provider_name: str, timeout_value: int) -> dict[str, Any]:
    """Build a command hook entry for non-Claude providers.

    ``timeout_value`` is provider-defined: Codex hooks.json uses seconds,
    Gemini settings.json uses milliseconds. Callers must pass the unit the
    target schema expects.
    """
    return {
        "name": "ccgram-session-tracker",
        "type": "command",
        "command": _current_hook_command(provider_name),
        "timeout": timeout_value,
    }


def _install_json_hooks(
    path: Path, provider_name: str, events: tuple[str, ...], timeout_value: int
) -> int:
    """Install ccgram command hooks into a JSON settings file.

    ``timeout_value`` is provider-defined (seconds for Codex, ms for Gemini).
    """
    settings = _load_json_settings(path)
    if settings is None:
        return 1
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        print(f"Error reading {path}: hooks must be an object", file=sys.stderr)
        return 1

    installed_count = 0
    already_count = 0
    predicate = _json_hook_command_predicate(provider_name)
    for event_type in events:
        event_hooks = hooks.setdefault(event_type, [])
        if not isinstance(event_hooks, list):
            print(
                f"Error reading {path}: hooks.{event_type} must be an array",
                file=sys.stderr,
            )
            return 1
        if _has_matching_hook(settings, event_type, predicate):
            already_count += 1
            continue
        event_hooks.append({"hooks": [_hook_entry(provider_name, timeout_value)]})
        installed_count += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Lazy: utils brings in subprocess + structlog at import time; not worth
        # paying that cost on `ccgram --help`, only at hook-install.
        from .utils import atomic_write_json

        atomic_write_json(path, settings)
    except OSError as e:
        print(f"Error writing {path}: {e}", file=sys.stderr)
        return 1
    print(
        f"{provider_name} hooks installed in {path}: "
        f"{installed_count} new, {already_count} already present"
    )
    return 0


_CODEX_HOOKS_KEY_RE = re.compile(r"^\s*codex_hooks\s*=\s*(\S+)", re.MULTILINE)


def _ensure_codex_feature_flag() -> int:
    """Ensure user Codex config enables the hooks feature flag.

    Detects any existing ``codex_hooks =`` line (spacing-tolerant). If it's
    already truthy, no-op. If it's explicitly false, warn and refuse to
    overwrite — the user opted out. Otherwise insert under ``[features]``.
    """
    config_file = _codex_config_file()
    if not config_file.exists():
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text("[features]\ncodex_hooks = true\n")
        except OSError as e:
            print(f"Error creating {config_file}: {e}", file=sys.stderr)
            return 1
        return 0
    try:
        text = config_file.read_text()
    except OSError as e:
        print(f"Error reading {config_file}: {e}", file=sys.stderr)
        return 1
    match = _CODEX_HOOKS_KEY_RE.search(text)
    if match:
        value = match.group(1).rstrip(",")
        if value == "true":
            return 0
        print(
            f"{config_file} has codex_hooks = {value}; set it to true and rerun.",
            file=sys.stderr,
        )
        return 1
    if "[features]" in text:
        text = text.replace("[features]", "[features]\ncodex_hooks = true", 1)
    else:
        text = text.rstrip() + "\n\n[features]\ncodex_hooks = true\n"
    try:
        config_file.write_text(text)
    except OSError as e:
        print(f"Error writing {config_file}: {e}", file=sys.stderr)
        return 1
    return 0


_CODEX_HOOK_TIMEOUT_SECONDS = 5
_GEMINI_HOOK_TIMEOUT_MS = 5_000


def _install_codex_hook() -> int:
    """Install user-level Codex hooks and enable the feature flag."""
    if _ensure_codex_feature_flag() != 0:
        return 1
    return _install_json_hooks(
        _codex_hooks_file(), "codex", _CODEX_HOOK_EVENTS, _CODEX_HOOK_TIMEOUT_SECONDS
    )


def _install_gemini_hook() -> int:
    """Install user-level Gemini hooks."""
    return _install_json_hooks(
        _gemini_settings_file(), "gemini", _GEMINI_HOOK_EVENTS, _GEMINI_HOOK_TIMEOUT_MS
    )


def _uninstall_json_hooks(path: Path, provider_name: str) -> int:
    """Remove provider-specific ccgram hooks from a JSON settings file."""
    settings = _load_json_settings(path)
    if settings is None:
        return 1
    if not settings:
        print(f"No {path} found — nothing to uninstall.")
        return 0
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, dict):
        return 0
    predicate = _json_hook_command_predicate(provider_name)
    removed = 0
    for event_hooks in hooks.values():
        if not isinstance(event_hooks, list):
            continue
        for group in event_hooks:
            if not isinstance(group, dict):
                continue
            inner_hooks = group.get("hooks", [])
            if not isinstance(inner_hooks, list):
                continue
            kept = []
            for hook_config in inner_hooks:
                if isinstance(hook_config, dict) and predicate(
                    hook_config.get("command", "")
                ):
                    removed += 1
                    continue
                kept.append(hook_config)
            group["hooks"] = kept
    if removed == 0:
        print(f"No {provider_name} hooks found in {path} — nothing to remove.")
        return 0
    try:
        # Lazy: same rationale as _install_json_hooks.
        from .utils import atomic_write_json

        atomic_write_json(path, settings)
    except OSError as e:
        print(f"Error writing {path}: {e}", file=sys.stderr)
        return 1
    print(f"{provider_name} hooks removed from {path}: {removed}")
    return 0


def _json_hook_status(path: Path, provider_name: str, events: tuple[str, ...]) -> int:
    """Print provider-specific JSON hook status."""
    settings = _load_json_settings(path)
    if settings is None:
        return 1
    if not settings:
        print(f"Not installed ({path} does not exist)")
        return 1
    predicate = _json_hook_command_predicate(provider_name)
    statuses = {
        event_type: _has_matching_hook(settings, event_type, predicate)
        for event_type in events
    }
    for event_type, installed in statuses.items():
        status_str = "installed" if installed else "MISSING"
        print(f"  {event_type}: {status_str}")
    if all(statuses.values()):
        print("All hooks installed")
        return 0
    missing = [
        event_type for event_type, installed in statuses.items() if not installed
    ]
    print(f"Missing hooks: {', '.join(missing)}")
    return 1


def _install_hook(provider_name: str = "claude") -> int:  # noqa: PLR0912
    """Install ccgram hooks for all event types into provider settings.

    Returns 0 on success, 1 on error.
    """
    match provider_name:
        case "codex":
            return _install_codex_hook()
        case "gemini":
            return _install_gemini_hook()
        case "pi":
            print(
                "Pi hooks are provided by the hook-runner extension; nothing to install."
            )
            return 0
        case "claude":
            pass
        case _:
            print(f"Unsupported hook provider: {provider_name}", file=sys.stderr)
            return 1
    settings_file = _claude_settings_file()
    settings_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing settings
    settings: dict = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading {settings_file}: {e}", file=sys.stderr)
            return 1

    if "hooks" not in settings:
        settings["hooks"] = {}

    installed_count = 0
    already_count = 0
    current_command = _current_hook_command("claude")

    for event_type in _HOOK_EVENT_TYPES:
        has_current = _has_matching_hook(settings, event_type, _is_current_hook_command)
        has_known = _has_matching_hook(
            settings, event_type, _is_any_ccgram_hook_command
        )

        if has_known and not has_current:
            _replace_hook_commands(
                settings,
                event_type,
                _is_any_ccgram_hook_command,
                current_command,
            )
            installed_count += 1
            continue

        if has_current:
            already_count += 1
            continue

        hook_config: dict[str, Any] = {
            "type": "command",
            "command": current_command,
            "timeout": 5,
        }
        if event_type in _ASYNC_EVENTS:
            hook_config["async"] = True

        if event_type not in settings["hooks"]:
            settings["hooks"][event_type] = []

        event_hooks = settings["hooks"][event_type]
        if event_hooks:
            first_entry = event_hooks[0]
            if isinstance(first_entry, dict):
                first_entry.setdefault("hooks", []).append(hook_config)
            else:
                event_hooks.append({"hooks": [hook_config]})
        else:
            event_hooks.append({"hooks": [hook_config]})

        installed_count += 1

    if installed_count == 0 and already_count == len(_HOOK_EVENT_TYPES):
        print(f"All hooks already installed in {settings_file}")
        return 0

    # Write back
    try:
        settings_file.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        )
    except OSError as e:
        print(f"Error writing {settings_file}: {e}", file=sys.stderr)
        return 1

    print(
        f"Hooks installed in {settings_file}: "
        f"{installed_count} new, {already_count} already present"
    )
    return 0


def _uninstall_hook(provider_name: str = "claude") -> int:  # noqa: PLR0911
    """Remove ccgram hooks from provider settings.

    Returns 0 on success, 1 on error.
    """
    match provider_name:
        case "codex":
            return _uninstall_json_hooks(_codex_hooks_file(), "codex")
        case "gemini":
            return _uninstall_json_hooks(_gemini_settings_file(), "gemini")
        case "pi":
            print(
                "Pi hooks are managed by the hook-runner extension; nothing to uninstall."
            )
            return 0
        case "claude":
            pass
        case _:
            print(f"Unsupported hook provider: {provider_name}", file=sys.stderr)
            return 1
    settings_file = _claude_settings_file()
    if not settings_file.exists():
        print("No settings.json found — nothing to uninstall.")
        return 0

    try:
        settings = json.loads(settings_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading {settings_file}: {e}", file=sys.stderr)
        return 1

    # Check if any ccgram hooks are installed
    any_installed = any(
        _has_ccgram_hook(settings, event) for event in _HOOK_EVENT_TYPES
    )
    if not any_installed:
        print("Hook not installed — nothing to uninstall.")
        return 0

    # Remove ccgram hook entries from all event types
    hooks_section = settings.get("hooks", {})
    for event_type in _HOOK_EVENT_TYPES:
        event_hooks = hooks_section.get(event_type, [])
        if not event_hooks:
            continue

        new_event_hooks = []
        for entry in event_hooks:
            if not isinstance(entry, dict):
                new_event_hooks.append(entry)
                continue
            inner_hooks = entry.get("hooks", [])
            filtered = [
                h
                for h in inner_hooks
                if not isinstance(h, dict)
                or not _is_any_ccgram_hook_command(h.get("command", ""))
            ]
            if filtered:
                entry["hooks"] = filtered
                new_event_hooks.append(entry)

        hooks_section[event_type] = new_event_hooks

    try:
        settings_file.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        )
    except OSError as e:
        print(f"Error writing {settings_file}: {e}", file=sys.stderr)
        return 1

    print(f"Hooks uninstalled from {settings_file}")
    return 0


def _hook_status(provider_name: str = "claude") -> int:  # noqa: PLR0911
    """Show per-event hook installation status.

    Returns 0 if all installed, 1 if any missing.
    """
    match provider_name:
        case "codex":
            return _json_hook_status(_codex_hooks_file(), "codex", _CODEX_HOOK_EVENTS)
        case "gemini":
            return _json_hook_status(
                _gemini_settings_file(), "gemini", _GEMINI_HOOK_EVENTS
            )
        case "pi":
            print("Pi hook status depends on the hook-runner extension.")
            print(
                "Expected built-in hook-runner ccgram events: "
                "SessionStart, Stop, SessionEnd, SubagentStart"
            )
            return 0
        case "claude":
            pass
        case _:
            print(f"Unsupported hook provider: {provider_name}", file=sys.stderr)
            return 1
    settings_file = _claude_settings_file()
    if not settings_file.exists():
        print(f"Not installed ({settings_file} does not exist)")
        return 1

    try:
        settings = json.loads(settings_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading {settings_file}: {e}", file=sys.stderr)
        return 1

    event_status = get_installed_events(settings)
    all_installed = all(event_status.values())

    for event_type, installed in event_status.items():
        status_str = "installed" if installed else "MISSING"
        print(f"  {event_type}: {status_str}")

    if all_installed:
        print("All hooks installed")
        return 0

    missing = [e for e, v in event_status.items() if not v]
    print(f"Missing hooks: {', '.join(missing)}")
    return 1


def _resolve_window_id(pane_id: str) -> tuple[str, str, str, str] | None:
    """Resolve tmux pane ID to (session_window_key, window_id, window_name, pane_tty).

    Returns None if resolution fails. pane_tty is the pane's controlling tty path
    (e.g. ``/dev/ttys012``) or "" when older tmux mocks omit the field.
    """
    try:
        result = subprocess.run(
            [
                "tmux",
                "display-message",
                "-t",
                pane_id,
                "-p",
                "#{session_name}\t#{window_id}\t#{window_name}\t#{pane_tty}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        logger.warning("tmux display-message timed out for pane %s", pane_id)
        return None
    raw_output = result.stdout.strip()
    parts = raw_output.split("\t", 3)
    if len(parts) < _TMUX_FORMAT_PARTS:
        logger.warning(
            "Failed to parse session:window_id:window_name from tmux "
            "(pane=%s, output=%s)",
            pane_id,
            raw_output,
        )
        return None

    tmux_session_name, window_id, window_name = parts[0], parts[1], parts[2]
    pane_tty = parts[3] if len(parts) >= _TMUX_FORMAT_PARTS_WITH_TTY else ""
    session_window_key = f"{tmux_session_name}:{window_id}"
    return session_window_key, window_id, window_name, pane_tty


def _ps_snapshot() -> dict[int, tuple[int, int, str, str]]:
    """Return ``{pid: (ppid, pgid, stat, command_basename)}`` for all processes.

    Empty dict on subprocess failure or unparseable output — callers must
    fail-open when the snapshot is empty.
    """
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,pgid=,stat=,command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired, OSError:
        return {}
    snapshot: dict[int, tuple[int, int, str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, _PS_SNAPSHOT_FIELDS - 1)
        if len(parts) < _PS_SNAPSHOT_FIELDS:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pgid = int(parts[2])
        except ValueError:
            continue
        stat = parts[3]
        cmd_argv0 = parts[4].split(None, 1)[0] if parts[4] else ""
        cmd_base = cmd_argv0.rsplit("/", 1)[-1]
        snapshot[pid] = (ppid, pgid, stat, cmd_base)
    return snapshot


def _foreground_pgid_on_tty(
    snapshot: dict[int, tuple[int, int, str, str]], pane_tty: str
) -> int | None:
    """Return the foreground process group id on ``pane_tty``, or None."""
    if not pane_tty or not snapshot:
        return None
    tty_name = pane_tty.removeprefix("/dev/")
    if not tty_name:
        return None
    try:
        result = subprocess.run(
            ["ps", "-t", tty_name, "-o", "pid="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired, OSError:
        return None
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        info = snapshot.get(pid)
        if info and "+" in info[2]:
            return info[1]
    return None


def _closest_claude_ancestor(
    snapshot: dict[int, tuple[int, int, str, str]], start_pid: int
) -> int | None:
    """Walk parent chain from ``start_pid``; return the closest claude PID, or None."""
    cur = start_pid
    visited: set[int] = set()
    for _ in range(40):
        if cur <= 1 or cur in visited:
            return None
        visited.add(cur)
        info = snapshot.get(cur)
        if info is None:
            return None
        ppid, _pgid, _stat, cmd_base = info
        if cmd_base == "claude":
            return cur
        cur = ppid
    return None


def _is_nested_session(pane_tty: str) -> bool:
    """Return True if the hook was fired by a nested (non-foreground) claude.

    The "primary" claude in a tmux pane is launched by the user's shell, so
    its PID equals the foreground process group id on the pane's tty. Any
    claude spawned beneath that primary (e.g. an MCP-server-launched observer
    such as claude-mem) is a *descendant* — its PID differs from the
    foreground PGID even though it shares the pgid via inheritance.

    Fails open: returns False on any subprocess error or missing data so
    hook delivery is never made *more* fragile than the status quo.
    """
    if not pane_tty:
        return False
    snapshot = _ps_snapshot()
    if not snapshot:
        return False
    fg_pgid = _foreground_pgid_on_tty(snapshot, pane_tty)
    if fg_pgid is None:
        return False
    owner = _closest_claude_ancestor(snapshot, os.getpid())
    if owner is None:
        return False
    return owner != fg_pgid


def _write_event(
    event_type: str,
    session_id: str,
    window_key: str,
    data: dict[str, Any],
) -> None:
    """Append one JSONL event line to events.jsonl with file locking."""
    # Lazy: hook.py runs as `python -m ccgram.hook` from Claude Code on
    # every notification; deferring utils import until an event actually
    # fires keeps the latency-sensitive fast path lean.
    # Lazy: utils.ccgram_dir resolves $CCGRAM_DIR at runtime
    from .utils import ccgram_dir

    events_file = ccgram_dir() / "events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)

    event_line = json.dumps(
        {
            "ts": time.time(),
            "event": event_type,
            "window_key": window_key,
            "session_id": session_id,
            "data": data,
        },
        separators=(",", ":"),
    )

    try:
        with open(events_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(event_line + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        logger.exception("Failed to write event to %s", events_file)


def _update_session_map(
    session_window_key: str,
    session_id: str,
    cwd: str,
    window_name: str,
    transcript_path: str,
    tmux_session_name: str,
    provider_name: str = "claude",
) -> None:
    """Update session_map.json for a SessionStart event."""
    # Lazy: same hook fast-path rationale as ``_write_event``.
    from .utils import ccgram_dir, atomic_write_json

    map_file = ccgram_dir() / "session_map.json"
    map_file.parent.mkdir(parents=True, exist_ok=True)

    lock_path = map_file.with_suffix(".lock")
    try:
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                session_map: dict[str, dict[str, str]] = {}
                if map_file.exists():
                    try:
                        raw = map_file.read_text()
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            session_map = parsed
                        else:
                            logger.warning(
                                "session_map.json has unexpected type %s, ignoring",
                                type(parsed).__name__,
                            )
                    except json.JSONDecodeError:
                        # Corrupted JSON — preserve the file for inspection
                        # instead of silently overwriting with near-empty data.
                        backup = map_file.with_suffix(".json.corrupt")
                        try:
                            # Lazy: shutil only needed in the error path of
                            # backing up a corrupted session_map.json.
                            import shutil

                            shutil.copy2(map_file, backup)
                            logger.warning(
                                "Corrupted session_map.json backed up to %s",
                                backup,
                            )
                        except OSError:
                            logger.warning("Corrupted session_map.json (backup failed)")
                    except OSError:
                        logger.warning("Failed to read session_map.json")

                session_map[session_window_key] = {
                    "session_id": session_id,
                    "cwd": cwd,
                    "window_name": window_name,
                    "transcript_path": transcript_path,
                    "provider_name": provider_name,
                }

                # Clean up old-format key ("session:window_name") if it exists
                old_key = f"{tmux_session_name}:{window_name}"
                if old_key != session_window_key and old_key in session_map:
                    del session_map[old_key]
                    logger.info("Removed old-format session_map key: %s", old_key)

                atomic_write_json(map_file, session_map)
                logger.info(
                    "Updated session_map: %s -> session_id=%s, cwd=%s",
                    session_window_key,
                    session_id,
                    cwd,
                )
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
    except OSError:
        logger.exception("Failed to write session_map")


def _encode_pi_cwd_dirname(cwd: str) -> str:
    """Encode cwd using Pi's session directory convention."""
    stripped = cwd.lstrip("/\\").rstrip("/\\")
    encoded = stripped.replace("/", "-").replace("\\", "-").replace(":", "-")
    return f"--{encoded}--"


def _resolve_pi_transcript_path(session_id: str, cwd: str) -> str:
    """Find a Pi transcript path when hook-runner omitted it."""
    if not cwd:
        return ""
    session_dir = (
        Path.home() / ".pi" / "agent" / "sessions" / _encode_pi_cwd_dirname(cwd)
    )
    if not session_dir.is_dir():
        return ""
    candidates: list[tuple[float, Path]] = []
    try:
        for entry in session_dir.iterdir():
            if entry.suffix != ".jsonl" or not entry.is_file():
                continue
            try:
                candidates.append((entry.stat().st_mtime, entry))
            except OSError:
                continue
    except OSError:
        return ""
    candidates.sort(reverse=True)
    for _mtime, path in candidates:
        if session_id and session_id in path.name:
            return str(path)
    return str(candidates[0][1]) if candidates else ""


def _resolve_transcript_path(
    provider_name: str, session_id: str, cwd: str, transcript_path: str
) -> str:
    """Return transcript path from payload or provider-specific fallback."""
    if transcript_path:
        return transcript_path
    if provider_name == "pi":
        return _resolve_pi_transcript_path(session_id, cwd)
    return ""


def _read_session_map_entry(session_window_key: str) -> dict[str, str]:
    """Return the current session_map entry for ``session_window_key`` or {}."""
    # Lazy: same hook fast-path rationale as ``_write_event``.
    from .utils import ccgram_dir

    map_file = ccgram_dir() / "session_map.json"
    if not map_file.exists():
        return {}
    try:
        raw = json.loads(map_file.read_text())
    except OSError, json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    entry = raw.get(session_window_key)
    return entry if isinstance(entry, dict) else {}


def _refresh_session_map_if_stale(
    session_window_key: str,
    session_id: str,
    provider_name: str,
    window_name: str,
    payload_cwd: str,
    payload_transcript_path: str,
) -> None:
    """Refresh ``session_map.json`` when a non-SessionStart event reports a
    different session_id or provider than the stored entry.

    Some installs (notably Pi via cc-thingz hook-runner) deliver Stop/Subagent
    hooks without a matching SessionStart through this hook path, so the map
    can keep pointing at the previous provider's session. We use values the
    hook payload already carries — no external scanning — to avoid the
    recovery anti-pattern called out in PR #51.
    """
    existing = _read_session_map_entry(session_window_key)
    if not existing:
        # SessionStart owns initial creation; never extend the map from a
        # non-SessionStart event. Missing entry means no prior session was
        # tracked here — leave the fallback (cwd-based discovery in
        # SessionMonitor) to handle it.
        return
    if (
        existing.get("session_id") == session_id
        and existing.get("provider_name") == provider_name
    ):
        return
    cwd = payload_cwd or existing.get("cwd", "")
    transcript_path = _resolve_transcript_path(
        provider_name, session_id, cwd, payload_transcript_path
    )
    tmux_session_name = session_window_key.rsplit(":", 1)[0]
    _update_session_map(
        session_window_key,
        session_id,
        cwd,
        window_name,
        transcript_path,
        tmux_session_name,
        provider_name,
    )
    logger.info(
        "Refreshed stale session_map for %s: %s/%s -> %s/%s",
        session_window_key,
        existing.get("provider_name") or "<none>",
        (existing.get("session_id") or "<none>")[:8],
        provider_name,
        session_id[:8],
    )


def _provider_from_pane_tty(pane_tty: str) -> ProviderName | None:
    """Best-effort provider detection from foreground tty process commands.

    This is a last-resort fallback; the primary paths are the explicit
    ``provider_name`` field and the ``/.provider/`` transcript path prefix
    checked in ``detect_provider_from_payload``.  JS-wrapped Pi (e.g.
    ``node ~/.pi/agent/cli.js``) is not matched here — it is caught by the
    ``/.pi/`` transcript path check instead.
    """
    if not pane_tty:
        return None
    tty_name = pane_tty.removeprefix("/dev/")
    try:
        result = subprocess.run(
            ["ps", "-t", tty_name, "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired, OSError:
        return None
    text = result.stdout.lower()
    if "gemini" in text:
        return "gemini"
    if "codex" in text:
        return "codex"
    if "claude" in text:
        return "claude"
    if any(tok == "pi" or tok.endswith("/pi") for tok in text.split()):
        return "pi"
    return None


def _locate_primary_window(
    session_id: str, event: str, provider_name: str = "claude"
) -> tuple[str, str, str] | None:
    """Resolve TMUX_PANE → primary window, or None to drop the hook.

    Returns ``(session_window_key, window_id, window_name)`` for the foreground
    claude in the pane. Returns ``None`` when the pane can't be resolved or
    when a nested claude (e.g. claude-mem observer) fired the hook — the
    nested case is logged at info so the rejection is visible to operators.
    """
    pane_id = os.environ.get("TMUX_PANE", "")
    if not pane_id:
        logger.warning("TMUX_PANE not set, cannot determine window")
        return None
    resolved = _resolve_window_id(pane_id)
    if not resolved:
        return None
    session_window_key, window_id, window_name, pane_tty = resolved
    logger.debug(
        "tmux key=%s, window_name=%s, session_id=%s, event=%s",
        session_window_key,
        window_name,
        session_id,
        event,
    )
    if provider_name == "claude" and _is_nested_session(pane_tty):
        logger.info(
            "Skipping hook from nested claude (window_key=%s, session_id=%s, event=%s)",
            session_window_key,
            session_id,
            event,
        )
        return None
    return session_window_key, window_id, window_name


def _process_hook_stdin(provider_name: str | None = None) -> None:
    """Process an agent hook event from stdin."""
    logger.debug("Processing hook event from stdin")
    try:
        raw_payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse stdin JSON: %s", e)
        return
    if not isinstance(raw_payload, dict):
        logger.warning("Hook stdin JSON must be an object")
        return
    payload: dict[str, object] = raw_payload

    payload_provider = detect_provider_from_payload(payload)
    if provider_name and payload_provider and payload_provider != provider_name:
        logger.warning(
            "Hook --provider=%s but payload looks like %s; using %s",
            provider_name,
            payload_provider,
            provider_name,
        )
    detected_provider = provider_name or payload_provider
    if detected_provider is None:
        pane_id = os.environ.get("TMUX_PANE", "")
        resolved = _resolve_window_id(pane_id) if pane_id else None
        if resolved:
            detected_provider = _provider_from_pane_tty(resolved[3])
    if detected_provider is None:
        detected_provider = "claude"

    adapter = get_hook_adapter(detected_provider)
    if adapter is None:
        logger.debug("Ignoring hook for unsupported provider: %s", detected_provider)
        return
    normalized = adapter.normalize(payload)
    if normalized is None:
        logger.debug(
            "Ignoring invalid hook payload for provider: %s", detected_provider
        )
        return

    event = normalized.canonical_event_name
    if event not in _HOOK_EVENT_TYPES and event not in {"PreCompact", "PostCompact"}:
        logger.debug("Ignoring unhandled event: %s", event)
        return

    located = _locate_primary_window(normalized.session_id, event, detected_provider)
    if located is None:
        return
    session_window_key, _window_id, window_name = located

    if event == "SessionStart":
        tmux_session_name = session_window_key.rsplit(":", 1)[0]
        transcript_path = _resolve_transcript_path(
            detected_provider,
            normalized.session_id,
            str(normalized.cwd) if normalized.cwd else "",
            str(normalized.transcript_path) if normalized.transcript_path else "",
        )
        cwd = str(normalized.cwd) if normalized.cwd else ""
        _update_session_map(
            session_window_key,
            normalized.session_id,
            cwd,
            window_name,
            transcript_path,
            tmux_session_name,
            detected_provider,
        )
        data = dict(normalized.data)
        data.update(
            {
                "cwd": cwd,
                "transcript_path": transcript_path,
                "window_name": window_name,
            }
        )
        _write_event(event, normalized.session_id, session_window_key, data)
        return

    _refresh_session_map_if_stale(
        session_window_key,
        normalized.session_id,
        detected_provider,
        window_name,
        str(normalized.cwd) if normalized.cwd else "",
        str(normalized.transcript_path) if normalized.transcript_path else "",
    )
    _write_event(event, normalized.session_id, session_window_key, normalized.data)


def hook_main(
    install: bool = False,
    uninstall: bool = False,
    status: bool = False,
    provider_name: str = "claude",
) -> None:
    """Process a Claude Code hook event from stdin, or manage hook installation."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG,
        stream=sys.stderr,
    )

    if install:
        logger.info("Hook install requested")
        sys.exit(_install_hook(provider_name))

    if uninstall:
        sys.exit(_uninstall_hook(provider_name))

    if status:
        sys.exit(_hook_status(provider_name))

    # Pass None for the implicit Claude default so detect_provider_from_payload
    # gets first say (an explicit `--provider claude` invocation deliberately
    # keeps the explicit flag to surface the mismatch warning when payload
    # heuristics disagree). The CLI default also resolves to "claude", so the
    # None path covers the common case of an unannotated hook command.
    _process_hook_stdin(provider_name if provider_name != "claude" else None)
