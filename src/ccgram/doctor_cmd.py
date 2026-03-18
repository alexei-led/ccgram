"""CLI `ccgram doctor [--fix]` — validate ccgram setup.

Checks environment, dependencies, and configuration without requiring
a bot token. With --fix, auto-repairs what it can (install hook, kill orphans).

Provider-aware: reads CCGRAM_PROVIDER env to determine which checks apply
(e.g. hook checks are skipped for providers without hook support).
No Config import needed — uses utils.ccgram_dir() and subprocess.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from collections.abc import Callable

from .providers import resolve_capabilities
from .utils import ccgram_dir, tmux_session_name

_PASS = "pass"
_FAIL = "fail"
_WARN = "warn"

_SYMBOLS = {_PASS: "\u2713", _FAIL: "\u2717", _WARN: "\u26a0"}

_TMUX_FORMAT_PARTS = 2
_MAIN_WINDOW_NAME = "__main__"


def _print_check(status: str, message: str) -> None:
    """Print a single check result."""
    sym = _SYMBOLS.get(status, "?")
    print(f"  {sym} {message}")


def _check_tmux() -> tuple[str, str]:
    """Check tmux binary and version."""
    path = shutil.which("tmux")
    if not path:
        return _FAIL, "tmux not found in PATH"
    try:
        result = subprocess.run(
            ["tmux", "-V"], capture_output=True, text=True, timeout=5
        )
        version = result.stdout.strip()
        return _PASS, f"{version} found"
    except (OSError, subprocess.TimeoutExpired):  # fmt: skip
        return _PASS, "tmux found (version unknown)"


def _check_provider_command(provider_name: str) -> tuple[str, str]:
    """Check provider CLI command availability."""
    from ccgram.providers import resolve_launch_command

    cmd = resolve_launch_command(provider_name)
    executable = cmd.split()[0]
    path = shutil.which(executable)
    if path:
        label = cmd if cmd != executable else executable
        return _PASS, f"{label} found at {path}"
    return _FAIL, f"'{executable}' not found in PATH"


def _check_tmux_session() -> tuple[str, str]:
    """Check if tmux session exists."""
    session_name = tmux_session_name()
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return _PASS, f'tmux session "{session_name}" exists'
        return _FAIL, f'tmux session "{session_name}" not found'
    except (OSError, subprocess.TimeoutExpired):  # fmt: skip
        return _FAIL, "cannot connect to tmux server"


def _check_hooks() -> tuple[str, str, dict[str, bool]]:
    """Check hook installation for all event types.

    Returns (status, message, event_status_dict).
    """
    from .hook import _claude_settings_file, get_installed_events

    settings_file = _claude_settings_file()
    if not settings_file.exists():
        return _FAIL, f"hooks not installed ({settings_file} missing)", {}
    try:
        settings = json.loads(settings_file.read_text())
    except (json.JSONDecodeError, OSError):  # fmt: skip
        return _FAIL, "hooks not installed (settings.json unreadable)", {}

    event_status = get_installed_events(settings)
    installed = [e for e, v in event_status.items() if v]
    missing = [e for e, v in event_status.items() if not v]

    if not missing:
        return _PASS, f"all {len(installed)} hook events installed", event_status
    if not installed:
        return _FAIL, "no hook events installed", event_status
    return (
        _WARN,
        f"{len(installed)} installed, {len(missing)} missing: {', '.join(missing)}",
        event_status,
    )


def _check_hook() -> tuple[str, str, bool]:
    """Check hook installation (backward compat wrapper).

    Returns (status, message, is_installed).
    """
    status, message, event_status = _check_hooks()
    any_installed = any(event_status.values()) if event_status else False
    return status, message, any_installed


def _check_config_dir() -> tuple[str, str]:
    """Check config directory exists."""
    config_dir = ccgram_dir()
    if config_dir.is_dir():
        return _PASS, f"config dir {config_dir} exists"
    return _FAIL, f"config dir {config_dir} not found"


def _check_bot_token() -> tuple[str, str]:
    """Check bot token is set (without printing it)."""
    from dotenv import load_dotenv

    config_dir = ccgram_dir()
    local_env = Path(".env")
    global_env = config_dir / ".env"
    if local_env.is_file():
        load_dotenv(local_env)
    if global_env.is_file():
        load_dotenv(global_env)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if token:
        return _PASS, "TELEGRAM_BOT_TOKEN set"
    return _FAIL, "TELEGRAM_BOT_TOKEN not set"


def _check_allowed_users() -> tuple[str, str]:
    """Check allowed users configured."""
    users_str = os.environ.get("ALLOWED_USERS", "")
    if not users_str:
        return _FAIL, "ALLOWED_USERS not set"
    try:
        users = [int(u.strip()) for u in users_str.split(",") if u.strip()]
        return _PASS, f"ALLOWED_USERS: {len(users)} user(s)"
    except ValueError:
        return _FAIL, "ALLOWED_USERS contains non-numeric values"


def _check_events_file() -> tuple[str, str]:
    """Check events.jsonl is writable in config dir."""
    config_dir = ccgram_dir()
    events_file = config_dir / "events.jsonl"
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        # Test write access
        with open(events_file, "a") as f:
            f.write("")
        return _PASS, f"events file {events_file} writable"
    except OSError as e:
        return _WARN, f"events file not writable: {e}"


def _list_live_windows(session_name: str) -> dict[str, str]:
    """List live tmux windows, excluding __main__. Returns {window_id: window_name}."""
    try:
        result = subprocess.run(
            [
                "tmux",
                "list-windows",
                "-t",
                session_name,
                "-F",
                "#{window_id}\t#{window_name}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
    except (OSError, subprocess.TimeoutExpired):  # fmt: skip
        return {}

    windows: dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == _TMUX_FORMAT_PARTS and parts[1] != _MAIN_WINDOW_NAME:
            windows[parts[0]] = parts[1]
    return windows


def _get_known_window_ids(config_dir: Path, session_name: str) -> set[str]:
    """Get window IDs known from state.json bindings and session_map.json."""
    known: set[str] = set()

    state_file = config_dir / "state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            for bindings in state.get("thread_bindings", {}).values():
                known.update(bindings.values())
        except (json.JSONDecodeError, OSError):  # fmt: skip
            pass

    session_map_file = config_dir / "session_map.json"
    prefix = f"{session_name}:"
    if session_map_file.exists():
        try:
            session_map = json.loads(session_map_file.read_text())
            for key in session_map:
                if key.startswith(prefix):
                    known.add(key[len(prefix) :])
        except (json.JSONDecodeError, OSError):  # fmt: skip
            pass

    return known


def _find_orphaned_windows() -> list[tuple[str, str]]:
    """Find tmux windows not bound to any topic and not in session_map."""
    session_name = tmux_session_name()
    live_windows = _list_live_windows(session_name)
    if not live_windows:
        return []

    known_ids = _get_known_window_ids(ccgram_dir(), session_name)
    return [(wid, wname) for wid, wname in live_windows.items() if wid not in known_ids]


def _run_check(check_fn: Callable[[], tuple[str, str]]) -> tuple[str, str, bool]:
    """Run a check function and return (status, message, is_failure)."""
    result = check_fn()
    status, msg = result[0], result[1]
    _print_check(status, msg)
    return status, msg, status == _FAIL


def _fix_hooks(event_status: dict[str, bool], fix: bool) -> None:
    """Attempt to install missing hooks if --fix is set."""
    if not fix:
        return
    missing = [e for e, v in event_status.items() if not v]
    if not missing:
        return
    from .hook import _install_hook

    result = _install_hook()
    if result == 0:
        _print_check(_PASS, "hooks installed (fixed)")
    else:
        _print_check(_FAIL, "failed to install hooks")


def _fix_hook(hook_installed: bool, fix: bool) -> None:
    """Attempt to fix missing hook if --fix is set (backward compat)."""
    if not fix or hook_installed:
        return
    _fix_hooks({}, fix)


def _fix_orphans(orphans: list[tuple[str, str]], fix: bool) -> None:
    """Kill orphaned windows if --fix is set."""
    if not fix:
        return
    session_name = tmux_session_name()
    for wid, wname in orphans:
        try:
            subprocess.run(
                ["tmux", "kill-window", "-t", f"{session_name}:{wid}"],
                capture_output=True,
                timeout=5,
            )
            _print_check(_PASS, f"killed orphaned window {wid} ({wname})")
        except (OSError, subprocess.TimeoutExpired):  # fmt: skip
            _print_check(_FAIL, f"failed to kill window {wid}")


def doctor_main(fix: bool = False) -> None:
    """Entry point for `ccgram doctor [--fix]`."""
    caps = resolve_capabilities()
    has_failures = False

    print(f"Provider: {caps.name}")

    # Core checks
    _, _, failed = _run_check(_check_tmux)
    has_failures = has_failures or failed

    _, _, failed = _run_check(lambda: _check_provider_command(caps.name))
    has_failures = has_failures or failed

    _, _, failed = _run_check(_check_tmux_session)
    has_failures = has_failures or failed

    # Hook checks — only relevant for providers with hook support
    if caps.supports_hook:
        hook_status, hook_msg, event_status = _check_hooks()
        _print_check(hook_status, hook_msg)
        if hook_status == _FAIL:
            has_failures = True
        _fix_hooks(event_status, fix)
    else:
        _print_check(_PASS, f"hook check skipped ({caps.name} has no hook support)")

    for check_fn in (_check_config_dir, _check_bot_token, _check_allowed_users):
        _, _, failed = _run_check(check_fn)
        has_failures = has_failures or failed

    # Events file check
    _, _, failed = _run_check(_check_events_file)
    has_failures = has_failures or failed

    # Orphaned windows
    orphans = _find_orphaned_windows()
    if orphans:
        names = ", ".join(f"{wid} ({wname})" for wid, wname in orphans)
        _print_check(_WARN, f"{len(orphans)} orphaned window(s): {names}")
        _fix_orphans(orphans, fix)
    else:
        _print_check(_PASS, "no orphaned windows")

    sys.exit(1 if has_failures else 0)
