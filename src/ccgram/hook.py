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
import structlog
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = structlog.get_logger()

# Validate session_id looks like a UUID
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _claude_settings_file() -> Path:
    """Resolve Claude settings.json path, respecting CLAUDE_CONFIG_DIR."""
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / "settings.json"
    return Path.home() / ".claude" / "settings.json"


# Substring marker for detecting ccgram hook in command strings
_HOOK_COMMAND_MARKER = "ccgram hook"
# Legacy marker from pre-rename ccbot — used for detection and cleanup
_LEGACY_HOOK_MARKER = "ccbot hook"

# Expected number of parts when parsing "session_name\t@id\twindow_name"
_TMUX_FORMAT_PARTS = 3

# Hook event types ccgram handles (order matters for status display)
_HOOK_EVENT_TYPES: tuple[str, ...] = (
    "SessionStart",
    "Notification",
    "Stop",
    "SubagentStart",
    "SubagentStop",
    "TeammateIdle",
    "TaskCompleted",
)

# Events that should not block the agent (async: true)
_ASYNC_EVENTS: frozenset[str] = frozenset(
    {
        "SubagentStart",
        "SubagentStop",
        "TeammateIdle",
        "TaskCompleted",
    }
)


def _has_hook_marker(settings: dict, event_type: str, marker: str) -> bool:
    """Check if a hook with the given marker is installed for an event type."""
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
            if marker in cmd:
                return True
    return False


def _has_ccgram_hook(settings: dict, event_type: str) -> bool:
    """Check if ccgram hook (or legacy ccbot hook) is installed."""
    return _has_hook_marker(
        settings, event_type, _HOOK_COMMAND_MARKER
    ) or _has_hook_marker(settings, event_type, _LEGACY_HOOK_MARKER)


def _is_hook_installed(settings: dict) -> bool:
    """Check if ccgram hook is installed for SessionStart (backward compat)."""
    return _has_ccgram_hook(settings, "SessionStart")


def get_installed_events(settings: dict) -> dict[str, bool]:
    """Return installation status for each expected hook event type."""
    return {event: _has_ccgram_hook(settings, event) for event in _HOOK_EVENT_TYPES}


def _replace_legacy_hooks(settings: dict, event_type: str) -> None:
    """Replace legacy 'ccbot hook' command strings with 'ccgram hook'."""
    event_hooks = settings.get("hooks", {}).get(event_type, [])
    for entry in event_hooks:
        if not isinstance(entry, dict):
            continue
        for h in entry.get("hooks", []):
            if not isinstance(h, dict):
                continue
            cmd = h.get("command", "")
            if _LEGACY_HOOK_MARKER in cmd:
                h["command"] = cmd.replace(_LEGACY_HOOK_MARKER, _HOOK_COMMAND_MARKER)


def _install_hook() -> int:
    """Install ccgram hooks for all event types into Claude's settings.json.

    Returns 0 on success, 1 on error.
    """
    settings_file = _claude_settings_file()
    settings_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing settings
    settings: dict = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.exception("Error reading %s", settings_file)
            print(f"Error reading {settings_file}: {e}", file=sys.stderr)
            return 1

    if "hooks" not in settings:
        settings["hooks"] = {}

    installed_count = 0
    already_count = 0

    for event_type in _HOOK_EVENT_TYPES:
        has_new = _has_hook_marker(settings, event_type, _HOOK_COMMAND_MARKER)
        has_legacy = _has_hook_marker(settings, event_type, _LEGACY_HOOK_MARKER)

        if has_legacy and not has_new:
            # Replace legacy "ccbot hook" entries with "ccgram hook"
            _replace_legacy_hooks(settings, event_type)
            installed_count += 1
            continue

        if has_new:
            already_count += 1
            continue

        hook_config: dict[str, Any] = {
            "type": "command",
            "command": "ccgram hook",
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
        logger.exception("Error writing %s", settings_file)
        print(f"Error writing {settings_file}: {e}", file=sys.stderr)
        return 1

    print(
        f"Hooks installed in {settings_file}: "
        f"{installed_count} new, {already_count} already present"
    )
    return 0


def _uninstall_hook() -> int:
    """Remove ccgram hooks from all event types in Claude's settings.json.

    Returns 0 on success, 1 on error.
    """
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
                or (
                    _HOOK_COMMAND_MARKER not in h.get("command", "")
                    and _LEGACY_HOOK_MARKER not in h.get("command", "")
                )
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


def _hook_status() -> int:
    """Show per-event hook installation status.

    Returns 0 if all installed, 1 if any missing.
    """
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


def _resolve_window_id(pane_id: str) -> tuple[str, str, str] | None:
    """Resolve tmux pane ID to (session_window_key, window_id, window_name).

    Returns None if resolution fails.
    """
    try:
        result = subprocess.run(
            [
                "tmux",
                "display-message",
                "-t",
                pane_id,
                "-p",
                "#{session_name}\t#{window_id}\t#{window_name}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        logger.warning("tmux display-message timed out for pane %s", pane_id)
        return None
    raw_output = result.stdout.strip()
    parts = raw_output.split("\t", 2)
    if len(parts) < _TMUX_FORMAT_PARTS:
        logger.warning(
            "Failed to parse session:window_id:window_name from tmux "
            "(pane=%s, output=%s)",
            pane_id,
            raw_output,
        )
        return None

    tmux_session_name, window_id, window_name = parts
    session_window_key = f"{tmux_session_name}:{window_id}"
    return session_window_key, window_id, window_name


def _write_event(
    event_type: str,
    session_id: str,
    window_key: str,
    data: dict[str, Any],
) -> None:
    """Append one JSONL event line to events.jsonl with file locking."""
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


def _extract_notification_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract data from a Notification hook payload."""
    return {
        "tool_name": payload.get("tool_name", ""),
        "message": payload.get("message", ""),
    }


def _extract_stop_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract data from a Stop hook payload."""
    return {
        "stop_reason": payload.get("stop_reason", ""),
        "num_turns": payload.get("num_turns", 0),
    }


def _extract_subagent_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract data from a SubagentStart/SubagentStop hook payload."""
    return {
        "subagent_id": payload.get("subagent_id", ""),
        "description": payload.get("description", ""),
        "name": payload.get("name", ""),
    }


def _extract_teammate_idle_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract data from a TeammateIdle hook payload."""
    return {
        "teammate_name": payload.get("teammate_name", ""),
        "team_name": payload.get("team_name", ""),
    }


def _extract_task_completed_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract data from a TaskCompleted hook payload."""
    return {
        "task_id": payload.get("task_id", ""),
        "task_subject": payload.get("task_subject", ""),
        "task_description": payload.get("task_description", ""),
        "teammate_name": payload.get("teammate_name", ""),
        "team_name": payload.get("team_name", ""),
    }


# Map event types to their data extractor functions
_EVENT_DATA_EXTRACTORS: dict[str, Any] = {
    "Notification": _extract_notification_data,
    "Stop": _extract_stop_data,
    "SubagentStart": _extract_subagent_data,
    "SubagentStop": _extract_subagent_data,
    "TeammateIdle": _extract_teammate_idle_data,
    "TaskCompleted": _extract_task_completed_data,
}


def _update_session_map(
    session_window_key: str,
    session_id: str,
    cwd: str,
    window_name: str,
    transcript_path: str,
    tmux_session_name: str,
) -> None:
    """Update session_map.json for a SessionStart event."""
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
                        session_map = json.loads(map_file.read_text())
                    except (json.JSONDecodeError, OSError):  # fmt: skip
                        logger.warning(
                            "Failed to read existing session_map, starting fresh"
                        )

                session_map[session_window_key] = {
                    "session_id": session_id,
                    "cwd": cwd,
                    "window_name": window_name,
                    "transcript_path": transcript_path,
                    "provider_name": "claude",
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


def _process_hook_stdin() -> None:
    """Process a Claude Code hook event from stdin."""
    logger.debug("Processing hook event from stdin")
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse stdin JSON: %s", e)
        return

    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")
    transcript_path = payload.get("transcript_path", "")
    event = payload.get("hook_event_name", "")

    if not session_id or not event:
        logger.debug("Empty session_id or event, ignoring")
        return

    # Validate session_id format
    if not UUID_RE.match(session_id):
        logger.warning("Invalid session_id format: %s", session_id)
        return

    # Validate cwd is an absolute path (if provided)
    if cwd and not os.path.isabs(cwd):
        logger.warning("cwd is not absolute: %s", cwd)
        return

    # Only process events we handle
    if event not in _HOOK_EVENT_TYPES:
        logger.debug("Ignoring unhandled event: %s", event)
        return

    # Get tmux session:window key for the pane running this hook.
    # TMUX_PANE is set by tmux for every process inside a pane.
    pane_id = os.environ.get("TMUX_PANE", "")
    if not pane_id:
        logger.warning("TMUX_PANE not set, cannot determine window")
        return

    resolved = _resolve_window_id(pane_id)
    if not resolved:
        return
    session_window_key, window_id, window_name = resolved

    logger.debug(
        "tmux key=%s, window_name=%s, session_id=%s, event=%s",
        session_window_key,
        window_name,
        session_id,
        event,
    )

    # SessionStart: update session_map.json AND write event
    if event == "SessionStart":
        tmux_session_name = session_window_key.rsplit(":", 1)[0]
        _update_session_map(
            session_window_key,
            session_id,
            cwd,
            window_name,
            transcript_path,
            tmux_session_name,
        )
        _write_event(
            event,
            session_id,
            session_window_key,
            {
                "cwd": cwd,
                "transcript_path": transcript_path,
                "window_name": window_name,
            },
        )
        return

    # Other events: write event only
    extractor = _EVENT_DATA_EXTRACTORS.get(event)
    data = extractor(payload) if extractor else {}
    _write_event(event, session_id, session_window_key, data)


def hook_main(
    install: bool = False, uninstall: bool = False, status: bool = False
) -> None:
    """Process a Claude Code hook event from stdin, or manage hook installation."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG,
        stream=sys.stderr,
    )

    if install:
        logger.info("Hook install requested")
        sys.exit(_install_hook())

    if uninstall:
        sys.exit(_uninstall_hook())

    if status:
        sys.exit(_hook_status())

    _process_hook_stdin()
