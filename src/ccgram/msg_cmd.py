"""CLI subcommand group for inter-agent messaging.

Provides ``ccgram msg`` with subcommands for peer discovery, message
send/receive, broadcast, registration, and mailbox maintenance.
Window self-identification via ``CCGRAM_WINDOW_ID`` env var or tmux fallback.

Key entry point: msg_group (Click group registered in cli.py).

Subcommand bodies lazy-load their workers (``msg_discovery``,
``spawn_request``); ``ccgram msg --help`` only walks the Click tree
and avoids materialising peer-discovery state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import click
import structlog

from .mailbox import Mailbox, Message
from .utils import ccgram_dir, tmux_session_name

if TYPE_CHECKING:
    from .msg_discovery import PeerInfo, WindowInfo

_CONTEXT_CACHE: dict[str, str] | None = None

logger = structlog.get_logger()

_RATE_WINDOW_SECONDS = 300  # 5 minutes


def _get_mailbox_dir() -> Path:
    return ccgram_dir() / "mailbox"


def _infer_tmux_session() -> str:
    """Infer the tmux session name for qualifying bare window IDs.

    Uses CCGRAM_WINDOW_ID session prefix (most reliable in agent context),
    falls back to TMUX_SESSION_NAME env var, then 'ccgram' default.
    """
    env_id = os.environ.get("CCGRAM_WINDOW_ID", "")
    if env_id and ":" in env_id:
        return env_id.rsplit(":", 1)[0]
    return tmux_session_name()


def _get_my_window_id() -> str:
    """Resolve this window's qualified ID.

    Priority: CCGRAM_WINDOW_ID env var > tmux runtime detection.
    """
    env_id = os.environ.get("CCGRAM_WINDOW_ID", "")
    if env_id:
        return env_id

    tmux_pane = os.environ.get("TMUX_PANE", "")
    if not tmux_pane:
        click.echo(
            "Error: not in a tmux session and CCGRAM_WINDOW_ID not set", err=True
        )
        sys.exit(1)

    try:
        result = subprocess.run(
            [
                "tmux",
                "display-message",
                "-p",
                "-t",
                tmux_pane,
                "#{session_name}:#{window_id}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError, subprocess.TimeoutExpired:
        pass

    click.echo("Error: could not detect window ID from tmux", err=True)
    sys.exit(1)


def _build_message_context(my_id: str) -> dict[str, str]:
    """Build context dict for outbound messages (window name, branch)."""
    global _CONTEXT_CACHE  # noqa: PLW0603
    if _CONTEXT_CACHE is not None:
        return _CONTEXT_CACHE

    ctx: dict[str, str] = {}
    states = _load_window_states()
    # Try qualified ID first, then bare ID (state.json keys are bare local IDs).
    # Only fall back to bare ID for local-session windows.
    ws = states.get(my_id)
    if ws is None and ":" in my_id:
        session_prefix = my_id.rsplit(":", 1)[0]
        if session_prefix == _infer_tmux_session():
            bare_id = my_id.rsplit(":", 1)[-1]
            ws = states.get(bare_id)
    if ws:
        if ws.window_name:
            ctx["window_name"] = ws.window_name
        if ws.cwd:
            # Lazy: msg_discovery ↔ msg_cmd cycle through CLI registration.
            from .msg_discovery import detect_branch

            branch = detect_branch(ws.cwd)
            if branch:
                ctx["branch"] = branch
    _CONTEXT_CACHE = ctx
    return ctx


def _load_window_states() -> dict[str, WindowInfo]:
    """Load window states from state.json."""
    # Lazy: msg_discovery ↔ msg_cmd cycle through CLI registration
    from .msg_discovery import export_window_info

    return export_window_info()


def _format_peers_table(peers: list[PeerInfo]) -> str:
    if not peers:
        return "No peers found."
    headers = ["ID", "Name", "Provider", "CWD", "Branch", "Task", "Team"]
    rows = [
        [p.window_id, p.name, p.provider, p.cwd, p.branch, p.task or "-", p.team or "-"]
        for p in peers
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows)
    return f"{header_line}\n{sep}\n{body}"


def _format_inbox_table(messages: list[Message]) -> str:
    if not messages:
        return "Inbox empty."
    headers = ["ID", "From", "Type", "Subject", "Status", "Created"]
    rows = [
        [m.id, m.from_id, m.type, m.subject or "-", m.status, m.created_at[:19]]
        for m in messages
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows)
    return f"{header_line}\n{sep}\n{body}"


def _check_rate_limit(mailbox: Mailbox, window_id: str, limit: int) -> bool:
    """Return True if the window has NOT exceeded its send rate limit."""
    cutoff = time.time() - _RATE_WINDOW_SECONDS
    count = 0
    base_dir = mailbox.base_dir
    skip_dirs = {"tmp", "spawns"}
    for entry in base_dir.iterdir() if base_dir.exists() else []:
        if not entry.is_dir() or entry.name in skip_dirs:
            continue
        for msg_file in entry.iterdir():
            if not msg_file.name.endswith(".json"):
                continue
            try:
                msg = Message.from_dict(json.loads(msg_file.read_text()))
            except json.JSONDecodeError, OSError, KeyError:
                continue
            if msg.from_id == window_id:
                try:
                    ts = float(msg.id.split("-")[0]) / 1e9
                except ValueError, IndexError:
                    continue
                if ts >= cutoff:
                    count += 1
    return count < limit


def _get_msg_rate_limit() -> int:
    return int(os.environ.get("CCGRAM_MSG_RATE_LIMIT", "10"))


def _get_wait_timeout() -> int:
    return int(os.environ.get("CCGRAM_MSG_WAIT_TIMEOUT", "60"))


def _wait_for_reply(mailbox: Mailbox, my_id: str, msg_id: str, wait_file: Path) -> None:
    """Block until a reply to msg_id arrives or timeout expires."""
    timeout = _get_wait_timeout()
    deadline = time.monotonic() + timeout
    click.echo(f"Sent {msg_id}, waiting for reply (timeout: {timeout}s)...")
    try:
        wait_file.write_text(msg_id)
        while time.monotonic() < deadline:
            messages = mailbox.inbox(my_id)
            for m in messages:
                if m.reply_to == msg_id:
                    click.echo(f"Reply from {m.from_id}: {m.body}")
                    sys.exit(0)
            time.sleep(1)
        click.echo("Error: wait timeout — no reply received", err=True)
        sys.exit(1)
    finally:
        wait_file.unlink(missing_ok=True)


@click.group("msg", help="Inter-agent messaging commands.")
def msg_group() -> None:
    pass


@msg_group.command("list-peers")
@click.option(
    "--json-output", "--json", "as_json", is_flag=True, help="Output as JSON."
)
def list_peers_cmd(as_json: bool) -> None:
    """List all known peer agent windows."""
    # Lazy: msg_discovery ↔ msg_cmd cycle through CLI registration
    from .msg_discovery import list_peers

    window_states = _load_window_states()
    session = _infer_tmux_session()
    declared_path = _get_mailbox_dir() / "declared.json"
    peers = list_peers(
        window_states=window_states,
        tmux_session=session,
        declared_path=declared_path,
    )
    if as_json:
        click.echo(json.dumps([asdict(p) for p in peers], indent=2))
    else:
        click.echo(_format_peers_table(peers))


@msg_group.command("find")
@click.option("--provider", default=None, help="Filter by provider name.")
@click.option("--team", default=None, help="Filter by team.")
@click.option("--cwd", default=None, help="Filter by cwd glob pattern.")
@click.option(
    "--json-output", "--json", "as_json", is_flag=True, help="Output as JSON."
)
def find_cmd(
    provider: str | None, team: str | None, cwd: str | None, as_json: bool
) -> None:
    """Find peers matching filters."""
    # Lazy: msg_discovery ↔ msg_cmd cycle through CLI registration
    from .msg_discovery import list_peers

    window_states = _load_window_states()
    session = _infer_tmux_session()
    declared_path = _get_mailbox_dir() / "declared.json"
    peers = list_peers(
        window_states=window_states,
        tmux_session=session,
        declared_path=declared_path,
        filter_provider=provider,
        filter_team=team,
        filter_cwd=cwd,
    )
    if as_json:
        click.echo(json.dumps([asdict(p) for p in peers], indent=2))
    else:
        click.echo(_format_peers_table(peers))


@msg_group.command("send")
@click.argument("to")
@click.argument("body")
@click.option("--subject", "-s", default="", help="Message subject.")
@click.option("--notify", is_flag=True, help="Send as notify type (fire-and-forget).")
@click.option("--wait", "wait_reply", is_flag=True, help="Block until reply received.")
@click.option("--ttl", type=int, default=None, help="TTL in minutes.")
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True),
    default=None,
    help="Attach file.",
)
def send_cmd(
    to: str,
    body: str,
    subject: str,
    notify: bool,
    wait_reply: bool,
    ttl: int | None,
    file_path: str | None,
) -> None:
    """Send a message to a peer window."""
    if ":" not in to:
        to = f"{_infer_tmux_session()}:{to}"
    my_id = _get_my_window_id()
    mailbox = Mailbox(_get_mailbox_dir())
    rate_limit = _get_msg_rate_limit()

    if not _check_rate_limit(mailbox, my_id, rate_limit):
        click.echo(
            f"Error: rate limit exceeded ({rate_limit} messages per 5 min)", err=True
        )
        sys.exit(1)

    wait_file: Path | None = None
    if wait_reply:
        wait_file = _get_mailbox_dir() / f".waiting-{my_id.replace(':', '-')}"
        wait_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(wait_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            click.echo(
                "Error: already waiting for a reply (one outstanding --wait per window)",
                err=True,
            )
            sys.exit(1)

    msg_type = "notify" if notify else "request"

    kwargs: dict = {
        "from_id": my_id,
        "to_id": to,
        "body": body,
        "msg_type": msg_type,
        "subject": subject,
        "context": _build_message_context(my_id),
    }
    if ttl is not None:
        kwargs["ttl_minutes"] = ttl
    if file_path is not None:
        kwargs["file_path"] = file_path

    try:
        msg = mailbox.send(**kwargs)
    except (ValueError, FileNotFoundError, OSError) as exc:
        if wait_file is not None:
            wait_file.unlink(missing_ok=True)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if wait_file is not None:
        _wait_for_reply(mailbox, my_id, msg.id, wait_file)
    else:
        click.echo(f"Sent {msg.id}")


@msg_group.command("inbox")
@click.option(
    "--json-output", "--json", "as_json", is_flag=True, help="Output as JSON."
)
def inbox_cmd(as_json: bool) -> None:
    """Show pending messages in this window's inbox."""
    my_id = _get_my_window_id()
    mailbox = Mailbox(_get_mailbox_dir())
    messages = mailbox.inbox(my_id)
    if as_json:
        click.echo(json.dumps([m.to_dict() for m in messages], indent=2))
    else:
        click.echo(_format_inbox_table(messages))


@msg_group.command("read")
@click.argument("msg_id")
def read_cmd(msg_id: str) -> None:
    """Mark a message as read and display it."""
    my_id = _get_my_window_id()
    mailbox = Mailbox(_get_mailbox_dir())
    msg = mailbox.read(msg_id, my_id)
    if msg is None:
        click.echo(f"Error: message {msg_id} not found", err=True)
        sys.exit(1)
    click.echo(f"From: {msg.from_id}")
    click.echo(f"Type: {msg.type}")
    if msg.subject:
        click.echo(f"Subject: {msg.subject}")
    click.echo(f"Body: {msg.body}")


@msg_group.command("reply")
@click.argument("msg_id")
@click.argument("body")
def reply_cmd(msg_id: str, body: str) -> None:
    """Reply to a message."""
    my_id = _get_my_window_id()
    mailbox = Mailbox(_get_mailbox_dir())
    reply_msg = mailbox.reply(msg_id, my_id, body)
    if reply_msg is None:
        click.echo(f"Error: message {msg_id} not found in inbox", err=True)
        sys.exit(1)
    click.echo(f"Replied {reply_msg.id}")


@msg_group.command("broadcast")
@click.argument("body")
@click.option("--subject", "-s", default="", help="Message subject.")
@click.option("--team", default=None, help="Filter recipients by team.")
@click.option("--provider", default=None, help="Filter recipients by provider.")
@click.option("--cwd", default=None, help="Filter recipients by cwd glob pattern.")
@click.option("--ttl", type=int, default=None, help="TTL in minutes (default: 480).")
def broadcast_cmd(
    body: str,
    subject: str,
    team: str | None,
    provider: str | None,
    cwd: str | None,
    ttl: int | None,
) -> None:
    """Broadcast a message to all matching peers."""
    # Lazy: msg_discovery ↔ msg_cmd cycle through CLI registration
    from .msg_discovery import list_peers

    my_id = _get_my_window_id()
    mailbox = Mailbox(_get_mailbox_dir())
    rate_limit = _get_msg_rate_limit()

    if not _check_rate_limit(mailbox, my_id, rate_limit):
        click.echo(
            f"Error: rate limit exceeded ({rate_limit} messages per 5 min)", err=True
        )
        sys.exit(1)

    window_states = _load_window_states()
    session = _infer_tmux_session()
    declared_path = _get_mailbox_dir() / "declared.json"

    peers = list_peers(
        window_states=window_states,
        tmux_session=session,
        declared_path=declared_path,
        filter_provider=provider,
        filter_team=team,
        filter_cwd=cwd,
    )

    recipient_ids = [p.window_id for p in peers if p.window_id != my_id]
    if not recipient_ids:
        click.echo("No matching recipients.")
        return

    sent = mailbox.broadcast(
        from_id=my_id,
        recipient_ids=recipient_ids,
        body=body,
        subject=subject,
        ttl_minutes=ttl if ttl is not None else None,
        context=_build_message_context(my_id),
    )

    click.echo(f"Broadcast to {len(sent)} recipient(s).")


@msg_group.command("register")
@click.option("--task", default=None, help="Current task description.")
@click.option("--team", default=None, help="Team name.")
def register_cmd(task: str | None, team: str | None) -> None:
    """Register self-declared metadata (task, team)."""
    # Lazy: msg_discovery ↔ msg_cmd cycle through CLI registration
    from .msg_discovery import register_declared

    if task is None and team is None:
        click.echo("Error: at least one of --task or --team required", err=True)
        sys.exit(1)
    my_id = _get_my_window_id()
    declared_path = _get_mailbox_dir() / "declared.json"
    register_declared(my_id, task=task, team=team, path=declared_path)
    parts = []
    if task is not None:
        parts.append(f"task={task!r}")
    if team is not None:
        parts.append(f"team={team!r}")
    click.echo(f"Registered {', '.join(parts)}")


@msg_group.command("spawn")
@click.option("--provider", "-p", default="claude", help="Provider for the new agent.")
@click.option("--cwd", "-d", default=None, help="Working directory for the new agent.")
@click.option("--prompt", required=True, help="Initial prompt for the new agent.")
@click.option(
    "--context",
    "context_file",
    type=click.Path(exists=True),
    default=None,
    help="Context file to attach to the spawn.",
)
@click.option("--auto", is_flag=True, help="Bypass approval (respects rate limits).")
def spawn_cmd(
    provider: str,
    cwd: str | None,
    prompt: str,
    context_file: str | None,
    auto: bool,
) -> None:
    """Request spawning a new agent window."""
    # Lazy: spawn request module pulls PTB indirectly
    from .spawn_request import (
        check_max_windows,
        check_spawn_rate,
        create_spawn_request,
    )

    my_id = _get_my_window_id()
    window_states = _load_window_states()

    max_windows = int(os.environ.get("CCGRAM_MSG_MAX_WINDOWS", "10"))
    if not check_max_windows(window_states, max_windows):
        click.echo(
            f"Error: max windows reached ({max_windows})",
            err=True,
        )
        sys.exit(1)

    spawn_rate = int(os.environ.get("CCGRAM_MSG_SPAWN_RATE", "3"))
    if not check_spawn_rate(my_id, spawn_rate):
        click.echo(
            f"Error: spawn rate limit exceeded ({spawn_rate} per hour)",
            err=True,
        )
        sys.exit(1)

    work_dir = cwd or os.getcwd()

    try:
        req = create_spawn_request(
            requester_window=my_id,
            provider=provider,
            cwd=work_dir,
            prompt=prompt,
            context_file=context_file,
            auto=auto,
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if auto:
        click.echo(f"Spawn request {req.id} created (auto-approve mode)")
    else:
        click.echo(f"Spawn request {req.id} created (awaiting Telegram approval)")


@msg_group.command("sweep")
def sweep_cmd() -> None:
    """Clean up expired and old messages."""
    mailbox = Mailbox(_get_mailbox_dir())
    removed = mailbox.sweep()
    click.echo(f"Swept {removed} message(s).")
