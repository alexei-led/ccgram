"""``ccgram admin`` CLI: host-side topic lifecycle commands (TASK-128).

Appends an admin command for the bridge to execute and prints the
result. Every identifier is an explicit argument; nothing is inferred
from names. The channel never touches Telegram polling: the bridge
consumes the command file on its own loop and replies through the
results file.
"""

from __future__ import annotations

import sys

import click

from .admin import submit_admin_command, wait_for_result


def _run(command: str, **args: object) -> None:
    command_id = submit_admin_command(command, **args)
    click.echo(f"submitted {command} ({command_id}), waiting for the bridge…")
    result = wait_for_result(command_id)
    if result is None:
        click.echo(
            "no response from the bridge within the timeout; is ccgram "
            "running? the command stays queued"
        )
        sys.exit(2)
    click.echo(f"{'ok' if result.get('ok') else 'FAILED'}: {result.get('detail')}")
    sys.exit(0 if result.get("ok") else 1)


@click.group("admin")
def admin_cli() -> None:
    """Host-side topic lifecycle commands (needs the bridge running)."""


@admin_cli.command("bind")
@click.option("--user-id", required=True, type=int)
@click.option("--chat-id", required=True, type=int)
@click.option("--thread-id", required=True, type=int)
@click.option("--window", "window_id", required=True)
def admin_bind(user_id: int, chat_id: int, thread_id: int, window_id: str) -> None:
    """Bind a topic to a live window id."""
    _run(
        "bind",
        user_id=user_id,
        chat_id=chat_id,
        thread_id=thread_id,
        window_id=window_id,
    )


@admin_cli.command("unbind")
@click.option("--user-id", required=True, type=int)
@click.option("--chat-id", required=True, type=int)
@click.option("--thread-id", required=True, type=int)
@click.option("--delete", "delete_topic", is_flag=True)
def admin_unbind(
    user_id: int, chat_id: int, thread_id: int, delete_topic: bool
) -> None:
    """Unbind a topic; keeps the topic unless --delete is given."""
    _run(
        "unbind",
        user_id=user_id,
        chat_id=chat_id,
        thread_id=thread_id,
        delete_topic=delete_topic,
    )


@admin_cli.command("rethread")
@click.option("--user-id", required=True, type=int)
@click.option("--chat-id", required=True, type=int)
@click.option("--from-thread", required=True, type=int)
@click.option("--to-thread", required=True, type=int)
def admin_rethread(
    user_id: int, chat_id: int, from_thread: int, to_thread: int
) -> None:
    """Move a topic binding to another topic id."""
    _run(
        "rethread",
        user_id=user_id,
        chat_id=chat_id,
        from_thread=from_thread,
        to_thread=to_thread,
    )


@admin_cli.command("sync")
def admin_sync() -> None:
    """Run the /sync audit and cleanup headless."""
    _run("sync")
