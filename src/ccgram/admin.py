"""Host-side admin surface for topic lifecycle (TASK-128).

Today topic lifecycle (bind, unbind, sync, rethread) exists only as
interactive Telegram commands, so recovering a topic whose window
changed identity means driving Telegram by hand. This module adds a
file-based admin channel with the same trust domain as the other
state files: a CLI process appends one JSON command to
``~/.ccgram/admin_commands.jsonl``; the bridge consumes it on its poll
loop, executes it through the REAL code paths (thread_router,
topic_deletion, the /sync audit), and appends a result record to
``~/.ccgram/admin_results.jsonl``. The channel never touches
getUpdates or polling: the bridge side only reads files and calls the
same internals the Telegram handlers call.

Explicit arguments only: every command names user, chat, and thread
ids verbatim. Nothing is inferred from names.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from .utils import ccgram_dir

logger = structlog.get_logger()

COMMANDS_FILE = "admin_commands.jsonl"
RESULTS_FILE = "admin_results.jsonl"
RESULT_WAIT_TIMEOUT_S = 30.0
_KNOWN_COMMANDS = {"bind", "unbind", "sync", "rethread"}


def _commands_path() -> Path:
    return ccgram_dir() / COMMANDS_FILE


def current_command_offset() -> int:
    """The byte offset a fresh consumer should start from (EOF).

    Commands from before the consumer started have no requester waiting
    (the CLI times out in 30s), and replaying them would re-run real
    lifecycle mutations on every restart.
    """
    try:
        return _commands_path().stat().st_size
    except OSError:
        return 0


def _results_path() -> Path:
    return ccgram_dir() / RESULTS_FILE


def submit_admin_command(command: str, **args: Any) -> str:
    """Append one admin command for the bridge to execute.

    Returns the command id the caller can wait on. No validation beyond
    the command name: argument validation is the executor's job so the
    reason lands in the result record the operator reads.
    """
    if command not in _KNOWN_COMMANDS:
        raise ValueError(f"unknown admin command: {command}")
    record = {
        "id": uuid.uuid4().hex,
        "command": command,
        "args": args,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = _commands_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as commands_f:
        commands_f.write(json.dumps(record) + "\n")
    return record["id"]


def read_new_commands(path: Path, offset: int) -> tuple[list[dict], int]:
    """Incremental read of the command file; malformed lines are skipped."""
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    start = offset
    if size < offset:
        # Truncated or rotated: re-read the retained lines from the start;
        # the consumer's id dedup skips the ones already executed, so a
        # command appended into the shorter file before the next poll is
        # still picked up instead of being lost past the stale offset.
        start = 0
    try:
        with open(path) as commands_f:
            commands_f.seek(start)
            lines = commands_f.read().splitlines()
            new_offset = commands_f.tell()
    except OSError, UnicodeDecodeError:
        return [], offset
    records: list[dict] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("admin command line is not JSON, skipping")
            continue
        if isinstance(record, dict) and record.get("id"):
            records.append(record)
    return records, new_offset


def append_admin_result(record: dict) -> None:
    path = _results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as results_f:
        results_f.write(json.dumps(record) + "\n")


def wait_for_result(
    command_id: str, timeout: float = RESULT_WAIT_TIMEOUT_S
) -> dict | None:
    """Block until the bridge posts this command's result, or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lines = _results_path().read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("id") == command_id:
                return record
        time.sleep(0.5)
    return None


def _result(command_id: str, command: str, ok: bool, detail: str, **extra: Any) -> dict:
    record = {
        "id": command_id,
        "command": command,
        "ok": ok,
        "detail": detail,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    record.update(extra)
    return record


async def execute_admin_command(record: dict, client: Any) -> dict:
    """Dispatch one command through the real lifecycle code paths."""
    command = record.get("command")
    args = record.get("args")
    if not isinstance(args, dict):
        args = {}
    command_id = str(record.get("id", ""))
    try:
        if command == "bind":
            return await _cmd_bind(command_id, args)
        if command == "unbind":
            return await _cmd_unbind(command_id, args, client)
        if command == "rethread":
            return await _cmd_rethread(command_id, args)
        if command == "sync":
            return await _cmd_sync(command_id, client)
        return _result(command_id, str(command), False, "unknown command")
    except Exception as exc:  # noqa: BLE001  # the result record carries it
        logger.exception("admin command failed", command=command)
        return _result(command_id, str(command), False, f"error: {exc}")


def _binding_owner_mismatch(user_id: int, chat_id: int, thread_id: int) -> bool:
    """True when the (chat, thread) binding exists but belongs to another user.

    The chat-scoped lookup ignores the owner, so a mistyped user id must
    not mutate someone else's binding.
    """
    # Lazy: keeps the CLI import path free of the runtime stack
    from .thread_router import thread_router

    for (
        owner,
        bound_chat,
        bound_thread,
        _wid,
    ) in thread_router.iter_thread_bindings_with_chat():
        if bound_chat == chat_id and bound_thread == thread_id:
            return owner != user_id
    return False


def _required_ints(args: dict, names: list[str]) -> tuple[dict, str | None]:
    """Return the named integer arguments, or an error message."""
    values: dict[str, int] = {}
    for name in names:
        raw = args.get(name)
        try:
            values[name] = int(raw)  # type: ignore[arg-type]
        except TypeError, ValueError:
            return {}, f"missing or invalid argument: {name}"
    return values, None


async def _cmd_bind(command_id: str, args: dict) -> dict:
    values, error = _required_ints(args, ["user_id", "chat_id", "thread_id"])
    if error:
        return _result(command_id, "bind", False, error)
    window_id = args.get("window_id")
    if not isinstance(window_id, str) or not window_id:
        return _result(command_id, "bind", False, "missing argument: window_id")

    # Lazy: multiplexer proxy import keeps this module CLI-importable.
    from .multiplexer import multiplexer

    # Lazy: keeps the CLI import path free of the runtime stack
    from .multiplexer.reconciliation import window_presence

    # Lazy: keeps the CLI import path free of the runtime stack
    from .thread_router import thread_router

    presence = await window_presence(window_id, multiplexer)
    if presence is not True:
        # Confirmed dead OR unknown (backend unreachable, or an id outside
        # this backend's namespace, which is how a truncated id looks):
        # refuse. An explicit-request bind must still name a window the
        # backend actually attests (2026-09-22: a truncated digest bound
        # through the previous unknown-passes hole).
        return _result(
            command_id,
            "bind",
            False,
            f"window {window_id} is not confirmed live (presence: {presence})",
        )
    previous = thread_router.get_window_for_chat_thread(
        values["chat_id"], values["thread_id"]
    )
    thread_router.bind_thread(
        values["user_id"],
        values["thread_id"],
        window_id,
        chat_id=values["chat_id"],
    )
    detail = f"topic {values['thread_id']} bound to {window_id}"
    if previous and previous != window_id:
        detail += f" (was {previous})"
    if presence is None:
        # Tri-state liveness: None is an unknown verdict (backend
        # unreachable or the id outside this backend's namespace), not a
        # confirmation. The explicit-argument contract still binds, but
        # the operator sees the uncertainty.
        detail += "; liveness UNKNOWN, bound on explicit request"
    return _result(command_id, "bind", True, detail, window_id=window_id)


async def _cmd_unbind(command_id: str, args: dict, client: Any) -> dict:
    values, error = _required_ints(args, ["user_id", "chat_id", "thread_id"])
    if error:
        return _result(command_id, "unbind", False, error)
    # Lazy: keeps the CLI import path free of the runtime stack
    from .thread_router import thread_router

    window_id = thread_router.get_window_for_chat_thread(
        values["chat_id"], values["thread_id"]
    )
    if not window_id:
        return _result(
            command_id,
            "unbind",
            False,
            f"topic {values['thread_id']} is not bound",
        )
    if _binding_owner_mismatch(
        values["user_id"], values["chat_id"], values["thread_id"]
    ):
        return _result(
            command_id,
            "unbind",
            False,
            f"topic {values['thread_id']} belongs to another user",
        )
    delete_topic = bool(args.get("delete_topic", False))
    if delete_topic:
        # Lazy: topic_deletion pulls PTB types.
        from .handlers.topics.topic_deletion import retire_topic_binding

        outcome = await retire_topic_binding(
            client,
            values["user_id"],
            values["thread_id"],
            window_id,
            chat_id=values["chat_id"],
        )
        # Only an actual close (or the topic already being gone) counts
        # as success; protected, failed, and retryable outcomes surface.
        ok = outcome in {"closed", "already_gone", "deleted"}
        return _result(
            command_id,
            "unbind",
            ok,
            f"topic {values['thread_id']} unbind --delete outcome: "
            f"{outcome} (was {window_id})",
        )
    thread_router.unbind_thread(
        values["user_id"],
        values["thread_id"],
        chat_id=values["chat_id"],
        retirement_reason="admin_unbind",
        cleanup_eligible=True,
    )
    return _result(
        command_id,
        "unbind",
        True,
        f"topic {values['thread_id']} unbound (was {window_id}); topic kept",
    )


async def _cmd_rethread(command_id: str, args: dict) -> dict:
    values, error = _required_ints(
        args, ["user_id", "chat_id", "from_thread", "to_thread"]
    )
    if error:
        return _result(command_id, "rethread", False, error)
    # Lazy: keeps the CLI import path free of the runtime stack
    from .thread_router import thread_router

    chat_id = values["chat_id"]
    from_thread = values["from_thread"]
    to_thread = values["to_thread"]
    window_id = thread_router.get_window_for_chat_thread(chat_id, from_thread)
    if not window_id:
        return _result(
            command_id, "rethread", False, f"topic {from_thread} is not bound"
        )
    if thread_router.get_window_for_chat_thread(chat_id, to_thread):
        return _result(
            command_id, "rethread", False, f"topic {to_thread} is already bound"
        )
    if _binding_owner_mismatch(values["user_id"], chat_id, from_thread):
        return _result(
            command_id,
            "rethread",
            False,
            f"topic {from_thread} belongs to another user",
        )
    thread_router.bind_thread(values["user_id"], to_thread, window_id, chat_id=chat_id)
    thread_router.unbind_thread(
        values["user_id"],
        from_thread,
        chat_id=chat_id,
        retirement_reason="admin_rethread",
        cleanup_eligible=False,
    )
    # Legacy (non-chat-scoped) from-bindings are neither evicted by the
    # chat-scoped bind nor seen by the chat-scoped unbind: sweep any one
    # of them that still points at the window, or the old topic keeps
    # forwarding.
    for user, thread, bound_wid in thread_router.iter_thread_bindings():
        if (
            user == values["user_id"]
            and thread == from_thread
            and bound_wid == window_id
        ):
            thread_router.unbind_thread(
                values["user_id"],
                from_thread,
                retirement_reason="admin_rethread_legacy",
                cleanup_eligible=False,
            )
            break
    return _result(
        command_id,
        "rethread",
        True,
        f"binding moved from topic {from_thread} to topic {to_thread} "
        f"(window {window_id})",
    )


async def _cmd_sync(command_id: str, client: Any) -> dict:
    """Headless /sync: same audit and cleanup, report as a result record."""
    # Lazy: handlers.sync_command pulls the handler stack.
    from .handlers.sync_command import (
        _cleanup_stale_topics,
        _retired_topic_issues,
        _run_audit,
    )

    audit = await _run_audit()
    if audit is None:
        return _result(command_id, "sync", False, "multiplexer unavailable")
    cleanup_issues = [*audit.issues, *_retired_topic_issues()]
    closed, manual, retired = await _cleanup_stale_topics(client, cleanup_issues)
    post_audit = await _run_audit()
    remaining = len(post_audit.issues) if post_audit is not None else -1
    retired_text = ", ".join(
        f"{count} {outcome}"
        for outcome, count in (retired.items() if isinstance(retired, dict) else {})
    )
    return _result(
        command_id,
        "sync",
        True,
        f"sync complete: {closed} topics closed, {manual} need manual "
        f"close ({retired_text}); {remaining} issues remain",
        closed=closed,
        manual_close=manual,
    )


# Executed-command ids, in-memory: the restart EOF-skip covers history,
# and this set makes a post-truncation re-read of retained lines a no-op.
_EXECUTED_COMMAND_IDS: set[str] = set()
_EXECUTED_COMMAND_IDS_CAP = 4096


async def consume_admin_commands(client: Any, offset: int = 0) -> int:
    """One pass: execute every pending command, append every result."""
    records, new_offset = read_new_commands(_commands_path(), offset)
    fresh = [r for r in records if r["id"] not in _EXECUTED_COMMAND_IDS]
    if len(_EXECUTED_COMMAND_IDS) > _EXECUTED_COMMAND_IDS_CAP:
        _EXECUTED_COMMAND_IDS.clear()
    for record in records:
        _EXECUTED_COMMAND_IDS.add(record["id"])
    for record in fresh:
        result = await execute_admin_command(record, client)
        try:
            append_admin_result(result)
        except OSError:
            # The command already ran; a failed reply must not kill the
            # consumer loop. The result is still in the log line below.
            logger.error(
                "admin result could not be persisted",
                command=result.get("command"),
                detail=result.get("detail"),
            )
        logger.info(
            "admin command executed",
            command=result.get("command"),
            ok=result.get("ok"),
            detail=result.get("detail"),
        )
    return new_offset
