"""Run fresh, follow-up and resumed agent turns through an isolated Telegram sink."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
import itertools
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import tempfile
import time

USER_ID = 999001
THREAD_ID = 42
FAKE_TOKEN = "123456:isolated-smoke-not-a-real-token"


def control(*args: str) -> dict:
    window = os.environ.get("AGTERM_WINDOW_ID")
    addressing = ["--window", window] if window else []
    result = subprocess.run(
        ["agtermctl", *args, *addressing, "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"agterm {' '.join(args)} failed: {result.stderr.strip()}")
    response = json.loads(result.stdout)
    if not response.get("ok"):
        raise RuntimeError(f"agterm rejected smoke operation: {response.get('error')}")
    return response["result"]


async def ready_identity(service, target_id, config, timeout):
    from ccgram.session_map import session_map_sync

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if config.session_map_file.exists():
            raw = json.loads(config.session_map_file.read_text(encoding="utf-8"))
            matches = {
                key: value
                for key, value in raw.items()
                if key.casefold() == ("agterm:" + target_id).casefold()
            }
            if matches:
                if len(raw) != 1:
                    raise RuntimeError("Smoke config contains a foreign target")
                await session_map_sync.load_session_map(raw)
                target, snapshot = await service.for_target(target_id)
                if (
                    target.foreground_verified
                    and "message.send" in snapshot.capabilities
                ):
                    return next(iter(matches.values())), target, snapshot
        await asyncio.sleep(0.2)
    raise RuntimeError("No attested ready hook identity; input was not sent")


def acknowledged_reply(mapping, nonce, monitor):
    from ccgram.session_protocol.messages import parse_records, read_entries

    tracked = monitor.state.get_session(mapping["session_id"])
    if tracked is None or not tracked.file_path:
        return None
    source = Path(tracked.file_path)
    if not source.is_file():
        return None
    data = source.read_bytes()
    entries = read_entries(data, mapping["provider_name"])
    records, _ = parse_records(entries, mapping["provider_name"])
    reply_entries = [
        entry
        for entry, record in records
        if record.role == "agent"
        and any(part.get("text") == nonce for part in record.parts)
    ]
    if not reply_entries:
        return None
    start = reply_entries[-1].offset
    end = next((entry.offset for entry in entries if entry.offset > start), len(data))
    if tracked.last_byte_offset < end:
        return None
    return {
        "reply_end": end,
        "watermark": tracked.last_byte_offset,
        "internal_request_records": sum(
            entry.data.get("type") == "api-request-blob" for entry in entries
        ),
    }


async def send_turn(
    adapter, service, monitor, client, mapping, target_id, session_id, timeout
):
    from ccgram.handlers.telegram_origin import send_telegram_to_window

    nonce = "CCGRAM_SMOKE_" + secrets.token_hex(6)
    prompt = (
        "Reply with exactly " + nonce + ". Do not use tools or read or write files."
    )
    call_start = len(client.calls)
    success, _ = await send_telegram_to_window(
        USER_ID,
        target_id,
        THREAD_ID,
        prompt,
        USER_ID,
        adapter=adapter,
        action_id="smoke:" + nonce,
    )
    if not success:
        raise RuntimeError("Core did not confirm input; no retry permitted")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        monitor.commit_delivered_watermarks()
        sent = [
            call.kwargs["text"]
            for call in client.calls[call_start:]
            if call.method == "send_message"
        ]
        history = await service.history(session_id, None, 100)
        replies = [
            part.text
            for message in history.messages
            if message.role == "agent"
            for part in message.parts
            if part.type == "text"
        ]
        if nonce in replies and nonce in sent:
            receipt = acknowledged_reply(mapping, nonce, monitor)
            if receipt is not None:
                if sent != [nonce]:
                    raise RuntimeError(
                        "Telegram received unexpected conversation content"
                    )
                return {
                    "input_confirmed": True,
                    "native_reply": True,
                    "simulated_telegram_ack": True,
                    **receipt,
                }
        await asyncio.sleep(0.2)
    raise RuntimeError("Expected native reply and Telegram receipt were not observed")


async def relay_turns(backend, store, mapping, target_id, client, timeout):
    from ccgram.handlers.messaging_pipeline.message_queue import shutdown_workers
    from ccgram.handlers.messaging_pipeline.message_routing import handle_new_message
    from ccgram.session_monitor import SessionMonitor
    from ccgram.session_protocol.service import LocalSessionService
    from ccgram.session_protocol.telegram import TelegramSessionAdapter

    async def deliver(message):
        await handle_new_message(message, client)

    service = LocalSessionService(backend, store)
    monitor = SessionMonitor(poll_interval=0.2, event_hub=service.hub)
    monitor.set_message_callback(deliver)
    monitor.start()
    turns = []
    try:
        for phase in ("initial", "followup", "after-service-restart"):
            if phase == "after-service-restart":
                await monitor.stop_and_wait()
                await shutdown_workers()
                monitor.commit_delivered_watermarks()
                old_epoch = service.hub.epoch
                service = LocalSessionService(backend, store)
                if service.hub.replay(old_epoch, 0) != (False, []):
                    raise RuntimeError("Old event epoch survived service restart")
                monitor = SessionMonitor(poll_interval=0.2, event_hub=service.hub)
                monitor.set_message_callback(deliver)
                monitor.start()
            _, snapshot = await service.for_target(target_id)
            result = await send_turn(
                TelegramSessionAdapter(service),
                service,
                monitor,
                client,
                mapping,
                target_id,
                snapshot.id,
                timeout,
            )
            turns.append({"phase": phase, **result})
        return turns
    finally:
        await monitor.stop_and_wait()
        await shutdown_workers(drain_timeout=1)
        monitor.commit_delivered_watermarks()


async def exercise(args, state, workspace, workspace_id, executable):
    # Import only after isolated config and the fake Telegram identity are set.
    import ccgram.session  # noqa: F401
    from ccgram.config import config
    from ccgram.multiplexer import install_multiplexer
    from ccgram.multiplexer.agterm import AgtermManager
    from ccgram.session_protocol.backend import MultiplexerSessionBackend
    from ccgram.session_protocol.service import LocalSessionService
    from ccgram.session_protocol.store import ProtocolStore
    from ccgram.telegram_client import FakeTelegramClient
    from telegram import Chat, Message
    from ccgram.thread_router import thread_router

    if config.config_dir != state or config.telegram_bot_token != FAKE_TOKEN:
        raise RuntimeError("Smoke isolation failed; refusing to create a session")
    name = workspace + "-" + args.provider
    command = [
        "/usr/bin/env",
        "-u",
        "VIRTUAL_ENV",
        "-u",
        "PI_SUBAGENT_CHILD",
        "CCGRAM_DIR=" + str(state),
        "TELEGRAM_BOT_TOKEN=" + FAKE_TOKEN,
        "ALLOWED_USERS=" + str(USER_ID),
        "CCGRAM_MULTIPLEXER=agterm",
        "CCGRAM_AGTERM_WORKSPACES=" + workspace,
        "PATH=" + os.environ["PATH"],
        executable,
    ]
    mux = AgtermManager(workspaces=(workspace,))
    install_multiplexer(mux)
    store = ProtocolStore(state / "smoke.sqlite3")
    backend = MultiplexerSessionBackend(mux, str(args.cwd))
    service = LocalSessionService(backend, store)
    message_ids = itertools.count(1)
    client = FakeTelegramClient(
        returns={
            "send_message": lambda chat_id, text, **_kwargs: Message(
                message_id=next(message_ids),
                date=datetime.now(UTC),
                chat=Chat(id=chat_id, type="private"),
                text=text,
            ),
        }
    )
    target_id = ""
    try:
        target_id = control(
            "session",
            "new",
            "--workspace",
            workspace_id,
            "--cwd",
            str(args.cwd),
            "--name",
            name,
            "--command",
            shlex.join(command),
            "--no-select",
            "--wait",
        )["id"]
        manifest = {"target_id": target_id, "workspace": workspace, "state": str(state)}
        (state / "owned-session.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        mapping, target, _ = await ready_identity(
            service, target_id, config, args.timeout
        )
        if (
            mapping["provider_name"] != args.provider
            or mapping["session_id"] != target.conversation_id
        ):
            raise RuntimeError("Attested source and live target identity disagree")
        thread_router.bind_thread(USER_ID, THREAD_ID, target_id, chat_id=USER_ID)

        turns = await relay_turns(
            backend, store, mapping, target_id, client, args.timeout
        )
        return {
            "provider": args.provider,
            **manifest,
            "production_bot_requests": 0,
            "turns": turns,
            "telegram_messages": client.call_count("send_message"),
        }
    finally:
        store.close()
        await close_owned(mux, target_id, name, workspace_id)


async def close_owned(mux, target_id, name, workspace_id):
    windows = await mux.list_windows_for_reconciliation()
    if windows is None:
        raise RuntimeError("Cannot confirm smoke ownership for cleanup")
    owned = next((w for w in windows if target_id and w.matches(target_id)), None)
    if owned and (owned.window_name != name or not await mux.kill_window(target_id)):
        raise RuntimeError("Owned smoke session could not safely be closed")
    tree = control("tree").get("tree")
    workspaces = tree.get("workspaces") if isinstance(tree, dict) else None
    if not isinstance(workspaces, list):
        raise RuntimeError("Cannot confirm smoke workspace for cleanup")
    workspace = next(
        (item for item in workspaces if item.get("id") == workspace_id), None
    )
    if workspace is None:
        return
    sessions = workspace.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("Cannot confirm smoke workspace sessions")
    if not sessions:
        control("workspace", "delete", "--target", workspace_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("claude", "pi", "codex"), required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    args.cwd = args.cwd.resolve(strict=True)
    if os.environ.get("AGTERM_ENABLED") != "1":
        parser.error("Run inside agterm")
    executable = shutil.which(args.provider)
    if not executable:
        parser.error("Requested provider CLI is not installed")
    os.umask(0o077)
    state = Path(tempfile.mkdtemp(prefix="ccgram-relay-smoke-"))
    workspace = "ccgram-smoke-" + secrets.token_hex(5)
    os.environ.update(
        {
            "CCGRAM_DIR": str(state),
            "TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
            "ALLOWED_USERS": str(USER_ID),
            "CCGRAM_MULTIPLEXER": "agterm",
            "CCGRAM_AGTERM_WORKSPACES": workspace,
        }
    )
    workspace_id = control("workspace", "new", workspace, "--collapsed")["id"]
    from ccgram.session_protocol.models import ProtocolError

    try:
        report = asyncio.run(exercise(args, state, workspace, workspace_id, executable))
    except (
        AssertionError,
        OSError,
        RuntimeError,
        ValueError,
        ProtocolError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            json.dumps(
                {
                    "provider": args.provider,
                    "state": str(state),
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        )
        raise SystemExit(1) from None
    result = {"passed": True, **report}
    (state / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
