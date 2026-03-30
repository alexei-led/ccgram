"""Agent spawn request handling with Telegram approval.

Manages spawn requests from agents: validation, rate limiting,
approval/denial flow, window creation, and topic auto-creation.
Uses callback_registry self-registration for inline keyboard dispatch.

Spawn requests are persisted to disk (``spawns/`` dir inside mailbox)
so the bot process can read requests written by CLI subprocesses.

Key components:
  - SpawnRequest: dataclass for pending spawn requests
  - create_spawn_request: validate and store a new request
  - handle_spawn_approval: create window + topic on approval
  - handle_spawn_denial: reject and clean up
  - scan_spawn_requests: read pending requests from disk (for broker)
  - Telegram callback handlers for [Approve] / [Deny] buttons
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..providers import resolve_launch_command
from ..session import session_manager
from ..tmux_manager import tmux_manager
from ..utils import ccgram_dir
from .callback_registry import register
from .message_sender import rate_limit_send_message

if TYPE_CHECKING:
    from telegram import Bot

logger = structlog.get_logger()

CB_SPAWN_APPROVE = "sp:ok:"
CB_SPAWN_DENY = "sp:no:"

_SPAWN_RATE_WINDOW_SECONDS = 3600  # 1 hour


@dataclass
class SpawnRequest:
    id: str
    requester_window: str
    provider: str
    cwd: str
    prompt: str
    context_file: str | None = None
    auto: bool = False
    created_at: float = field(default_factory=time.time)

    def is_expired(self, timeout: int = 300) -> bool:
        return time.time() - self.created_at > timeout

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SpawnRequest:
        return cls(
            id=data["id"],
            requester_window=data["requester_window"],
            provider=data.get("provider", "claude"),
            cwd=data["cwd"],
            prompt=data.get("prompt", ""),
            context_file=data.get("context_file"),
            auto=data.get("auto", False),
            created_at=data.get("created_at", 0.0),
        )


@dataclass
class SpawnResult:
    window_id: str
    window_name: str


# In-memory cache of requests loaded from disk (bot process only).
_pending_requests: dict[str, SpawnRequest] = {}


def _spawns_dir() -> Path:
    return ccgram_dir() / "mailbox" / "spawns"


def _spawn_timeout() -> int:
    from ..config import config

    return config.msg_spawn_timeout


def reset_spawn_state() -> None:
    _pending_requests.clear()


def clear_spawn_state(window_id: str) -> None:
    to_remove = [
        rid
        for rid, req in _pending_requests.items()
        if req.requester_window == window_id
    ]
    for rid in to_remove:
        del _pending_requests[rid]
    # Also clean up any spawn files from this requester
    sdir = _spawns_dir()
    if sdir.is_dir():
        for entry in sdir.iterdir():
            if not entry.name.endswith(".json"):
                continue
            try:
                data = json.loads(entry.read_text())
                if data.get("requester_window") == window_id:
                    entry.unlink(missing_ok=True)
            except json.JSONDecodeError, OSError:
                continue


def check_max_windows(
    window_states: dict,
    max_windows: int,
) -> bool:
    return len(window_states) < max_windows


def _load_spawn_log() -> dict[str, list[float]]:
    """Load spawn rate log from disk."""
    path = _spawns_dir() / "rate_log.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError, OSError:
            return {}
    return {}


def _save_spawn_log(log: dict[str, list[float]]) -> None:
    """Save spawn rate log to disk."""
    sdir = _spawns_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / "rate_log.json"
    path.write_text(json.dumps(log))


def check_spawn_rate(window_id: str, max_rate: int) -> bool:
    log = _load_spawn_log()
    cutoff = time.time() - _SPAWN_RATE_WINDOW_SECONDS
    timestamps = log.get(window_id, [])
    recent = [t for t in timestamps if t > cutoff]
    return len(recent) < max_rate


def record_spawn(window_id: str) -> None:
    log = _load_spawn_log()
    log.setdefault(window_id, []).append(time.time())
    # Prune old entries
    cutoff = time.time() - _SPAWN_RATE_WINDOW_SECONDS
    for wid in log:
        log[wid] = [t for t in log[wid] if t > cutoff]
    _save_spawn_log(log)


def create_spawn_request(
    requester_window: str,
    provider: str,
    cwd: str,
    prompt: str,
    context_file: str | None = None,
    auto: bool = False,
) -> SpawnRequest:
    if not Path(cwd).is_dir():
        raise ValueError(f"cwd does not exist: {cwd}")

    request_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    req = SpawnRequest(
        id=request_id,
        requester_window=requester_window,
        provider=provider,
        cwd=cwd,
        prompt=prompt,
        context_file=context_file,
        auto=auto,
    )

    # Store in memory (for same-process access) and persist to disk
    # (so the bot process can find requests from CLI subprocesses).
    _pending_requests[request_id] = req
    sdir = _spawns_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / f"{request_id}.json").write_text(json.dumps(req.to_dict(), indent=2))

    # Record for rate limiting (file-based, shared across processes)
    record_spawn(requester_window)

    return req


def scan_spawn_requests() -> list[SpawnRequest]:
    """Read pending spawn request files from disk.

    Called by the broker cycle in the bot process. Loads new requests
    into ``_pending_requests`` and returns them for keyboard posting.
    """
    sdir = _spawns_dir()
    if not sdir.is_dir():
        return []

    new_requests: list[SpawnRequest] = []
    for entry in sdir.iterdir():
        if not entry.name.endswith(".json") or entry.name == "rate_log.json":
            continue
        try:
            data = json.loads(entry.read_text())
            req = SpawnRequest.from_dict(data)
        except json.JSONDecodeError, OSError, KeyError:
            continue

        if req.id in _pending_requests:
            continue

        if req.is_expired(timeout=_spawn_timeout()):
            with contextlib.suppress(OSError):
                entry.unlink()
            continue

        _pending_requests[req.id] = req
        new_requests.append(req)

    return new_requests


async def handle_spawn_approval(
    request_id: str,
    bot: Bot,
) -> SpawnResult | None:
    req = _pending_requests.pop(request_id, None)
    if req is None:
        logger.warning(
            "Spawn request %s not found (expired or already handled)", request_id
        )
        return None

    # Remove the file from disk
    spawn_file = _spawns_dir() / f"{request_id}.json"
    spawn_file.unlink(missing_ok=True)

    launch_command = resolve_launch_command(req.provider)

    success, message, window_name, window_id = await tmux_manager.create_window(
        req.cwd,
        launch_command=launch_command,
    )
    if not success:
        logger.error("Spawn window creation failed: %s", message)
        return None

    session_manager.set_window_provider(window_id, req.provider, cwd=req.cwd)

    await _create_topic_for_spawn(bot, window_id, window_name, req)

    if req.provider == "claude":
        from ..msg_skill import ensure_skill_installed

        ensure_skill_installed(req.cwd)

    if req.prompt:
        prompt_text = req.prompt
        if req.context_file:
            prompt_text = f"{req.prompt} (context: {req.context_file})"
        await tmux_manager.send_keys(window_id, prompt_text)

    logger.info(
        "Spawned window %s (%s) for %s (provider=%s)",
        window_id,
        window_name,
        req.requester_window,
        req.provider,
    )

    return SpawnResult(window_id=window_id, window_name=window_name)


def handle_spawn_denial(request_id: str) -> None:
    req = _pending_requests.pop(request_id, None)
    if req is not None:
        logger.info("Spawn request %s denied", request_id)
    # Remove the file from disk
    spawn_file = _spawns_dir() / f"{request_id}.json"
    spawn_file.unlink(missing_ok=True)


async def post_spawn_approval_keyboard(
    bot: Bot,
    requester_window: str,
    request: SpawnRequest,
) -> None:
    from .msg_telegram import _resolve_topic

    topic = _resolve_topic(requester_window)
    if topic is None:
        return

    _, thread_id, chat_id, _ = topic

    text = (
        f"\U0001f680 Spawn request: {request.provider} at {request.cwd}\n"
        f"Prompt: {request.prompt}"
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Approve",
                    callback_data=f"{CB_SPAWN_APPROVE}{request.id}",
                ),
                InlineKeyboardButton(
                    "Deny",
                    callback_data=f"{CB_SPAWN_DENY}{request.id}",
                ),
            ]
        ]
    )

    await rate_limit_send_message(
        bot,
        chat_id,
        text,
        message_thread_id=thread_id,
        reply_markup=keyboard,
    )


async def _create_topic_for_spawn(
    bot: Bot,
    window_id: str,
    window_name: str,
    req: SpawnRequest,
) -> None:
    from .msg_telegram import _resolve_topic
    from .topic_orchestration import _collect_target_chats, _create_topic_in_chat

    target_chats = _collect_target_chats(window_id)
    for chat_id in target_chats:
        await _create_topic_in_chat(bot, chat_id, window_id, window_name)

    topic_info = _resolve_topic(req.requester_window)
    if topic_info:
        _, thread_id, chat_id, _ = topic_info
        text = f"\u2705 Spawned {window_name} ({window_id}) for: {req.prompt}"
        await rate_limit_send_message(
            bot,
            chat_id,
            text,
            message_thread_id=thread_id,
            disable_notification=True,
        )


# ── Callback handlers for spawn approval buttons ───────��─────────────────


@register(CB_SPAWN_APPROVE, CB_SPAWN_DENY)
async def _handle_spawn_callback(
    update: Update,
    _context: ContextTypes.DEFAULT_TYPE,
) -> None:
    import contextlib as _contextlib

    from telegram.error import TelegramError

    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()

    data = query.data

    if data.startswith(CB_SPAWN_APPROVE):
        request_id = data[len(CB_SPAWN_APPROVE) :]
        bot = update.get_bot()
        result = await handle_spawn_approval(request_id, bot)
        if result:
            text = f"\u2705 Spawned: {result.window_name} ({result.window_id})"
        else:
            text = "\u274c Spawn failed (request expired or window creation error)"
        with _contextlib.suppress(TelegramError):
            await query.edit_message_text(text)

    elif data.startswith(CB_SPAWN_DENY):
        request_id = data[len(CB_SPAWN_DENY) :]
        handle_spawn_denial(request_id)
        with _contextlib.suppress(TelegramError):
            await query.edit_message_text("\u274c Spawn request denied")
