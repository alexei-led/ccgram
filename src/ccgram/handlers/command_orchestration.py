"""Command forwarding orchestration for provider slash commands.

Routes unknown Telegram /commands to the active provider session via tmux.
Manages scoped command menu caching, command metadata resolution, and
post-send failure probing (transcript + pane delta).

Core responsibilities:
  - forward_command_handler(): the main entry point for unrecognized /commands
  - Provider command menu sync (per-user, per-chat, global scopes)
  - Command failure probing (transcript-based + pane-delta fallback)
  - Status snapshot fallback for /status and /stats
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path
import re

import structlog
from telegram import (
    BotCommandScopeChat,
    BotCommandScopeChatMember,
    Message,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import Application, ContextTypes

from ..cc_commands import discover_provider_commands, register_commands
from ..providers import (
    AgentProvider,
    get_provider,
    get_provider_for_window,
    registry,
)
from ..session import session_manager
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from ..utils import task_done_callback
from .callback_helpers import get_thread_id as _get_thread_id
from .message_sender import safe_reply

logger = structlog.get_logger()

_CommandRefreshError = (TelegramError, OSError)

_CODEX_STATUS_FALLBACK_DELAY_SECONDS = 1.2
_COMMAND_ERROR_PROBE_DELAY_SECONDS = 1.0
_COMMAND_ERROR_RE = re.compile(
    r"(?i)\b(?:"
    r"unrecognized command|"
    r"unknown command|"
    r"invalid command|"
    r"unsupported command|"
    r"no such command|"
    r"command not found|"
    r"not recognized"
    r")\b"
)

# --- Menu cache state ---

_scoped_provider_menu: OrderedDict[tuple[int, int], str] = OrderedDict()
_chat_scoped_provider_menu: OrderedDict[int, str] = OrderedDict()
_global_provider_menu: str | None = None
_MAX_SCOPED_PROVIDER_MENU_ENTRIES = 512
_MAX_CHAT_PROVIDER_MENU_ENTRIES = 256


# --- Helpers ---


def _set_bounded_cache_entry[K, V](
    cache: OrderedDict[K, V],
    key: K,
    value: V,
    *,
    max_entries: int,
) -> None:
    if key in cache:
        cache.pop(key, None)
    cache[key] = value
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _get_lru_cache_entry[K, V](
    cache: OrderedDict[K, V],
    key: K,
) -> V | None:
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def _normalize_slash_token(command: str) -> str:
    parts = command.strip().split(None, 1)
    if not parts:
        return "/"
    token = parts[0].lower()
    return token if token.startswith("/") else f"/{token}"


def _extract_probe_error_line(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _COMMAND_ERROR_RE.search(line):
            return line
        if "error" in line.lower() and "command" in line.lower():
            return line
    return None


def _extract_pane_delta(before: str | None, after: str | None) -> str:
    """Return the likely newly-added pane text after a command send."""
    if not after:
        return ""
    if not before:
        return after
    if before == after:
        return ""

    before_lines = before.splitlines()
    after_lines = after.splitlines()
    max_overlap = min(len(before_lines), len(after_lines))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if before_lines[-size:] == after_lines[:size]:
            overlap = size
            break
    return "\n".join(after_lines[overlap:]).strip()


def _short_supported_commands(supported_commands: set[str], limit: int = 8) -> str:
    supported = sorted(supported_commands)
    if not supported:
        return "Use /commands to list available commands."
    shown = supported[:limit]
    suffix = "" if len(supported) <= limit else " …"
    return "Try: " + ", ".join(shown) + suffix


# --- Provider command metadata ---


def _build_provider_command_metadata(
    provider: AgentProvider,
) -> tuple[dict[str, str], set[str]]:
    mapping: dict[str, str] = {}
    supported: set[str] = set()
    for cmd in discover_provider_commands(provider):
        if cmd.telegram_name and cmd.telegram_name not in mapping:
            mapping[cmd.telegram_name] = cmd.name
        token = cmd.name if cmd.name.startswith("/") else f"/{cmd.name}"
        supported.add(token.lower())
    for builtin in provider.capabilities.builtin_commands:
        if not builtin:
            continue
        token = builtin if builtin.startswith("/") else f"/{builtin}"
        supported.add(token.lower())
    return mapping, supported


# --- Scoped command menu sync ---


async def sync_scoped_provider_menu(
    message: Message,
    user_id: int,
    provider: AgentProvider,
) -> None:
    """Update per-user command menu for the current chat/provider context."""
    global _global_provider_menu

    chat_id = message.chat.id
    provider_name = provider.capabilities.name
    cache_key = (chat_id, user_id)
    if _get_lru_cache_entry(_scoped_provider_menu, cache_key) == provider_name:
        return

    try:
        member_scope = BotCommandScopeChatMember(chat_id=chat_id, user_id=user_id)
        await register_commands(
            message.get_bot(), provider=provider, scope=member_scope
        )
        _set_bounded_cache_entry(
            _scoped_provider_menu,
            cache_key,
            provider_name,
            max_entries=_MAX_SCOPED_PROVIDER_MENU_ENTRIES,
        )
        _set_bounded_cache_entry(
            _chat_scoped_provider_menu,
            chat_id,
            provider_name,
            max_entries=_MAX_CHAT_PROVIDER_MENU_ENTRIES,
        )
        return
    except _CommandRefreshError:
        logger.debug(
            "Failed to update member-scoped command menu (chat=%s user=%s provider=%s)",
            chat_id,
            user_id,
            provider_name,
        )

    if _get_lru_cache_entry(_chat_scoped_provider_menu, chat_id) != provider_name:
        try:
            chat_scope = BotCommandScopeChat(chat_id=chat_id)
            await register_commands(
                message.get_bot(), provider=provider, scope=chat_scope
            )
            _set_bounded_cache_entry(
                _chat_scoped_provider_menu,
                chat_id,
                provider_name,
                max_entries=_MAX_CHAT_PROVIDER_MENU_ENTRIES,
            )
            _set_bounded_cache_entry(
                _scoped_provider_menu,
                cache_key,
                provider_name,
                max_entries=_MAX_SCOPED_PROVIDER_MENU_ENTRIES,
            )
            return
        except _CommandRefreshError:
            logger.debug(
                "Failed to update chat-scoped command menu (chat=%s provider=%s)",
                chat_id,
                provider_name,
            )

    if _global_provider_menu == provider_name:
        _set_bounded_cache_entry(
            _scoped_provider_menu,
            cache_key,
            provider_name,
            max_entries=_MAX_SCOPED_PROVIDER_MENU_ENTRIES,
        )
        return
    try:
        await register_commands(message.get_bot(), provider=provider)
        _global_provider_menu = provider_name
        _set_bounded_cache_entry(
            _scoped_provider_menu,
            cache_key,
            provider_name,
            max_entries=_MAX_SCOPED_PROVIDER_MENU_ENTRIES,
        )
    except _CommandRefreshError:
        logger.debug(
            "Failed to update global provider command menu (provider=%s)",
            provider_name,
        )


async def sync_scoped_menu_for_text_context(update: Update, user_id: int) -> None:
    """Sync scoped menu when a bound topic receives plain text."""
    message = update.message
    if not message:
        return
    thread_id = _get_thread_id(update)
    if thread_id is None:
        return
    window_id = thread_router.resolve_window_for_thread(user_id, thread_id)
    if not window_id:
        return
    provider = get_provider_for_window(window_id)
    await sync_scoped_provider_menu(message, user_id, provider)


def get_global_provider_menu() -> str | None:
    """Return the current global provider menu name."""
    return _global_provider_menu


def set_global_provider_menu(provider_name: str) -> None:
    """Set the global provider menu name."""
    global _global_provider_menu
    _global_provider_menu = provider_name


# --- Command failure probing ---


async def _capture_command_probe_context(
    window_id: str,
    provider: AgentProvider,
) -> tuple[str | None, int | None, str | None]:
    """Capture transcript offset + pane snapshot before sending a command."""
    transcript_path = session_manager.get_window_state(window_id).transcript_path
    since_offset: int | None = None
    if transcript_path:
        try:
            if provider.capabilities.supports_incremental_read:
                since_offset = Path(transcript_path).stat().st_size
            else:
                _, since_offset = await asyncio.to_thread(
                    provider.read_transcript_file,
                    transcript_path,
                    0,
                )
        except OSError:
            since_offset = None
    pane_before = await tmux_manager.capture_pane(window_id)
    return transcript_path or None, since_offset, pane_before


async def _probe_transcript_command_error(
    provider: AgentProvider,
    transcript_path: str | None,
    since_offset: int | None,
) -> str | None:
    """Return first command-like error line found in transcript delta."""
    if not transcript_path or since_offset is None:
        return None

    def _read_incremental_entries(path: str, offset: int) -> list[dict]:
        entries: list[dict] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            fh.seek(offset)
            for line in fh:
                parsed = provider.parse_transcript_line(line)
                if parsed:
                    entries.append(parsed)
        return entries

    try:
        if provider.capabilities.supports_incremental_read:
            entries = await asyncio.to_thread(
                _read_incremental_entries,
                transcript_path,
                since_offset,
            )
        else:
            entries, _ = await asyncio.to_thread(
                provider.read_transcript_file,
                transcript_path,
                since_offset,
            )
    except OSError, NotImplementedError:
        return None

    messages, _ = provider.parse_transcript_entries(entries, pending_tools={})
    for msg in messages:
        if msg.role != "assistant":
            continue
        found = _extract_probe_error_line(msg.text)
        if found:
            return found
    return None


async def _maybe_send_command_failure_message(
    message: Message,
    window_id: str,
    display: str,
    cc_slash: str,
    *,
    provider: AgentProvider,
    transcript_path: str | None,
    since_offset: int | None,
    pane_before: str | None,
) -> None:
    """Probe transcript/pane for quick command failures and surface them."""
    await asyncio.sleep(_COMMAND_ERROR_PROBE_DELAY_SECONDS)

    error_line = await _probe_transcript_command_error(
        provider,
        transcript_path,
        since_offset,
    )
    if not error_line:
        pane_after = await tmux_manager.capture_pane(window_id)
        pane_delta = _extract_pane_delta(pane_before, pane_after)
        error_line = _extract_probe_error_line(pane_delta)
    if error_line:
        await safe_reply(
            message,
            f"\u274c [{display}] `{cc_slash}` failed\n> {error_line}",
        )


def _spawn_command_failure_probe(
    message: Message,
    window_id: str,
    display: str,
    cc_slash: str,
    *,
    provider: AgentProvider,
    transcript_path: str | None,
    since_offset: int | None,
    pane_before: str | None,
) -> None:
    async def _run() -> None:
        await _maybe_send_command_failure_message(
            message,
            window_id,
            display,
            cc_slash,
            provider=provider,
            transcript_path=transcript_path,
            since_offset=since_offset,
            pane_before=pane_before,
        )

    task = asyncio.create_task(_run())
    task.add_done_callback(task_done_callback)


# --- Status snapshot ---


def _status_snapshot_probe_offset(window_id: str, cc_slash: str) -> int | None:
    """Return transcript file offset before sending a /status(/stats) command."""
    command = cc_slash.split(None, 1)[0].lower()
    if command not in ("/status", "/stats"):
        return None

    provider = get_provider_for_window(window_id)
    if not provider.capabilities.supports_status_snapshot:
        return None

    transcript_path = session_manager.get_window_state(window_id).transcript_path
    if not transcript_path:
        return None

    try:
        return Path(transcript_path).stat().st_size
    except OSError:
        return None


async def _maybe_send_status_snapshot(
    message: Message,
    window_id: str,
    display: str,
    cc_slash: str,
    *,
    since_offset: int | None = None,
) -> None:
    """Send transcript-based status snapshot fallback for /status and /stats."""
    command = cc_slash.split(None, 1)[0].lower()
    if command not in ("/status", "/stats"):
        return

    provider = get_provider_for_window(window_id)
    if not provider.capabilities.supports_status_snapshot:
        return

    state = session_manager.get_window_state(window_id)
    transcript_path = state.transcript_path
    if not transcript_path:
        await safe_reply(
            message,
            f"[{display}] Status snapshot unavailable (no transcript path).",
        )
        return

    if since_offset is not None:
        await asyncio.sleep(_CODEX_STATUS_FALLBACK_DELAY_SECONDS)
        has_native_output = await asyncio.to_thread(
            provider.has_output_since,
            transcript_path,
            since_offset,
        )
        if has_native_output:
            return

    snapshot = await asyncio.to_thread(
        provider.build_status_snapshot,
        transcript_path,
        display_name=display,
        session_id=state.session_id,
        cwd=state.cwd,
    )
    if snapshot:
        await safe_reply(message, snapshot)
        return

    await safe_reply(
        message,
        f"[{display}] Status snapshot unavailable (transcript unreadable).",
    )


def _command_known_in_other_provider(
    command_token: str,
    current_provider: AgentProvider,
    *,
    supported_cache: dict[str, set[str]] | None = None,
) -> bool:
    """Return True when command exists in any provider except the current one."""
    current_name = current_provider.capabilities.name
    for name in registry.provider_names():
        if name == current_name:
            continue
        if supported_cache is not None and name in supported_cache:
            supported = supported_cache[name]
        else:
            provider = registry.get(name)
            _, supported = _build_provider_command_metadata(provider)
            if supported_cache is not None:
                supported_cache[name] = supported
        if command_token in supported:
            return True
    return False


async def _handle_clear_command(
    update: Update,
    user_id: int,
    window_id: str,
    display: str,
    cc_slash: str,
    thread_id: int | None,
) -> None:
    """Handle post-send cleanup when /clear is forwarded."""
    if cc_slash.strip().lower() != "/clear":
        return
    logger.info("Clearing session for window %s after /clear", display)
    session_manager.clear_window_session(window_id)
    from .message_queue import enqueue_status_update
    from .polling_strategies import (
        clear_screen_buffer,
        clear_seen_status,
    )

    await enqueue_status_update(
        update.get_bot(), user_id, window_id, None, thread_id=thread_id
    )
    clear_seen_status(window_id)
    clear_screen_buffer(window_id)


# --- Main command handler ---


async def forward_command_handler(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Forward any non-bot command as a slash command to the topic provider session."""
    from ..bot import is_user_allowed

    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    chat = update.message.chat
    thread_id = _get_thread_id(update)
    if chat.type in ("group", "supergroup") and thread_id is not None:
        thread_router.set_group_chat_id(user.id, thread_id, chat.id)

    cmd_text = update.message.text or ""
    parts = cmd_text.split(None, 1)
    raw_cmd = parts[0].split("@")[0] if parts else ""
    tg_cmd = raw_cmd.lstrip("/")
    args = parts[1] if len(parts) > 1 else ""
    window_id = thread_router.resolve_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "\u274c No session bound to this topic.")
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        display = thread_router.get_display_name(window_id)
        await safe_reply(update.message, f"\u274c Window '{display}' no longer exists.")
        return

    display = thread_router.get_display_name(window_id)
    provider = get_provider_for_window(window_id)
    await sync_scoped_provider_menu(update.message, user.id, provider)
    provider_map, current_supported = _build_provider_command_metadata(provider)
    resolved_name = provider_map.get(tg_cmd, tg_cmd)
    cc_name = resolved_name.lstrip("/")
    if not args and cc_name in ("remote-control", "rc"):
        args = display
    cc_slash = f"/{cc_name} {args}".rstrip() if args else f"/{cc_name}"
    command_token = _normalize_slash_token(cc_slash)

    supported_cache: dict[str, set[str]] = {
        provider.capabilities.name: current_supported
    }
    if command_token not in current_supported and _command_known_in_other_provider(
        command_token,
        provider,
        supported_cache=supported_cache,
    ):
        await safe_reply(
            update.message,
            f"\u274c [{display}] `{command_token}` is not supported by "
            f"`{provider.capabilities.name}`.\n"
            f"{_short_supported_commands(current_supported)}\n"
            "Use /commands for the full list.",
        )
        return

    logger.info(
        "Forwarding command %s to window %s (user=%d)", cc_slash, display, user.id
    )
    await update.message.get_bot().send_chat_action(
        chat_id=update.message.chat.id,
        message_thread_id=thread_id,
        action=ChatAction.TYPING,
    )
    (
        probe_transcript_path,
        probe_transcript_offset,
        probe_pane_before,
    ) = await _capture_command_probe_context(window_id, provider)
    status_probe_offset = _status_snapshot_probe_offset(window_id, cc_slash)
    from .polling_strategies import clear_probe_failures

    clear_probe_failures(window_id)
    success, error_msg = await session_manager.send_to_window(window_id, cc_slash)
    if not success:
        await safe_reply(update.message, f"\u274c {error_msg}")
        return

    if thread_id is not None:
        from .command_history import record_command

        record_command(user.id, thread_id, cc_slash)
    await safe_reply(update.message, f"\u26a1 [{display}] Sent: {cc_slash}")
    await _maybe_send_status_snapshot(
        update.message,
        window_id,
        display,
        cc_slash,
        since_offset=status_probe_offset,
    )
    _spawn_command_failure_probe(
        update.message,
        window_id,
        display,
        cc_slash,
        provider=provider,
        transcript_path=probe_transcript_path,
        since_offset=probe_transcript_offset,
        pane_before=probe_pane_before,
    )
    await _handle_clear_command(
        update, user.id, window_id, display, cc_slash, thread_id
    )


def setup_menu_refresh_job(application: "Application") -> None:
    """Register the periodic command menu refresh job."""
    global _global_provider_menu

    default_provider = get_provider()
    _global_provider_menu = default_provider.capabilities.name

    async def _refresh_commands(context: ContextTypes.DEFAULT_TYPE) -> None:
        global _global_provider_menu
        if context.bot:
            try:
                refreshed_provider = get_provider()
                await register_commands(context.bot, provider=refreshed_provider)
                _global_provider_menu = refreshed_provider.capabilities.name
            except _CommandRefreshError:
                logger.exception("Failed to refresh CC commands, keeping previous menu")

    jq = getattr(application, "job_queue", None)
    if jq is not None:
        jq.run_repeating(_refresh_commands, interval=600, first=600)
