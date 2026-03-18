"""Telegram bot handlers — the main UI layer of CCGram.

Registers all command/callback/message handlers and manages the bot lifecycle.
Each Telegram topic maps 1:1 to a tmux window (Claude session).

Core responsibilities:
  - Command handlers: /new (+ /start alias), /history, /sessions, /resume,
    /screenshot, /panes, /restore, plus forwarding unknown /commands to Claude Code via tmux.
  - Callback query handler: thin dispatcher routing to dedicated handler modules.
  - Topic-based routing: each named topic binds to one tmux window.
    Unbound topics trigger the directory browser to create a new session.
  - Topic lifecycle: closing a topic unbinds the window (kept alive for
    rebinding). Unbound windows are auto-killed after TTL by status polling.
    Unsupported content (images, stickers, etc.) is rejected with a warning.
  - Bot lifecycle management: post_init, post_shutdown, create_bot.

Key functions: create_bot(), handle_new_message().
"""

import asyncio
from collections import OrderedDict
import contextlib
import structlog
import os
import re
import signal
import time
from pathlib import Path

from telegram import (
    Bot,
    BotCommandScopeChat,
    BotCommandScopeChatMember,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import Conflict, RetryAfter, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from .cc_commands import (
    discover_provider_commands,
    register_commands,
)
from .providers import (
    AgentProvider,
    detect_provider_from_command,
    detect_provider_from_runtime,
    get_provider,
    get_provider_for_window,
    registry,
    should_probe_pane_title_for_provider_detection,
)
from .config import config
from .codex_status import (
    build_codex_status_snapshot,
    has_codex_assistant_output_since,
)
from .handlers.callback_data import (
    CB_DIR_CANCEL,
    CB_DIR_CONFIRM,
    CB_DIR_FAV,
    CB_DIR_PAGE,
    CB_DIR_SELECT,
    CB_DIR_STAR,
    CB_DIR_UP,
    CB_MODE_SELECT,
    CB_PROV_SELECT,
    CB_HISTORY_NEXT,
    CB_HISTORY_PREV,
    CB_KEYS_PREFIX,
    CB_PANE_SCREENSHOT,
    CB_RECOVERY_BACK,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_PICK,
    CB_RECOVERY_RESUME,
    CB_RESUME_CANCEL,
    CB_RESUME_PAGE,
    CB_RESUME_PICK,
    CB_SCREENSHOT_REFRESH,
    CB_SESSIONS_KILL,
    CB_SESSIONS_KILL_CONFIRM,
    CB_SESSIONS_NEW,
    CB_SESSIONS_REFRESH,
    CB_STATUS_ESC,
    CB_STATUS_NOTIFY,
    CB_STATUS_RECALL,
    CB_STATUS_SCREENSHOT,
    CB_SYNC_DISMISS,
    CB_SYNC_FIX,
    CB_VOICE,
    CB_WIN_BIND,
    CB_WIN_CANCEL,
    CB_WIN_NEW,
)
from .handlers.callback_helpers import get_thread_id as _get_thread_id
from .handlers.callback_helpers import user_owns_window as _user_owns_window
from .handlers.directory_callbacks import handle_directory_callback
from .handlers.history_callbacks import handle_history_callback
from .handlers.interactive_callbacks import (
    handle_interactive_callback,
    match_interactive_prefix as _match_interactive_prefix,
)
from .handlers.recovery_callbacks import handle_recovery_callback
from .handlers.restore_command import restore_command
from .handlers.resume_command import handle_resume_command_callback, resume_command
from .handlers.screenshot_callbacks import handle_screenshot_callback
from .handlers.voice_callbacks import handle_voice_callback
from .handlers.window_callbacks import handle_window_callback
from .handlers.directory_browser import clear_browse_state
from .handlers.cleanup import clear_topic_state
from .handlers.topic_emoji import strip_emoji_prefix, update_stored_topic_name
from .handlers.history import send_history
from .handlers.sessions_dashboard import (
    handle_sessions_kill,
    handle_sessions_kill_confirm,
    handle_sessions_refresh,
    sessions_command,
)
from .handlers.sync_command import (
    handle_sync_dismiss,
    handle_sync_fix,
    sync_command,
)
from .handlers.upgrade import upgrade_command
from .handlers.interactive_ui import (
    INTERACTIVE_TOOL_NAMES,
    clear_interactive_mode,
    clear_interactive_msg,
    get_interactive_msg_id,
    handle_interactive_ui,
    set_interactive_mode,
)
from .handlers.message_queue import (
    enqueue_content_message,
    enqueue_status_update,
    get_message_queue,
    shutdown_workers,
)
from .handlers.message_sender import safe_reply
from .handlers.response_builder import build_response_parts
from .handlers.status_polling import status_poll_loop
from .handlers.file_handler import handle_document_message, handle_photo_message
from .handlers.voice_handler import handle_voice_message
from .handlers.text_handler import handle_text_message
from .session import session_manager
from .session_monitor import NewMessage, NewWindowEvent, SessionMonitor
from .tmux_manager import tmux_manager
from .utils import task_done_callback

logger = structlog.get_logger()

_CommandRefreshError = (TelegramError, OSError)

# Error keyword pattern for errors_only notification mode (word boundaries)
_ERROR_KEYWORDS_RE = re.compile(
    r"\b(?:error|exception|failed|traceback|stderr|assertion)\b", re.IGNORECASE
)

# Max label length for /recall command buttons (wider than status bar buttons)
_RECALL_LABEL_MAX = 40
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

# Session monitor instance
session_monitor: SessionMonitor | None = None

# Status polling task
_status_poll_task: asyncio.Task | None = None

# Per-chat backoff for auto topic creation after Telegram flood control.
# chat_id -> monotonic timestamp when next attempt is allowed.
_topic_create_retry_until: dict[int, float] = {}
_TOPIC_CREATE_RETRY_BUFFER_SECONDS = 1
_scoped_provider_menu: OrderedDict[tuple[int, int], str] = OrderedDict()
_chat_scoped_provider_menu: OrderedDict[int, str] = OrderedDict()
_global_provider_menu: str | None = None
_MAX_SCOPED_PROVIDER_MENU_ENTRIES = 512
_MAX_CHAT_PROVIDER_MENU_ENTRIES = 256


def is_user_allowed(user_id: int | None) -> bool:
    return user_id is not None and config.is_user_allowed(user_id)


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


def _get_provider_command_metadata(
    provider: AgentProvider,
) -> tuple[dict[str, str], set[str]]:
    return _build_provider_command_metadata(provider)


async def _sync_scoped_provider_menu(
    message: Message,
    user_id: int,
    provider: AgentProvider,
) -> None:
    """Update per-user command menu for the current chat/provider context."""
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

    # Fallback for Telegram deployments where chat-member scopes are restricted.
    # Chat scope still keeps command hints contextual for this chat.
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

    # Last fallback: update global menu so slash suggestions still include
    # provider commands (same UX as classic single-provider mode).
    global _global_provider_menu
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


async def _sync_scoped_menu_for_text_context(update: Update, user_id: int) -> None:
    """Sync scoped menu when a bound topic receives plain text."""
    message = update.message
    if not message:
        return
    thread_id = _get_thread_id(update)
    if thread_id is None:
        return
    window_id = session_manager.resolve_window_for_thread(user_id, thread_id)
    if not window_id:
        return
    provider = get_provider_for_window(window_id)
    await _sync_scoped_provider_menu(message, user_id, provider)


# Group filter: when CCBOT_GROUP_ID is set, only process updates from that group.
# filters.ALL is a no-op — single-instance backward compat.
_group_filter: filters.BaseFilter = (
    filters.Chat(chat_id=config.group_id) if config.group_id else filters.ALL
)


# --- Command handlers ---


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        if update.message:
            await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    clear_browse_state(context.user_data)

    if update.message:
        await safe_reply(
            update.message,
            "\U0001f916 *Claude Code Monitor*\n\n"
            "Each topic is a session. Create a new topic to start.",
        )


async def history_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show message history for the active session or bound thread."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    window_id = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "\u274c No session bound to this topic.")
        return

    await send_history(update.message, window_id)


async def commands_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show provider-specific slash commands for the current topic."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    window_id = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "\u274c No session bound to this topic.")
        return

    provider = get_provider_for_window(window_id)
    await _sync_scoped_provider_menu(update.message, user.id, provider)
    commands = discover_provider_commands(provider)
    if not commands:
        await safe_reply(
            update.message,
            f"Provider: `{provider.capabilities.name}`\nNo discoverable commands.",
        )
        return

    lines = [f"Provider: `{provider.capabilities.name}`", "Supported commands:"]
    for cmd in sorted(commands, key=lambda c: c.telegram_name):
        if not cmd.telegram_name:
            continue
        original = cmd.name if cmd.name.startswith("/") else f"/{cmd.name}"
        lines.append(f"- `/{cmd.telegram_name}` \u2192 `{original}`")
    await safe_reply(update.message, "\n".join(lines))


async def topic_closed_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle topic closure — unbind thread but keep the tmux window alive.

    The window becomes "unbound" and is available for rebinding via the window
    picker when a new topic is created. Unbound windows are auto-killed after
    the configured TTL (autoclose_done_minutes) by the status polling loop.
    """
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        return

    window_id = session_manager.get_window_for_thread(user.id, thread_id)
    if window_id:
        display = session_manager.get_display_name(window_id)
        session_manager.unbind_thread(user.id, thread_id)
        # Clean up all memory state for this topic
        await clear_topic_state(
            user.id, thread_id, context.bot, context.user_data, window_id=window_id
        )
        logger.info(
            "Topic closed: window %s unbound (kept alive for rebinding, user=%d, thread=%d)",
            display,
            user.id,
            thread_id,
        )
    else:
        logger.debug(
            "Topic closed: no binding (user=%d, thread=%d)", user.id, thread_id
        )


async def topic_edited_handler(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle topic rename — sync new name to tmux window and emoji cache.

    Ignores icon-only edits (name is None) and emoji-only changes from the bot
    itself (clean name unchanged after stripping prefixes).
    """
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message or not update.message.forum_topic_edited:
        return

    new_name = update.message.forum_topic_edited.name
    if not new_name:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    window_id = session_manager.get_window_for_chat_thread(chat_id, thread_id)
    if not window_id:
        logger.debug("Topic edited: no binding (thread=%d)", thread_id)
        return

    clean_name = strip_emoji_prefix(new_name)

    # Loop guard: if clean name matches current display name, this was a
    # bot-originated emoji/mode change — skip to prevent rename loops.
    current_display = session_manager.get_display_name(window_id)
    if current_display and strip_emoji_prefix(current_display) == clean_name:
        logger.debug(
            "Topic edited: name unchanged after strip, skipping (thread=%d)", thread_id
        )
        return

    renamed = await tmux_manager.rename_window(window_id, clean_name)
    if renamed:
        session_manager.set_display_name(window_id, clean_name)
        update_stored_topic_name(chat_id, thread_id, clean_name)
        logger.info(
            "Topic renamed: window %s → %r (thread=%d)",
            window_id,
            clean_name,
            thread_id,
        )


async def unbind_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Disconnect a topic from its tmux window without killing the session."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = session_manager.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    display = session_manager.get_display_name(window_id)
    # Enqueue a status clear to actually delete the Telegram status message
    # (clear_topic_state only clears the tracking dict, leaving a ghost)
    await enqueue_status_update(context.bot, user.id, window_id, None, thread_id)
    await clear_topic_state(
        user.id, thread_id, context.bot, context.user_data, window_id=window_id
    )
    session_manager.unbind_thread(user.id, thread_id)
    await safe_reply(
        update.message,
        f"\u2702 Unbound from window `{display}`. The session is still running.\n"
        "Send a message in this topic to rebind or create a new session.",
    )


async def forward_command_handler(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Forward any non-bot command as a slash command to the topic provider session."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    # Store group chat_id for forum topic message routing
    chat = update.message.chat
    thread_id = _get_thread_id(update)
    if chat.type in ("group", "supergroup") and thread_id is not None:
        session_manager.set_group_chat_id(user.id, thread_id, chat.id)

    cmd_text = update.message.text or ""
    # Split into command word + arguments, then strip @botname from command
    parts = cmd_text.split(None, 1)  # ["/cmd@botname", "optional args"]
    raw_cmd = parts[0].split("@")[0] if parts else ""  # strip @botname
    tg_cmd = raw_cmd.lstrip("/")
    args = parts[1] if len(parts) > 1 else ""
    window_id = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "\u274c No session bound to this topic.")
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        display = session_manager.get_display_name(window_id)
        await safe_reply(update.message, f"\u274c Window '{display}' no longer exists.")
        return

    display = session_manager.get_display_name(window_id)
    provider = get_provider_for_window(window_id)
    await _sync_scoped_provider_menu(update.message, user.id, provider)
    provider_map, current_supported = _get_provider_command_metadata(provider)
    resolved_name = provider_map.get(tg_cmd, tg_cmd)
    cc_name = resolved_name.lstrip("/")
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
    status_probe_offset = _codex_status_probe_offset(window_id, cc_slash)
    success, message = await session_manager.send_to_window(window_id, cc_slash)
    if success:
        if thread_id is not None:
            from .handlers.command_history import record_command

            record_command(user.id, thread_id, cc_slash)
        await safe_reply(update.message, f"\u26a1 [{display}] Sent: {cc_slash}")
        await _maybe_send_codex_status_snapshot(
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
        # If /clear command was sent, clear the session association
        # so we can detect the new session after first message
        if cc_slash.strip().lower() == "/clear":
            logger.info("Clearing session for window %s after /clear", display)
            session_manager.clear_window_session(window_id)
            from .handlers.message_queue import enqueue_status_update
            from .handlers.status_polling import (
                clear_screen_buffer,
                clear_seen_status,
            )

            await enqueue_status_update(
                update.get_bot(), user.id, window_id, None, thread_id=thread_id
            )
            clear_seen_status(window_id)
            clear_screen_buffer(window_id)
    else:
        await safe_reply(update.message, f"\u274c {message}")


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
            _, supported = _get_provider_command_metadata(provider)
            if supported_cache is not None:
                supported_cache[name] = supported
        if command_token in supported:
            return True
    return False


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
    if not error_line:
        return

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


def _codex_status_probe_offset(window_id: str, cc_slash: str) -> int | None:
    """Return transcript file offset before sending codex /status(/stats)."""
    command = cc_slash.split(None, 1)[0].lower()
    if command not in ("/status", "/stats"):
        return None

    provider = get_provider_for_window(window_id)
    if provider.capabilities.name != "codex":
        return None

    transcript_path = session_manager.get_window_state(window_id).transcript_path
    if not transcript_path:
        return None

    try:
        return Path(transcript_path).stat().st_size
    except OSError:
        return None


async def _maybe_send_codex_status_snapshot(
    message: Message,
    window_id: str,
    display: str,
    cc_slash: str,
    *,
    since_offset: int | None = None,
) -> None:
    """Send transcript-based Codex status fallback for /status and /stats."""
    command = cc_slash.split(None, 1)[0].lower()
    if command not in ("/status", "/stats"):
        return

    provider = get_provider_for_window(window_id)
    if provider.capabilities.name != "codex":
        return

    state = session_manager.get_window_state(window_id)
    transcript_path = state.transcript_path
    if not transcript_path:
        await safe_reply(
            message,
            f"[{display}] Codex status snapshot unavailable (no transcript path).",
        )
        return

    if since_offset is not None:
        await asyncio.sleep(_CODEX_STATUS_FALLBACK_DELAY_SECONDS)
        has_native_output = await asyncio.to_thread(
            has_codex_assistant_output_since,
            transcript_path,
            since_offset,
        )
        if has_native_output:
            return

    snapshot = await asyncio.to_thread(
        build_codex_status_snapshot,
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
        f"[{display}] Codex status snapshot unavailable (transcript unreadable).",
    )


async def screenshot_command(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Capture and send a terminal screenshot for the current topic."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = session_manager.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await safe_reply(update.message, "\u274c Window no longer exists.")
        return

    pane_text = await tmux_manager.capture_pane(w.window_id, with_ansi=True)
    if not pane_text:
        await safe_reply(update.message, "\u274c Failed to capture terminal.")
        return

    import io

    from .handlers.screenshot_callbacks import build_screenshot_keyboard
    from .screenshot import text_to_image

    png_bytes = await text_to_image(pane_text, with_ansi=True)
    keyboard = build_screenshot_keyboard(window_id)
    chat_id = session_manager.resolve_chat_id(user.id, thread_id)
    try:
        await update.message.get_bot().send_document(
            chat_id=chat_id,
            document=io.BytesIO(png_bytes),
            filename="screenshot.png",
            reply_markup=keyboard,
            message_thread_id=thread_id,
        )
    except TelegramError as e:
        logger.error("Failed to send screenshot: %s", e)
        await safe_reply(update.message, "\u274c Failed to send screenshot.")


async def panes_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all panes in the current topic's window."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    window_id = session_manager.get_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(
            update.message, "\u274c This topic is not bound to any session."
        )
        return

    panes = await tmux_manager.list_panes(window_id)
    if len(panes) <= 1:
        await safe_reply(
            update.message,
            "\U0001f4d0 Single pane \u2014 no multi-pane layout detected.",
        )
        return

    from .handlers.status_polling import has_pane_alert

    lines = [f"\U0001f4d0 {len(panes)} panes in window\n"]
    buttons: list[InlineKeyboardButton] = []
    for pane in panes:
        prefix = "\U0001f4cd" if pane.active else "  "
        label = f"Pane {pane.index} ({pane.command})"
        suffix_parts: list[str] = []
        if pane.active:
            suffix_parts.append("active")
        if has_pane_alert(pane.pane_id):
            prefix = "\u26a0\ufe0f"
            suffix_parts.append("blocked")
        elif not pane.active:
            suffix_parts.append("running")
        suffix = f" \u2014 {', '.join(suffix_parts)}" if suffix_parts else ""
        lines.append(f"{prefix} {label}{suffix}")
        buttons.append(
            InlineKeyboardButton(
                f"\U0001f4f7 {pane.index}",
                callback_data=f"{CB_PANE_SCREENSHOT}{window_id}:{pane.pane_id}"[:64],
            )
        )

    keyboard = InlineKeyboardMarkup([buttons]) if buttons else None
    await safe_reply(update.message, "\n".join(lines), reply_markup=keyboard)


async def recall_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent command history for the current topic."""
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    if not update.message:
        return

    thread_id = _get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "\u274c Use this command inside a topic.")
        return

    from .handlers.command_history import (
        INLINE_QUERY_MAX,
        get_history,
        truncate_for_display,
    )

    history = get_history(user.id, thread_id, limit=10)
    if not history:
        await safe_reply(update.message, "\U0001f4cb No command history yet.")
        return

    rows = []
    for cmd in history:
        label = truncate_for_display(cmd, _RECALL_LABEL_MAX)
        query = cmd[:INLINE_QUERY_MAX]
        rows.append(
            [InlineKeyboardButton(label, switch_inline_query_current_chat=query)]
        )
    keyboard = InlineKeyboardMarkup(rows)
    await safe_reply(
        update.message, "\U0001f4cb Recent commands:", reply_markup=keyboard
    )


async def inline_query_handler(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Echo query text as a sendable inline result."""
    if not update.inline_query:
        return
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    text = update.inline_query.query.strip()
    if not text:
        await update.inline_query.answer([])
        return

    result = InlineQueryResultArticle(
        id="cmd",
        title=text,
        description="Tap to send",
        input_message_content=InputTextMessageContent(message_text=text),
    )
    await update.inline_query.answer([result], cache_time=0, is_personal=True)


async def unsupported_content_handler(
    update: Update,
    _context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Reply to non-text messages (images, stickers, voice, etc.)."""
    if not update.message:
        return
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        return
    logger.debug("Unsupported content from user %d", user.id)
    # Omit "voice" from the list when whisper is configured (has its own handler)
    media_list = (
        "Stickers, voice, video" if not config.whisper_provider else "Stickers, video"
    )
    await safe_reply(
        update.message,
        f"\u26a0 {media_list}, and similar media are not supported. Use text, photos, or documents.",
    )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        if update.message:
            await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    if not update.message or not update.message.text:
        return

    await _sync_scoped_menu_for_text_context(update, user.id)
    await handle_text_message(update, context)


# --- Callback query handler (thin dispatcher) ---

# Callback prefixes that route to dedicated handler modules.
# Order matters: prefixes checked via startswith must be longest-first
# to avoid false matches (e.g. CB_SESSIONS_KILL_CONFIRM before CB_SESSIONS_KILL).
_CB_HISTORY = (CB_HISTORY_PREV, CB_HISTORY_NEXT)
_CB_DIRECTORY = (
    CB_DIR_FAV,
    CB_DIR_STAR,
    CB_DIR_SELECT,
    CB_DIR_UP,
    CB_DIR_PAGE,
    CB_DIR_CONFIRM,
    CB_PROV_SELECT,
    CB_MODE_SELECT,
    CB_DIR_CANCEL,
)
_CB_WINDOW = (CB_WIN_BIND, CB_WIN_NEW, CB_WIN_CANCEL)
_CB_SCREENSHOT = (
    CB_SCREENSHOT_REFRESH,
    CB_STATUS_RECALL,
    CB_STATUS_ESC,
    CB_STATUS_NOTIFY,
    CB_STATUS_SCREENSHOT,
    CB_KEYS_PREFIX,
    CB_PANE_SCREENSHOT,
)
_CB_RECOVERY = (
    CB_RECOVERY_BACK,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_RESUME,
    CB_RECOVERY_PICK,
    CB_RECOVERY_CANCEL,
)
_CB_VOICE = (CB_VOICE,)
_CB_RESUME = (CB_RESUME_PICK, CB_RESUME_PAGE, CB_RESUME_CANCEL)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch callback queries to dedicated handler modules."""
    # CallbackQueryHandler doesn't support filters= param, so check inline.
    if config.group_id:
        chat = update.effective_chat
        if not chat or chat.id != config.group_id:
            return

    query = update.callback_query
    if not query or not query.data:
        return

    user = update.effective_user
    if not user or not is_user_allowed(user.id):
        await query.answer("Not authorized")
        return

    # Store group chat_id for forum topic message routing
    if query.message and query.message.chat.type in ("group", "supergroup"):
        cb_thread_id = _get_thread_id(update)
        if cb_thread_id is not None:
            session_manager.set_group_chat_id(
                user.id, cb_thread_id, query.message.chat.id
            )

    data = query.data

    # History pagination
    if data.startswith(_CB_HISTORY):
        await handle_history_callback(query, user.id, data, update, context)

    # Directory browser
    elif data.startswith(_CB_DIRECTORY):
        await handle_directory_callback(query, user.id, data, update, context)

    # Window picker
    elif data.startswith(_CB_WINDOW):
        await handle_window_callback(query, user.id, data, update, context)

    # Screenshot / status buttons / quick keys
    elif data.startswith(_CB_SCREENSHOT):
        await handle_screenshot_callback(query, user.id, data, update, context)

    # No-op
    elif data == "noop":
        await query.answer()

    # Interactive UI (AskUserQuestion / ExitPlanMode navigation)
    elif _match_interactive_prefix(data):
        await handle_interactive_callback(query, user.id, data, update, context)

    # Recovery UI
    elif data.startswith(_CB_RECOVERY):
        await handle_recovery_callback(query, user.id, data, update, context)

    # Resume command UI
    elif data.startswith(_CB_RESUME):
        await handle_resume_command_callback(query, user.id, data, update, context)

    # Sessions dashboard
    elif data == CB_SESSIONS_REFRESH:
        await handle_sessions_refresh(query, user.id)
        await query.answer("Refreshed")
    elif data == CB_SESSIONS_NEW:
        await query.answer("Create a new topic to start a session.")
    elif data.startswith(CB_SESSIONS_KILL_CONFIRM):
        window_id = data[len(CB_SESSIONS_KILL_CONFIRM) :]
        if not _user_owns_window(user.id, window_id):
            await query.answer("Not your session", show_alert=True)
            return
        await handle_sessions_kill_confirm(query, user.id, window_id, context.bot)
        await query.answer("Killed")
    elif data.startswith(CB_SESSIONS_KILL):
        window_id = data[len(CB_SESSIONS_KILL) :]
        if not _user_owns_window(user.id, window_id):
            await query.answer("Not your session", show_alert=True)
            return
        await handle_sessions_kill(query, user.id, window_id)
        await query.answer()

    # Voice message callbacks
    elif data.startswith(_CB_VOICE):
        await handle_voice_callback(update, context)

    # Sync command
    elif data == CB_SYNC_FIX:
        await handle_sync_fix(query)
        await query.answer("Fixed")
    elif data == CB_SYNC_DISMISS:
        await handle_sync_dismiss(query)
        await query.answer("Dismissed")


# --- Streaming response / notifications ---


async def handle_new_message(msg: NewMessage, bot: Bot) -> None:
    """Handle a new assistant message — enqueue for sequential processing.

    Messages are queued per-user to ensure status messages always appear last.
    Routes via thread_bindings to deliver to the correct topic.
    """
    status = "complete" if msg.is_complete else "streaming"
    logger.info(
        "handle_new_message [%s]: session=%s, text_len=%d",
        status,
        msg.session_id,
        len(msg.text),
    )

    # Find users whose thread-bound window matches this session
    active_users = session_manager.find_users_for_session(msg.session_id)

    if not active_users:
        logger.info("No active users for session %s", msg.session_id)
        return

    for user_id, window_id, thread_id in active_users:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            window_id=window_id, session_id=msg.session_id
        )
        # Check notification mode — skip suppressed messages.
        # All tool_use/tool_result MUST pass through regardless of mode: the message
        # queue edits tool_use messages in-place when tool_result arrives, so filtering
        # one half would break pairing and leave orphaned messages. This means muted/
        # errors_only sessions still deliver tool flow — an accepted trade-off.
        notif_mode = session_manager.get_notification_mode(window_id)
        is_tool_flow = msg.tool_name in INTERACTIVE_TOOL_NAMES or msg.content_type in (
            "tool_use",
            "tool_result",
        )
        if not is_tool_flow:
            if notif_mode == "muted":
                continue
            if notif_mode == "errors_only" and not _ERROR_KEYWORDS_RE.search(
                msg.text or ""
            ):
                continue

        # Handle interactive tools specially - capture terminal and send UI
        if msg.tool_name in INTERACTIVE_TOOL_NAMES and msg.content_type == "tool_use":
            # Mark interactive mode BEFORE sleeping so polling skips this window
            set_interactive_mode(user_id, window_id, thread_id)
            # Flush pending messages (e.g. plan content) before sending interactive UI
            queue = get_message_queue(user_id)
            if queue:
                await queue.join()
            # Wait briefly for Claude Code to render the question UI
            await asyncio.sleep(0.3)
            handled = await handle_interactive_ui(bot, user_id, window_id, thread_id)
            if handled:
                # Update user's read offset
                session = await session_manager.resolve_session_for_window(window_id)
                if session and session.file_path:
                    try:
                        file_size = Path(session.file_path).stat().st_size
                        session_manager.update_user_window_offset(
                            user_id, window_id, file_size
                        )
                    except OSError:
                        pass
                continue  # Don't send the normal tool_use message
            else:
                # UI not rendered — clear the early-set mode
                clear_interactive_mode(user_id, thread_id)

        # Any non-interactive message means the interaction is complete — delete the UI message
        if get_interactive_msg_id(user_id, thread_id):
            await clear_interactive_msg(user_id, bot, thread_id)

        parts = build_response_parts(
            msg.text,
            msg.is_complete,
            msg.content_type,
            msg.role,
        )

        if msg.is_complete:
            # Enqueue content message task
            # Note: tool_result editing is handled inside _process_content_task
            # to ensure sequential processing with tool_use message sending
            await enqueue_content_message(
                bot=bot,
                user_id=user_id,
                window_id=window_id,
                parts=parts,
                tool_use_id=msg.tool_use_id,
                content_type=msg.content_type,
                text=msg.text,
                thread_id=thread_id,
            )

            # Update user's read offset to current file position
            # This marks these messages as "read" for this user
            session = await session_manager.resolve_session_for_window(window_id)
            if session and session.file_path:
                try:
                    file_size = Path(session.file_path).stat().st_size
                    session_manager.update_user_window_offset(
                        user_id, window_id, file_size
                    )
                except OSError:
                    pass


# --- Auto-create topic for new tmux windows ---


async def _handle_new_window(event: NewWindowEvent, bot: Bot) -> None:
    """Create a Telegram forum topic for a newly detected tmux window.

    Skips if the window is already bound to a topic. Creates one topic per
    unique group chat, binds all users in that chat.
    """

    # Check if this window is already bound to any topic
    for _, _, bound_wid in session_manager.iter_thread_bindings():
        if bound_wid == event.window_id:
            logger.debug(
                "New window %s already bound, skipping topic creation", event.window_id
            )
            return

    # Auto-detect provider from the running process (only if not already set).
    # detect_provider_from_command returns "" for unrecognized commands (shells),
    # so we only persist when a known CLI is confidently identified.
    existing_provider = session_manager.get_window_state(event.window_id).provider_name
    if not existing_provider:
        w = await tmux_manager.find_window_by_id(event.window_id)
        if w and w.pane_current_command:
            detected = detect_provider_from_command(w.pane_current_command)
            if not detected and should_probe_pane_title_for_provider_detection(
                w.pane_current_command
            ):
                pane_title = await tmux_manager.get_pane_title(event.window_id)
                detected = detect_provider_from_runtime(
                    w.pane_current_command,
                    pane_title=pane_title,
                )
            if detected:
                session_manager.set_window_provider(event.window_id, detected)
                logger.info(
                    "Auto-detected provider %r for window %s (command=%s)",
                    detected,
                    event.window_id,
                    w.pane_current_command,
                )

    topic_name = event.window_name or Path(event.cwd).name or event.window_id

    # Collect unique chat_ids from existing bindings
    seen_chats: set[int] = set()
    for user_id, thread_id, _ in session_manager.iter_thread_bindings():
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        if chat_id != user_id:  # Only group chats (not fallback to user_id)
            seen_chats.add(chat_id)

    # Fallback: use preserved group_chat_ids (post-restart, no bindings left)
    if not seen_chats:
        seen_chats.update(
            cid for cid in session_manager.group_chat_ids.values() if cid < 0
        )

    if not seen_chats:
        if config.group_id:
            seen_chats.add(config.group_id)
            logger.info(
                "Cold-start: using CCBOT_GROUP_ID=%d for auto-topic (window %s)",
                config.group_id,
                event.window_id,
            )
        else:
            logger.debug(
                "No group chats found for auto-topic creation (window %s)",
                event.window_id,
            )
            return

    for chat_id in seen_chats:
        retry_until = _topic_create_retry_until.get(chat_id, 0.0)
        now = time.monotonic()
        if now < retry_until:
            wait_seconds = max(1, int(retry_until - now))
            logger.debug(
                "Skipping auto-topic creation for chat %d (window %s), "
                "backoff active for %ss",
                chat_id,
                event.window_id,
                wait_seconds,
            )
            continue

        try:
            topic = await bot.create_forum_topic(chat_id=chat_id, name=topic_name)
            _topic_create_retry_until.pop(chat_id, None)
            logger.info(
                "Auto-created topic '%s' (thread=%d) in chat %d for window %s",
                topic_name,
                topic.message_thread_id,
                chat_id,
                event.window_id,
            )
            # Bind one user to establish the route for this chat.
            # In cold-start (no existing bindings), use the first allowed user.
            bound = False
            for user_id, thread_id, _ in session_manager.iter_thread_bindings():
                if session_manager.resolve_chat_id(user_id, thread_id) == chat_id:
                    session_manager.bind_thread(
                        user_id,
                        topic.message_thread_id,
                        event.window_id,
                        window_name=topic_name,
                    )
                    session_manager.set_group_chat_id(
                        user_id, topic.message_thread_id, chat_id
                    )
                    bound = True
                    break
            if not bound and config.allowed_users:
                first_user_id = next(iter(config.allowed_users))
                session_manager.bind_thread(
                    first_user_id,
                    topic.message_thread_id,
                    event.window_id,
                    window_name=topic_name,
                )
                session_manager.set_group_chat_id(
                    first_user_id, topic.message_thread_id, chat_id
                )
        except RetryAfter as e:
            retry_after_seconds = (
                e.retry_after
                if isinstance(e.retry_after, int)
                else int(e.retry_after.total_seconds())
            )
            retry_after_seconds = max(1, retry_after_seconds)
            _topic_create_retry_until[chat_id] = (
                time.monotonic()
                + retry_after_seconds
                + _TOPIC_CREATE_RETRY_BUFFER_SECONDS
            )
            logger.warning(
                "Flood control creating topic for window %s in chat %d, "
                "backing off %ss",
                event.window_id,
                chat_id,
                retry_after_seconds,
            )
        except TelegramError:
            logger.exception(
                "Failed to create topic for window %s in chat %d",
                event.window_id,
                chat_id,
            )


# --- App lifecycle ---


async def _adopt_unbound_windows(bot: Bot) -> None:
    """Auto-adopt known-but-unbound windows (post-restart recovery)."""
    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]
    audit = session_manager.audit_state(live_ids, live_pairs)
    orphaned = [i for i in audit.issues if i.category == "orphaned_window"]
    if orphaned:
        from .handlers.sync_command import _adopt_orphaned_windows

        await _adopt_orphaned_windows(bot, orphaned)
        logger.info("Startup: adopted %d unbound window(s)", len(orphaned))


async def post_init(application: Application) -> None:
    global session_monitor, _status_poll_task, _global_provider_menu

    default_provider = get_provider()
    await register_commands(application.bot, provider=default_provider)
    _global_provider_menu = default_provider.capabilities.name

    # Refresh bot command menu every 10 minutes.
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

    # Re-resolve stale window IDs from persisted state against live tmux windows
    await session_manager.resolve_stale_ids()

    await _adopt_unbound_windows(application.bot)

    # Warn if Claude Code hooks are not installed (provider-aware, non-blocking)
    provider = get_provider()
    if provider.capabilities.supports_hook:
        from .hook import _claude_settings_file, get_installed_events

        settings_file = _claude_settings_file()
        import json

        if settings_file.exists():
            try:
                settings = json.loads(settings_file.read_text())
                events = get_installed_events(settings)
                missing = [e for e, ok in events.items() if not ok]
                if missing:
                    logger.warning(
                        "Claude Code hooks incomplete — %d missing: %s. "
                        "Run: ccgram hook --install",
                        len(missing),
                        ", ".join(missing),
                    )
            except (json.JSONDecodeError, OSError):  # fmt: skip
                logger.warning(
                    "Claude Code hooks not installed. Run: ccgram hook --install"
                )
        else:
            logger.warning(
                "Claude Code hooks not installed (%s missing). "
                "Run: ccgram hook --install",
                settings_file,
            )

    monitor = SessionMonitor()
    # Expose to other modules (status_polling activity heuristic)
    from ccgram.session_monitor import set_active_monitor

    set_active_monitor(monitor)

    async def message_callback(msg: NewMessage) -> None:
        await handle_new_message(msg, application.bot)

    monitor.set_message_callback(message_callback)

    async def new_window_callback(event: NewWindowEvent) -> None:
        await _handle_new_window(event, application.bot)

    monitor.set_new_window_callback(new_window_callback)

    # Wire hook event dispatcher for structured Claude Code events
    from ccgram.handlers.hook_events import HookEvent, dispatch_hook_event

    async def hook_event_callback(event: HookEvent) -> None:
        await dispatch_hook_event(event, application.bot)

    monitor.set_hook_event_callback(hook_event_callback)
    monitor.start()
    session_monitor = monitor
    logger.info("Session monitor started")

    # Start status polling task (routed through PTB error handler)
    _status_poll_task = asyncio.create_task(status_poll_loop(application.bot))
    _status_poll_task.add_done_callback(task_done_callback)
    logger.info("Status polling task started")


async def post_shutdown(_application: Application) -> None:
    global _status_poll_task

    # Stop status polling
    if _status_poll_task:
        _status_poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _status_poll_task
        _status_poll_task = None
        logger.info("Status polling stopped")

    # Stop all queue workers
    await shutdown_workers()

    if session_monitor:
        session_monitor.stop()
        logger.info("Session monitor stopped")

    # Flush debounced state to disk AFTER workers/monitor stop (captures final mutations)
    session_manager.flush_state()


async def _error_handler(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle bot-level errors from updater and handlers."""
    if isinstance(context.error, Conflict):
        logger.critical(
            "Another bot instance is polling with the same token. "
            "Shutting down to avoid conflicts."
        )
        os.kill(os.getpid(), signal.SIGINT)
        return
    logger.error("Unhandled bot error", exc_info=context.error)


def create_bot() -> Application:
    # Suppress PTBUserWarning about JobQueue (we intentionally don't use it for core tasks)
    import warnings

    warnings.filterwarnings("ignore", message=".*JobQueue.*", category=UserWarning)
    application = (
        Application.builder()
        .token(config.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_error_handler(_error_handler)
    application.add_handler(CommandHandler("new", new_command, filters=_group_filter))
    application.add_handler(
        CommandHandler("start", new_command, filters=_group_filter)  # compat alias
    )
    application.add_handler(
        CommandHandler("history", history_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("commands", commands_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("sessions", sessions_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("resume", resume_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("unbind", unbind_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("upgrade", upgrade_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("recall", recall_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("screenshot", screenshot_command, filters=_group_filter)
    )
    application.add_handler(
        CommandHandler("panes", panes_command, filters=_group_filter)
    )
    application.add_handler(CommandHandler("sync", sync_command, filters=_group_filter))
    application.add_handler(
        CommandHandler("restore", restore_command, filters=_group_filter)
    )
    application.add_handler(CallbackQueryHandler(callback_handler))
    # Topic closed event — unbind window (kept alive for rebinding)
    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_CLOSED & _group_filter,
            topic_closed_handler,
        )
    )
    # Topic renamed event — sync name to tmux window
    application.add_handler(
        MessageHandler(
            filters.StatusUpdate.FORUM_TOPIC_EDITED & _group_filter,
            topic_edited_handler,
        )
    )
    # Forward any other /command to the topic's provider CLI
    application.add_handler(
        MessageHandler(filters.COMMAND & _group_filter, forward_command_handler)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & _group_filter, text_handler)
    )
    # Photos
    application.add_handler(
        MessageHandler(filters.PHOTO & _group_filter, handle_photo_message)
    )
    # Documents
    application.add_handler(
        MessageHandler(filters.Document.ALL & _group_filter, handle_document_message)
    )
    # Voice messages (transcription when configured)
    application.add_handler(
        MessageHandler(filters.VOICE & _group_filter, handle_voice_message)
    )
    # Catch-all: unsupported content (stickers, voice, video, etc.)
    application.add_handler(
        MessageHandler(
            ~filters.COMMAND
            & ~filters.TEXT
            & ~filters.PHOTO
            & ~filters.Document.ALL
            & ~filters.VOICE
            & ~filters.StatusUpdate.ALL
            & _group_filter,
            unsupported_content_handler,
        )
    )
    # Inline query handler (serves switch_inline_query_current_chat from history buttons)
    application.add_handler(InlineQueryHandler(inline_query_handler))

    return application
