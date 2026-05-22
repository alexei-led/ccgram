"""Provider command menu cache + scoped registration.

Handles per-user / per-chat / global Telegram command menus for the
active provider, plus the periodic refresh job. Owns the bounded LRU
caches that prevent unbounded growth across long-running deployments.

Core responsibilities:
  - sync_scoped_provider_menu(): keep the visible /-menu in sync with the
    topic's provider, falling back chat → global on permission errors
  - sync_scoped_menu_for_text_context(): same but triggered by inbound
    plain text in a bound topic
  - setup_menu_refresh_job(): periodic global menu refresh
  - _build_provider_command_metadata(): translate AgentProvider commands
    into a Telegram-name → original-name mapping used by forward.py
"""

from __future__ import annotations


from typing import TYPE_CHECKING
from collections import OrderedDict

import structlog
from telegram import (
    BotCommandScopeChat,
    BotCommandScopeChatMember,
    Message,
    Update,
)
from telegram.error import TelegramError

from ...cc_commands import discover_provider_commands, register_commands
from ...providers import (
    AgentProvider,
    get_provider,
    get_provider_for_window,
)
from ... import window_query
from ...thread_router import thread_router
from ..callback_helpers import get_thread_id as _get_thread_id

if TYPE_CHECKING:
    from telegram.ext import Application
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_CommandRefreshError = (TelegramError, OSError)

# --- Menu cache state ---

_scoped_provider_menu: OrderedDict[tuple[int, int], str] = OrderedDict()
_chat_scoped_provider_menu: OrderedDict[int, str] = OrderedDict()
_global_provider_menu: str | None = None
_MAX_SCOPED_PROVIDER_MENU_ENTRIES = 512
_MAX_CHAT_PROVIDER_MENU_ENTRIES = 256


# --- Bounded LRU helpers ---


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


# --- Provider command metadata ---


def _build_provider_command_metadata(
    provider: AgentProvider,
) -> dict[str, str]:
    """Map Telegram-friendly /-name (e.g. ``spec_work``) → provider native (``spec:work``)."""
    mapping: dict[str, str] = {}
    for cmd in discover_provider_commands(provider):
        if cmd.telegram_name and cmd.telegram_name not in mapping:
            mapping[cmd.telegram_name] = cmd.name
    return mapping


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
    provider = get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    )
    await sync_scoped_provider_menu(message, user_id, provider)


def get_global_provider_menu() -> str | None:
    """Return the current global provider menu name."""
    return _global_provider_menu


def set_global_provider_menu(provider_name: str) -> None:
    """Set the global provider menu name."""
    global _global_provider_menu
    _global_provider_menu = provider_name


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
                # Recoverable: the previous menu stays in place, so this is a
                # warning, not an ERROR-level exception.
                logger.warning(
                    "Failed to refresh CC commands, keeping previous menu",
                    exc_info=True,
                )

    jq = getattr(application, "job_queue", None)
    if jq is not None:
        jq.run_repeating(_refresh_commands, interval=600, first=600)
