"""Shared control-menu registration and provider-command metadata.

Telegram scopes belong to chats/users, not topics. All default and scoped
menus expose the same four controls; topic panels own session/agent commands.
The existing bounded caches avoid repeated per-message API calls.
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
from ...config import config
from ...providers import AgentProvider
from ...thread_router import thread_router

if TYPE_CHECKING:
    from telegram import Bot, BotCommandScope
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
    _provider: AgentProvider | None = None,
) -> None:
    """Keep the shared control commands visible in every known chat context."""
    global _global_provider_menu

    chat_id = message.chat.id
    provider_name = "control"
    cache_key = (chat_id, user_id)
    if _get_lru_cache_entry(_scoped_provider_menu, cache_key) == provider_name:
        return

    bot = message.get_bot()
    try:
        await register_commands(
            bot,
            include_cc_commands=False,
            scope=BotCommandScopeChatMember(chat_id=chat_id, user_id=user_id),
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
            "Failed to update member control menu (chat=%s user=%s)", chat_id, user_id
        )

    if _get_lru_cache_entry(_chat_scoped_provider_menu, chat_id) != provider_name:
        try:
            await register_commands(
                bot,
                include_cc_commands=False,
                scope=BotCommandScopeChat(chat_id=chat_id),
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
            logger.debug("Failed to update chat control menu (chat=%s)", chat_id)

    if _global_provider_menu == provider_name:
        _set_bounded_cache_entry(
            _scoped_provider_menu,
            cache_key,
            provider_name,
            max_entries=_MAX_SCOPED_PROVIDER_MENU_ENTRIES,
        )
        return
    try:
        await register_commands(bot, include_cc_commands=False)
        _global_provider_menu = provider_name
        _set_bounded_cache_entry(
            _scoped_provider_menu,
            cache_key,
            provider_name,
            max_entries=_MAX_SCOPED_PROVIDER_MENU_ENTRIES,
        )
    except _CommandRefreshError:
        logger.debug("Failed to update global control menu")


async def sync_scoped_menu_for_text_context(update: Update, user_id: int) -> None:
    """Keep the same control menu in General, session, and unbound topics."""
    message = update.message
    if not message:
        return
    await sync_scoped_provider_menu(message, user_id)


def get_global_provider_menu() -> str | None:
    """Return the current global provider menu name."""
    return _global_provider_menu


def set_global_provider_menu(provider_name: str) -> None:
    """Set the global provider menu name."""
    global _global_provider_menu
    _global_provider_menu = provider_name


async def register_control_menus(bot: Bot) -> None:
    """Overwrite known legacy scopes; changing the default alone cannot hide them."""
    chats = set(config.allowed_users)
    chats.update(thread_router.iter_private_topic_chat_ids())
    chats.update(
        chat_id for chat_id in thread_router.group_chat_ids.values() if chat_id < 0
    )
    chats.update(
        chat_id
        for _uid, chat_id, _tid, _wid in thread_router.iter_thread_bindings_with_chat()
        if chat_id is not None
    )
    if config.group_id:
        chats.add(config.group_id)
    scopes: list[BotCommandScope | None] = [None]
    for chat_id in sorted(chats):
        scopes.append(BotCommandScopeChat(chat_id=chat_id))
        if chat_id < 0:
            chat_users = config.allowed_users
            scopes.extend(
                BotCommandScopeChatMember(chat_id=chat_id, user_id=user_id)
                for user_id in sorted(chat_users)
            )
    for scope in scopes:
        try:
            await register_commands(bot, include_cc_commands=False, scope=scope)
        except _CommandRefreshError:
            logger.warning("Failed to refresh control command scope %s", scope)


def setup_menu_refresh_job(application: "Application") -> None:
    """Register the periodic command menu refresh job."""

    async def _refresh_commands(context: ContextTypes.DEFAULT_TYPE) -> None:
        if context.bot:
            try:
                await register_control_menus(context.bot)
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
