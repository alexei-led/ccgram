"""Tests for menu_sync — provider command menu cache + scoped registration."""

from typing import TYPE_CHECKING, cast
from types import SimpleNamespace

if TYPE_CHECKING:
    from ccgram.providers import AgentProvider
from unittest.mock import AsyncMock, patch

import pytest
from telegram import BotCommandScopeChat, BotCommandScopeChatMember

import ccgram.handlers.commands.menu_sync as menu_sync_mod
from ccgram.handlers.commands.menu_sync import (
    _build_provider_command_metadata,
    _chat_scoped_provider_menu,
    _scoped_provider_menu,
    get_global_provider_menu,
    set_global_provider_menu,
    sync_scoped_provider_menu as _sync_scoped_provider_menu,
)


_MS = "ccgram.handlers.commands.menu_sync"


@pytest.fixture(autouse=True)
def _allow_user():
    with patch("ccgram.config.Config.is_user_allowed", return_value=True):
        yield


@pytest.fixture(autouse=True)
def _clean_scoped_caches():
    _scoped_provider_menu.clear()
    _chat_scoped_provider_menu.clear()
    menu_sync_mod._global_provider_menu = None
    yield
    _scoped_provider_menu.clear()
    _chat_scoped_provider_menu.clear()
    menu_sync_mod._global_provider_menu = None


class TestBuildProviderCommandMetadata:
    def test_builds_telegram_to_native_mapping(self) -> None:
        provider = SimpleNamespace(
            capabilities=SimpleNamespace(name="codex", builtin_commands=("/builtin",))
        )
        discovered = [
            SimpleNamespace(name="/status", telegram_name="status"),
            SimpleNamespace(name="spec:work", telegram_name="spec_work"),
        ]

        with patch(f"{_MS}.discover_provider_commands", return_value=discovered):
            mapping = _build_provider_command_metadata(provider)  # type: ignore[arg-type]

        assert mapping == {"status": "/status", "spec_work": "spec:work"}


class TestScopedProviderMenuSync:
    async def test_caches_provider_menu_per_chat_user(self) -> None:
        message = AsyncMock()
        message.chat.id = -100999
        message.get_bot.return_value = object()
        provider = cast(
            "AgentProvider",
            SimpleNamespace(capabilities=SimpleNamespace(name="codex")),
        )

        with patch(f"{_MS}.register_commands", new_callable=AsyncMock) as mock_reg:
            await _sync_scoped_provider_menu(message, 100, provider)
            await _sync_scoped_provider_menu(message, 100, provider)

        mock_reg.assert_called_once()
        assert _scoped_provider_menu[(-100999, 100)] == "codex"

    async def test_cache_updates_when_provider_changes(self) -> None:
        message = AsyncMock()
        message.chat.id = -100999
        message.get_bot.return_value = object()
        codex = cast(
            "AgentProvider", SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        )
        claude = cast(
            "AgentProvider",
            SimpleNamespace(capabilities=SimpleNamespace(name="claude")),
        )

        with patch(f"{_MS}.register_commands", new_callable=AsyncMock) as mock_reg:
            await _sync_scoped_provider_menu(message, 100, codex)
            await _sync_scoped_provider_menu(message, 100, claude)

        assert mock_reg.call_count == 2
        assert _scoped_provider_menu[(-100999, 100)] == "claude"

    async def test_register_failure_does_not_update_cache(self) -> None:
        message = AsyncMock()
        message.chat.id = -100999
        message.get_bot.return_value = object()
        provider = cast(
            "AgentProvider",
            SimpleNamespace(capabilities=SimpleNamespace(name="codex")),
        )

        with patch(
            f"{_MS}.register_commands",
            new_callable=AsyncMock,
            side_effect=OSError("boom"),
        ):
            await _sync_scoped_provider_menu(message, 100, provider)

        assert (-100999, 100) not in _scoped_provider_menu

    async def test_falls_back_to_chat_scope_when_member_scope_fails(self) -> None:
        message = AsyncMock()
        message.chat.id = -100999
        message.get_bot.return_value = object()
        provider = cast(
            "AgentProvider",
            SimpleNamespace(capabilities=SimpleNamespace(name="codex")),
        )

        with patch(
            f"{_MS}.register_commands",
            new_callable=AsyncMock,
            side_effect=[OSError("member"), None],
        ) as mock_reg:
            await _sync_scoped_provider_menu(message, 100, provider)

        assert mock_reg.call_count == 2
        first_scope = mock_reg.call_args_list[0].kwargs["scope"]
        second_scope = mock_reg.call_args_list[1].kwargs["scope"]
        assert isinstance(first_scope, BotCommandScopeChatMember)
        assert isinstance(second_scope, BotCommandScopeChat)
        assert _chat_scoped_provider_menu[-100999] == "codex"
        assert _scoped_provider_menu[(-100999, 100)] == "codex"

    async def test_falls_back_to_global_when_member_and_chat_scope_fail(self) -> None:
        message = AsyncMock()
        message.chat.id = -100999
        message.get_bot.return_value = object()
        provider = cast(
            "AgentProvider",
            SimpleNamespace(capabilities=SimpleNamespace(name="codex")),
        )

        with patch(
            f"{_MS}.register_commands",
            new_callable=AsyncMock,
            side_effect=[OSError("member"), OSError("chat"), None],
        ) as mock_reg:
            await _sync_scoped_provider_menu(message, 100, provider)

        assert mock_reg.call_count == 3
        assert "scope" in mock_reg.call_args_list[0].kwargs
        assert "scope" in mock_reg.call_args_list[1].kwargs
        assert "scope" not in mock_reg.call_args_list[2].kwargs
        assert _scoped_provider_menu[(-100999, 100)] == "codex"

    async def test_scoped_menu_cache_is_bounded(self) -> None:
        message = AsyncMock()
        message.chat.id = -100999
        message.get_bot.return_value = object()
        provider = cast(
            "AgentProvider",
            SimpleNamespace(capabilities=SimpleNamespace(name="codex")),
        )

        with (
            patch(f"{_MS}._MAX_SCOPED_PROVIDER_MENU_ENTRIES", 1),
            patch(f"{_MS}.register_commands", new_callable=AsyncMock),
        ):
            await _sync_scoped_provider_menu(message, 100, provider)
            await _sync_scoped_provider_menu(message, 101, provider)

        assert len(_scoped_provider_menu) == 1


class TestMenuCacheInvalidation:
    async def test_menu_cache_invalidated_on_provider_change(self) -> None:
        set_global_provider_menu("old")
        try:
            message = AsyncMock()
            message.chat.id = -100
            message.get_bot.return_value = object()
            codex = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
            claude = SimpleNamespace(capabilities=SimpleNamespace(name="claude"))

            with patch(f"{_MS}.register_commands", new_callable=AsyncMock) as mock_reg:
                await _sync_scoped_provider_menu(message, 1, codex)  # type: ignore[arg-type]
                await _sync_scoped_provider_menu(message, 1, claude)  # type: ignore[arg-type]

            assert mock_reg.call_count == 2
            assert _scoped_provider_menu[(-100, 1)] == "claude"
        finally:
            set_global_provider_menu("claude")


class TestGlobalProviderMenu:
    def test_get_set_global_provider_menu(self) -> None:
        old = get_global_provider_menu()
        try:
            set_global_provider_menu("test-provider")
            assert get_global_provider_menu() == "test-provider"
        finally:
            if old is not None:
                set_global_provider_menu(old)
