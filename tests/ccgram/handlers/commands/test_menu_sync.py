"""Tests for menu_sync — provider command menu cache + scoped registration."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
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

if TYPE_CHECKING:
    from ccgram.providers import AgentProvider

_MS = "ccgram.handlers.commands.menu_sync"
_CHAT_ID = -100999
_USER_ID = 100


def _provider(name: str) -> "AgentProvider":
    return cast(
        "AgentProvider", SimpleNamespace(capabilities=SimpleNamespace(name=name))
    )


@pytest.fixture(autouse=True)
def _allow_user() -> Iterator[None]:
    with patch("ccgram.config.Config.is_user_allowed", return_value=True):
        yield


@pytest.fixture(autouse=True)
def _clean_scoped_caches() -> Iterator[None]:
    def _reset() -> None:
        _scoped_provider_menu.clear()
        _chat_scoped_provider_menu.clear()
        menu_sync_mod._global_provider_menu = None

    _reset()
    yield
    _reset()


@pytest.fixture
def message() -> AsyncMock:
    message = AsyncMock()
    message.chat.id = _CHAT_ID
    message.get_bot.return_value = object()
    return message


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
    async def test_caches_provider_menu_per_chat_user(self, message: AsyncMock) -> None:
        with patch(f"{_MS}.register_commands", new_callable=AsyncMock) as mock_reg:
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))

        mock_reg.assert_called_once()
        assert _scoped_provider_menu[(_CHAT_ID, _USER_ID)] == "codex"

    async def test_resyncs_when_provider_changes(self, message: AsyncMock) -> None:
        with patch(f"{_MS}.register_commands", new_callable=AsyncMock) as mock_reg:
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("claude"))

        assert mock_reg.call_count == 2
        assert _scoped_provider_menu[(_CHAT_ID, _USER_ID)] == "claude"

    async def test_register_failure_does_not_update_cache(
        self, message: AsyncMock
    ) -> None:
        with patch(
            f"{_MS}.register_commands",
            new_callable=AsyncMock,
            side_effect=OSError("boom"),
        ):
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))

        assert (_CHAT_ID, _USER_ID) not in _scoped_provider_menu

    async def test_falls_back_to_chat_scope_when_member_scope_fails(
        self, message: AsyncMock
    ) -> None:
        with patch(
            f"{_MS}.register_commands",
            new_callable=AsyncMock,
            side_effect=[OSError("member"), None],
        ) as mock_reg:
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))

        assert mock_reg.call_count == 2
        assert isinstance(
            mock_reg.call_args_list[0].kwargs["scope"], BotCommandScopeChatMember
        )
        assert isinstance(
            mock_reg.call_args_list[1].kwargs["scope"], BotCommandScopeChat
        )
        assert _chat_scoped_provider_menu[_CHAT_ID] == "codex"
        assert _scoped_provider_menu[(_CHAT_ID, _USER_ID)] == "codex"

    async def test_falls_back_to_global_when_both_scopes_fail(
        self, message: AsyncMock
    ) -> None:
        with patch(
            f"{_MS}.register_commands",
            new_callable=AsyncMock,
            side_effect=[OSError("member"), OSError("chat"), None],
        ) as mock_reg:
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))

        assert mock_reg.call_count == 3
        assert "scope" in mock_reg.call_args_list[0].kwargs
        assert "scope" in mock_reg.call_args_list[1].kwargs
        assert "scope" not in mock_reg.call_args_list[2].kwargs
        assert _scoped_provider_menu[(_CHAT_ID, _USER_ID)] == "codex"

    async def test_scoped_menu_cache_is_bounded(self, message: AsyncMock) -> None:
        with (
            patch(f"{_MS}._MAX_SCOPED_PROVIDER_MENU_ENTRIES", 1),
            patch(f"{_MS}.register_commands", new_callable=AsyncMock),
        ):
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))
            await _sync_scoped_provider_menu(message, _USER_ID + 1, _provider("codex"))

        assert len(_scoped_provider_menu) == 1


class TestGlobalProviderMenu:
    def test_round_trips_the_registered_menu(self) -> None:
        assert get_global_provider_menu() is None

        set_global_provider_menu("test-provider")

        assert get_global_provider_menu() == "test-provider"
