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
        assert _scoped_provider_menu[(_CHAT_ID, _USER_ID)] == "control"

    async def test_provider_changes_do_not_change_shared_control_menu(
        self, message: AsyncMock
    ) -> None:
        with patch(f"{_MS}.register_commands", new_callable=AsyncMock) as mock_reg:
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("claude"))

        mock_reg.assert_awaited_once()
        assert mock_reg.await_args is not None
        assert mock_reg.await_args.kwargs["include_cc_commands"] is False
        assert isinstance(
            mock_reg.await_args.kwargs["scope"], BotCommandScopeChatMember
        )
        assert _scoped_provider_menu[(_CHAT_ID, _USER_ID)] == "control"

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
        assert _chat_scoped_provider_menu[_CHAT_ID] == "control"
        assert _scoped_provider_menu[(_CHAT_ID, _USER_ID)] == "control"

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
        assert _scoped_provider_menu[(_CHAT_ID, _USER_ID)] == "control"

    async def test_scoped_menu_cache_is_bounded(self, message: AsyncMock) -> None:
        with (
            patch(f"{_MS}._MAX_SCOPED_PROVIDER_MENU_ENTRIES", 1),
            patch(f"{_MS}.register_commands", new_callable=AsyncMock),
        ):
            await _sync_scoped_provider_menu(message, _USER_ID, _provider("codex"))
            await _sync_scoped_provider_menu(message, _USER_ID + 1, _provider("codex"))

        assert len(_scoped_provider_menu) == 1


async def test_startup_replaces_legacy_global_chat_and_member_menus(monkeypatch):
    monkeypatch.setattr(menu_sync_mod.config, "allowed_users", {100, 200})
    monkeypatch.setattr(menu_sync_mod.config, "group_id", -100999)
    monkeypatch.setattr(
        menu_sync_mod.thread_router, "group_chat_ids", {"100:42": -100888}
    )
    monkeypatch.setattr(
        menu_sync_mod.thread_router,
        "iter_thread_bindings_with_chat",
        lambda: iter([(100, -100888, 42, "@1")]),
    )
    monkeypatch.setattr(
        menu_sync_mod.thread_router, "iter_private_topic_chat_ids", lambda: iter([100])
    )
    bot = AsyncMock()

    await menu_sync_mod.register_control_menus(bot)

    scopes = [call.kwargs.get("scope") for call in bot.set_my_commands.call_args_list]
    assert None in scopes
    assert BotCommandScopeChat(chat_id=100) in scopes
    assert BotCommandScopeChat(chat_id=-100999) in scopes
    assert BotCommandScopeChat(chat_id=-100888) in scopes
    assert BotCommandScopeChatMember(chat_id=-100888, user_id=100) in scopes
    assert BotCommandScopeChatMember(chat_id=-100888, user_id=200) in scopes
    assert all(
        [item.command for item in entry.args[0]]
        == ["commands", "sessions", "sync", "upgrade"]
        for entry in bot.set_my_commands.call_args_list
    )


class TestGlobalProviderMenu:
    def test_round_trips_the_registered_menu(self) -> None:
        assert get_global_provider_menu() is None

        set_global_provider_menu("test-provider")

        assert get_global_provider_menu() == "test-provider"
